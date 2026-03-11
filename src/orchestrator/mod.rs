pub mod prompts;

use crate::context::{ProjectContext, ProjectTask, TaskStatus, AgentRole};
use crate::agents::{Worker, GenericWorker};
use anyhow::Result;
use rig::completion::Prompt;
use rig::providers::openai;
use std::env;
use crate::orchestrator::prompts::{ORCHESTRATOR_SYSTEM_PROMPT, INFRASTRUCTURE_SYSTEM_PROMPT, BACKEND_SYSTEM_PROMPT, FRONTEND_SYSTEM_PROMPT, QA_SYSTEM_PROMPT};
use crate::tools::SpawnAgent;

pub struct Orchestrator {
    pub model: String,
    pub design_doc: String,
}

impl Orchestrator {
    pub fn new(model: String, design_doc: String) -> Self {
        Self { model, design_doc }
    }

    fn get_system_prompt(&self, role: &AgentRole) -> String {
        match role {
            AgentRole::Infrastructure => INFRASTRUCTURE_SYSTEM_PROMPT.to_string(),
            AgentRole::Backend => BACKEND_SYSTEM_PROMPT.to_string(),
            AgentRole::Frontend => FRONTEND_SYSTEM_PROMPT.to_string(),
            AgentRole::QA => QA_SYSTEM_PROMPT.to_string(),
            AgentRole::Orchestrator => ORCHESTRATOR_SYSTEM_PROMPT.to_string(),
        }
    }

    /// Analyze a design document and generate an initial list of tasks
    pub async fn bootstrap_project(&self, design: &str) -> Result<Vec<ProjectTask>> {
        let openai_key = env::var("OPENAI_API_KEY").unwrap_or_else(|_| "stub".to_string());
        
        if openai_key == "stub" {
            // Return a simple list of tasks if no key is present
            return Ok(vec![
                ProjectTask {
                    id: "TASK-1".into(),
                    title: "Initialize Project Structure".into(),
                    description: "Set up the basic folder structure and main.rs".into(),
                    dependencies: vec![],
                    status: TaskStatus::Pending,
                    assigned_role: AgentRole::Infrastructure,
                    feedback: None,
                },
                ProjectTask {
                    id: "TASK-2".into(),
                    title: "Implement Backend API".into(),
                    description: "Create a basic REST API using Axum or similar".into(),
                    dependencies: vec!["TASK-1".into()],
                    status: TaskStatus::Pending,
                    assigned_role: AgentRole::Backend,
                    feedback: None,
                },
            ]);
        }

        let client = openai::Client::new(&openai_key);
        let agent = client.agent(&self.model)
            .preamble(ORCHESTRATOR_SYSTEM_PROMPT)
            .build();
            
        let prompt = format!(
            "Analyze the following design document and decompose it into a JSON list of ProjectTasks.\n\
            Each task must have:\n\
            - id: String (e.g., TASK-1)\n\
            - title: String\n\
            - description: String\n\
            - dependencies: Vec<String> (list of IDs it depends on)\n\
            - status: \"Pending\"\n\
            - assigned_role: One of \"Infrastructure\", \"Backend\", \"Frontend\", \"QA\"\n\n\
            Output ONLY raw JSON. No markdown blocks.\n\n\
            Design Document:\n{}", 
            design
        );
        
        let response = agent.prompt(&prompt).await?;
        
        // Attempt to parse JSON from the response
        let tasks: Vec<ProjectTask> = if let Some(start) = response.find('[') {
            let end = response.rfind(']').unwrap_or(response.len() - 1);
            serde_json::from_str(&response[start..=end])?
        } else {
            return Err(anyhow::anyhow!("LLM response did not contain a JSON list: {}", response));
        };
        
        Ok(tasks)
    }

    pub async fn run_iteration(&self, context: &mut ProjectContext) -> Result<()> {
        let ready_task_ids: Vec<String> = context.get_ready_tasks().iter().map(|t| t.id.clone()).collect();
        
        for task_id in ready_task_ids {
            tracing::info!("Orchestrator: Dispatching task {}", task_id);
            
            // Mark as InProgress and get task info
            let (title, description, role, feedback) = if let Some(t) = context.get_task_mut(&task_id) {
                t.status = TaskStatus::InProgress("GenericWorker".to_string());
                (t.title.clone(), t.description.clone(), t.assigned_role.clone(), t.feedback.clone())
            } else {
                continue;
            };

            // Create a worker based on the role
            let worker = GenericWorker::new(
                role.clone(),
                self.model.clone(),
                self.get_system_prompt(&role),
                self.design_doc.clone()
            );

            // Execute the task - We need a temporary ProjectTask to pass to execute_task
            // since we can't pass the one in context while we have a mut reference to context
            // or we just pass the info we have.
            // Let's create a temporary task for the worker.
            let temp_task = ProjectTask {
                id: task_id.clone(),
                title,
                description,
                dependencies: vec![],
                status: TaskStatus::Pending,
                assigned_role: role,
                feedback,
            };

            let result = worker.execute_task(&temp_task, &context.working_directory.to_string_lossy()).await?;
            
            // Update context with result
            if let Some(t) = context.get_task_mut(&task_id) {
                t.status = TaskStatus::ReadyForReview {
                    output_summary: result.summary,
                };
            }
        }

        // 2. Review Phase
        let review_task_ids: Vec<String> = context.get_tasks_for_review().iter().map(|t| t.id.clone()).collect();
        for task_id in review_task_ids {
            tracing::info!("Orchestrator: Dispatching task {} for review", task_id);

            let (title, description, summary) = if let Some(t) = context.get_task_mut(&task_id) {
                if let TaskStatus::ReadyForReview { output_summary } = &t.status {
                    let summary = output_summary.clone();
                    t.status = TaskStatus::Reviewing("GenericQA".to_string());
                    (t.title.clone(), t.description.clone(), summary)
                } else {
                    continue;
                }
            } else {
                continue;
            };

            let qa_worker = GenericWorker::new(
                AgentRole::QA,
                self.model.clone(),
                "You are a specialized QA Reviewer. Your job is to critique the work of other agents. \
                Respond with 'PASSED' if the work is satisfactory, or provide constructive feedback if it is not.".to_string(),
                self.design_doc.clone()
            );

            // Execute review
            let prompt = format!(
                "Review the following task implementation:\n\
                Title: {}\n\
                Description: {}\n\
                Implementation Summary: {}\n\n\
                Does this implementation satisfy the requirements? Respond with 'PASSED' or feedback.",
                title, description, summary
            );

            let openai_key = env::var("OPENAI_API_KEY").unwrap_or_else(|_| "stub".to_string());
            let review_result = if openai_key == "stub" {
                "PASSED".to_string()
            } else {
                let client = openai::Client::new(&openai_key);
                let agent = client.agent(&self.model)
                    .preamble(&qa_worker.system_prompt)
                    .build();
                agent.prompt(&prompt).await?
            };

            if let Some(t) = context.get_task_mut(&task_id) {
                if review_result.contains("PASSED") {
                    tracing::info!("Orchestrator: Task {} passed review. Moving to human approval.", task_id);
                    t.status = TaskStatus::RequiresHumanApproval {
                        output_summary: summary,
                        feedback: None,
                    };
                    t.feedback = None;
                } else {
                    tracing::warn!("Orchestrator: Task {} failed review. Feedback: {}", task_id, review_result);
                    t.status = TaskStatus::Pending;
                    t.feedback = Some(review_result);
                }
            }
        }

        // 3. Human in the Loop Phase
        let approval_task_ids: Vec<String> = context.get_tasks_for_approval().iter().map(|t| t.id.clone()).collect();
        for task_id in approval_task_ids {
            tracing::info!("Orchestrator: Waiting for human approval for task {}", task_id);

            let summary = if let Some(t) = context.get_task(&task_id) {
                if let TaskStatus::RequiresHumanApproval { output_summary, .. } = &t.status {
                    output_summary.clone()
                } else {
                    continue;
                }
            } else {
                continue;
            };

            println!("\n--- HUMAN APPROVAL REQUIRED ---");
            println!("Task: {}", task_id);
            println!("Summary: {}", summary);
            println!("Approve? (y/n): ");

            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;

            if input.trim().to_lowercase() == "y" {
                if let Some(t) = context.get_task_mut(&task_id) {
                    t.status = TaskStatus::Completed {
                        output_summary: summary,
                    };
                    tracing::info!("Orchestrator: Task {} approved by human.", task_id);
                }
            } else {
                println!("Provide feedback for the agent: ");
                let mut feedback = String::new();
                std::io::stdin().read_line(&mut feedback)?;
                
                if let Some(t) = context.get_task_mut(&task_id) {
                    t.status = TaskStatus::Pending;
                    tracing::info!("Orchestrator: Task {} rejected by human. Feedback: {}", task_id, feedback.trim());
                    t.feedback = Some(feedback.trim().to_string());
                }
            }
        }
        
        Ok(())
    }
}
