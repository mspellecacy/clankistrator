pub mod prompts;

use crate::context::{ProjectContext, ProjectTask, TaskStatus, AgentRole};
use crate::agents::{Worker, GenericWorker};
use anyhow::Result;
use rig::completion::Prompt;
use rig::prelude::CompletionClient;
use rig::providers::openai;
use std::env;
use std::sync::Arc;
use tokio::sync::Mutex;
use crate::orchestrator::prompts::{ORCHESTRATOR_SYSTEM_PROMPT, INFRASTRUCTURE_SYSTEM_PROMPT, BACKEND_SYSTEM_PROMPT, FRONTEND_SYSTEM_PROMPT, QA_SYSTEM_PROMPT};

pub struct Orchestrator {
    pub model: String,
    pub design_doc: String,
    pub use_tui: bool,
    pub max_steps: usize,
}

impl Orchestrator {
    pub fn new(model: String, design_doc: String, use_tui: bool, max_steps: usize) -> Self {
        Self { model, design_doc, use_tui, max_steps }
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

        let client = openai::Client::new(&openai_key)?;
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
        
        let response: String = agent.prompt(&prompt).await?;
        
        // Attempt to parse JSON from the response
        let json_str = if let Some(start) = response.find("```json") {
            let rest = &response[start + 7..];
            if let Some(end) = rest.find("```") {
                &rest[..end]
            } else {
                &response[start + 7..]
            }
        } else if let Some(start) = response.find('[') {
            let end = response.rfind(']').unwrap_or(response.len() - 1);
            &response[start..=end]
        } else {
            tracing::error!("LLM response did not contain a JSON list: {}", response);
            return Err(anyhow::anyhow!("LLM response did not contain a JSON list"));
        };

        let tasks: Vec<ProjectTask> = serde_json::from_str(json_str.trim())
            .map_err(|e| {
                tracing::error!("Failed to parse tasks JSON: {}. JSON: {}", e, json_str);
                anyhow::anyhow!("Failed to parse tasks JSON: {}", e)
            })?;
        
        if tasks.is_empty() {
            tracing::warn!("LLM returned an empty list of tasks for the design document.");
        } else {
            tracing::info!("Successfully parsed {} tasks from LLM response.", tasks.len());
        }
        
        Ok(tasks)
    }

    pub async fn run_iteration(&self, context: Arc<Mutex<ProjectContext>>) -> Result<()> {
        let ready_task_ids: Vec<String> = {
            let ctx = context.lock().await;
            ctx.get_ready_tasks().iter().map(|t| t.id.clone()).collect()
        };
        
        for task_id in ready_task_ids {
            // Mark as InProgress and get task info
            let (title, description, role, feedback, working_dir) = {
                let mut ctx = context.lock().await;
                tracing::info!("Orchestrator: Dispatching task {}", task_id);
                self.update_usage(&mut ctx);
                
                if let Some(t) = ctx.get_task_mut(&task_id) {
                    t.status = TaskStatus::InProgress("GenericWorker".to_string());
                    (t.title.clone(), t.description.clone(), t.assigned_role.clone(), t.feedback.clone(), ctx.working_directory.clone())
                } else {
                    continue;
                }
            };

            // Create a worker based on the role
            let worker = GenericWorker::new(
                role.clone(),
                self.model.clone(),
                self.get_system_prompt(&role),
                self.design_doc.clone(),
                if self.max_steps == 0 { 1000 } else { self.max_steps }
            );

            // Execute the task
            let temp_task = ProjectTask {
                id: task_id.clone(),
                title,
                description,
                dependencies: vec![],
                status: TaskStatus::Pending,
                assigned_role: role,
                feedback,
            };

            let result = worker.execute_task(&temp_task, &working_dir.to_string_lossy()).await?;
            
            // Update context with result
            {
                let mut ctx = context.lock().await;
                if let Some(t) = ctx.get_task_mut(&task_id) {
                    if result.status == "Completed" {
                        t.status = TaskStatus::ReadyForReview {
                            output_summary: result.summary,
                        };
                    } else {
                        tracing::warn!("Orchestrator: Task {} failed execution. Feedback: {}", task_id, result.summary);
                        t.status = TaskStatus::Pending;
                        t.feedback = Some(result.summary);
                    }
                }
            }
        }

        // 2. Review Phase
        let review_task_ids: Vec<String> = {
            let ctx = context.lock().await;
            ctx.get_tasks_for_review().iter().map(|t| t.id.clone()).collect()
        };
        for task_id in review_task_ids {
            let (title, description, summary) = {
                let mut ctx = context.lock().await;
                tracing::info!("Orchestrator: Dispatching task {} for review", task_id);
                self.update_usage(&mut ctx);

                if let Some(t) = ctx.get_task_mut(&task_id) {
                    if let TaskStatus::ReadyForReview { output_summary } = &t.status {
                        let summary = output_summary.clone();
                        t.status = TaskStatus::Reviewing("GenericQA".to_string());
                        (t.title.clone(), t.description.clone(), summary)
                    } else {
                        continue;
                    }
                } else {
                    continue;
                }
            };

            let qa_worker = GenericWorker::new(
                AgentRole::QA,
                self.model.clone(),
                self.get_system_prompt(&AgentRole::QA),
                self.design_doc.clone(),
                if self.max_steps == 0 { 1000 } else { self.max_steps }
            );

            // Execute review
            let prompt = format!(
                "Review the following task implementation:\n\
                Title: {}\n\
                Description: {}\n\
                Implementation Summary: {}\n\n\
                Does this implementation satisfy the requirements? Respond with 'PASSED' or provide constructive feedback.\n\
                If it satisfies requirements, your response MUST start with 'PASSED'.",
                title, description, summary
            );

            let openai_key = env::var("OPENAI_API_KEY").unwrap_or_else(|_| "stub".to_string());
            let review_result = if openai_key == "stub" {
                "PASSED".to_string()
            } else {
                let client = openai::Client::new(&openai_key)?;
                let agent = client.agent(&self.model)
                    .preamble(&qa_worker.system_prompt)
                    .build();
                let response: String = agent.prompt(&prompt).await?;
                response
            };

            {
                let mut ctx = context.lock().await;
                if let Some(t) = ctx.get_task_mut(&task_id) {
                    if review_result.to_uppercase().trim().starts_with("PASSED") {
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
        }

        // 3. Human in the Loop Phase
        let approval_task_ids: Vec<String> = {
            let ctx = context.lock().await;
            ctx.get_tasks_for_approval().iter().map(|t| t.id.clone()).collect()
        };
        for task_id in approval_task_ids {
            let (brave_mode, _title, _description, summary) = {
                let ctx = context.lock().await;
                if let Some(t) = ctx.get_task(&task_id) {
                    if let TaskStatus::RequiresHumanApproval { output_summary, .. } = &t.status {
                        (ctx.brave_mode, t.title.clone(), t.description.clone(), output_summary.clone())
                    } else {
                        continue;
                    }
                } else {
                    continue;
                }
            };

            if brave_mode {
                tracing::info!("Orchestrator: Brave Mode active. Automatically approving task {}", task_id);
                let mut ctx = context.lock().await;
                if let Some(t) = ctx.get_task_mut(&task_id) {
                    t.status = TaskStatus::Completed { output_summary: summary };
                }
                continue;
            }

            if self.use_tui {
                tracing::info!("Orchestrator: Waiting for human approval for task {} via TUI.", task_id);
                continue;
            }

            tracing::info!("Orchestrator: Requesting human approval for task {}", task_id);
            println!("\n--- HUMAN APPROVAL REQUIRED ---");
            println!("Task: {}", task_id);
            println!("Summary: {}", summary);
            println!("Approve? (y/n): ");

            let mut input = String::new();
            if std::io::stdin().read_line(&mut input).is_ok() && input.trim().to_lowercase() == "y" {
                let mut ctx = context.lock().await;
                if let Some(t) = ctx.get_task_mut(&task_id) {
                    t.status = TaskStatus::Completed { output_summary: summary };
                    tracing::info!("Orchestrator: Task {} approved by human.", task_id);
                }
            } else {
                println!("Provide feedback for the agent: ");
                let mut feedback = String::new();
                if std::io::stdin().read_line(&mut feedback).is_ok() {
                    let mut ctx = context.lock().await;
                    if let Some(t) = ctx.get_task_mut(&task_id) {
                        t.status = TaskStatus::Pending;
                        tracing::info!("Orchestrator: Task {} rejected by human. Feedback: {}", task_id, feedback.trim());
                        t.feedback = Some(feedback.trim().to_string());
                    }
                }
            }
        }
        
        Ok(())
    }

    fn update_usage(&self, context: &mut ProjectContext) {
        context.token_usage += 1500;
        context.total_cost += 0.03;
    }
}
