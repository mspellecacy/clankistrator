pub mod prompts;

use crate::context::{ProjectContext, ProjectTask, TaskStatus, AgentRole, TaskResult};
use crate::agents::{Worker, GenericWorker};
use anyhow::Result;
use rig::completion::Prompt;
use rig::prelude::CompletionClient;
use rig::providers::openai;
use std::env;
use std::sync::Arc;
use tokio::sync::{Mutex, Semaphore};
use tokio::task::JoinSet;
use crate::orchestrator::prompts::{ORCHESTRATOR_SYSTEM_PROMPT, INFRASTRUCTURE_SYSTEM_PROMPT, BACKEND_SYSTEM_PROMPT, FRONTEND_SYSTEM_PROMPT, QA_SYSTEM_PROMPT, CRITIC_SYSTEM_PROMPT};

const MAX_CONCURRENT_REQUESTS: usize = 4;

pub struct Orchestrator {
    pub model: String,
    pub design_doc: String,
    pub use_tui: bool,
    pub max_steps: usize,
    semaphore: Arc<Semaphore>,
}

impl Orchestrator {
    pub fn new(model: String, design_doc: String, use_tui: bool, max_steps: usize) -> Self {
        Self {
            model,
            design_doc,
            use_tui,
            max_steps,
            semaphore: Arc::new(Semaphore::new(MAX_CONCURRENT_REQUESTS)),
        }
    }

    fn get_system_prompt(&self, role: &AgentRole) -> String {
        match role {
            AgentRole::Infrastructure => INFRASTRUCTURE_SYSTEM_PROMPT.to_string(),
            AgentRole::Backend => BACKEND_SYSTEM_PROMPT.to_string(),
            AgentRole::Frontend => FRONTEND_SYSTEM_PROMPT.to_string(),
            AgentRole::QA => QA_SYSTEM_PROMPT.to_string(),
            AgentRole::Orchestrator => ORCHESTRATOR_SYSTEM_PROMPT.to_string(),
            AgentRole::Critic => CRITIC_SYSTEM_PROMPT.to_string(),
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
                    retries: 0,
                },
                ProjectTask {
                    id: "TASK-2".into(),
                    title: "Implement Backend API".into(),
                    description: "Create a basic REST API using Axum or similar".into(),
                    dependencies: vec!["TASK-1".into()],
                    status: TaskStatus::Pending,
                    assigned_role: AgentRole::Backend,
                    feedback: None,
                    retries: 0,
                },
            ]);
        }

        let client = openai::Client::new(&openai_key)?;
        let agent = client.agent(&self.model)
            .preamble(ORCHESTRATOR_SYSTEM_PROMPT)
            .build();
            
        let prompt = format!(
            "Analyze the following design document and decompose it into a small JSON list of high-level ProjectTasks (MAXIMUM 3-5 tasks total).\n\
            Do NOT overcomplicate or create micro-tasks.\n\
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
        
        let response: String = agent.prompt(&prompt)
            .max_turns(if self.max_steps == 0 { 1000 } else { self.max_steps })
            .await?;
        
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
        let (ready_task_ids, model) = {
            let ctx = context.lock().await;
            (
                ctx.get_ready_tasks().iter().map(|t| t.id.clone()).collect::<Vec<String>>(),
                ctx.model.clone(),
            )
        };
        
        let (tasks_for_review, tasks_for_approval) = {
            let ctx = context.lock().await;
            (
                ctx.get_tasks_for_review().iter().map(|t| t.id.clone()).collect::<Vec<String>>(),
                ctx.get_tasks_for_approval().iter().map(|t| t.id.clone()).collect::<Vec<String>>(),
            )
        };

        if ready_task_ids.is_empty() && tasks_for_review.is_empty() && tasks_for_approval.is_empty() {
            // Check if there are any pending tasks that are blocked by failed tasks
            let ctx = context.lock().await;
            let has_pending = ctx.tasks.iter().any(|t| matches!(t.status, TaskStatus::Pending));
            if has_pending {
                tracing::warn!("Orchestrator: No ready tasks but pending tasks exist. They might be blocked by failed dependencies.");
            }
            return Ok(());
        }
        let mut execution_set = JoinSet::new();

        for task_id in ready_task_ids {
            // Mark as InProgress and get task info
            let (title, description, role, feedback, working_dir) = {
                let mut ctx = context.lock().await;
                tracing::info!("Orchestrator [{}]: Dispatching task", task_id);
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
                model.clone(),
                self.get_system_prompt(&role),
                self.design_doc.clone(),
                10 // Limit each task to a maximum of 10 turns to prevent endless loops
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
                retries: 0, // In this temporary copy, retries don't matter much
            };

            let semaphore = Arc::clone(&self.semaphore);
            let working_dir_str = working_dir.to_string_lossy().to_string();
            execution_set.spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                let res = worker.execute_task(&temp_task, &working_dir_str).await;
                (task_id, res)
            });
        }

        while let Some(res) = execution_set.join_next().await {
            let (task_id, result) = res.unwrap();
            let result = match result {
                Ok(r) => r,
                Err(e) => {
                    tracing::error!("Orchestrator [{}]: Worker execution error: {}", task_id, e);
                    TaskResult {
                        task_id: task_id.clone(),
                        status: "Failed".to_string(),
                        summary: format!("Internal worker error: {}", e),
                        files_created: vec![],
                        feedback: Some(format!("The worker encountered an internal error: {}", e)),
                        new_tasks: None,
                    }
                }
            };
            
            // Update context with result
            {
                let mut ctx = context.lock().await;
                if let Some(t) = ctx.get_task_mut(&task_id) {
                    if result.status == "Completed" {
                        t.status = TaskStatus::ReadyForReview {
                            output_summary: result.summary,
                        };
                        tracing::info!("Orchestrator [{}]: Task execution completed, ready for review", task_id);
                    } else {
                        t.retries += 1;
                        if t.retries > 3 {
                            tracing::error!("Orchestrator [{}]: Task failed execution too many times. Marking as Failed permanently.", task_id);
                            t.status = TaskStatus::Failed { error: result.summary.clone() };
                        } else {
                            tracing::warn!("Orchestrator [{}]: Task failed execution. Feedback: {}", task_id, result.summary);
                            t.status = TaskStatus::Pending;
                            t.feedback = Some(result.summary);
                        }
                    }
                }
                if result.status == "Completed" {
                    if let Some(new_tasks) = result.new_tasks {
                        for nt in new_tasks {
                            let new_id = format!("TASK-{}", ctx.tasks.len() + 1);
                            tracing::info!("Orchestrator [{}]: Worker injected new task {}", task_id, new_id);
                            let injected = ProjectTask {
                                id: new_id,
                                title: nt.title,
                                description: nt.description,
                                dependencies: nt.dependencies,
                                status: TaskStatus::Pending,
                                assigned_role: nt.assigned_role,
                                feedback: None,
                                retries: 0,
                            };
                            ctx.tasks.push(injected);
                        }
                    }
                }
            }
        }

        // 2. Review Phase
        let review_task_ids = tasks_for_review;
        
        let mut review_set = JoinSet::new();

        for task_id in review_task_ids {
            let (title, description, summary) = {
                let mut ctx = context.lock().await;
                tracing::info!("Orchestrator [{}]: Dispatching task for review", task_id);
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

            let semaphore = Arc::clone(&self.semaphore);
            let design_doc = self.design_doc.clone();
            let qa_system_prompt = self.get_system_prompt(&AgentRole::QA);
            let model_clone = model.clone();

            review_set.spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                
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
                    let client = openai::Client::new(&openai_key).map_err(|e| anyhow::anyhow!(e))?;
                    let agent = client.agent(&model_clone)
                        .preamble(&qa_system_prompt)
                        .tool(crate::tools::QueryDesignDoc { design_content: design_doc })
                        .build();
                    
                    match agent.prompt(&prompt)
                        .max_turns(5)
                        .await 
                    {
                        Ok(res) => res,
                        Err(e) => {
                            let err_str = e.to_string();
                            if err_str.contains("max turn order reached") || err_str.contains("turn limit") {
                                format!("FAILED: QA Agent reached turn limit (5). The implementation might be too complex.")
                            } else {
                                return Err(anyhow::anyhow!(e));
                            }
                        }
                    }
                };
                
                Ok::<(String, String, String), anyhow::Error>((task_id, review_result, summary))
            });
        }

        while let Some(res) = review_set.join_next().await {
            let (task_id, review_result, summary) = res.unwrap()?;
            
            {
                let mut ctx = context.lock().await;
                if let Some(t) = ctx.get_task_mut(&task_id) {
                    if review_result.to_uppercase().trim().starts_with("PASSED") {
                        tracing::info!("Orchestrator [{}]: Task passed review. Moving to human approval.", task_id);
                        t.status = TaskStatus::RequiresHumanApproval {
                            output_summary: summary,
                            feedback: None,
                        };
                        t.feedback = None;
                        t.retries = 0;
                    } else {
                        t.retries += 1;
                        if t.retries > 3 {
                            tracing::error!("Orchestrator [{}]: Task failed review too many times. Marking as Failed.", task_id);
                            t.status = TaskStatus::Failed { error: review_result.clone() };
                        } else {
                            tracing::warn!("Orchestrator [{}]: Task failed review. Feedback: {}", task_id, review_result);
                            t.status = TaskStatus::Pending;
                            t.feedback = Some(review_result);
                        }
                    }
                }
            }
        }

        // 3. Human in the Loop Phase
        let (brave_mode, approval_task_ids) = {
            let ctx = context.lock().await;
            (ctx.brave_mode, tasks_for_approval)
        };

        if brave_mode && !approval_task_ids.is_empty() {
            let mut ctx = context.lock().await;
            for task_id in approval_task_ids {
                if let Some(t) = ctx.get_task_mut(&task_id) {
                    if let TaskStatus::RequiresHumanApproval { output_summary, .. } = &t.status {
                        tracing::info!("Orchestrator [{}]: Brave Mode auto-approving task", task_id);
                        t.status = TaskStatus::Completed { output_summary: output_summary.clone() };
                    }
                }
            }
            return Ok(());
        }

        for task_id in approval_task_ids {
            let (_title, _description, summary) = {
                let ctx = context.lock().await;
                if let Some(t) = ctx.get_task(&task_id) {
                    if let TaskStatus::RequiresHumanApproval { output_summary, .. } = &t.status {
                        (t.title.clone(), t.description.clone(), output_summary.clone())
                    } else {
                        continue;
                    }
                } else {
                    continue;
                }
            };

            if self.use_tui {
                tracing::info!("Orchestrator [{}]: Waiting for human approval via TUI.", task_id);
                continue;
            }

            tracing::info!("Orchestrator [{}]: Requesting human approval", task_id);
            println!("\n--- HUMAN APPROVAL REQUIRED ---");
            println!("Task: {}", task_id);
            println!("Summary: {}", summary);
            println!("Approve? (y/n): ");

            let mut input = String::new();
            if std::io::stdin().read_line(&mut input).is_ok() && input.trim().to_lowercase() == "y" {
                let mut ctx = context.lock().await;
                if let Some(t) = ctx.get_task_mut(&task_id) {
                    t.status = TaskStatus::Completed { output_summary: summary };
                    tracing::info!("Orchestrator [{}]: Task approved by human.", task_id);
                }
            } else {
                println!("Provide feedback for the agent: ");
                let mut feedback = String::new();
                if std::io::stdin().read_line(&mut feedback).is_ok() {
                    let mut ctx = context.lock().await;
                    if let Some(t) = ctx.get_task_mut(&task_id) {
                        t.status = TaskStatus::Pending;
                        tracing::info!("Orchestrator [{}]: Task rejected by human. Feedback: {}", task_id, feedback.trim());
                        t.feedback = Some(feedback.trim().to_string());
                    }
                }
            }
        }

        // 4. Critic Evaluation Phase (Meta-Agency)
        // Let Critic review tasks that have failed multiple times or analyze overall progress.
        let critic_tasks: Vec<_> = {
            let ctx = context.lock().await;
            ctx.tasks.iter()
                .filter(|t| matches!(t.status, TaskStatus::Failed { .. }) || t.retries > 1)
                .cloned()
                .collect()
        };

        if !critic_tasks.is_empty() {
            tracing::info!("Orchestrator: Engaging Critic for {} struggling tasks...", critic_tasks.len());
            let openai_key = env::var("OPENAI_API_KEY").unwrap_or_else(|_| "stub".to_string());
            if openai_key != "stub" {
                let client = openai::Client::new(&openai_key)?;
                
                let preamble = format!(
                    "{}\n\
                    You are working in the directory: {}.\n\
                    The following tasks are struggling:\n\
                    {:?}\n\
                    You can use the ExecuteCommand tool to write and run scripts to help them.",
                    CRITIC_SYSTEM_PROMPT,
                    context.lock().await.working_directory.to_string_lossy(),
                    critic_tasks
                );
                
                let agent = client.agent(&self.model)
                    .preamble(&preamble)
                    .tool(crate::tools::ExecuteCommand)
                    .tool(crate::tools::WriteFile)
                    .build();

                let prompt = "Analyze the struggling tasks and provide your JSON output.";
                match agent.prompt(prompt).max_turns(3).await {
                    Ok(response) => {
                        tracing::info!("Critic output: {}", response);
                        // In a full implementation, we'd parse this JSON,
                        // apply `suggested_prompt_updates`, etc.
                    }
                    Err(e) => {
                        tracing::warn!("Critic failed: {}", e);
                    }
                }
            }
        }
        
        Ok(())
    }

    fn update_usage(&self, context: &mut ProjectContext) {
        // Estimate token usage and cost based on the model.
        // These are rough estimates and should be replaced with actual usage data from the LLM provider if available.
        let (tokens, cost) = match self.model.as_str() {
            "gpt-4o" => (1000, 0.01),
            "gpt-4o-mini" => (1000, 0.00015),
            "o1" | "o1-preview" => (1000, 0.015),
            _ => (1000, 0.01), // Default estimate
        };
        context.token_usage += tokens;
        context.total_cost += cost;
    }
}
