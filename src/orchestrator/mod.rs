pub mod prompts;

use crate::context::{ProjectContext, ProjectTask, TaskStatus, AgentRole};
use crate::agents::{Worker, GenericWorker};
use anyhow::Result;
use rig::completion::Prompt;
use rig::providers::openai;
use std::env;
use crate::orchestrator::prompts::ORCHESTRATOR_SYSTEM_PROMPT;

pub struct Orchestrator {
    pub model: String,
}

impl Orchestrator {
    pub fn new(model: String) -> Self {
        Self { model }
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
                },
                ProjectTask {
                    id: "TASK-2".into(),
                    title: "Implement Backend API".into(),
                    description: "Create a basic REST API using Axum or similar".into(),
                    dependencies: vec!["TASK-1".into()],
                    status: TaskStatus::Pending,
                    assigned_role: AgentRole::Backend,
                },
            ]);
        }

        let client = openai::Client::new(&openai_key);
        let agent = client.agent(&self.model)
            .preamble(ORCHESTRATOR_SYSTEM_PROMPT)
            .build();
            
        let prompt = format!(
            "Analyze the following design document and decompose it into a JSON list of ProjectTasks.\n\
            Each task must have: id, title, description, dependencies (list of IDs), and assigned_role (Infrastructure, Backend, Frontend, or QA).\n\
            Design Document:\n{}", 
            design
        );
        
        let response = agent.prompt(&prompt).await?;
        
        // In a real implementation, we would parse JSON from the response.
        // For this step, we'll assume the LLM might fail or return messy text,
        // and for the sake of the exercise, we'll return a stub or attempt parsing if it looks like JSON.
        
        // Stub implementation for now to ensure it works
        Ok(vec![
            ProjectTask {
                id: "TASK-1".into(),
                title: "Initialize Project Structure".into(),
                description: "Set up the basic folder structure and main.rs".into(),
                dependencies: vec![],
                status: TaskStatus::Pending,
                assigned_role: AgentRole::Infrastructure,
            },
        ])
    }

    pub async fn run_iteration(&self, context: &mut ProjectContext) -> Result<()> {
        let ready_task_ids: Vec<String> = context.get_ready_tasks().iter().map(|t| t.id.clone()).collect();
        
        for task_id in ready_task_ids {
            println!("Orchestrator: Dispatching task {}", task_id);
            
            // Mark as InProgress and get task info
            let (title, description, role) = if let Some(t) = context.get_task_mut(&task_id) {
                t.status = TaskStatus::InProgress("GenericWorker".to_string());
                (t.title.clone(), t.description.clone(), t.assigned_role.clone())
            } else {
                continue;
            };

            // Create a worker based on the role
            let worker = GenericWorker::new(
                role.clone(),
                self.model.clone(),
                format!("You are a specialized {:?} agent.", role)
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
            };

            let result = worker.execute_task(&temp_task).await?;
            
            // Update context with result
            if let Some(t) = context.get_task_mut(&task_id) {
                t.status = TaskStatus::Completed {
                    output_summary: result.summary,
                };
            }
        }
        
        Ok(())
    }
}
