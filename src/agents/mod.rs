use crate::context::{ProjectTask, TaskResult, AgentRole};
use anyhow::Result;
use rig::completion::Prompt;
use rig::providers::openai;
use std::env;
use crate::tools::{ReadFile, WriteFile, ListFiles, ExecuteCommand};

pub trait Worker {
    async fn execute_task(&self, task: &ProjectTask) -> Result<TaskResult>;
}

pub struct GenericWorker {
    pub role: AgentRole,
    pub model: String,
    pub system_prompt: String,
}

impl GenericWorker {
    pub fn new(role: AgentRole, model: String, system_prompt: String) -> Self {
        Self {
            role,
            model,
            system_prompt,
        }
    }
}

impl Worker for GenericWorker {
    async fn execute_task(&self, task: &ProjectTask) -> Result<TaskResult> {
        let openai_key = env::var("OPENAI_API_KEY").unwrap_or_else(|_| "stub".to_string());
        
        if openai_key == "stub" {
            return Ok(TaskResult {
                task_id: task.id.clone(),
                status: "Completed".to_string(),
                summary: format!("STUB implementation for {} by GenericWorker ({:?})", task.title, self.role),
                files_created: vec![],
                feedback: Some("No OPENAI_API_KEY found, used stub.".to_string()),
            });
        }

        let client = openai::Client::new(&openai_key);
        let agent = client.agent(&self.model)
            .preamble(&self.system_prompt)
            .tool(ReadFile)
            .tool(WriteFile)
            .tool(ListFiles)
            .tool(ExecuteCommand)
            .build();
            
        let prompt = format!("Execute the following task:\nTitle: {}\nDescription: {}", task.title, task.description);
        let response = agent.prompt(&prompt).await?;

        // Simple parsing of the response to create a TaskResult
        // For now, assume the whole response is the summary
        Ok(TaskResult {
            task_id: task.id.clone(),
            status: "Completed".to_string(),
            summary: response,
            files_created: vec![],
            feedback: None,
        })
    }
}
