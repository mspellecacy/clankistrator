use crate::context::{ProjectTask, TaskResult, AgentRole};
use anyhow::Result;
use rig::completion::Prompt;
use rig::providers::openai;
use std::env;
use crate::tools::{ReadFile, WriteFile, ListFiles, ExecuteCommand, QueryDesignDoc};

pub trait Worker {
    async fn execute_task(&self, task: &ProjectTask, working_directory: &str) -> Result<TaskResult>;
}

pub struct GenericWorker {
    pub role: AgentRole,
    pub model: String,
    pub system_prompt: String,
    pub design_doc: String,
}

impl GenericWorker {
    pub fn new(role: AgentRole, model: String, system_prompt: String, design_doc: String) -> Self {
        Self {
            role,
            model,
            system_prompt,
            design_doc,
        }
    }
}

impl Worker for GenericWorker {
    async fn execute_task(&self, task: &ProjectTask, working_directory: &str) -> Result<TaskResult> {
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
        let preamble = format!(
            "{}\n\
            You are currently working in the directory: {}.\n\
            You MUST use the provided tools (write_file, read_file, etc.) to implement the task.\n\
            Always prefix your file paths with the working directory if you are creating new files for the project.", 
            self.system_prompt, working_directory
        );

        let agent = client.agent(&self.model)
            .preamble(&preamble)
            .tool(ReadFile)
            .tool(WriteFile)
            .tool(ListFiles)
            .tool(ExecuteCommand)
            .tool(QueryDesignDoc { design_content: self.design_doc.clone() })
            .build();
            
        let mut prompt = format!(
            "Execute the following task:\n\
            Title: {}\n\
            Description: {}\n\n\
            Implementation requirements:\n\
            - Use the tools to create any necessary files in {}.\n\
            - After you are done, respond with a JSON object containing:\n\
              - summary: String (detailed description of what you did)\n\
              - files_created: Vec<String> (list of files you created or modified)\n\n\
            Output ONLY the raw JSON object.", 
            task.title, task.description, working_directory
        );

        if let Some(feedback) = &task.feedback {
            prompt = format!(
                "{}\n\n\
                IMPORTANT: Your previous attempt failed review with the following feedback:\n\
                {}\n\
                Please address this feedback in your current attempt.", 
                prompt, feedback
            );
        }

        let response = agent.prompt(&prompt).await?;

        // Attempt to parse JSON from the response
        let result: TaskResult = if let Some(start) = response.find('{') {
            let end = response.rfind('}').unwrap_or(response.len() - 1);
            let json_str = &response[start..=end];
            let parsed: serde_json::Value = serde_json::from_str(json_str)?;
            TaskResult {
                task_id: task.id.clone(),
                status: "Completed".to_string(),
                summary: parsed["summary"].as_str().unwrap_or(&response).to_string(),
                files_created: parsed["files_created"]
                    .as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect(),
                feedback: None,
            }
        } else {
            TaskResult {
                task_id: task.id.clone(),
                status: "Completed".to_string(),
                summary: response,
                files_created: vec![],
                feedback: None,
            }
        };

        Ok(result)
    }
}
