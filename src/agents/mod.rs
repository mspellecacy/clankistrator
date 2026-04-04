use crate::context::{ProjectTask, TaskResult, AgentRole};
use anyhow::Result;
use rig::prelude::*;
use rig::providers::openai;
use rig::completion::Prompt;
use std::env;
use crate::tools::{ReadFile, WriteFile, ListFiles, ExecuteCommand, QueryDesignDoc, SearchProjectMemory};

pub trait Worker {
    async fn execute_task(&self, task: &ProjectTask, working_directory: &str) -> Result<TaskResult>;
}

#[derive(Clone)]
pub struct GenericWorker {
    pub role: AgentRole,
    pub model: String,
    pub system_prompt: String,
    pub design_doc: String,
    pub max_turns: usize,
}

impl GenericWorker {
    pub fn new(role: AgentRole, model: String, system_prompt: String, design_doc: String, max_turns: usize) -> Self {
        Self {
            role,
            model,
            system_prompt,
            design_doc,
            max_turns,
        }
    }
}

impl Worker for GenericWorker {
    async fn execute_task(&self, task: &ProjectTask, working_directory: &str) -> Result<TaskResult> {
        let openai_key = env::var("OPENAI_API_KEY").unwrap_or_else(|_| "stub".to_string());
        
        if openai_key == "stub" {
            return Ok(TaskResult {
                task_id: task.id.clone(),
                status: "Failed".to_string(),
                summary: format!("STUB implementation for {} by GenericWorker ({:?})", task.title, self.role),
                files_created: vec![],
                feedback: Some("No OPENAI_API_KEY found, used stub. Please provide an OPENAI_API_KEY in .env.local.".to_string()),
                new_tasks: None,
            });
        }

        let client = openai::Client::new(&openai_key)?;
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
            .tool(SearchProjectMemory { qdrant_url: "http://localhost:6334".to_string() })
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
            Output ONLY the raw JSON object. If you use markdown, use ```json blocks.",
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

        tracing::info!("Agent ({:?}): Prompting LLM for task: {}", self.role, task.id);
        let response: String = match agent.prompt(&prompt)
            .max_turns(self.max_turns)
            .await {
                Ok(res) => res,
                Err(e) => {
                    let err_str = e.to_string();
                    if err_str.contains("max turn order reached") || err_str.contains("turn limit") {
                        return Ok(TaskResult {
                            task_id: task.id.clone(),
                            status: "Failed".to_string(),
                            summary: format!("Agent reached the maximum turn limit ({}).", self.max_turns),
                            files_created: vec![],
                            feedback: Some(format!(
                                "The agent reached the maximum turn limit of {}. \
                                This often means the task is too complex or the agent is in a loop. \
                                Try breaking down the task into smaller sub-tasks or increasing the limit with --max-steps.", 
                                self.max_turns
                            )),
                            new_tasks: None,
                        });
                    }
                    return Err(e.into());
                }
            };
        tracing::debug!("Agent ({:?}): LLM response: {}", self.role, response);

        // Attempt to parse JSON from the response
        let json_str = if let Some(start) = response.find("```json") {
            let rest = &response[start + 7..];
            if let Some(end) = rest.find("```") {
                &rest[..end]
            } else {
                &response[start + 7..]
            }
        } else if let Some(start) = response.find('{') {
            let end = response.rfind('}').unwrap_or(response.len() - 1);
            &response[start..=end]
        } else {
            ""
        };

        let result: TaskResult = if !json_str.is_empty() {
            match serde_json::from_str::<serde_json::Value>(json_str.trim()) {
                Ok(parsed) => {
                    let mut new_tasks = None;
                    if let Some(tasks_arr) = parsed.get("new_tasks").and_then(|t| t.as_array()) {
                        let parsed_tasks: Vec<crate::context::NewTaskDef> = tasks_arr.iter().filter_map(|t| serde_json::from_value(t.clone()).ok()).collect();
                        if !parsed_tasks.is_empty() {
                            new_tasks = Some(parsed_tasks);
                        }
                    }

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
                        new_tasks,
                    }
                }
                Err(e) => {
                    tracing::warn!("Agent ({:?}): Failed to parse JSON from response: {}. Using raw response as failure.", self.role, e);
                    TaskResult {
                        task_id: task.id.clone(),
                        status: "Failed".to_string(),
                        summary: format!("Failed to parse JSON: {}. Raw response: {}", e, response),
                        files_created: vec![],
                        feedback: Some(format!("The agent failed to provide a valid JSON response: {}", e)),
                        new_tasks: None,
                    }
                }
            }
        } else {
            tracing::warn!("Agent ({:?}): No JSON object found in response. Using raw response as failure.", self.role);
            TaskResult {
                task_id: task.id.clone(),
                status: "Failed".to_string(),
                summary: format!("No JSON found in response: {}", response),
                files_created: vec![],
                feedback: Some("The agent failed to provide a structured JSON response.".to_string()),
                new_tasks: None,
            }
        };

        Ok(result)
    }
}
