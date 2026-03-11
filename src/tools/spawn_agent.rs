use serde::{Deserialize, Serialize};
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use crate::context::AgentRole;

#[derive(Deserialize, Serialize)]
pub struct SpawnAgentArgs {
    pub role: AgentRole,
    pub task_id: String,
}

#[derive(Debug)]
pub struct ToolError(String);
impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for ToolError {}

pub struct SpawnAgent;

impl Tool for SpawnAgent {
    const NAME: &'static str = "spawn_agent";
    type Error = ToolError;
    type Args = SpawnAgentArgs;
    type Output = String;

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        // In a real implementation, this would trigger an agent execution
        // For now, it's a marker for the Orchestrator's logic
        Ok(format!("Agent with role {:?} spawned for task {}", args.role, args.task_id))
    }

    async fn definition(&self, _path: String) -> ToolDefinition {
        serde_json::from_value(serde_json::json!({
            "name": "spawn_agent",
            "description": "Spawn a specialized worker agent to execute a specific task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": ["Infrastructure", "Backend", "Frontend", "QA"]
                    },
                    "task_id": {
                        "type": "string"
                    }
                },
                "required": ["role", "task_id"]
            }
        })).expect("Invalid tool definition")
    }
}
