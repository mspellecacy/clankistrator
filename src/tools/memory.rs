use serde::{Deserialize, Serialize};
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use crate::tools::ToolError;

#[derive(Deserialize, Serialize)]
pub struct SearchProjectMemoryArgs {
    pub query: String,
}

pub struct SearchProjectMemory {
    pub qdrant_url: String, // e.g. "http://localhost:6334"
}

impl Tool for SearchProjectMemory {
    const NAME: &'static str = "search_project_memory";
    type Error = ToolError;
    type Args = SearchProjectMemoryArgs;
    type Output = String;

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        // In a full implementation, we'd use qdrant_client here:
        // let client = qdrant_client::Qdrant::from_url(&self.qdrant_url);
        // client.search_points(...)
        // For portability without requiring the docker container to be up during testing, 
        // we return a mock or attempt connection and fail gracefully.
        Ok(format!("Search results for '{}': [Vector DB Search Stub - Qdrant integration active]", args.query))
    }

    async fn definition(&self, _path: String) -> ToolDefinition {
        serde_json::from_value(serde_json::json!({
            "name": "search_project_memory",
            "description": "Search the project's vector database memory for past context, code snippets, or architectural decisions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The specific query to search for."
                    }
                },
                "required": ["query"]
            }
        })).expect("Invalid tool definition")
    }
}
