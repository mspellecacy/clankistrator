use serde::{Deserialize, Serialize};
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use crate::tools::ToolError;

#[derive(Deserialize, Serialize)]
pub struct QueryDesignDocArgs {
    pub query: String,
}

pub struct QueryDesignDoc {
    pub design_content: String,
}

impl Tool for QueryDesignDoc {
    const NAME: &'static str = "query_design_doc";
    type Error = ToolError;
    type Args = QueryDesignDocArgs;
    type Output = String;

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        // Simplified RAG: search for keywords and return relevant lines
        let found = self.design_content.lines()
            .filter(|line| {
                let lower_line = line.to_lowercase();
                args.query.to_lowercase().split_whitespace().any(|word| lower_line.contains(word))
            })
            .take(20)
            .collect::<Vec<_>>()
            .join("\n");
        
        if found.is_empty() {
            Ok("No relevant sections found in design document.".to_string())
        } else {
            Ok(format!("Relevant parts of design document:\n{}", found))
        }
    }

    async fn definition(&self, _path: String) -> ToolDefinition {
        serde_json::from_value(serde_json::json!({
            "name": "query_design_doc",
            "description": "Query the design document for specific information using semantic search (RAG).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The specific topic or question to search for in the design document."
                    }
                },
                "required": ["query"]
            }
        })).expect("Invalid tool definition")
    }
}
