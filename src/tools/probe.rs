//! A tool with no access to anything, used only to find out whether a model
//! can complete a tool-call round trip.

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};

use crate::tools::ToolError;

#[derive(Deserialize, Serialize)]
pub struct GetCodeArgs {
    /// A list on purpose: list arguments are where providers and models differ.
    pub words: Vec<String>,
}

/// Returns a code that the caller could not have guessed, so a final answer
/// containing it proves the model called the tool and read the result.
pub struct GetCode {
    pub token: u32,
}

impl GetCode {
    pub fn code(token: u32) -> String {
        format!("PROBE-{token}")
    }
}

impl Tool for GetCode {
    const NAME: &'static str = "get_code";
    type Error = ToolError;
    type Args = GetCodeArgs;
    type Output = String;

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if args.words.len() < 2 {
            return Ok("Error: pass at least two words in the list.".to_string());
        }
        Ok(Self::code(self.token))
    }

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Returns a code. Takes a list of words.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "words": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Any two or more words."
                    }
                },
                "required": ["words"]
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn returns_the_code_only_for_a_real_list() {
        let tool = GetCode { token: 4242 };
        let code = tool.call(GetCodeArgs { words: vec!["alpha".into(), "beta".into()] }).await.unwrap();
        assert_eq!(code, "PROBE-4242");
        let refused = tool.call(GetCodeArgs { words: vec!["alpha".into()] }).await.unwrap();
        assert!(refused.starts_with("Error"));
        assert!(serde_json::from_str::<GetCodeArgs>(r#"{"words":"alpha beta"}"#).is_err());
    }
}
