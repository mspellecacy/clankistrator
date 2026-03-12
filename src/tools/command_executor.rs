use serde::{Deserialize, Serialize};
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use crate::tools::ToolError;
use tokio::process::Command;

#[derive(Deserialize, Serialize)]
pub struct ExecuteCommandArgs {
    pub command: String,
    pub args: Vec<String>,
}

pub struct ExecuteCommand;

impl Tool for ExecuteCommand {
    const NAME: &'static str = "execute_command";
    type Error = ToolError;
    type Args = ExecuteCommandArgs;
    type Output = String;

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        tracing::info!("Executing command: {} {:?}", args.command, args.args);
        let output_res = Command::new(&args.command)
            .args(&args.args)
            .output()
            .await;

        match output_res {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                if output.status.success() {
                    Ok(format!("Command executed successfully.\nStdout:\n{}\nStderr:\n{}", stdout, stderr))
                } else {
                    Ok(format!("Command failed with exit code {:?}.\nStdout:\n{}\nStderr:\n{}", output.status.code(), stdout, stderr))
                }
            }
            Err(e) => {
                Ok(format!("Failed to execute command {} {:?}: {}", args.command, args.args, e))
            }
        }
    }

    async fn definition(&self, _path: String) -> ToolDefinition {
        serde_json::from_value(serde_json::json!({
            "name": "execute_command",
            "description": "Execute a shell command and return its output. Use this for building and testing the project. PREFER `cargo` for Rust projects. Avoid using scripting languages like Python or Bash unless absolutely necessary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute (e.g., 'cargo')."
                    },
                    "args": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "The arguments to pass to the command (e.g., ['check'])."
                    }
                },
                "required": ["command", "args"]
            }
        })).expect("Invalid tool definition")
    }
}
