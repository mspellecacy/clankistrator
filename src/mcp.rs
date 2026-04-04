use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{self, Write};
use tokio::io::{AsyncBufReadExt, BufReader};

/// Represents an MCP Server that exposes Clankistrator
pub struct McpServer {
    pub name: String,
    pub version: String,
}

impl Default for McpServer {
    fn default() -> Self {
        Self::new()
    }
}

impl McpServer {
    pub fn new() -> Self {
        Self {
            name: "clankistrator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Run the server, listening on stdio
    pub async fn run(&self) -> anyhow::Result<()> {
        let stdin = tokio::io::stdin();
        let mut reader = BufReader::new(stdin).lines();

        while let Some(line) = reader.next_line().await? {
            if line.trim().is_empty() {
                continue;
            }

            match serde_json::from_str::<Value>(&line) {
                Ok(req) => {
                    if let Some(id) = req.get("id") {
                        let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
                        let response = self.handle_request(id, method, &req).await;
                        if let Some(res) = response {
                            let mut out = io::stdout();
                            let res_str = serde_json::to_string(&res)?;
                            writeln!(out, "{}", res_str)?;
                            out.flush()?;
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to parse JSON-RPC request: {}", e);
                }
            }
        }
        Ok(())
    }

    async fn handle_request(&self, id: &Value, method: &str, req: &Value) -> Option<Value> {
        match method {
            "initialize" => {
                Some(json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": self.name,
                            "version": self.version
                        }
                    }
                }))
            }
            "tools/list" => {
                Some(json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "tools": [
                            {
                                "name": "run_command",
                                "description": "Execute a terminal command locally.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "command": {
                                            "type": "string",
                                            "description": "The command to run in the terminal."
                                        }
                                    },
                                    "required": ["command"]
                                }
                            },
                            {
                                "name": "parse_markdown_design",
                                "description": "Parse a markdown file into sections.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "path": {
                                            "type": "string",
                                            "description": "Path to the markdown file to parse."
                                        }
                                    },
                                    "required": ["path"]
                                }
                            }
                        ]
                    }
                }))
            }
            "tools/call" => {
                let params = req.get("params")?;
                let name = params.get("name")?.as_str()?;
                let args = params.get("arguments").cloned().unwrap_or(json!({}));
                
                let result = match name {
                    "run_command" => {
                        let command = args.get("command").and_then(|c| c.as_str()).unwrap_or("");
                        // Use std::process::Command directly for now
                        match std::process::Command::new("sh").arg("-c").arg(command).output() {
                            Ok(output) => {
                                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                                format!("STDOUT:\n{}\nSTDERR:\n{}", stdout, stderr)
                            }
                            Err(e) => format!("Failed to run command: {}", e)
                        }
                    }
                    "parse_markdown_design" => {
                        let path = args.get("path").and_then(|p| p.as_str()).unwrap_or("");
                        match std::fs::read_to_string(path) {
                            Ok(content) => content,
                            Err(e) => format!("Failed to read file: {}", e)
                        }
                    }
                    _ => format!("Unknown tool: {}", name)
                };

                Some(json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": result
                            }
                        ]
                    }
                }))
            }
            _ => {
                Some(json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {
                        "code": -32601,
                        "message": "Method not found"
                    }
                }))
            }
        }
    }
}
