//! A small MCP server (JSON-RPC over stdio) exposing Clankistrator's planning side.
//!
//! It deliberately has no tool that runs commands or writes files.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::context::ProjectContext;
use crate::llm::{Llm, check_credentials};
use crate::orchestrator::plan_tasks;
use crate::parser::markdown::MarkdownParser;
use crate::tools::Sandbox;

const PROTOCOL_VERSION: &str = "2024-11-05";

pub struct McpServer {
    name: String,
    version: String,
    llm: Arc<dyn Llm>,
    default_model: String,
}

fn text_arg<'a>(args: &'a Value, name: &str) -> Result<&'a str, String> {
    args.get(name)
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| format!("missing required string argument '{name}'"))
}

impl McpServer {
    pub fn new(llm: Arc<dyn Llm>, default_model: String) -> Self {
        Self {
            name: "clankistrator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            llm,
            default_model,
        }
    }

    /// Serve requests from stdin until it closes. Responses go to stdout, one JSON object per line.
    pub async fn run(&self) -> anyhow::Result<()> {
        let mut lines = BufReader::new(tokio::io::stdin()).lines();
        let mut stdout = tokio::io::stdout();

        while let Some(line) = lines.next_line().await? {
            if line.trim().is_empty() {
                continue;
            }
            if let Some(response) = self.handle_line(&line).await {
                stdout.write_all(serde_json::to_string(&response)?.as_bytes()).await?;
                stdout.write_all(b"\n").await?;
                stdout.flush().await?;
            }
        }
        Ok(())
    }

    /// Handle one message. Returns `None` for notifications, which get no reply.
    pub async fn handle_line(&self, line: &str) -> Option<Value> {
        let request: Value = match serde_json::from_str(line) {
            Ok(request) => request,
            Err(e) => {
                return Some(json!({
                    "jsonrpc": "2.0",
                    "id": null,
                    "error": { "code": -32700, "message": format!("Parse error: {e}") }
                }));
            }
        };
        let id = request.get("id").filter(|id| !id.is_null())?.clone();
        let method = request.get("method").and_then(Value::as_str).unwrap_or("");
        let params = request.get("params").cloned().unwrap_or_else(|| json!({}));

        let outcome = match method {
            "initialize" => Ok(json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": { "tools": {} },
                "serverInfo": { "name": self.name, "version": self.version }
            })),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({ "tools": Self::tool_list() })),
            "tools/call" => self.call_tool(&params).await,
            _ => Err((-32601, format!("Method not found: {method}"))),
        };

        Some(match outcome {
            Ok(result) => json!({ "jsonrpc": "2.0", "id": id, "result": result }),
            Err((code, message)) => {
                json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
            }
        })
    }

    fn tool_list() -> Value {
        json!([
            {
                "name": "parse_markdown_design",
                "description": "Split a Markdown design document into its sections (heading and text). Returns a JSON list.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Path to the Markdown file." }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "plan_tasks",
                "description": "Ask the configured model to decompose a Markdown design document into a validated task graph (ids, roles, dependencies). Calls the model provider, so it costs tokens. Nothing is executed or written.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Path to the Markdown design document." },
                        "model": { "type": "string", "description": "Model to use instead of the server default." }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "project_status",
                "description": "Report the saved state of a Clankistrator run: phase, tasks with their statuses, and token usage.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "output_dir": { "type": "string", "description": "The --output directory of the run." }
                    },
                    "required": ["output_dir"]
                }
            }
        ])
    }

    async fn call_tool(&self, params: &Value) -> Result<Value, (i64, String)> {
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .ok_or((-32602, "tools/call requires a 'name'".to_string()))?;
        let args = params.get("arguments").cloned().unwrap_or_else(|| json!({}));

        let result = match name {
            "parse_markdown_design" => Self::parse_design(&args),
            "plan_tasks" => self.plan(&args).await,
            "project_status" => Self::status(&args),
            _ => return Err((-32602, format!("Unknown tool: {name}"))),
        };
        // Tool failures are reported in-band, as the MCP spec asks.
        Ok(match result {
            Ok(text) => json!({ "content": [{ "type": "text", "text": text }] }),
            Err(message) => json!({ "content": [{ "type": "text", "text": message }], "isError": true }),
        })
    }

    fn read_design(args: &Value) -> Result<String, String> {
        let path = text_arg(args, "path")?;
        std::fs::read_to_string(path).map_err(|e| format!("Failed to read {path}: {e}"))
    }

    fn parse_design(args: &Value) -> Result<String, String> {
        let sections: Vec<Value> = MarkdownParser::parse_sections(&Self::read_design(args)?)
            .into_iter()
            .map(|s| json!({ "title": s.title, "content": s.content }))
            .collect();
        serde_json::to_string_pretty(&sections).map_err(|e| e.to_string())
    }

    async fn plan(&self, args: &Value) -> Result<String, String> {
        let design = Self::read_design(args)?;
        let model = args.get("model").and_then(Value::as_str).unwrap_or(&self.default_model);
        check_credentials(model).map_err(|e| e.to_string())?;
        // Planning uses no tools; the sandbox is only a formality here.
        let sandbox = Sandbox::new(std::env::temp_dir(), Duration::from_secs(1)).map_err(|e| e.to_string())?;
        let (tasks, usage, _) = plan_tasks(self.llm.as_ref(), model, &design, &sandbox)
            .await
            .map_err(|e| format!("{e:#}"))?;
        let tasks: Vec<Value> = tasks
            .iter()
            .map(|t| {
                json!({
                    "id": t.id,
                    "title": t.title,
                    "description": t.description,
                    "assigned_role": t.assigned_role,
                    "dependencies": t.dependencies,
                    "acceptance_criteria": t.acceptance_criteria,
                })
            })
            .collect();
        serde_json::to_string_pretty(&json!({
            "model": model,
            "tasks": tasks,
            "usage": {
                "input_tokens": usage.input_tokens,
                "cached_input_tokens": usage.cached_input_tokens,
                "output_tokens": usage.output_tokens
            }
        }))
        .map_err(|e| e.to_string())
    }

    fn status(args: &Value) -> Result<String, String> {
        let dir = text_arg(args, "output_dir")?;
        let ctx = ProjectContext::load(Path::new(dir)).map_err(|e| format!("{e:#}"))?;
        let tasks: Vec<Value> = ctx
            .tasks
            .iter()
            .map(|t| {
                json!({
                    "id": t.id,
                    "title": t.title,
                    "status": t.status,
                    "retries": t.retries,
                    "files": t.files,
                })
            })
            .collect();
        serde_json::to_string_pretty(&json!({
            "project": ctx.project_name,
            "phase": ctx.phase.label(),
            "model": ctx.model,
            "steps_taken": ctx.steps_taken,
            "tasks": tasks,
            "usage": {
                "model_calls": ctx.llm_calls,
                "input_tokens": ctx.input_tokens,
                "cached_input_tokens": ctx.cached_input_tokens,
                "output_tokens": ctx.output_tokens
            }
        }))
        .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{AgentRequest, LlmFuture, LlmReply, TokenUsage};

    struct FixedLlm(&'static str);
    impl Llm for FixedLlm {
        fn run<'a>(&'a self, _request: AgentRequest) -> LlmFuture<'a> {
            Box::pin(async move { Ok(LlmReply { text: self.0.to_string(), usage: TokenUsage::default() }) })
        }
    }

    fn server(reply: &'static str) -> McpServer {
        McpServer::new(Arc::new(FixedLlm(reply)), "ollama:test".into())
    }

    async fn call(server: &McpServer, request: Value) -> Value {
        server.handle_line(&request.to_string()).await.expect("a response")
    }

    fn tool_text(response: &Value) -> &str {
        response["result"]["content"][0]["text"].as_str().unwrap()
    }

    #[tokio::test]
    async fn handshake_and_tool_listing() {
        let server = server("");
        let init = call(&server, json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}})).await;
        assert_eq!(init["result"]["serverInfo"]["name"], "clankistrator");

        // Notifications get no reply.
        let note = json!({"jsonrpc":"2.0","method":"notifications/initialized"}).to_string();
        assert!(server.handle_line(&note).await.is_none());

        let list = call(&server, json!({"jsonrpc":"2.0","id":2,"method":"tools/list"})).await;
        let names: Vec<&str> =
            list["result"]["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert_eq!(names, ["parse_markdown_design", "plan_tasks", "project_status"]);
    }

    #[tokio::test]
    async fn every_request_with_an_id_gets_an_answer() {
        let server = server("");
        let unknown = call(&server, json!({"jsonrpc":"2.0","id":"a","method":"nope"})).await;
        assert_eq!(unknown["error"]["code"], -32601);
        assert_eq!(unknown["id"], "a");

        let no_params = call(&server, json!({"jsonrpc":"2.0","id":3,"method":"tools/call"})).await;
        assert_eq!(no_params["error"]["code"], -32602);

        let bad_tool = call(
            &server,
            json!({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"run_command","arguments":{"command":"id"}}}),
        )
        .await;
        assert_eq!(bad_tool["error"]["code"], -32602);

        let garbage = server.handle_line("{not json").await.unwrap();
        assert_eq!(garbage["error"]["code"], -32700);

        let missing_arg = call(
            &server,
            json!({"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"parse_markdown_design"}}),
        )
        .await;
        assert_eq!(missing_arg["result"]["isError"], true);
    }

    #[tokio::test]
    async fn tools_do_real_work() {
        let dir = tempfile::tempdir().unwrap();
        let design = dir.path().join("design.md");
        std::fs::write(&design, "# Vision\nA calculator.\n\n## Features\n- Add\n").unwrap();
        let server = server(
            r#"[{"id":"TASK-1","title":"Build","description":"Build it","dependencies":[],"assigned_role":"Backend"}]"#,
        );

        let parsed = call(
            &server,
            json!({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"parse_markdown_design","arguments":{"path":design}}}),
        )
        .await;
        let sections: Value = serde_json::from_str(tool_text(&parsed)).unwrap();
        assert_eq!(sections[0]["title"], "Vision");
        assert_eq!(sections[1]["title"], "Features");

        let planned = call(
            &server,
            json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"plan_tasks","arguments":{"path":design}}}),
        )
        .await;
        let plan: Value = serde_json::from_str(tool_text(&planned)).unwrap();
        assert_eq!(plan["tasks"][0]["id"], "TASK-1");
        assert_eq!(plan["tasks"][0]["assigned_role"], "Backend");

        let mut ctx = ProjectContext::new("p".into(), dir.path().to_path_buf(), 5, 5, false, "m".into());
        ctx.tasks = vec![crate::context::tests::task("TASK-1", &[])];
        ctx.save().unwrap();
        let status = call(
            &server,
            json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"project_status","arguments":{"output_dir":dir.path()}}}),
        )
        .await;
        let status: Value = serde_json::from_str(tool_text(&status)).unwrap();
        assert_eq!(status["tasks"][0]["status"], "Pending");

        let missing = call(
            &server,
            json!({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"project_status","arguments":{"output_dir":"/nonexistent/dir"}}}),
        )
        .await;
        assert_eq!(missing["result"]["isError"], true);
    }
}
