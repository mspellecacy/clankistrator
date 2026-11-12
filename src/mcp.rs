//! A small MCP server (JSON-RPC over stdio) with one tool, `run_agent`: it
//! runs an agent in a project directory for the MCP client (see
//! [`crate::delegate`]).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;
use tokio::task::{AbortHandle, JoinSet};

use crate::delegate::{self, Delegation};
use crate::llm::Llm;

const PROTOCOL_VERSION: &str = "2024-11-05";
const RUN_AGENT: &str = "run_agent";
/// How often a running agent tells the client that it is still at work. A
/// client may give up on a call that stays silent (Claude Code does after 30
/// minutes), and an agent run can take that long.
const HEARTBEAT: Duration = Duration::from_secs(60);

/// What `run_agent` may do, decided by whoever starts the server.
#[derive(Debug, Clone)]
pub struct Delegating {
    /// The only models an agent may run on, the default first. The client
    /// chooses among them; it cannot name another, so which providers get to
    /// see a project is settled on the server's command line.
    pub models: Vec<String>,
    pub max_turns: usize,
    pub command_timeout: Duration,
    /// The jail agents' commands run in. `None` runs them as ordinary
    /// processes. A call cannot change it, or what is hidden.
    pub jail: Option<crate::tools::JailPolicy>,
    /// Paths in a project, relative to its root, that agents may not see.
    pub hidden: Vec<PathBuf>,
    /// A file that gets one line per agent run (see [`RunRecord`]).
    pub usage_log: Option<PathBuf>,
}

/// One line of the usage log: what ran, on which model, for how long, at what
/// cost in tokens, and how it ended. Never the prompt, the reply or an error
/// text. A run that ends without [`finish`](Self::finish), because its client
/// cancelled it or went away, is written as "cancelled" when it is dropped.
struct RunRecord {
    log: Option<PathBuf>,
    summary: delegate::Summary,
    started: std::time::Instant,
    written: bool,
}

/// Seconds since 1970 as an ISO 8601 UTC timestamp.
fn utc_timestamp(seconds: u64) -> String {
    let (days, rest) = (seconds / 86_400, seconds % 86_400);
    // Days to a civil date (Howard Hinnant's algorithm), for dates from 1970 on.
    let z = days + 719_468;
    let (era, day_of_era) = (z / 146_097, z % 146_097);
    let year_of_era = (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 { shifted_month + 3 } else { shifted_month - 9 };
    let year = year_of_era + era * 400 + u64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z", rest / 3_600, rest % 3_600 / 60, rest % 60)
}

impl RunRecord {
    fn start(log: Option<&Path>, summary: &delegate::Summary) -> Self {
        Self { log: log.map(Path::to_path_buf), summary: summary.clone(), started: std::time::Instant::now(), written: false }
    }

    /// Write the run's line and return how many seconds it took.
    fn finish(mut self, outcome: &Result<delegate::Outcome, String>) -> u64 {
        match outcome {
            Ok(outcome) => self.write("ok", Some(&outcome.usage)),
            Err(_) => self.write("failed", None),
        }
        self.started.elapsed().as_secs()
    }

    fn write(&mut self, outcome: &str, usage: Option<&crate::llm::TokenUsage>) {
        self.written = true;
        let Some(log) = &self.log else { return };
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |since| since.as_secs());
        let line = json!({
            "time": utc_timestamp(now),
            "directory": self.summary.directory,
            "agent": self.summary.agent,
            "model": self.summary.model,
            "access": self.summary.access,
            "outcome": outcome,
            "seconds": self.started.elapsed().as_secs(),
            "input_tokens": usage.map(|usage| usage.input_tokens),
            "cached_input_tokens": usage.map(|usage| usage.cached_input_tokens),
            "output_tokens": usage.map(|usage| usage.output_tokens),
        });
        let appended = std::fs::OpenOptions::new().create(true).append(true).open(log).and_then(|mut file| {
            use std::io::Write as _;
            writeln!(file, "{line}")
        });
        // The log is bookkeeping: a run never fails over it.
        if let Err(e) = appended {
            tracing::warn!("could not write to the usage log {}: {e}", log.display());
        }
    }
}

impl Drop for RunRecord {
    fn drop(&mut self) {
        if !self.written {
            self.write("cancelled", None);
        }
    }
}

pub struct McpServer {
    name: String,
    version: String,
    llm: Arc<dyn Llm>,
    delegating: Delegating,
}

/// What a message turned out to ask for.
enum Begun {
    /// Answered already (`None` for a notification, which gets no answer).
    Reply(Option<Value>),
    /// The client no longer wants the answer to this request.
    Cancel(String),
    /// An agent run that passed its checks and is ready to start.
    Agent { id: Value, progress_token: Option<Value>, prepared: Box<delegate::Prepared> },
}

fn respond(id: &Value, outcome: Result<Value, (i64, String)>) -> Value {
    match outcome {
        Ok(result) => json!({ "jsonrpc": "2.0", "id": id, "result": result }),
        Err((code, message)) => {
            json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
        }
    }
}

/// Tool failures are reported in-band, as the MCP spec asks.
fn tool_result(result: Result<String, String>) -> Value {
    match result {
        Ok(text) => json!({ "content": [{ "type": "text", "text": text }] }),
        Err(message) => json!({ "content": [{ "type": "text", "text": message }], "isError": true }),
    }
}

fn agent_result(outcome: Result<delegate::Outcome, String>, seconds: u64) -> Value {
    tool_result(outcome.and_then(|outcome| {
        serde_json::to_string_pretty(&json!({
            "model": outcome.model,
            "seconds": seconds,
            "reply": outcome.reply,
            "report_path": outcome.report.as_ref().map(|report| &report.path),
            "report_written_by": outcome.report.as_ref().map(|report| {
                if report.written_by_agent { "the agent" } else { "clankistrator, from the agent's reply" }
            }),
            "usage": {
                "input_tokens": outcome.usage.input_tokens,
                "cached_input_tokens": outcome.usage.cached_input_tokens,
                "output_tokens": outcome.usage.output_tokens
            }
        }))
        .map_err(|e| e.to_string())
    }))
}

/// The key under which a running request is remembered, so that a
/// cancellation (which repeats the id) finds it.
fn request_key(id: &Value) -> String {
    id.to_string()
}

fn text_arg<'a>(args: &'a Value, name: &str) -> Result<&'a str, String> {
    args.get(name)
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| format!("missing required string argument '{name}'"))
}

impl McpServer {
    pub fn new(llm: Arc<dyn Llm>, delegating: Delegating) -> Self {
        Self { name: "clankistrator".to_string(), version: env!("CARGO_PKG_VERSION").to_string(), llm, delegating }
    }

    /// Serve requests from stdin until it closes. Responses go to stdout, one JSON object per line.
    ///
    /// Requests are answered one at a time, except agent runs: those take
    /// minutes, so each runs as a task of its own while the server goes on
    /// answering. An agent run ends early when the client cancels its request
    /// and when stdin closes, because an agent should not go on changing a
    /// project for a client that has gone away.
    pub async fn run(&self) -> anyhow::Result<()> {
        self.serve(BufReader::new(tokio::io::stdin()), tokio::io::stdout()).await
    }

    /// [`run`](Self::run) on any pair of streams.
    pub async fn serve(
        &self,
        input: impl AsyncBufRead + Unpin,
        mut stdout: impl AsyncWrite + Unpin,
    ) -> anyhow::Result<()> {
        async fn write(stdout: &mut (impl AsyncWrite + Unpin), message: &Value) -> anyhow::Result<()> {
            stdout.write_all(serde_json::to_string(message)?.as_bytes()).await?;
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;
            Ok(())
        }

        let mut lines = input.lines();
        // Everything a running agent sends: heartbeats, then its response.
        let (outgoing, mut queued) = mpsc::unbounded_channel::<Value>();
        let mut agents: JoinSet<()> = JoinSet::new();
        let mut running: HashMap<String, (AbortHandle, Value)> = HashMap::new();

        loop {
            tokio::select! {
                line = lines.next_line() => {
                    let Some(line) = line? else { break };
                    if line.trim().is_empty() {
                        continue;
                    }
                    match self.begin(&line).await {
                        Begun::Reply(Some(response)) => write(&mut stdout, &response).await?,
                        Begun::Reply(None) => {}
                        Begun::Cancel(key) => {
                            if let Some((agent, _)) = running.remove(&key) {
                                tracing::info!("Request {key} was cancelled by the client; stopping its agent");
                                agent.abort();
                            }
                        }
                        Begun::Agent { id, progress_token, prepared } => {
                            // Which messages a given client sends is best learnt from a real run.
                            tracing::info!(
                                "Request {id}: starting an agent; the client {} progress notifications",
                                if progress_token.is_some() { "asked for" } else { "did not ask for" }
                            );
                            let llm = self.llm.clone();
                            let outgoing = outgoing.clone();
                            let response_id = id.clone();
                            let record = RunRecord::start(self.usage_log(), prepared.summary());
                            let agent = agents.spawn(async move {
                                let run = delegate::run(llm.as_ref(), *prepared);
                                tokio::pin!(run);
                                let mut beat = tokio::time::interval(HEARTBEAT);
                                beat.tick().await; // the first tick is immediate
                                let mut minutes = 0u64;
                                let outcome = loop {
                                    tokio::select! {
                                        outcome = &mut run => break outcome,
                                        _ = beat.tick() => {
                                            minutes += 1;
                                            if let Some(token) = &progress_token {
                                                let _ = outgoing.send(json!({
                                                    "jsonrpc": "2.0",
                                                    "method": "notifications/progress",
                                                    "params": {
                                                        "progressToken": token,
                                                        "progress": minutes,
                                                        "message": format!("The agent has been working for {minutes} min")
                                                    }
                                                }));
                                            }
                                        }
                                    }
                                };
                                let seconds = record.finish(&outcome);
                                let _ = outgoing.send(respond(&response_id, Ok(agent_result(outcome, seconds))));
                            });
                            running.insert(request_key(&id), (agent, id));
                        }
                    }
                }
                Some(message) = queued.recv() => write(&mut stdout, &message).await?,
                Some(ended) = agents.join_next(), if !agents.is_empty() => {
                    // A run that panicked sent no response; its client must still get one.
                    if let Err(e) = &ended
                        && e.is_panic()
                        && let Some((_, id)) = running.values().find(|(agent, _)| agent.id() == e.id())
                    {
                        let error = (-32603, format!("The agent run failed inside Clankistrator: {e}"));
                        write(&mut stdout, &respond(id, Err(error))).await?;
                    }
                    running.retain(|_, (agent, _)| !agent.is_finished());
                }
            }
        }

        // Stops the agents and waits until they are gone, so that the commands
        // they started are killed before the program leaves.
        agents.shutdown().await;
        while let Ok(message) = queued.try_recv() {
            write(&mut stdout, &message).await?;
        }
        Ok(())
    }

    /// Handle one message to its end. Returns `None` for notifications, which get no reply.
    pub async fn handle_line(&self, line: &str) -> Option<Value> {
        match self.begin(line).await {
            Begun::Reply(response) => response,
            Begun::Cancel(_) => None,
            Begun::Agent { id, prepared, .. } => {
                let record = RunRecord::start(self.usage_log(), prepared.summary());
                let outcome = delegate::run(self.llm.as_ref(), *prepared).await;
                let seconds = record.finish(&outcome);
                Some(respond(&id, Ok(agent_result(outcome, seconds))))
            }
        }
    }

    /// Answer a message, or, for an agent run, check it and hand it back to be started.
    async fn begin(&self, line: &str) -> Begun {
        let request: Value = match serde_json::from_str(line) {
            Ok(request) => request,
            Err(e) => {
                return Begun::Reply(Some(json!({
                    "jsonrpc": "2.0",
                    "id": null,
                    "error": { "code": -32700, "message": format!("Parse error: {e}") }
                })));
            }
        };
        let method = request.get("method").and_then(Value::as_str).unwrap_or("");
        let params = request.get("params").cloned().unwrap_or_else(|| json!({}));
        let Some(id) = request.get("id").filter(|id| !id.is_null()).cloned() else {
            return match (method, params.get("requestId")) {
                ("notifications/cancelled", Some(cancelled)) => Begun::Cancel(request_key(cancelled)),
                _ => Begun::Reply(None),
            };
        };

        let outcome = match method {
            "initialize" => Ok(json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": { "tools": {} },
                "serverInfo": { "name": self.name, "version": self.version }
            })),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({ "tools": self.tool_list() })),
            "tools/call" if params.get("name").and_then(Value::as_str) == Some(RUN_AGENT) => {
                let args = params.get("arguments").cloned().unwrap_or_else(|| json!({}));
                match self.prepare_agent(&args) {
                    Ok(prepared) => {
                        let progress_token = params.pointer("/_meta/progressToken").cloned();
                        return Begun::Agent { id, progress_token, prepared: Box::new(prepared) };
                    }
                    Err(refusal) => Ok(tool_result(Err(refusal))),
                }
            }
            "tools/call" => match params.get("name").and_then(Value::as_str) {
                Some(name) => Err((-32602, format!("Unknown tool: {name}"))),
                None => Err((-32602, "tools/call requires a 'name'".to_string())),
            },
            _ => Err((-32601, format!("Method not found: {method}"))),
        };
        Begun::Reply(Some(respond(&id, outcome)))
    }

    fn usage_log(&self) -> Option<&Path> {
        self.delegating.usage_log.as_deref()
    }

    /// Check a `run_agent` call against what the server was started to allow.
    fn prepare_agent(&self, args: &Value) -> Result<delegate::Prepared, String> {
        let delegating = &self.delegating;
        let text = |name: &str| args.get(name).and_then(Value::as_str).filter(|s| !s.trim().is_empty());
        let model = match text("model") {
            Some(model) if delegating.models.iter().any(|allowed| allowed == model.trim()) => model.trim(),
            Some(model) => {
                return Err(format!(
                    "model {model:?} is not one this server may use. It was started with: {}. \
                     Another model has to be named on the server's command line; it cannot be chosen here.",
                    delegating.models.join(", ")
                ));
            }
            None => delegating.models.first().ok_or("the server was started without a model for agents")?,
        };
        let max_turns = match args.get("max_turns") {
            None | Some(Value::Null) => delegating.max_turns,
            Some(turns) => turns
                .as_u64()
                .filter(|turns| *turns > 0)
                .and_then(|turns| usize::try_from(turns).ok())
                .ok_or("max_turns must be a positive whole number")?,
        };
        let prepared = delegate::prepare(Delegation {
            directory: PathBuf::from(text_arg(args, "directory")?.trim()),
            agent: text("agent").map(str::to_string),
            prompt: text_arg(args, "prompt")?.to_string(),
            model: model.to_string(),
            write_access: args.get("write_access").and_then(Value::as_bool).unwrap_or(false),
            scratch: args.get("scratch").and_then(Value::as_bool).unwrap_or(false),
            report_path: text("report_path").map(str::to_string),
            max_turns,
            command_timeout: delegating.command_timeout,
            jail: delegating.jail.clone(),
            hidden: delegating.hidden.clone(),
        })?;
        // The log names every project the server has worked in, and an agent reads its project.
        if let Some(log) = &delegating.usage_log
            && log.starts_with(&prepared.summary().directory)
        {
            return Err(format!(
                "the usage log {} is inside the project; start the server with a --usage-log outside it",
                log.display()
            ));
        }
        Ok(prepared)
    }

    fn run_agent_tool(delegating: &Delegating) -> Value {
        let models = delegating.models.join(", ");
        let (commands, without_write) = match &delegating.jail {
            Some(jail) => (
                format!(
                    "Commands it runs are jailed: they see the project (read-only without write_access) but not \
                     your home directory, environment files or the paths this server hides, and they have {}. \
                     They are not checked by your own permission rules or hooks. With scratch, its commands \
                     may change files in a throwaway copy of the project and nothing reaches the real one: \
                     use that for a reviewer that has to change code to test it.",
                    if jail.network { "the network" } else { "no network" }
                ),
                "Without them the project is read-only to its commands as well.",
            ),
            None => (
                "Commands it runs are ordinary processes, not jailed: they are not checked by your own \
                 permission rules or hooks."
                    .to_string(),
                "Without them it can still run commands, and a command can change files.",
            ),
        };
        json!({
            "name": RUN_AGENT,
            "description": format!(
                "Run an agent on one of this server's models, in a project directory, and return its final \
                 reply. The agent gets a fresh context: it knows only its agent file and your prompt. It can \
                 read files and run commands in the directory, and with write_access also write and edit \
                 files. It has no skills, slash commands, subagents or MCP tools, so an agent file that only \
                 wraps one of those cannot be run this way. What the agent reads is sent to its model's \
                 provider. {commands} A run takes minutes and costs tokens. Models: {models}."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Absolute path of the project the agent works in. It must be the root of the repository or of a worktree: git is not visible from a subdirectory."
                    },
                    "prompt": { "type": "string", "description": "The task, complete in itself." },
                    "agent": {
                        "type": "string",
                        "description": "The agent's standing instructions: a Markdown file (path relative to the directory, or absolute), or a bare name, which is looked up as .claude/agents/<name>.md in the directory. Its front matter is ignored."
                    },
                    "model": {
                        "type": "string",
                        "description": format!("One of: {models}. Defaults to the first.")
                    },
                    "write_access": {
                        "type": "boolean",
                        "description": format!("Give the agent the tools that write and edit files. Defaults to false. {without_write}")
                    },
                    "scratch": {
                        "type": "boolean",
                        "description": "Let the agent's commands change files in a throwaway copy of the project, deleted when the run ends. Nothing they write reaches the project. Defaults to false. Needs the jail, and cannot be combined with write_access."
                    },
                    "report_path": {
                        "type": "string",
                        "description": "File inside the directory for the agent's report. If the agent does not write it, its final reply is saved there."
                    },
                    "max_turns": {
                        "type": "integer",
                        "description": format!("Model turns (tool-call rounds) the agent may take. Defaults to {}.", delegating.max_turns)
                    }
                },
                "required": ["directory", "prompt"]
            }
        })
    }

    fn tool_list(&self) -> Value {
        json!([Self::run_agent_tool(&self.delegating)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delegate::tests::ActingLlm;
    use crate::tools::Sandbox;
    use crate::llm::{AgentRequest, LlmFuture, LlmReply, TokenUsage};

    struct FixedLlm(&'static str);
    impl Llm for FixedLlm {
        fn run<'a>(&'a self, _request: AgentRequest) -> LlmFuture<'a> {
            Box::pin(async move { Ok(LlmReply { text: self.0.to_string(), usage: TokenUsage::default() }) })
        }
    }

    fn server(reply: &'static str) -> McpServer {
        delegating_server(Arc::new(FixedLlm(reply)))
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
        assert_eq!(names, ["run_agent"]);
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

        let missing_arg =
            call(&server, json!({"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"run_agent"}})).await;
        assert_eq!(missing_arg["result"]["isError"], true);
    }

    #[test]
    fn timestamps_are_iso_utc() {
        assert_eq!(utc_timestamp(0), "1970-01-01T00:00:00Z");
        assert_eq!(utc_timestamp(1_700_000_000), "2023-11-14T22:13:20Z");
        assert_eq!(utc_timestamp(1_709_164_800), "2024-02-29T00:00:00Z");
        assert_eq!(utc_timestamp(1_735_689_599), "2024-12-31T23:59:59Z");
    }

    #[tokio::test]
    async fn every_run_gets_one_line_in_the_usage_log_and_nothing_of_what_was_said() {
        let project = crate::delegate::tests::project();
        let elsewhere = tempfile::tempdir().unwrap();
        let log = elsewhere.path().join("usage.jsonl");
        let lines = || -> Vec<Value> {
            std::fs::read_to_string(&log)
                .unwrap_or_default()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect()
        };
        let server_with = |llm: Arc<dyn Llm>, log: &Path| {
            McpServer::new(
                llm,
                Delegating {
                    models: vec!["ollama:small".into()],
                    max_turns: 3,
                    command_timeout: Duration::from_secs(30),
                    jail: None,
                    hidden: Vec::new(),
                    usage_log: Some(log.to_path_buf()),
                },
            )
        };
        let arguments = json!({"directory": project.path(), "agent": "reviewer", "prompt": "SECRET-PROMPT"});

        let server = server_with(Arc::new(ActingLlm::replying("SECRET-REPLY")), &log);
        let done = call(&server, run_agent(1, arguments.clone())).await;
        let result: Value = serde_json::from_str(tool_text(&done)).unwrap();
        assert!(result["seconds"].is_u64(), "the caller is told how long the run took: {result}");

        let server = server_with(Arc::new(ActingLlm::new(|_| Err("401 SECRET-ERROR".to_string()))), &log);
        let failed = call(&server, run_agent(2, arguments.clone())).await;
        assert_eq!(failed["result"]["isError"], true);

        let written = lines();
        assert_eq!(written.len(), 2);
        let ok = &written[0];
        assert_eq!(ok["outcome"], "ok");
        assert_eq!(ok["agent"], "reviewer");
        assert_eq!(ok["model"], "ollama:small");
        assert_eq!(ok["access"], "read");
        assert_eq!(ok["directory"], json!(project.path().canonicalize().unwrap()));
        let tokens = |name: &str| ok[name].as_u64();
        assert_eq!((tokens("input_tokens"), tokens("cached_input_tokens"), tokens("output_tokens")), (Some(10), Some(4), Some(2)));
        assert!(ok["time"].as_str().unwrap().ends_with('Z') && ok["seconds"].is_u64());
        assert_eq!(written[1]["outcome"], "failed");
        assert!(written[1]["input_tokens"].is_null(), "a failed run's tokens are not known");
        let text = std::fs::read_to_string(&log).unwrap();
        for said in ["SECRET-PROMPT", "SECRET-REPLY", "SECRET-ERROR"] {
            assert!(!text.contains(said), "{said} must not be logged");
        }

        // A run that never reaches its end is written when it is dropped.
        let summary = crate::delegate::Summary {
            directory: project.path().to_path_buf(),
            agent: None,
            model: "ollama:small".into(),
            access: "scratch",
        };
        drop(RunRecord::start(Some(&log), &summary));
        assert_eq!(lines()[2]["outcome"], "cancelled");
        assert_eq!(lines()[2]["access"], "scratch");

        // A log that cannot be written does not fail the run.
        let server = server_with(Arc::new(ActingLlm::replying("fine")), &elsewhere.path().join("no/such/dir/usage.jsonl"));
        let done = call(&server, run_agent(3, arguments.clone())).await;
        assert_ne!(done["result"]["isError"], true, "{done}");

        // A log inside the project would be read by the agents working there.
        let inside = project.path().canonicalize().unwrap().join("usage.jsonl");
        let server = server_with(Arc::new(ActingLlm::replying("unused")), &inside);
        let refused = call(&server, run_agent(4, arguments)).await;
        assert!(tool_text(&refused).contains("inside the project"), "{refused}");
        assert!(!inside.exists());
    }

    #[test]
    fn the_tool_description_says_whether_commands_are_jailed() {
        let mut delegating = Delegating {
            models: vec!["ollama:test".into()],
            max_turns: 1,
            command_timeout: Duration::from_secs(1),
            jail: None,
            hidden: Vec::new(),
            usage_log: None,
        };
        let text = |delegating: &Delegating| McpServer::run_agent_tool(delegating).to_string();
        assert!(text(&delegating).contains("ordinary processes, not jailed"));
        assert!(text(&delegating).contains("a command can change files"));

        delegating.jail = Some(crate::tools::JailPolicy::default());
        let jailed = text(&delegating);
        assert!(jailed.contains("Commands it runs are jailed") && jailed.contains("they have no network"), "{jailed}");
        assert!(jailed.contains("read-only to its commands as well"), "{jailed}");

        delegating.jail = Some(crate::tools::JailPolicy { network: true, ..Default::default() });
        assert!(text(&delegating).contains("they have the network"));
    }

    fn delegating_server(llm: Arc<dyn Llm>) -> McpServer {
        McpServer::new(
            llm,
            Delegating {
                models: vec!["ollama:small".into(), "ollama:large".into()],
                max_turns: 12,
                command_timeout: Duration::from_secs(30),
                jail: None,
                hidden: Vec::new(),
                usage_log: None,
            },
        )
    }

    fn run_agent(id: u64, arguments: Value) -> Value {
        json!({"jsonrpc":"2.0","id":id,"method":"tools/call","params":{"name":"run_agent","arguments":arguments}})
    }

    #[tokio::test]
    async fn run_agent_is_the_only_tool_and_names_the_servers_models() {
        let llm = Arc::new(ActingLlm::replying("unused"));
        let server = delegating_server(llm.clone());
        let list = call(&server, json!({"jsonrpc":"2.0","id":2,"method":"tools/list"})).await;
        let tools = list["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "run_agent");
        assert!(tools[0]["description"].as_str().unwrap().contains("ollama:small, ollama:large"));
        assert!(llm.requests.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn run_agent_runs_the_agent_and_returns_its_reply_and_report() {
        let project = crate::delegate::tests::project();
        let llm = Arc::new(ActingLlm::replying("VERDICT: clean"));
        let server = delegating_server(llm.clone());

        let done = call(
            &server,
            run_agent(
                1,
                json!({
                    "directory": project.path(),
                    "agent": "reviewer",
                    "prompt": "- **Diff:** `git diff abc123`",
                    "model": "ollama:large",
                    "report_path": "tmp/review.md",
                    "max_turns": 80
                }),
            ),
        )
        .await;
        assert!(done["result"].get("isError").is_none(), "{done}");
        let result: Value = serde_json::from_str(tool_text(&done)).unwrap();
        assert_eq!(result["model"], "ollama:large");
        assert_eq!(result["reply"], "VERDICT: clean");
        assert_eq!(result["report_path"], "tmp/review.md");
        assert_eq!(result["report_written_by"], "clankistrator, from the agent's reply");
        assert_eq!(result["usage"]["input_tokens"], 10);
        assert_eq!(std::fs::read_to_string(project.path().join("tmp/review.md")).unwrap(), "VERDICT: clean");

        let requests = llm.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].model, "ollama:large");
        assert_eq!(requests[0].max_turns, 80);
        assert_eq!(requests[0].tools, crate::tools::ToolAccess::Inspect, "no write tools unless asked for");
        assert_eq!(requests[0].sandbox.command_timeout, Duration::from_secs(30));
        assert!(requests[0].preamble.contains("You review a change you did not write."));
    }

    #[tokio::test]
    async fn run_agent_uses_only_the_servers_models_and_defaults_to_the_first() {
        let project = crate::delegate::tests::project();
        let llm = Arc::new(ActingLlm::replying("ok"));
        let server = delegating_server(llm.clone());
        let with = |extra: Value| {
            let mut arguments = json!({"directory": project.path(), "prompt": "Do it."});
            arguments.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
            run_agent(1, arguments)
        };

        // Not even the server's own default model, unless it was named for agents.
        for model in ["gpt-anything", "ollama:test"] {
            let refused = call(&server, with(json!({"model": model}))).await;
            assert_eq!(refused["result"]["isError"], true, "{model}");
            assert!(tool_text(&refused).contains("ollama:small, ollama:large"));
        }
        for bad in [json!({"max_turns": 0}), json!({"max_turns": "many"}), json!({"directory": "/nonexistent/dir"})] {
            let refused = call(&server, with(bad.clone())).await;
            assert_eq!(refused["result"]["isError"], true, "{bad}");
        }
        let no_prompt = call(&server, run_agent(2, json!({"directory": project.path()}))).await;
        assert_eq!(no_prompt["result"]["isError"], true);
        assert!(llm.requests.lock().unwrap().is_empty(), "a refused call reaches no model");

        let done = call(&server, with(json!({"write_access": true}))).await;
        assert!(done["result"].get("isError").is_none(), "{done}");
        let requests = llm.requests.lock().unwrap();
        assert_eq!(requests[0].model, "ollama:small");
        assert_eq!(requests[0].max_turns, 12);
        assert_eq!(requests[0].tools, crate::tools::ToolAccess::Full);
    }

    #[tokio::test]
    async fn a_failed_agent_run_is_an_error_result_not_a_protocol_error() {
        let project = crate::delegate::tests::project();
        let llm = Arc::new(ActingLlm::new(|_| Err("401 Unauthorized".to_string())));
        let server = delegating_server(llm);
        let failed = call(&server, run_agent(1, json!({"directory": project.path(), "prompt": "Do it."}))).await;
        assert_eq!(failed["result"]["isError"], true);
        assert!(tool_text(&failed).contains("half changed"));
    }

    #[tokio::test]
    async fn a_cancellation_names_the_request_it_cancels() {
        let server = server("");
        let begun = server
            .begin(&json!({"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":7}}).to_string())
            .await;
        assert!(matches!(begun, Begun::Cancel(key) if key == request_key(&json!(7))));
        let begun = server
            .begin(&json!({"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":"a-1"}}).to_string())
            .await;
        assert!(matches!(begun, Begun::Cancel(key) if key == request_key(&json!("a-1"))));

        // A run that is ready to start carries the token its heartbeats need.
        let project = crate::delegate::tests::project();
        let delegating = delegating_server(Arc::new(ActingLlm::replying("ok")));
        let mut request = run_agent(7, json!({"directory": project.path(), "prompt": "Do it."}));
        request["params"]["_meta"] = json!({"progressToken": "tok-1"});
        let begun = delegating.begin(&request.to_string()).await;
        assert!(
            matches!(begun, Begun::Agent { id, progress_token: Some(token), .. } if id == json!(7) && token == json!("tok-1"))
        );
    }

    /// Never answers, like a model in the middle of a long run, and hands
    /// out the sandbox of the run it was given.
    #[derive(Default)]
    struct Hanging(std::sync::Mutex<Option<Sandbox>>);
    impl Llm for Hanging {
        fn run<'a>(&'a self, request: AgentRequest) -> LlmFuture<'a> {
            *self.0.lock().unwrap() = Some(request.sandbox.clone());
            Box::pin(std::future::pending())
        }
    }

    /// A served connection: what the client writes, what it reads, and the server's end.
    struct Connection {
        to_server: Option<tokio::io::DuplexStream>,
        from_server: tokio::io::Lines<BufReader<tokio::io::DuplexStream>>,
        served: tokio::task::JoinHandle<anyhow::Result<()>>,
    }

    impl Connection {
        fn open(server: McpServer) -> Self {
            let (to_server, input) = tokio::io::duplex(64 * 1024);
            let (output, from_server) = tokio::io::duplex(64 * 1024);
            let served = tokio::spawn(async move { server.serve(BufReader::new(input), output).await });
            Self { to_server: Some(to_server), from_server: BufReader::new(from_server).lines(), served }
        }

        /// Close the client's end, as a client that exits does.
        fn hang_up(&mut self) {
            self.to_server.take();
        }

        async fn send(&mut self, message: Value) {
            let to_server = self.to_server.as_mut().expect("the client has not hung up");
            to_server.write_all(format!("{message}\n").as_bytes()).await.unwrap();
        }

        async fn next(&mut self) -> Value {
            let line = tokio::time::timeout(Duration::from_secs(600), self.from_server.next_line())
                .await
                .expect("the server sends something")
                .unwrap()
                .expect("the server is still there");
            serde_json::from_str(&line).unwrap()
        }
    }

    async fn sandbox_of(llm: &Hanging) -> Sandbox {
        loop {
            if let Some(sandbox) = llm.0.lock().unwrap().clone() {
                return sandbox;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_running_agent_does_not_block_the_server_and_stops_when_cancelled() {
        let project = crate::delegate::tests::project();
        let llm = Arc::new(Hanging::default());
        let mut connection = Connection::open(delegating_server(llm.clone()));

        let mut request = run_agent(1, json!({"directory": project.path(), "prompt": "Do it."}));
        request["params"]["_meta"] = json!({"progressToken": "tok-1"});
        connection.send(request).await;
        let sandbox = sandbox_of(&llm).await;

        // Other requests are answered while the agent works.
        connection.send(json!({"jsonrpc":"2.0","id":2,"method":"ping"})).await;
        assert_eq!(connection.next().await["id"], 2);

        // A silent call is one the client may give up on.
        let beat = connection.next().await;
        assert_eq!(beat["method"], "notifications/progress");
        assert_eq!(beat["params"]["progressToken"], "tok-1");
        assert_eq!(beat["params"]["progress"], 1);

        assert!(!sandbox.commands_stopped());
        connection.send(json!({"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":1}})).await;
        while !sandbox.commands_stopped() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        // The cancelled request gets no answer, and no more heartbeats.
        connection.send(json!({"jsonrpc":"2.0","id":3,"method":"ping"})).await;
        tokio::time::sleep(HEARTBEAT * 3).await;
        assert_eq!(connection.next().await["id"], 3);
        connection.hang_up();
        assert!(connection.from_server.next_line().await.unwrap().is_none());
        connection.served.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn an_agent_stops_when_the_client_goes_away_and_a_finished_one_is_answered() {
        let project = crate::delegate::tests::project();

        let llm = Arc::new(Hanging::default());
        let mut connection = Connection::open(delegating_server(llm.clone()));
        connection.send(run_agent(1, json!({"directory": project.path(), "prompt": "Do it."}))).await;
        let sandbox = sandbox_of(&llm).await;
        connection.hang_up();
        connection.served.await.unwrap().unwrap();
        assert!(sandbox.commands_stopped(), "nothing goes on changing the project for a client that left");

        let mut connection = Connection::open(delegating_server(Arc::new(ActingLlm::replying("done"))));
        connection.send(run_agent(5, json!({"directory": project.path(), "prompt": "Do it."}))).await;
        let done = connection.next().await;
        assert_eq!(done["id"], 5);
        let result: Value = serde_json::from_str(tool_text(&done)).unwrap();
        assert_eq!(result["reply"], "done");
        // A refusal is answered even when the client closes right after asking.
        connection.send(run_agent(6, json!({"directory": "/nonexistent/dir", "prompt": "Do it."}))).await;
        connection.hang_up();
        let refused = connection.next().await;
        assert_eq!(refused["id"], 6);
        assert_eq!(refused["result"]["isError"], true);
        connection.served.await.unwrap().unwrap();
    }
}
