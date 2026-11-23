//! Models reached through the Codex CLI, on the account Codex is logged in
//! with (usually a ChatGPT plan), instead of an OpenAI API key.
//!
//! `codex app-server` speaks JSON-RPC, one JSON object per line, on stdio.
//! Codex runs the model loop; the agent's tools are handed to it as "dynamic
//! tools", and every call of one comes back here and runs in the agent's
//! sandbox (and jail), as it does for every other provider.
//!
//! Codex also offers the model tools of its own (a JavaScript `exec` through
//! which the agent's tools are reached, `apply_patch`, subagents, ...), and not
//! all of them can be switched off. So they are refused, in layers:
//!
//! - Codex starts with what can be switched off switched off, no MCP servers,
//!   and an empty directory of its own as its working directory; its thread
//!   gets no execution environment, a read-only sandbox and approval for
//!   everything. A thread that comes back otherwise is not used. Without an
//!   environment, Codex's file and command tools have nowhere to act: asked to
//!   use them, they failed, and the model carried on without them.
//!   Codex's note to the model on that sandbox is left out: it describes
//!   Codex's own tools, and a model with write access, told its sandbox only
//!   permits reading files, stopped before changing anything.
//! - Every approval Codex asks for is refused, and ends the run.
//! - The first item that is not a message, reasoning or a call of one of the
//!   agent's tools, and anything about another thread (a subagent's), ends the
//!   run, and Codex with it. Codex has no setting that removes subagents, and
//!   asked to, the model started one.
//!
//! The settings that switch things off are names Codex may rename in a later
//! release, and Codex ignores names it does not know; the checks on the thread
//! and the last two layers do not depend on them. Dynamic tools are an experimental part of Codex's
//! protocol. Written against codex-cli 0.171.0.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use rig::tool::ToolDyn;
use serde_json::{Value, json};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader, Lines};
use tokio::task::JoinSet;
use tokio::time::Instant;

use crate::llm::{AgentRequest, LlmError, LlmReply, Provider, TokenUsage, secret_env_vars};
use crate::tools::command_executor::OwnedProcessGroup;

/// The Codex program to run, if not `codex` on the PATH. An MCP client may start
/// the server with a PATH that does not include where Codex is installed.
pub const CODEX_BIN_VAR: &str = "CODEX_BIN";

/// Codex features that give the model tools, context or connections of
/// Codex's own. Each is passed as `--disable`.
const DISABLED_FEATURES: &[&str] = &[
    "apps",
    "browser_use",
    "browser_use_external",
    "computer_use",
    "goals",
    "hooks",
    "image_generation",
    "in_app_browser",
    "multi_agent",
    "plugins",
    "remote_plugin",
    "shell_tool",
    "skill_mcp_dependency_install",
    "skill_search",
    "sleep_tool",
    "tool_suggest",
    "unified_exec",
    "view_image",
];

/// Codex settings, each passed as `-c`. `mcp_servers={}` drops the MCP servers in
/// the user's Codex configuration, which would run outside the jail.
const SETTINGS: &[&str] = &[
    "mcp_servers={}",
    "project_doc_max_bytes=0",
    "web_search=\"disabled\"",
    "analytics.enabled=false",
    "skills.include_instructions=false",
    "tools.experimental_request_user_input.enabled=false",
    "tools.update_plan.enabled=false",
];

/// The kinds of item a turn may contain. Anything else is Codex acting on its own.
const ALLOWED_ITEMS: &[&str] = &["userMessage", "agentMessage", "reasoning", "dynamicToolCall", "contextCompaction"];

/// How the server introduces itself to Codex, which passes it on to OpenAI.
const CLIENT_NAME: &str = "clankistrator";

const DEVELOPER_INSTRUCTIONS: &str = "Only the tools described in your instructions work here (in `exec` they are \
     `tools.<name>`). Codex's own tools for files, commands, patches, subagents and the web are not available. \
     Codex's sandbox and approval settings are for those tools: what you may read and change is in your instructions.";

/// Codex sometimes reports its thread idle and never sends `turn/completed`
/// (seen with codex-cli 0.171.0). After this long idle, the turn is taken as over.
const TURN_COMPLETED_GRACE: Duration = Duration::from_secs(5);

fn program() -> String {
    std::env::var(CODEX_BIN_VAR)
        .ok()
        .map(|program| program.trim().to_string())
        .filter(|program| !program.is_empty())
        .unwrap_or_else(|| "codex".to_string())
}

/// Fail early if Codex is missing or not logged in.
pub fn check_login() -> Result<()> {
    let program = program();
    let output = std::process::Command::new(&program)
        .args(["login", "status"])
        .env("RUST_LOG", "error")
        .stdin(Stdio::null())
        .output()
        .map_err(|e| {
            anyhow!(
                "could not run `{program} login status`: {e}. codex: models need the Codex CLI; install it, \
                 or set {CODEX_BIN_VAR} to its path"
            )
        })?;
    if output.status.success() {
        return Ok(());
    }
    let said = [&output.stdout, &output.stderr]
        .iter()
        .flat_map(|bytes| String::from_utf8_lossy(bytes).lines().map(str::to_string).collect::<Vec<_>>())
        .filter(|line| !line.trim().is_empty() && !line.starts_with("WARNING"))
        .collect::<Vec<_>>()
        .join(" ");
    bail!("Codex is not logged in (`{program} login status` says: {said}). Run `codex login` as the user the server runs as")
}

/// A directory of Codex's own, empty, so that it finds no project
/// configuration, instructions or skills where it is started.
struct EmptyDir(PathBuf);

impl EmptyDir {
    fn new() -> std::io::Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let mut path = std::env::temp_dir().join(format!(
            "clankistrator-codex-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        // A directory left by a crashed run under a reused process id is not ours to empty.
        loop {
            match std::fs::create_dir(&path) {
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && NEXT.load(Ordering::Relaxed) < 1_000 => {
                    path.set_file_name(format!(
                        "clankistrator-codex-{}-{}",
                        std::process::id(),
                        NEXT.fetch_add(1, Ordering::Relaxed)
                    ));
                }
                other => break other?,
            }
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700))?;
        }
        Ok(Self(path))
    }
}

impl Drop for EmptyDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Run one agent on a model through Codex. Codex is started for this run
/// only, and killed when the run ends, however it ends.
pub async fn run(request: &AgentRequest, model: &str) -> Result<LlmReply, LlmError> {
    let other = LlmError::Other;
    let dir = EmptyDir::new().map_err(|e| other(anyhow!("could not make an empty directory for Codex: {e}")))?;
    let program = program();
    let mut command = tokio::process::Command::new(&program);
    command.arg("app-server");
    for feature in DISABLED_FEATURES {
        command.args(["--disable", feature]);
    }
    for setting in SETTINGS {
        command.args(["-c", setting]);
    }
    command
        .current_dir(&dir.0)
        // At trace level Codex logs request headers, which carry the account's token.
        .env("RUST_LOG", "error")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true);
    // The account Codex is logged in with pays, not an API key that happens to be set here.
    for var in secret_env_vars() {
        command.env_remove(var);
    }
    #[cfg(unix)]
    command.process_group(0);
    let mut child =
        command.spawn().map_err(|e| other(anyhow!("could not start `{program} app-server`: {e}")))?;
    let _group = OwnedProcessGroup::new(child.id());
    let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
        return Err(other(anyhow!("Codex was started without its stdin and stdout")));
    };
    converse(BufReader::new(stdout), stdin, request, model, &dir.0).await
}

/// One message from Codex.
enum Incoming<'a> {
    /// The answer to one of our requests.
    Response { id: Option<u64>, outcome: Result<&'a Value, String> },
    /// Codex asks something of its client.
    Request { id: &'a Value, method: &'a str, params: &'a Value },
    Notification { method: &'a str, params: &'a Value },
}

static NULL: Value = Value::Null;

fn classify(message: &Value) -> Incoming<'_> {
    let params = message.get("params").unwrap_or(&NULL);
    match (message.get("method").and_then(Value::as_str), message.get("id")) {
        (Some(method), Some(id)) => Incoming::Request { id, method, params },
        (Some(method), None) => Incoming::Notification { method, params },
        (None, id) => Incoming::Response {
            id: id.and_then(Value::as_u64),
            outcome: match message.get("error") {
                Some(error) => Err(error
                    .get("message")
                    .and_then(Value::as_str)
                    .map_or_else(|| error.to_string(), str::to_string)),
                None => Ok(message.get("result").unwrap_or(&NULL)),
            },
        },
    }
}

/// What a request from Codex was about, for the error, and what to answer it
/// with. Each one means the model reached for something of Codex's own.
fn refusal(method: &str) -> (&'static str, Value) {
    let cancel = json!({"result": {"decision": "cancel"}});
    // The older approval requests take a decision of another kind.
    let abort = json!({"result": {"decision": "abort"}});
    let unsupported = json!({"error": {"code": -32601, "message": "clankistrator does not allow this"}});
    match method {
        "item/commandExecution/requestApproval" => ("run a command with Codex's own tools", cancel),
        "execCommandApproval" => ("run a command with Codex's own tools", abort),
        "item/fileChange/requestApproval" => ("change files with Codex's own tools", cancel),
        "applyPatchApproval" => ("change files with Codex's own tools", abort),
        "item/permissions/requestApproval" => ("widen Codex's own permissions", unsupported),
        "item/tool/requestUserInput" | "mcpServer/elicitation/request" => ("ask the user a question", unsupported),
        _ => ("do something clankistrator does not offer", unsupported),
    }
}

struct Wire<R, W> {
    lines: Lines<R>,
    writer: W,
    next_id: u64,
}

impl<R: AsyncBufRead + Unpin, W: AsyncWrite + Unpin> Wire<R, W> {
    async fn send(&mut self, message: &Value) -> Result<()> {
        let mut line = serde_json::to_vec(message)?;
        line.push(b'\n');
        self.writer.write_all(&line).await?;
        self.writer.flush().await?;
        Ok(())
    }

    async fn request(&mut self, method: &str, params: Value) -> Result<u64> {
        self.next_id += 1;
        let id = self.next_id;
        self.send(&json!({"id": id, "method": method, "params": params})).await?;
        Ok(id)
    }

    async fn answer(&mut self, id: &Value, mut answer: Value) -> Result<()> {
        answer["id"] = id.clone();
        self.send(&answer).await
    }

    /// The next message. Cancel safe: nothing is lost if the future is dropped.
    async fn next(&mut self) -> Result<Value> {
        loop {
            match self.lines.next_line().await? {
                None => bail!("Codex stopped unexpectedly; what it said is in the server's log (stderr)"),
                Some(line) if line.trim().is_empty() => continue,
                Some(line) => {
                    return serde_json::from_str(&line).map_err(|e| anyhow!("Codex sent a line that is not JSON: {e}"));
                }
            }
        }
    }

    /// Refuse a request from Codex, and say why the run ends.
    async fn refuse(&mut self, id: &Value, method: &str) -> anyhow::Error {
        let (what, answer) = refusal(method);
        if let Err(e) = self.answer(id, answer).await {
            tracing::debug!("could not answer Codex's {method}: {e}");
        }
        anyhow!(
            "the model tried to {what} (Codex asked: {method}). Agents act only through clankistrator's tools, so \
             the run was stopped"
        )
    }

    /// Send a request and wait for its answer. Before the turn, Codex has no
    /// reason to ask anything, so a request from it ends the run.
    async fn call(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.request(method, params).await?;
        loop {
            let message = self.next().await?;
            match classify(&message) {
                Incoming::Response { id: answered, outcome } if answered == Some(id) => {
                    return outcome.cloned().map_err(|e| anyhow!("Codex refused {method}: {e}"));
                }
                Incoming::Request { id, method: asked, .. } => {
                    let id = id.clone();
                    let asked = asked.to_string();
                    return Err(self.refuse(&id, &asked).await);
                }
                _ => {}
            }
        }
    }
}

/// The tool's output as text: rig's tools return JSON, a plain string quoted.
fn unquote(output: String) -> String {
    serde_json::from_str::<String>(&output).unwrap_or(output)
}

fn tool_result(text: impl Into<String>, success: bool) -> Value {
    json!({"result": {"contentItems": [{"type": "inputText", "text": text.into()}], "success": success}})
}

/// The error for a turn Codex reports as failed.
fn turn_failure(error: &Value) -> anyhow::Error {
    let message = error.get("message").and_then(Value::as_str).unwrap_or("no reason given");
    match error.get("codexErrorInfo").and_then(Value::as_str) {
        Some("usageLimitExceeded") => anyhow!(
            "the usage limit of the plan Codex is logged in with is reached ({message}). It resets on the plan's \
             schedule; until then, use a model from another provider"
        ),
        Some("unauthorized") => anyhow!("Codex's login was refused ({message}); run `codex login` again"),
        Some(code) => anyhow!("Codex failed the turn ({code}): {message}"),
        None => anyhow!("Codex failed the turn: {message}"),
    }
}

/// What the turn has produced so far.
#[derive(Default)]
struct Turn {
    reply: Option<String>,
    usage: TokenUsage,
    /// Model responses so far, counted from Codex's usage reports.
    responses: usize,
    /// A failure Codex reported that it will not retry.
    error: Option<String>,
    idle_since: Option<Instant>,
}

impl Turn {
    fn reply(self) -> LlmReply {
        LlmReply { text: self.reply.unwrap_or_default(), usage: self.usage }
    }

    /// The turn as it stands when Codex has gone idle without completing it.
    fn ended_idle(self) -> Result<LlmReply, LlmError> {
        if let Some(error) = self.error {
            return Err(LlmError::Other(anyhow!("Codex failed the turn: {error}")));
        }
        if self.reply.is_none() {
            return Err(LlmError::Other(anyhow!("Codex stopped working on the turn without a reply")));
        }
        tracing::debug!("Codex went idle without completing the turn; taking its last message as the reply");
        Ok(self.reply())
    }
}

/// The conversation with Codex for one agent run: set up a thread, start one
/// turn with the prompt, answer the tool calls, and return the final reply.
async fn converse<R, W>(
    reader: R,
    writer: W,
    request: &AgentRequest,
    model: &str,
    cwd: &Path,
) -> Result<LlmReply, LlmError>
where
    R: AsyncBufRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let other = LlmError::Other;
    let mut wire = Wire { lines: reader.lines(), writer, next_id: 0 };

    // What Codex reports along the way goes into the run's transcript: never a
    // message of Codex's as it came, only the parts that say what the agent did.
    let note = |event: &str, fields: Value| {
        if let Some(transcript) = &request.transcript {
            transcript.record(event, fields);
        }
    };
    let tools: Arc<Vec<Box<dyn ToolDyn>>> = Arc::new(request.build_tools());
    let mut specs = Vec::with_capacity(tools.len());
    for tool in tools.iter() {
        let definition = tool.definition(String::new()).await;
        specs.push(json!({
            "type": "function",
            "name": definition.name,
            "description": definition.description,
            "inputSchema": definition.parameters,
        }));
    }

    let initialized = wire
        .call(
            "initialize",
            json!({
                "clientInfo": {"name": CLIENT_NAME, "title": null, "version": env!("CARGO_PKG_VERSION")},
                // Dynamic tools are part of the experimental API.
                "capabilities": {"experimentalApi": true},
            }),
        )
        .await
        .map_err(other)?;
    // It names Codex's version, which the protocol is still changing between.
    note("codex", json!({"user_agent": initialized.get("userAgent").and_then(Value::as_str)}));
    wire.send(&json!({"method": "initialized"})).await.map_err(other)?;

    let started = wire
        .call(
            "thread/start",
            json!({
                "cwd": cwd,
                "model": model,
                "ephemeral": true,
                "sandbox": "read-only",
                "approvalPolicy": "untrusted",
                "baseInstructions": request.preamble,
                "developerInstructions": DEVELOPER_INSTRUCTIONS,
                "dynamicTools": specs,
                // No environment: Codex's own tools have nowhere to run commands or apply patches.
                "environments": [],
                // Without this, Codex tells the model that its sandbox only permits reading
                // files. That is true of Codex's own tools, not of the agent's.
                "config": {"include_permissions_instructions": false},
            }),
        )
        .await
        .map_err(other)?;
    let Some(thread) = started.pointer("/thread/id").and_then(Value::as_str).map(str::to_string) else {
        return Err(other(anyhow!("Codex started a thread without saying its id")));
    };
    // Each of these must be said, and be what was asked for: a field Codex stops
    // sending is a change nobody has looked at.
    let sandbox = started.pointer("/sandbox/type").and_then(Value::as_str);
    let approvals = started.get("approvalPolicy").and_then(Value::as_str);
    let environments = started.pointer("/thread/environments").and_then(Value::as_array).map(Vec::len);
    if sandbox != Some("readOnly") || approvals != Some("untrusted") || environments != Some(0) {
        return Err(other(anyhow!(
            "Codex did not give the thread a read-only sandbox, approval for everything and no environment \
             (sandbox: {}, approvals: {}, environments: {}), so it is not used",
            sandbox.unwrap_or("not said"),
            approvals.unwrap_or("not said"),
            environments.map_or_else(|| "not said".to_string(), |n| n.to_string())
        )));
    }
    let Some(sources) = started.get("instructionSources").and_then(Value::as_array) else {
        return Err(other(anyhow!("Codex did not say which instruction files it would add, so the thread is not used")));
    };
    if !sources.is_empty() {
        let sources: Vec<String> =
            sources.iter().map(|s| s.as_str().map_or_else(|| s.to_string(), str::to_string)).collect();
        return Err(other(anyhow!(
            "Codex would add instructions of its own to the agent's ({}). An agent starts from its own instructions \
             only; move those files, or use another provider",
            sources.join(", ")
        )));
    }

    let servers = wire.call("mcpServerStatus/list", json!({})).await.map_err(other)?;
    let Some(servers) = servers.get("data").and_then(Value::as_array) else {
        return Err(other(anyhow!("Codex did not say which MCP servers it has")));
    };
    if !servers.is_empty() {
        let names: Vec<&str> = servers.iter().map(|s| s.get("name").and_then(Value::as_str).unwrap_or("?")).collect();
        return Err(other(anyhow!(
            "Codex would give the agent MCP servers ({}), which run outside the jail, so the run was not started",
            names.join(", ")
        )));
    }

    let turn_request = wire
        .request(
            "turn/start",
            json!({"threadId": thread, "input": [{"type": "text", "text": request.prompt}], "environments": []}),
        )
        .await
        .map_err(other)?;

    let mut turn = Turn::default();
    let mut calls: JoinSet<(Value, Result<String, String>)> = JoinSet::new();
    loop {
        let grace_over = turn.idle_since.map(|since| since + TURN_COMPLETED_GRACE);
        tokio::select! {
            message = wire.next() => {
                let message = message.map_err(other)?;
                match classify(&message) {
                    Incoming::Response { id, outcome } => {
                        if id == Some(turn_request)
                            && let Err(e) = outcome
                        {
                            return Err(other(anyhow!("Codex did not start the turn: {e}")));
                        }
                    }
                    Incoming::Request { id, method, params } => {
                        let id = id.clone();
                        if method != "item/tool/call" {
                            let method = method.to_string();
                            return Err(other(wire.refuse(&id, &method).await));
                        }
                        let name = params.get("tool").and_then(Value::as_str).unwrap_or_default().to_string();
                        let namespaced = params.get("namespace").is_some_and(|n| !n.is_null());
                        let ours = params.get("threadId").and_then(Value::as_str) == Some(thread.as_str());
                        let index =
                            if ours && !namespaced { tools.iter().position(|tool| tool.name() == name) } else { None };
                        let Some(index) = index else {
                            let answer = tool_result(format!("There is no tool named {name}."), false);
                            wire.answer(&id, answer).await.map_err(other)?;
                            continue;
                        };
                        // Each model response that calls tools counts as a turn, as with rig.
                        if turn.responses >= request.max_turns {
                            let answer = tool_result("The agent has used all of its turns.", false);
                            let _ = wire.answer(&id, answer).await;
                            return Err(LlmError::MaxTurns(request.max_turns));
                        }
                        let arguments = match params.get("arguments") {
                            Some(Value::String(text)) => text.clone(),
                            Some(arguments) => arguments.to_string(),
                            None => "{}".to_string(),
                        };
                        let tools = tools.clone();
                        calls.spawn(async move {
                            let outcome = tools[index].call(arguments).await.map(unquote).map_err(|e| e.to_string());
                            (id, outcome)
                        });
                    }
                    Incoming::Notification { method, params } => {
                        // A subagent works in a thread of its own. Anything about another
                        // thread (its start, items, status or end) is not this run's.
                        let about = match method {
                            "thread/started" => params.pointer("/thread/id"),
                            _ => params.get("threadId"),
                        };
                        if about.and_then(Value::as_str).is_some_and(|t| t != thread) {
                            return Err(other(anyhow!(
                                "Codex started work in another thread (a subagent), which clankistrator does not \
                                 allow; the run was stopped"
                            )));
                        }
                        match method {
                            "item/started" => {
                                let kind = params.pointer("/item/type").and_then(Value::as_str).unwrap_or("unknown");
                                if !ALLOWED_ITEMS.contains(&kind) {
                                    return Err(other(anyhow!(
                                        "the model used Codex's own {kind} instead of the agent's tools. Agents act \
                                         only through clankistrator's tools, so the run was stopped"
                                    )));
                                }
                            }
                            "item/completed" => match params.pointer("/item/type").and_then(Value::as_str) {
                                Some("agentMessage") => {
                                    if let Some(text) = params.pointer("/item/text").and_then(Value::as_str) {
                                        note("message", json!({"text": text}));
                                        turn.reply = Some(text.to_string());
                                    }
                                }
                                Some("reasoning") => {
                                    let joined = |field: &str| {
                                        let parts: Vec<&str> = params
                                            .pointer(&format!("/item/{field}"))
                                            .and_then(Value::as_array)
                                            .map(|parts| parts.iter().filter_map(Value::as_str).collect())
                                            .unwrap_or_default();
                                        (!parts.is_empty()).then(|| crate::transcript::capped(&parts.join("\n")).0)
                                    };
                                    let (summary, content) = (joined("summary"), joined("content"));
                                    if summary.is_some() || content.is_some() {
                                        note("reasoning", json!({"summary": summary, "content": content}));
                                    }
                                }
                                Some("contextCompaction") => note("compaction", json!({})),
                                _ => {}
                            },
                            "thread/tokenUsage/updated" => {
                                turn.responses += 1;
                                let total = params.pointer("/tokenUsage/total").unwrap_or(&NULL);
                                let count = |field: &str| total.get(field).and_then(Value::as_u64).unwrap_or(0);
                                turn.usage = TokenUsage::normalised(
                                    Provider::Codex,
                                    count("inputTokens"),
                                    count("cachedInputTokens"),
                                    0,
                                    count("outputTokens"),
                                );
                                let last = params.pointer("/tokenUsage/last").unwrap_or(&NULL);
                                let count = |field: &str| last.get(field).and_then(Value::as_u64).unwrap_or(0);
                                note(
                                    "response",
                                    json!({"usage": {
                                        "input_tokens": count("inputTokens"),
                                        "cached_input_tokens": count("cachedInputTokens"),
                                        "output_tokens": count("outputTokens"),
                                    }}),
                                );
                            }
                            "error" => {
                                let text =
                                    params.pointer("/error/message").and_then(Value::as_str).unwrap_or("no reason given");
                                // A retry can wait minutes (a model at capacity), which the
                                // transcript shows as the time to the next event.
                                if params.get("willRetry").and_then(Value::as_bool) == Some(true) {
                                    tracing::warn!("Codex: {text} (it is trying again)");
                                    note("codex_retry", json!({"message": text}));
                                } else {
                                    note("codex_error", json!({"message": text}));
                                    turn.error = Some(text.to_string());
                                }
                            }
                            "thread/status/changed" => match params.pointer("/status/type").and_then(Value::as_str) {
                                Some("idle") => turn.idle_since = Some(Instant::now()),
                                Some("systemError") => {
                                    return Err(other(anyhow!("Codex reported a system error in the thread")));
                                }
                                _ => turn.idle_since = None,
                            },
                            "turn/completed" => {
                                let finished = params.get("turn").unwrap_or(&NULL);
                                return match finished.get("status").and_then(Value::as_str) {
                                    Some("completed") => Ok(turn.reply()),
                                    Some("interrupted") => Err(other(anyhow!("Codex interrupted the turn"))),
                                    _ => Err(other(turn_failure(finished.get("error").unwrap_or(&NULL)))),
                                };
                            }
                            _ => {}
                        }
                    }
                }
            }
            Some(done) = calls.join_next(), if !calls.is_empty() => {
                let (id, outcome) = done.map_err(|e| other(anyhow!("a tool call did not finish: {e}")))?;
                let answer = match outcome {
                    Ok(text) => tool_result(text, true),
                    Err(error) => tool_result(error, false),
                };
                wire.answer(&id, answer).await.map_err(other)?;
            }
            _ = tokio::time::sleep_until(grace_over.unwrap_or_else(Instant::now)),
                if grace_over.is_some() && calls.is_empty() =>
            {
                return turn.ended_idle();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::future::Future;

    use tokio::io::{DuplexStream, ReadHalf, WriteHalf};

    use super::*;
    use crate::tools::{Sandbox, ToolAccess};

    /// Plays Codex's side of the protocol.
    struct FakeCodex {
        from_client: Lines<BufReader<ReadHalf<DuplexStream>>>,
        to_client: WriteHalf<DuplexStream>,
    }

    impl FakeCodex {
        async fn recv(&mut self) -> Value {
            let line = self.from_client.next_line().await.unwrap().expect("the client hung up");
            serde_json::from_str(&line).unwrap()
        }

        /// Sends one message. The client may already have gone, which some tests are about.
        async fn send(&mut self, message: Value) {
            let mut line = serde_json::to_vec(&message).unwrap();
            line.push(b'\n');
            let _ = self.to_client.write_all(&line).await;
        }

        /// Expects a request for `method`, answers it with `result`, and returns its params.
        async fn answer(&mut self, method: &str, result: Value) -> Value {
            let message = self.recv().await;
            assert_eq!(message["method"], method, "{message}");
            self.send(json!({"id": message["id"], "result": result})).await;
            message["params"].clone()
        }

        fn thread_started() -> Value {
            json!({
                "thread": {"id": "t1", "environments": []},
                "sandbox": {"type": "readOnly", "networkAccess": false},
                "approvalPolicy": "untrusted",
                "instructionSources": [],
            })
        }

        /// Everything up to the turn starting, with these MCP servers. Returns the thread's params.
        async fn start(&mut self) -> Value {
            self.answer("initialize", json!({"userAgent": "fake"})).await;
            assert_eq!(self.recv().await["method"], "initialized");
            let thread = self.answer("thread/start", Self::thread_started()).await;
            self.answer("mcpServerStatus/list", json!({"data": []})).await;
            let turn = self.answer("turn/start", json!({"turn": {"id": "u1", "status": "inProgress"}})).await;
            assert_eq!((&turn["threadId"], &turn["environments"]), (&json!("t1"), &json!([])));
            assert_eq!(turn["input"], json!([{"type": "text", "text": "Do the thing."}]));
            thread
        }

        async fn notify(&mut self, method: &str, params: Value) {
            self.send(json!({"method": method, "params": params})).await;
        }

        async fn item(&mut self, phase: &str, item: Value) {
            self.notify(&format!("item/{phase}"), json!({"threadId": "t1", "turnId": "u1", "item": item})).await;
        }

        async fn usage(&mut self, input: u64, cached: u64, output: u64) {
            let total = json!({"inputTokens": input, "cachedInputTokens": cached, "outputTokens": output,
                "reasoningOutputTokens": 0, "totalTokens": input + output});
            self.notify(
                "thread/tokenUsage/updated",
                json!({"threadId": "t1", "turnId": "u1", "tokenUsage": {"total": total, "last": total}}),
            )
            .await;
        }

        async fn idle(&mut self) {
            self.notify("thread/status/changed", json!({"threadId": "t1", "status": {"type": "idle"}})).await;
        }

        async fn complete(&mut self, status: &str, error: Value) {
            self.notify("turn/completed", json!({"threadId": "t1", "turn": {"id": "u1", "status": status, "error": error}}))
                .await;
        }

        async fn call_tool(&mut self, id: u64, tool: &str, arguments: Value) -> Value {
            self.send(json!({"id": id, "method": "item/tool/call", "params": {"threadId": "t1", "turnId": "u1",
                "callId": format!("c{id}"), "tool": tool, "namespace": null, "arguments": arguments}}))
                .await;
            self.recv().await
        }

        /// Wait until the client has gone.
        async fn until_closed(mut self) {
            while let Ok(Some(_)) = self.from_client.next_line().await {}
        }
    }

    fn request(tools: ToolAccess, max_turns: usize) -> (tempfile::TempDir, AgentRequest) {
        let dir = tempfile::tempdir().unwrap();
        let sandbox = Sandbox::new(dir.path(), Duration::from_secs(5)).unwrap();
        let request = AgentRequest {
            model: "codex:gpt-test".to_string(),
            preamble: "You are a test.".to_string(),
            prompt: "Do the thing.".to_string(),
            tools,
            sandbox,
            max_turns,
            transcript: None,
        };
        (dir, request)
    }

    /// Runs the client against a fake Codex that plays `script`, and returns
    /// the run's outcome and what the script returned.
    async fn against<F, Fut, T>(request: AgentRequest, script: F) -> (Result<LlmReply, LlmError>, T)
    where
        F: FnOnce(FakeCodex) -> Fut,
        Fut: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        let (client, server) = tokio::io::duplex(1 << 20);
        let (client_read, client_write) = tokio::io::split(client);
        let (server_read, server_write) = tokio::io::split(server);
        let fake = FakeCodex { from_client: BufReader::new(server_read).lines(), to_client: server_write };
        let played = tokio::spawn(script(fake));
        let outcome = converse(BufReader::new(client_read), client_write, &request, "gpt-test", Path::new("/empty")).await;
        (outcome, played.await.unwrap())
    }

    fn error_text(outcome: Result<LlmReply, LlmError>) -> String {
        match outcome {
            Ok(reply) => panic!("the run succeeded with {:?}", reply.text),
            Err(e) => e.to_string(),
        }
    }

    #[tokio::test]
    async fn the_agents_tools_are_called_through_codex_and_the_reply_comes_back() {
        let (_dir, request) = request(ToolAccess::Probe(4242), 4);
        let (outcome, (thread, answer)) = against(request, |mut codex| async move {
            let thread = codex.start().await;
            codex.item("started", json!({"type": "dynamicToolCall", "tool": "get_code"})).await;
            let answer = codex.call_tool(7, "get_code", json!({"words": ["alpha", "beta"]})).await;
            codex.usage(9_200, 9_000, 50).await;
            codex.item("started", json!({"type": "agentMessage", "text": ""})).await;
            codex.item("completed", json!({"type": "agentMessage", "text": "PROBE-4242"})).await;
            // The order Codex was seen to use: idle first, then the turn's end.
            codex.idle().await;
            codex.complete("completed", Value::Null).await;
            (thread, answer)
        })
        .await;

        let reply = outcome.unwrap();
        assert_eq!(reply.text, "PROBE-4242");
        let usage = reply.usage;
        assert_eq!((usage.input_tokens, usage.cached_input_tokens, usage.output_tokens), (9_200, 9_000, 50));
        // The tool ran here, and its output went back unquoted.
        assert_eq!(
            answer,
            json!({"id": 7, "result": {"contentItems": [{"type": "inputText", "text": "PROBE-4242"}], "success": true}})
        );
        // Codex's own reach is cut down before the turn starts.
        assert_eq!(thread["sandbox"], "read-only");
        assert_eq!(thread["approvalPolicy"], "untrusted");
        assert_eq!(thread["environments"], json!([]));
        assert_eq!(thread["config"], json!({"include_permissions_instructions": false}));
        assert_eq!(thread["ephemeral"], true);
        assert_eq!(thread["cwd"], "/empty");
        assert_eq!(thread["baseInstructions"], "You are a test.");
        assert_eq!(thread["model"], "gpt-test");
        let tools = thread["dynamicTools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!((&tools[0]["type"], &tools[0]["name"]), (&json!("function"), &json!("get_code")));
        assert_eq!(tools[0]["inputSchema"]["required"], json!(["words"]));
    }

    #[tokio::test]
    async fn what_the_agent_did_goes_into_the_transcript_and_codexs_own_messages_do_not() {
        let (dir, mut request) = request(ToolAccess::Probe(4242), 4);
        let transcript = crate::transcript::Transcript::create(&dir.path().join("t"), "run").unwrap();
        request.transcript = Some(transcript.clone());
        let (outcome, ()) = against(request, |mut codex| async move {
            codex.start().await;
            codex
                .notify("error", json!({"threadId": "t1", "turnId": "u1", "willRetry": true,
                    "error": {"message": "Selected model is at capacity."}}))
                .await;
            codex.item("completed", json!({"type": "reasoning", "summary": ["Call the tool."], "content": []})).await;
            codex.item("started", json!({"type": "dynamicToolCall", "tool": "get_code"})).await;
            codex.call_tool(7, "get_code", json!({"words": ["alpha", "beta"]})).await;
            codex.usage(9_200, 9_000, 50).await;
            codex.item("completed", json!({"type": "agentMessage", "text": "PROBE-4242"})).await;
            codex.complete("completed", Value::Null).await;
        })
        .await;
        assert_eq!(outcome.unwrap().text, "PROBE-4242");

        let text = std::fs::read_to_string(transcript.path()).unwrap();
        let events: Vec<Value> = text.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
        let names: Vec<&str> = events.iter().map(|event| event["event"].as_str().unwrap()).collect();
        assert_eq!(names, ["codex", "codex_retry", "reasoning", "tool_call", "tool_result", "response", "message"]);
        assert_eq!(events[0]["user_agent"], "fake");
        assert_eq!(events[1]["message"], "Selected model is at capacity.");
        assert_eq!((&events[2]["summary"], &events[2]["content"]), (&json!("Call the tool."), &Value::Null));
        assert_eq!((&events[3]["tool"], &events[3]["arguments"]), (&json!("get_code"), &json!({"words": ["alpha", "beta"]})));
        assert_eq!((&events[4]["ok"], &events[4]["output"]), (&json!(true), &json!("PROBE-4242")));
        assert_eq!(events[5]["usage"]["cached_input_tokens"], 9_000);
        assert_eq!(events[6]["text"], "PROBE-4242");
        assert!(!text.contains("thread/start") && !text.contains("\"threadId\""), "none of Codex's messages as sent");
    }

    #[tokio::test]
    async fn codex_doing_anything_of_its_own_ends_the_run() {
        // subAgentActivity is what Codex sent first when the model started a subagent.
        let kinds = ["commandExecution", "fileChange", "mcpToolCall", "collabAgentToolCall", "subAgentActivity", "webSearch", "plan"];
        for kind in kinds {
            let (_dir, request) = request(ToolAccess::None, 4);
            let (outcome, ()) = against(request, move |mut codex| async move {
                codex.start().await;
                codex.item("started", json!({"type": kind})).await;
                codex.until_closed().await;
            })
            .await;
            let error = error_text(outcome);
            assert!(error.contains(kind) && error.contains("the run was stopped"), "{error}");
        }
    }

    #[tokio::test]
    async fn an_approval_codex_asks_for_is_cancelled_and_ends_the_run() {
        let (_dir, request) = request(ToolAccess::None, 4);
        let (outcome, answer) = against(request, |mut codex| async move {
            codex.start().await;
            codex.send(json!({"id": "a1", "method": "item/fileChange/requestApproval", "params": {"threadId": "t1"}})).await;
            codex.recv().await
        })
        .await;
        assert_eq!(answer, json!({"id": "a1", "result": {"decision": "cancel"}}));
        assert!(error_text(outcome).contains("change files with Codex's own tools"));
    }

    #[test]
    fn older_approval_requests_get_a_decision_they_know() {
        assert_eq!(refusal("applyPatchApproval").1, json!({"result": {"decision": "abort"}}));
        assert_eq!(refusal("execCommandApproval").1, json!({"result": {"decision": "abort"}}));
        assert_eq!(refusal("item/commandExecution/requestApproval").1, json!({"result": {"decision": "cancel"}}));
    }

    #[tokio::test]
    async fn any_other_request_from_codex_is_refused_and_ends_the_run() {
        let (_dir, request) = request(ToolAccess::None, 4);
        let (outcome, answer) = against(request, |mut codex| async move {
            codex.start().await;
            codex.send(json!({"id": 5, "method": "item/tool/requestUserInput", "params": {"threadId": "t1"}})).await;
            codex.recv().await
        })
        .await;
        assert_eq!(answer["id"], 5);
        assert!(answer.get("error").is_some(), "{answer}");
        assert!(error_text(outcome).contains("ask the user a question"));
    }

    #[tokio::test]
    async fn mcp_servers_from_codexs_configuration_stop_the_run_before_its_turn() {
        let (_dir, request) = request(ToolAccess::None, 4);
        let (outcome, after) = against(request, |mut codex| async move {
            codex.answer("initialize", json!({})).await;
            codex.recv().await;
            codex.answer("thread/start", FakeCodex::thread_started()).await;
            codex.answer("mcpServerStatus/list", json!({"data": [{"name": "github"}]})).await;
            codex.from_client.next_line().await.unwrap()
        })
        .await;
        assert!(error_text(outcome).contains("MCP servers (github)"));
        // No turn was started.
        assert_eq!(after, None);
    }

    #[tokio::test]
    async fn a_thread_that_is_not_locked_down_or_brings_instructions_is_not_used() {
        let changed = |pointer: &str, value: Option<Value>| {
            let mut started = FakeCodex::thread_started();
            let (parent, field) = pointer.rsplit_once('/').unwrap();
            let parent = started.pointer_mut(parent).unwrap().as_object_mut().unwrap();
            match value {
                Some(value) => parent.insert(field.to_string(), value),
                None => parent.remove(field),
            };
            started
        };
        let cases = [
            (changed("/thread/environments", Some(json!([{"environmentId": "local", "cwd": "/"}]))), "environments: 1"),
            (changed("/thread/environments", None), "environments: not said"),
            (changed("/sandbox/type", Some(json!("workspaceWrite"))), "sandbox: workspaceWrite"),
            (changed("/approvalPolicy", Some(json!("never"))), "approvals: never"),
            (changed("/approvalPolicy", None), "approvals: not said"),
            (changed("/instructionSources", Some(json!(["/home/someone/.codex/AGENTS.md"]))), "AGENTS.md"),
            (changed("/instructionSources", None), "which instruction files"),
        ];
        for (started, expected) in cases {
            let (_dir, request) = request(ToolAccess::None, 4);
            let (outcome, after) = against(request, move |mut codex| async move {
                codex.answer("initialize", json!({})).await;
                codex.recv().await;
                codex.answer("thread/start", started).await;
                codex.from_client.next_line().await.unwrap()
            })
            .await;
            let error = error_text(outcome);
            assert!(error.contains(expected), "{error}");
            assert_eq!(after, None);
        }
    }

    #[tokio::test]
    async fn a_used_up_plan_is_reported_as_such() {
        let (_dir, request) = request(ToolAccess::None, 4);
        let (outcome, ()) = against(request, |mut codex| async move {
            codex.start().await;
            let error = json!({"message": "You've hit your usage limit.", "codexErrorInfo": "usageLimitExceeded"});
            codex.complete("failed", error).await;
        })
        .await;
        let error = error_text(outcome);
        assert!(error.contains("usage limit of the plan") && error.contains("You've hit your usage limit."), "{error}");
    }

    #[tokio::test(start_paused = true)]
    async fn a_turn_that_goes_idle_and_never_completes_ends_with_its_last_message() {
        let (_dir, request) = request(ToolAccess::None, 4);
        let (outcome, ()) = against(request, |mut codex| async move {
            codex.start().await;
            codex.item("completed", json!({"type": "agentMessage", "text": "READY"})).await;
            codex.usage(100, 0, 2).await;
            codex.idle().await;
            codex.until_closed().await;
        })
        .await;
        let reply = outcome.unwrap();
        assert_eq!((reply.text.as_str(), reply.usage.input_tokens), ("READY", 100));
    }

    #[tokio::test]
    async fn a_tool_call_after_the_last_turn_ends_the_run() {
        let (_dir, request) = request(ToolAccess::Probe(1), 1);
        let (outcome, answer) = against(request, |mut codex| async move {
            codex.start().await;
            codex.usage(10, 0, 1).await;
            codex.call_tool(3, "get_code", json!({"words": ["a", "b"]})).await
        })
        .await;
        assert!(matches!(outcome, Err(LlmError::MaxTurns(1))), "{:?}", outcome.map(|r| r.text));
        assert_eq!(answer["result"]["success"], false);
    }

    #[tokio::test]
    async fn a_tool_the_agent_was_not_given_is_answered_with_a_failure() {
        let (_dir, request) = request(ToolAccess::None, 4);
        let (outcome, answer) = against(request, |mut codex| async move {
            codex.start().await;
            let answer = codex.call_tool(2, "write_file", json!({"path": "x"})).await;
            codex.item("completed", json!({"type": "agentMessage", "text": "I could not write it."})).await;
            codex.complete("completed", Value::Null).await;
            answer
        })
        .await;
        assert_eq!(outcome.unwrap().text, "I could not write it.");
        assert_eq!(answer["result"]["success"], false);
        assert!(answer["result"]["contentItems"][0]["text"].as_str().unwrap().contains("no tool named write_file"));
    }

    #[tokio::test]
    async fn work_in_another_thread_ends_the_run() {
        // A subagent's thread starting, its items, its status, and its turn ending, which must
        // not pass for the end of this run's turn.
        let elsewhere = [
            ("thread/started", json!({"thread": {"id": "t2"}})),
            ("item/started", json!({"threadId": "t2", "turnId": "x", "item": {"type": "agentMessage"}})),
            ("thread/status/changed", json!({"threadId": "t2", "status": {"type": "idle"}})),
            ("turn/completed", json!({"threadId": "t2", "turn": {"id": "x", "status": "completed"}})),
        ];
        for (method, params) in elsewhere {
            let (_dir, request) = request(ToolAccess::None, 4);
            let (outcome, ()) = against(request, move |mut codex| async move {
                codex.start().await;
                codex.item("completed", json!({"type": "agentMessage", "text": "from the subagent"})).await;
                codex.notify(method, params).await;
                codex.until_closed().await;
            })
            .await;
            let error = error_text(outcome);
            assert!(error.contains("another thread"), "{method}: {error}");
        }
    }

    #[tokio::test]
    async fn codex_going_away_mid_turn_is_an_error() {
        let (_dir, request) = request(ToolAccess::None, 4);
        let (outcome, ()) = against(request, |mut codex| async move {
            codex.start().await;
        })
        .await;
        assert!(error_text(outcome).contains("Codex stopped unexpectedly"));
    }

    #[test]
    fn string_outputs_lose_their_json_quotes_and_others_stay_as_they_are() {
        assert_eq!(unquote("\"PROBE-1\"".to_string()), "PROBE-1");
        assert_eq!(unquote("{\"a\":1}".to_string()), "{\"a\":1}");
        assert_eq!(unquote("plain".to_string()), "plain");
    }
}
