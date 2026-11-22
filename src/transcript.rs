//! The transcript of one agent run, for auditing it afterwards: what the
//! agent was told, every tool call it made and what came back, what the model
//! said between calls, and how the run ended. One JSON object per line, each
//! with its `event` and the milliseconds since the run started.
//!
//! Written only with `--transcript-dir`. A transcript holds the project's code
//! and the output of the agent's commands, so it goes outside the project, into
//! a directory only its owner can enter, as a file only its owner can read. It
//! holds what Clankistrator itself saw and nothing of a provider's own traffic:
//! for a `codex:` model, none of the messages Codex sends, whose logs can carry
//! the account's token. Like the usage log it is bookkeeping, and a transcript
//! that cannot be written never fails a run.

use std::fs::File;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rig::OneOrMany;
use rig::completion::{AssistantContent, ToolDefinition};
use rig::message::ReasoningContent;
use rig::tool::{ToolDyn, ToolError};
use rig::wasm_compat::WasmBoxedFuture;
use serde_json::{Value, json};

use crate::llm::TokenUsage;

/// The most of one tool call's arguments, its output, or a model's reasoning
/// that a transcript keeps, in bytes. A file read or a test run can be
/// megabytes; the start of it says what it was, and the rest is in the
/// project or can be run again.
pub const TEXT_LIMIT: usize = 16 * 1024;

/// A run's transcript. Clones write to the same file.
#[derive(Clone)]
pub struct Transcript(Arc<Inner>);

struct Inner {
    path: PathBuf,
    /// `None` once a write has failed: the rest would only fail too.
    file: Mutex<Option<File>>,
    started: Instant,
    calls: AtomicU64,
}

impl std::fmt::Debug for Transcript {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Transcript").field(&self.0.path).finish()
    }
}

impl Transcript {
    /// Start the transcript of run `run_id` as `<dir>/<run_id>.jsonl`,
    /// creating the directory if it is not there. An existing file is never
    /// written over.
    pub fn create(dir: &Path, run_id: &str) -> std::io::Result<Self> {
        if !dir.is_dir() {
            std::fs::create_dir_all(dir)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
            }
        }
        let path = dir.join(format!("{run_id}.jsonl"));
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = options.open(&path)?;
        Ok(Self(Arc::new(Inner { path, file: Mutex::new(Some(file)), started: Instant::now(), calls: AtomicU64::new(0) })))
    }

    pub fn path(&self) -> &Path {
        &self.0.path
    }

    /// Append one event, with the fields of `fields` (a JSON object) after its
    /// name and time.
    pub fn record(&self, event: &str, fields: Value) {
        let ms = u64::try_from(self.0.started.elapsed().as_millis()).unwrap_or(u64::MAX);
        let mut line = json!({ "event": event, "ms": ms });
        if let (Some(line), Value::Object(fields)) = (line.as_object_mut(), fields) {
            line.extend(fields);
        }
        let mut file = self.0.file.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(open) = file.as_mut() else { return };
        if let Err(e) = writeln!(open, "{line}") {
            tracing::warn!("could not write to the transcript {}: {e}; the rest of the run is not recorded", self.0.path.display());
            *file = None;
        }
    }

    /// One model response, as rig hands it over: what the model said, its
    /// reasoning where the provider shows it, the tools it called (each call is
    /// recorded on its own as well), and what the response cost.
    pub fn response(&self, choice: &OneOrMany<AssistantContent>, usage: &TokenUsage) {
        let mut text = Vec::new();
        let mut reasoning = Vec::new();
        let mut tools = Vec::new();
        for part in choice.iter() {
            match part {
                AssistantContent::Text(said) if !said.text.trim().is_empty() => text.push(said.text.as_str()),
                AssistantContent::ToolCall(call) => tools.push(call.function.name.as_str()),
                AssistantContent::Reasoning(thought) => {
                    for block in &thought.content {
                        match block {
                            ReasoningContent::Text { text, .. } | ReasoningContent::Summary(text)
                                if !text.trim().is_empty() =>
                            {
                                reasoning.push(text.as_str());
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        let joined = |parts: Vec<&str>| (!parts.is_empty()).then(|| parts.join("\n"));
        self.record(
            "response",
            json!({
                "text": joined(text),
                "reasoning": joined(reasoning).map(|text| capped(&text).0),
                "tool_calls": tools,
                "usage": {
                    "input_tokens": usage.input_tokens,
                    "cached_input_tokens": usage.cached_input_tokens,
                    "output_tokens": usage.output_tokens
                }
            }),
        );
    }
}

/// `text` cut to [`TEXT_LIMIT`] bytes, on a character boundary, and whether it was cut.
pub fn capped(text: &str) -> (String, bool) {
    if text.len() <= TEXT_LIMIT {
        return (text.to_string(), false);
    }
    let mut end = TEXT_LIMIT;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    (text[..end].to_string(), true)
}

/// The tools, each call of which goes into `transcript`.
pub fn recorded(tools: Vec<Box<dyn ToolDyn>>, transcript: &Transcript) -> Vec<Box<dyn ToolDyn>> {
    tools
        .into_iter()
        .map(|tool| Box::new(Recorded { tool, transcript: transcript.clone() }) as Box<dyn ToolDyn>)
        .collect()
}

/// A tool whose calls are written to a transcript: one event when the call
/// starts, so that a call still running when the run is stopped shows, and
/// one with its result.
struct Recorded {
    tool: Box<dyn ToolDyn>,
    transcript: Transcript,
}

impl ToolDyn for Recorded {
    fn name(&self) -> String {
        self.tool.name()
    }

    fn definition<'a>(&'a self, prompt: String) -> WasmBoxedFuture<'a, ToolDefinition> {
        self.tool.definition(prompt)
    }

    fn call<'a>(&'a self, args: String) -> WasmBoxedFuture<'a, Result<String, ToolError>> {
        Box::pin(async move {
            let call = self.transcript.0.calls.fetch_add(1, Ordering::Relaxed) + 1;
            let mut started = json!({ "call": call, "tool": self.tool.name() });
            match serde_json::from_str::<Value>(&args) {
                Ok(arguments) if args.len() <= TEXT_LIMIT => started["arguments"] = arguments,
                _ => {
                    let (arguments, truncated) = capped(&args);
                    started["arguments"] = json!(arguments);
                    started["arguments_bytes"] = json!(args.len());
                    started["truncated"] = json!(truncated);
                }
            }
            self.transcript.record("tool_call", started);

            let began = Instant::now();
            let outcome = self.tool.call(args).await;
            // A tool's output comes back as a JSON string; the transcript keeps the text.
            let (ok, text) = match &outcome {
                Ok(output) => (true, serde_json::from_str::<String>(output).unwrap_or_else(|_| output.clone())),
                Err(error) => (false, error.to_string()),
            };
            let (output, truncated) = capped(&text);
            self.transcript.record(
                "tool_result",
                json!({
                    "call": call,
                    "ok": ok,
                    "took_ms": u64::try_from(began.elapsed().as_millis()).unwrap_or(u64::MAX),
                    "output": output,
                    "output_bytes": text.len(),
                    "truncated": truncated
                }),
            );
            outcome
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn events(transcript: &Transcript) -> Vec<Value> {
        std::fs::read_to_string(transcript.path())
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    #[test]
    fn a_transcript_is_a_new_file_only_its_owner_can_read() {
        let dir = tempfile::tempdir().unwrap();
        let place = dir.path().join("transcripts");
        let transcript = Transcript::create(&place, "run-1").unwrap();
        assert_eq!(transcript.path(), place.join("run-1.jsonl"));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = |path: &Path| std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode(&place), 0o700);
            assert_eq!(mode(transcript.path()), 0o600);
        }
        assert!(Transcript::create(&place, "run-1").is_err(), "an existing transcript is never written over");

        transcript.record("start", json!({"prompt": "Do it."}));
        transcript.clone().record("end", json!({"outcome": "ok"}));
        let events = events(&transcript);
        assert_eq!(events.len(), 2);
        assert_eq!((&events[0]["event"], &events[0]["prompt"]), (&json!("start"), &json!("Do it.")));
        assert!(events[0]["ms"].is_u64());
        assert_eq!(events[1]["outcome"], "ok");
    }

    #[test]
    fn long_text_is_cut_on_a_character_boundary() {
        assert_eq!(capped("short"), ("short".to_string(), false));
        let long = "é".repeat(TEXT_LIMIT);
        let (cut, truncated) = capped(&long);
        assert!(truncated && cut.len() <= TEXT_LIMIT && cut.len() > TEXT_LIMIT - 2);
        assert!(cut.chars().all(|c| c == 'é'));
    }

    #[test]
    fn a_response_records_what_the_model_said_and_called() {
        let dir = tempfile::tempdir().unwrap();
        let transcript = Transcript::create(dir.path(), "r").unwrap();
        let choice = OneOrMany::many(vec![
            AssistantContent::reasoning("I should look first."),
            AssistantContent::text("Looking at the tree."),
            AssistantContent::tool_call("c1", "list_files", json!({"path": "."})),
        ])
        .unwrap();
        let usage = TokenUsage { input_tokens: 100, cached_input_tokens: 40, output_tokens: 7 };
        transcript.response(&choice, &usage);
        let silent = OneOrMany::many(vec![AssistantContent::reasoning(""), AssistantContent::text("  ")]).unwrap();
        transcript.response(&silent, &usage);
        let event = &events(&transcript)[0];
        assert_eq!(event["event"], "response");
        assert_eq!(event["text"], "Looking at the tree.");
        assert_eq!(event["reasoning"], "I should look first.");
        assert_eq!(event["tool_calls"], json!(["list_files"]));
        assert_eq!(event["usage"], json!({"input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 7}));
        let silent = &events(&transcript)[1];
        assert!(silent["text"].is_null() && silent["reasoning"].is_null(), "empty blocks say nothing: {silent}");
    }

    #[tokio::test]
    async fn a_recorded_tool_writes_its_call_and_its_result() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path().join("project");
        std::fs::create_dir_all(&project).unwrap();
        std::fs::write(project.join("big.txt"), "x".repeat(TEXT_LIMIT * 2)).unwrap();
        let sandbox = crate::tools::Sandbox::new(&project, std::time::Duration::from_secs(5)).unwrap();
        let transcript = Transcript::create(&dir.path().join("t"), "r").unwrap();
        let tools = recorded(crate::tools::ToolAccess::Inspect.build(&sandbox), &transcript);
        let read = tools.iter().find(|tool| tool.name() == "read_file").unwrap();

        let output = read.call(json!({"path": "big.txt"}).to_string()).await.unwrap();
        assert!(output.len() > TEXT_LIMIT * 2, "the agent still gets everything");
        // Arguments the tool cannot take fail the call itself.
        let missing = read.call(json!({"file": "big.txt"}).to_string()).await;
        assert!(missing.is_err());

        let events = events(&transcript);
        assert_eq!(events.len(), 4);
        assert_eq!(events[0]["event"], "tool_call");
        assert_eq!((&events[0]["call"], &events[0]["tool"]), (&json!(1), &json!("read_file")));
        assert_eq!(events[0]["arguments"], json!({"path": "big.txt"}));
        assert_eq!((&events[1]["event"], &events[1]["call"], &events[1]["ok"]), (&json!("tool_result"), &json!(1), &json!(true)));
        assert_eq!(events[1]["truncated"], true);
        assert!(events[1]["output_bytes"].as_u64().unwrap() >= (TEXT_LIMIT * 2) as u64);
        assert!(events[1]["output"].as_str().unwrap().len() <= TEXT_LIMIT);
        assert!(events[1]["took_ms"].is_u64());
        assert_eq!((&events[3]["call"], &events[3]["ok"]), (&json!(2), &json!(false)));
        assert!(!events[3]["output"].as_str().unwrap().is_empty(), "a failed call records why");
    }
}
