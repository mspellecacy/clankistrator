//! The worker and reviewer agents: how a task is turned into a prompt, and how
//! the model's reply is turned back into an outcome.

use std::time::Duration;

use serde_json::Value;

use crate::context::{Finding, FindingStatus, MAX_RETRIES, ProjectTask, Requirement, TaskDef};
use crate::llm::{AgentRequest, Llm, LlmError, LlmReply, TokenUsage};
use crate::orchestrator::prompts;
use crate::tools::{Sandbox, ToolAccess};

/// Turns given to the reviewer: enough to read the files and run the tests.
const REVIEW_MAX_TURNS: usize = 20;
/// Turns for the whole-project design check, which has more to read.
const DESIGN_CHECK_MAX_TURNS: usize = 30;
/// Items given to the reviewer in one go. A long list gets shallow answers.
const VERIFY_BATCH: usize = 10;
/// The reviewer gets one chance to correct answers that fail the mechanical checks.
const VERIFY_ATTEMPTS: usize = 2;
/// A quoted line must have at least this many non-space characters to count.
const MIN_QUOTE_CHARS: usize = 10;
/// Ceiling on the requirement checklist extracted from a design document.
const MAX_REQUIREMENTS: usize = 40;

/// How many times one agent run is attempted when the provider call itself fails.
const CALL_ATTEMPTS: u32 = 3;

/// Everything an agent run needs besides the task itself.
#[derive(Clone)]
pub struct AgentSettings {
    pub model: String,
    /// Model for reviews and the design check.
    pub review_model: String,
    pub design_doc: String,
    pub sandbox: Sandbox,
    pub max_turns: usize,
    /// The whole task list (id, role, title, status), so agents know what is
    /// planned elsewhere and which ids exist.
    pub plan: String,
    /// Base delay before retrying a failed provider call; doubles each time.
    pub retry_delay: Duration,
}

/// Errors that will not go away by waiting. When the provider sent a status,
/// the status decides; words are only consulted when there is none, and
/// numbers never are: "Requested 40312 tokens" is not a 403.
fn is_permanent(error: &str) -> bool {
    if let Some(status) = crate::llm::http_status(error) {
        return matches!(status, 400..=404 | 422);
    }
    let lower = error.to_ascii_lowercase();
    [
        "unauthorized",
        "invalid_api_key",
        "invalid api key",
        "authentication",
        "is not set",
        // Out of credit, out of the day's free allowance, or a model the
        // settings forbid: waiting seconds does not help.
        "payment required",
        "allowance for today is used up",
        "is not a free openrouter model",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

/// Run an agent, waiting and trying again when the provider call fails (rate
/// limits, overload, network). Without this, parallel agents on a rate-limited
/// key would burn through every task's retries within seconds.
async fn run_with_backoff(
    llm: &dyn Llm,
    request: AgentRequest,
    retry_delay: Duration,
) -> Result<LlmReply, LlmError> {
    let mut attempt = 1;
    loop {
        match llm.run(request.clone()).await {
            Err(LlmError::Other(e)) if attempt < CALL_ATTEMPTS && !is_permanent(&format!("{e:#}")) => {
                let wait = retry_delay * 2u32.pow(attempt - 1);
                tracing::warn!(
                    "Model call failed (attempt {attempt} of {CALL_ATTEMPTS}), retrying in {}s: {e:#}",
                    wait.as_secs()
                );
                tokio::time::sleep(wait).await;
                attempt += 1;
            }
            other => return other,
        }
    }
}

/// What an agent run produced, plus what it cost and the model's raw reply
/// (kept so a run can be audited afterwards).
pub struct AgentRun<T> {
    pub outcome: T,
    pub usage: TokenUsage,
    /// Agent runs that finished (and were billed).
    pub calls: u64,
    pub reply: Option<String>,
}

/// One thing a reviewer has to decide.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckItem {
    pub text: String,
    /// Must be answered met or unmet: "cannot verify" is not accepted.
    pub decidable: bool,
    /// "Met" only counts with a line quoted from a project file, which is
    /// looked up in that file. Off for items whose evidence is command output.
    pub needs_quote: bool,
}

impl CheckItem {
    fn requirement(text: &str) -> Self {
        Self { text: text.to_string(), decidable: false, needs_quote: true }
    }
}

#[derive(Debug, PartialEq)]
pub enum WorkerOutcome {
    Completed { summary: String, files: Vec<String>, new_tasks: Vec<TaskDef> },
    Failed { reason: String },
    /// The model itself cannot be used (a dead key, a used-up allowance). The
    /// task is not at fault, and trying the same model again will not help.
    ModelUnavailable { reason: String },
}

#[derive(Debug, PartialEq)]
pub enum ReviewVerdict {
    /// Nothing was found unmet. `notes` lists what the reviewer could not verify.
    Passed { notes: Vec<String> },
    /// The reviewer found at least one item unmet.
    Rejected(String),
    /// The reviewer found nothing unmet, but did not show its evidence for
    /// some "met" answers, so they do not count. The work goes back, but
    /// nothing has been found wrong with it.
    Unproven(String),
}

/// Checked on every task, whatever its own criteria are. The flag says whether
/// "met" needs a quote from a project file.
const TASK_BASELINE: &[(&str, bool)] = &[
    ("The project builds without errors.", false),
    (
        "Logic added or changed by this task is covered by automated tests, and every test in the project passes. (Unmet if this task added logic and there are no tests for it. If this task added no logic at all, answer met and say so.)",
        false,
    ),
    ("The project's files are in the project root, not in a subdirectory of it.", false),
];

/// Checked on the finished project, in addition to the design's requirements.
const PROJECT_BASELINE: &[(&str, bool)] = &[
    ("The project builds without errors.", false),
    ("The project has automated tests for its logic, and all of them pass. (Zero tests is unmet.)", false),
    (
        "The program is complete from its entry point: started normally, it reaches working functionality, including whatever setup and teardown its environment needs (a terminal UI must put the terminal into raw mode and restore it on exit; a server must bind and serve). Read the entry point and trace the startup path; quote the line that performs the setup.",
        true,
    ),
    ("The project's files are in the project root, not in a subdirectory of it.", false),
];

fn baseline(items: &[(&str, bool)]) -> Vec<CheckItem> {
    items
        .iter()
        .map(|(text, needs_quote)| CheckItem { text: text.to_string(), decidable: true, needs_quote: *needs_quote })
        .collect()
}

/// Find a JSON value delimited by `open`/`close` in a model reply. Accepts a bare
/// value, a fenced code block, or a value surrounded by prose.
pub fn extract_json(text: &str, open: char, close: char) -> Option<Value> {
    let parse = |candidate: &str| -> Option<Value> {
        let candidate = candidate.trim();
        if !candidate.starts_with(open) {
            return None;
        }
        serde_json::from_str::<Value>(candidate).ok()
    };

    if let Some(value) = parse(text) {
        return Some(value);
    }
    // Fenced blocks, last one first: the final block is the answer.
    let mut fenced: Vec<&str> = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("```") {
        let after = &rest[start + 3..];
        let Some(end) = after.find("```") else { break };
        let block = &after[..end];
        // Drop the language tag ("json") on the opening line, if any.
        let body = match block.find('\n') {
            Some(newline) if !block[..newline].trim_start().starts_with(open) => &block[newline + 1..],
            _ => block,
        };
        fenced.push(body);
        rest = &after[end + 3..];
    }
    if let Some(value) = fenced.iter().rev().find_map(|block| parse(block)) {
        return Some(value);
    }
    // Outermost delimiters.
    let start = text.find(open)?;
    if let Some(end) = text.rfind(close)
        && end > start
        && let Some(value) = parse(&text[start..=end])
    {
        return Some(value);
    }
    // Several values with prose between them: a model that corrects itself
    // writes its answer twice, and the last one is the one it stands by. Only
    // values that start their own line count, and every one of them has to
    // parse: looking inside or past a value that did not would turn a
    // truncated answer into one of its fragments (a task's dependency list,
    // an entry of `new_tasks`, the `{}` in a line of code).
    let mut last = None;
    let mut line_start = 0;
    let mut skip_until = 0;
    for line in text.split_inclusive('\n') {
        let at = line_start;
        line_start += line.len();
        if at < skip_until || !line.starts_with(open) {
            continue;
        }
        let mut values = serde_json::Deserializer::from_str(&text[at..]).into_iter::<Value>();
        let Some(Ok(value)) = values.next() else { return None };
        last = Some(value);
        skip_until = at + values.byte_offset();
    }
    last
}

pub fn parse_worker_reply(reply: &str) -> WorkerOutcome {
    if reply.trim().is_empty() {
        return WorkerOutcome::Failed { reason: "The agent returned an empty reply.".into() };
    }
    let Some(report) = extract_json(reply, '{', '}') else {
        // The work may well be done; let the reviewer judge the files rather than
        // throwing the attempt away over formatting.
        tracing::warn!("Worker reply contained no JSON report; passing the raw reply on to review.");
        return WorkerOutcome::Completed { summary: reply.trim().to_string(), files: vec![], new_tasks: vec![] };
    };

    let summary = report
        .get("summary")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| reply.trim().to_string());

    let status = report.get("status").and_then(Value::as_str).unwrap_or("completed");
    if status.eq_ignore_ascii_case("failed") {
        return WorkerOutcome::Failed { reason: summary };
    }

    let files = report
        .get("files_created")
        .and_then(Value::as_array)
        .map(|files| files.iter().filter_map(|f| f.as_str().map(str::to_string)).collect())
        .unwrap_or_default();

    let new_tasks = report
        .get("new_tasks")
        .and_then(Value::as_array)
        .map(|tasks| {
            tasks
                .iter()
                .filter_map(|t| match serde_json::from_value::<TaskDef>(t.clone()) {
                    Ok(def) => Some(def),
                    Err(e) => {
                        tracing::warn!("Ignoring malformed follow-up task from worker: {e}");
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    WorkerOutcome::Completed { summary, files, new_tasks }
}

fn parse_status(raw: &str) -> Option<FindingStatus> {
    let key: String = raw.chars().filter(|c| c.is_ascii_alphabetic()).collect::<String>().to_ascii_lowercase();
    match key.as_str() {
        "met" | "pass" | "passed" | "yes" => Some(FindingStatus::Met),
        "unmet" | "notmet" | "fail" | "failed" | "missing" | "partial" | "partiallymet" | "no" => {
            Some(FindingStatus::Unmet)
        }
        "cannotverify" | "unverified" | "unverifiable" | "notverifiable" | "unknown" => {
            Some(FindingStatus::CannotVerify)
        }
        _ => None,
    }
}

/// A reviewer's answer to one item, before the mechanical checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Answer {
    pub finding: Finding,
    pub file: Option<String>,
    pub quote: Option<String>,
}

/// Parse a reviewer's item-by-item answer. Every item must be answered.
pub fn parse_findings(reply: &str, items: &[CheckItem]) -> Result<Vec<Answer>, String> {
    let value = extract_json(reply, '[', ']').ok_or("the reply did not contain a JSON list")?;
    let entries = value.as_array().ok_or("the reply did not contain a JSON list")?;

    let text = |entry: &Value, key: &str| {
        entry.get(key).and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty()).map(str::to_string)
    };
    let mut answers: Vec<Option<Answer>> = vec![None; items.len()];
    for entry in entries {
        let id = entry.get("id").and_then(|id| {
            id.as_u64().or_else(|| id.as_str().and_then(|s| s.trim().trim_end_matches('.').parse().ok()))
        });
        let Some(index) = id.and_then(|id| usize::try_from(id).ok()).and_then(|id| id.checked_sub(1)) else {
            continue;
        };
        if index >= items.len() {
            continue;
        }
        let raw_status = entry.get("status").and_then(Value::as_str).unwrap_or("");
        let status = parse_status(raw_status)
            .ok_or_else(|| format!("item {} has an unknown status {raw_status:?}", index + 1))?;
        answers[index] = Some(Answer {
            finding: Finding {
                requirement: items[index].text.clone(),
                status,
                evidence: text(entry, "evidence").unwrap_or_default(),
                overruled: false,
            },
            file: text(entry, "file"),
            quote: text(entry, "quote"),
        });
    }

    let missing: Vec<String> = answers
        .iter()
        .enumerate()
        .filter(|(_, answer)| answer.is_none())
        .map(|(i, _)| (i + 1).to_string())
        .collect();
    if !missing.is_empty() {
        return Err(format!("items {} were not answered", missing.join(", ")));
    }
    Ok(answers.into_iter().flatten().collect())
}

/// Why a reviewer's quote does not count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuoteProblem {
    /// The named file is not a readable file in the project.
    NoFile,
    /// The quote is in the file but is only a fragment that could be anywhere.
    TooShort,
    NotFound,
}

/// A line with fewer letters and digits than this (`}`, `});`) says nothing.
const MIN_LINE_ALPHANUMERICS: usize = 2;

/// Is a line of `quote` really in `file`? Whitespace is ignored, and one
/// matching line is enough, so indentation and a second line quoted for
/// context do not fail an honest quote.
///
/// A quote counts if it is long enough to identify code, or if it is a whole
/// line of the file (`fn main() {` is short, and is still the line). A file
/// with nothing in it to quote needs no quote: that it exists is checked here.
fn check_quote(sandbox: &Sandbox, file: &str, quote: &str) -> Result<(), QuoteProblem> {
    // Reviewers like to write "src/app.rs:52".
    let file = file.split(':').next().unwrap_or(file).trim();
    let path = sandbox.resolve(file).map_err(|_| QuoteProblem::NoFile)?;
    let content = std::fs::read_to_string(path).map_err(|_| QuoteProblem::NoFile)?;
    let squash = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
    let says_something = |line: &str| line.chars().filter(|c| c.is_alphanumeric()).count() >= MIN_LINE_ALPHANUMERICS;

    let file_lines: Vec<String> = content.lines().map(squash).filter(|line| says_something(line)).collect();
    if file_lines.is_empty() && quote.trim().is_empty() {
        return Ok(());
    }
    let haystack = squash(&content);
    let mut problem = QuoteProblem::NotFound;
    for line in quote.lines().map(squash).filter(|line| says_something(line)) {
        let long_enough = line.chars().filter(|c| !c.is_whitespace()).count() >= MIN_QUOTE_CHARS;
        if file_lines.contains(&line) || (long_enough && haystack.contains(&line)) {
            return Ok(());
        }
        if haystack.contains(&line) {
            problem = QuoteProblem::TooShort;
        }
    }
    Err(problem)
}

/// The checks a program can make on a reviewer's answers, so that "met" is
/// not just the reviewer's word:
/// - "met" on an item that needs a quote must quote a line that exists in the named file;
/// - "met" with no evidence at all is not accepted;
/// - "cannot verify" is not accepted on items that can be decided from the code.
///
/// Returns the findings with every failed answer turned into "unmet", and for
/// each failure the index of the item and a description for the reviewer.
pub fn audit(answers: Vec<Answer>, items: &[CheckItem], sandbox: &Sandbox) -> (Vec<Finding>, Vec<(usize, String)>) {
    let mut problems = Vec::new();
    let findings = answers
        .into_iter()
        .zip(items)
        .enumerate()
        .map(|(i, (answer, item))| {
            let n = i + 1;
            let Answer { mut finding, file, quote } = answer;
            match finding.status {
                FindingStatus::Met if item.needs_quote => {
                    let checked = match &file {
                        Some(file) => check_quote(sandbox, file, quote.as_deref().unwrap_or_default()),
                        None => Err(QuoteProblem::NoFile),
                    };
                    if let Err(problem) = checked {
                        let named = file.as_deref().unwrap_or("(no file was named)");
                        let what = match problem {
                            QuoteProblem::NoFile => format!(
                                "the file you named, {named}, is not a file in the project. Name the file, relative \
                                 to the project root, and quote one line from it"
                            ),
                            QuoteProblem::TooShort => format!(
                                "your quote is in {named} but is only a fragment of a line. Copy the whole line"
                            ),
                            QuoteProblem::NotFound => format!(
                                "the quote was not found in {named}. Copy one line exactly from the file that \
                                 implements the item (for an item about a file existing or a command succeeding: \
                                 a line of the file it concerns)"
                            ),
                        };
                        problems.push((i, format!("Item {n}: you answered met, but {what}, or answer unmet.")));
                        finding.status = FindingStatus::Unmet;
                        finding.overruled = true;
                        finding.evidence = format!(
                            "The reviewer called this met but could not quote code in the project that implements it. Its claim: {}",
                            if finding.evidence.is_empty() { "(none)" } else { &finding.evidence }
                        );
                    }
                }
                FindingStatus::Met if finding.evidence.is_empty() => {
                    problems.push((i, format!("Item {n}: you answered met without evidence. Give the command output that shows it, or answer unmet.")));
                    finding.status = FindingStatus::Unmet;
                    finding.overruled = true;
                    finding.evidence = "The reviewer called this met but gave no evidence.".to_string();
                }
                FindingStatus::CannotVerify if item.decidable => {
                    problems.push((i, format!(
                        "Item {n}: cannot_verify is not accepted for this item. It is behaviour, which the code or a test \
                         either demonstrates or does not: trace it through the code and answer met with a quote, or answer \
                         unmet and say what is missing (for an input and its expected result: a test that runs exactly that input)."
                    )));
                    finding.status = FindingStatus::Unmet;
                    finding.evidence = format!(
                        "Not established: nothing in the code or its tests was shown to demonstrate this. The reviewer said: {}",
                        finding.evidence
                    );
                }
                _ => {}
            }
            finding
        })
        .collect();
    (findings, problems)
}

fn design_section(design_doc: &str) -> String {
    format!("\n\n# Design document\n\nThis is the source of truth for the project.\n\n{design_doc}\n")
}

fn plan_section(plan: &str) -> String {
    if plan.is_empty() {
        String::new()
    } else {
        format!("\n# The whole plan\n\nThese are all the tasks of the project. Each is handled by its own agent.\n\n{plan}")
    }
}

fn numbered(items: &[String]) -> String {
    items
        .iter()
        .enumerate()
        .map(|(i, item)| format!("{}. {}\n", i + 1, item.split_whitespace().collect::<Vec<_>>().join(" ")))
        .collect()
}

/// Run a worker agent on one task. `prior_work` describes tasks already completed.
pub async fn execute_task(
    llm: &dyn Llm,
    task: &ProjectTask,
    prior_work: &str,
    settings: &AgentSettings,
) -> AgentRun<WorkerOutcome> {
    let preamble = format!(
        "{}\n{}{}",
        prompts::role_prompt(task.assigned_role),
        prompts::WORKER_RULES,
        design_section(&settings.design_doc)
    );

    let mut prompt = format!("# Your task: {} ({})\n\n{}\n", task.title, task.id, task.description);
    if !task.acceptance_criteria.is_empty() {
        prompt.push_str(&format!(
            "\n# Acceptance criteria\n\nA reviewer will check each of these against your files, one by one, \
             and reject the task if any is not met:\n\n{}",
            numbered(&task.acceptance_criteria)
        ));
    }
    prompt.push_str(&plan_section(&settings.plan));
    if !prior_work.is_empty() {
        prompt.push_str(&format!(
            "\n# Work already completed by other agents\n\n{prior_work}\n\
             Inspect the existing files before changing them and build on what is there.\n"
        ));
    }
    if let Some(feedback) = &task.feedback {
        prompt.push_str(&format!(
            "\n# Feedback on your previous attempt\n\n{feedback}\n\n\
             Your earlier files are still in the project directory. Fix the problems described above.\n"
        ));
    }
    if let Some(advice) = &task.advice {
        prompt.push_str(&format!(
            "\n# Advice from the lead engineer\n\n\
             The lead engineer looked at the project after an attempt at this task failed. Follow this advice:\n\n\
             {advice}\n"
        ));
    }
    prompt.push_str(prompts::WORKER_REPORT_FORMAT);

    tracing::info!("Worker ({:?}) [{}]: starting", task.assigned_role, task.id);
    let request = AgentRequest {
        model: settings.model.clone(),
        preamble,
        prompt,
        tools: ToolAccess::Full,
        sandbox: settings.sandbox.clone(),
        max_turns: settings.max_turns,
    };
    match run_with_backoff(llm, request, settings.retry_delay).await {
        Ok(reply) => AgentRun {
            outcome: parse_worker_reply(&reply.text),
            usage: reply.usage,
            calls: 1,
            reply: Some(reply.text),
        },
        Err(LlmError::MaxTurns(turns)) => AgentRun {
            outcome: WorkerOutcome::Failed {
                reason: format!(
                    "You used all {turns} turns without reporting a result. Work in fewer, larger steps: \
                     write complete files in one call, run the build once, then report. \
                     (The limit can be raised with --max-turns.)"
                ),
            },
            usage: TokenUsage::default(),
            calls: 0,
            reply: None,
        },
        Err(e) if crate::llm::model_unusable(&e.to_string()) => AgentRun {
            outcome: WorkerOutcome::ModelUnavailable { reason: e.to_string() },
            usage: TokenUsage::default(),
            calls: 0,
            reply: None,
        },
        Err(e) => AgentRun {
            outcome: WorkerOutcome::Failed { reason: format!("The model call failed: {e}") },
            usage: TokenUsage::default(),
            calls: 0,
            reply: None,
        },
    }
}

fn numbered_items(items: &[CheckItem]) -> String {
    numbered(&items.iter().map(|item| item.text.clone()).collect::<Vec<_>>())
}

/// Have the reviewer model answer a list of items against the files in the
/// project directory, a batch at a time. The model reports status and
/// evidence per item and is never asked for an overall verdict; its answers
/// are audited mechanically, and it gets one chance to correct what fails.
async fn verify_items(
    llm: &dyn Llm,
    subject: String,
    items: &[CheckItem],
    max_turns: usize,
    settings: &AgentSettings,
) -> AgentRun<Result<Vec<Finding>, LlmError>> {
    let mut run = AgentRun { outcome: Ok(Vec::new()), usage: TokenUsage::default(), calls: 0, reply: None };
    let mut findings = Vec::new();
    let mut replies: Vec<String> = Vec::new();
    let batches: Vec<&[CheckItem]> = items.chunks(VERIFY_BATCH).collect();

    for (number, batch) in batches.iter().enumerate() {
        let base_prompt = format!(
            "{subject}\n# Items to verify\n\n{}\n{}",
            numbered_items(batch),
            prompts::VERIFY_INSTRUCTIONS
        );
        let mut corrections = String::new();
        let mut attempt = 1;
        // Answers that passed the audit are final. Without this, a reviewer asked
        // to correct one answer may quietly change its mind about the others.
        let mut accepted: Vec<Option<Finding>> = vec![None; batch.len()];
        let audited = loop {
            let request = AgentRequest {
                model: settings.review_model.clone(),
                preamble: format!("{}{}", prompts::QA_SYSTEM_PROMPT, design_section(&settings.design_doc)),
                prompt: format!("{base_prompt}{corrections}"),
                tools: ToolAccess::Inspect,
                sandbox: settings.sandbox.clone(),
                max_turns,
            };
            let reply = match run_with_backoff(llm, request, settings.retry_delay).await {
                Ok(reply) => reply,
                Err(e) => break Err(e),
            };
            run.calls += 1;
            run.usage += reply.usage;
            tracing::debug!("Reviewer reply (batch {}, attempt {attempt}): {}", number + 1, reply.text);
            replies.push(reply.text.clone());

            let answers = match parse_findings(&reply.text, batch) {
                Ok(answers) => answers,
                Err(e) => break Err(LlmError::Other(anyhow::anyhow!("the reviewer's answer was unusable: {e}"))),
            };
            let (audited, problems) = audit(answers, batch, &settings.sandbox);
            let failed = |index: usize| problems.iter().any(|(i, _)| *i == index);
            let still_open: Vec<&(usize, String)> =
                problems.iter().filter(|(i, _)| accepted[*i].is_none()).collect();
            let last = attempt == VERIFY_ATTEMPTS;
            for (index, finding) in audited.into_iter().enumerate() {
                if accepted[index].is_none() && (last || !failed(index)) {
                    accepted[index] = Some(finding);
                }
            }
            if still_open.is_empty() || last {
                if !still_open.is_empty() {
                    tracing::warn!(
                        "Reviewer: {} answer(s) still failed the evidence check and count as unmet",
                        still_open.len()
                    );
                }
                break Ok(accepted.into_iter().flatten().collect::<Vec<Finding>>());
            }
            tracing::info!("Reviewer: {} answer(s) failed the evidence check; asking for corrections", still_open.len());
            corrections = format!(
                "\n# Problems with your previous answer\n\nA program checked your answer and found:\n{}\n\
                 Look at the code again for these items. Your answers to the other items were accepted and are final. \
                 Reply in the same format with every item; only the items listed above will be read.\n",
                still_open.iter().map(|(_, p)| format!("- {p}\n")).collect::<String>()
            );
            attempt += 1;
        };
        match audited {
            Ok(batch_findings) => findings.extend(batch_findings),
            Err(e) => {
                run.outcome = Err(e);
                run.reply = (!replies.is_empty()).then(|| replies.join("\n\n---\n\n"));
                return run;
            }
        }
    }
    run.outcome = Ok(findings);
    run.reply = Some(replies.join("\n\n---\n\n"));
    run
}

/// The items a task is reviewed against: the baseline plus its own criteria.
pub fn review_items(task: &ProjectTask) -> Vec<CheckItem> {
    let mut items = baseline(TASK_BASELINE);
    if task.acceptance_criteria.is_empty() {
        items.push(CheckItem::requirement(&format!(
            "Everything the task description asks for is implemented: {}",
            task.description
        )));
    } else {
        items.extend(task.acceptance_criteria.iter().map(|c| CheckItem::requirement(c)));
    }
    items
}

/// Pass or reject from the findings. This decision is made here, not by the model.
pub fn verdict_from(findings: &[Finding]) -> ReviewVerdict {
    let unmet: Vec<String> =
        findings.iter().filter(|f| f.status == FindingStatus::Unmet).map(Finding::line).collect();
    if unmet.is_empty() {
        ReviewVerdict::Passed {
            notes: findings
                .iter()
                .filter(|f| f.status == FindingStatus::CannotVerify)
                .map(Finding::line)
                .collect(),
        }
    } else {
        let feedback =
            format!("The QA reviewer checked your work item by item. These are not met:\n{}", unmet.join("\n"));
        if findings.iter().any(|f| f.status == FindingStatus::Unmet && !f.overruled) {
            ReviewVerdict::Rejected(feedback)
        } else {
            ReviewVerdict::Unproven(feedback)
        }
    }
}

/// What the advisor made of a failed attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Advice {
    /// Instructions for the worker's next attempt.
    pub guidance: String,
    /// The worker model the advisor wants for the next attempt, if it named one.
    pub model: Option<String>,
}

pub fn parse_advice(reply: &str) -> Result<Advice, String> {
    let value = extract_json(reply, '{', '}').ok_or("the reply did not contain a JSON object")?;
    let text = |key: &str| value.get(key).and_then(Value::as_str).map(str::trim).filter(|t| !t.is_empty());
    let guidance = text("guidance").ok_or("the reply has no guidance")?;
    let guidance = match text("diagnosis") {
        Some(diagnosis) => format!("What went wrong: {diagnosis}\n\nWhat to do: {guidance}"),
        None => guidance.to_string(),
    };
    Ok(Advice { guidance, model: text("model").map(str::to_string) })
}

/// A failed attempt, as put to the advisor.
pub struct FailedAttempt<'a> {
    pub task: &'a ProjectTask,
    /// The model that made the attempt.
    pub model: &'a str,
    /// The model the next attempt is set to run on.
    pub next_model: &'a str,
    /// What is known about the worker models.
    pub pool_facts: &'a str,
}

/// Ask the advisor why an attempt failed, what the worker should do, and on
/// which model. It can look at the project but not change it.
pub async fn advise(
    llm: &dyn Llm,
    advisor_model: &str,
    attempt: &FailedAttempt<'_>,
    settings: &AgentSettings,
) -> AgentRun<Result<Advice, LlmError>> {
    let task = attempt.task;
    let mut prompt = format!(
        "# Advice wanted\n\nA worker model attempted the task below and the attempt failed.\n\n\
         # The task: {} ({})\n\n{}\n",
        task.title, task.id, task.description
    );
    if !task.acceptance_criteria.is_empty() {
        prompt.push_str(&format!("\n# Acceptance criteria\n\n{}", numbered(&task.acceptance_criteria)));
    }
    if let Some(earlier) = &task.advice {
        prompt.push_str(&format!(
            "\n# Advice you gave after an earlier failed attempt\n\n\
             The worker had this in front of it for the attempt that has now failed.\n\n{earlier}\n"
        ));
    }
    // A task taken up again after an earlier run gave up on it has made no attempt in this one.
    let which = if task.retries == 0 {
        format!("an earlier run used up its attempts; this run allows {} more; last attempt made by {}", MAX_RETRIES + 1, attempt.model)
    } else {
        format!("failed attempt {} of at most {}, made by {}", task.retries, MAX_RETRIES + 1, attempt.model)
    };
    prompt.push_str(&format!(
        "\n# What went wrong ({which})\n\n{}\n\
         \n# Worker models\n\nListed from cheapest to most capable.\n\n{}\n\
         The next attempt is set to run on {}.\n\n{}",
        task.feedback.as_deref().unwrap_or("(no details were recorded)"),
        attempt.pool_facts,
        attempt.next_model,
        prompts::ADVICE_INSTRUCTIONS
    ));

    tracing::info!("Advisor [{}]: looking at the failed attempt by {}, on {advisor_model}", task.id, attempt.model);
    let request = AgentRequest {
        model: advisor_model.to_string(),
        preamble: format!("{}{}", prompts::ADVISOR_SYSTEM_PROMPT, design_section(&settings.design_doc)),
        prompt,
        tools: ToolAccess::Inspect,
        sandbox: settings.sandbox.clone(),
        max_turns: REVIEW_MAX_TURNS,
    };
    match run_with_backoff(llm, request, settings.retry_delay).await {
        Ok(reply) => AgentRun {
            outcome: parse_advice(&reply.text)
                .map_err(|e| LlmError::Other(anyhow::anyhow!("the advisor's answer was unusable: {e}"))),
            usage: reply.usage,
            calls: 1,
            reply: Some(reply.text),
        },
        Err(e) => AgentRun { outcome: Err(e), usage: TokenUsage::default(), calls: 0, reply: None },
    }
}

/// Run the QA reviewer on a task that a worker reported as complete.
pub async fn review_task(
    llm: &dyn Llm,
    task: &ProjectTask,
    summary: &str,
    settings: &AgentSettings,
) -> AgentRun<Result<ReviewVerdict, LlmError>> {
    let files = if task.files.is_empty() {
        "(the worker did not list any files; inspect the project directory)".to_string()
    } else {
        task.files.iter().map(|f| format!("- {f}")).collect::<Vec<_>>().join("\n")
    };
    let subject = format!(
        "# Task under review: {} ({})\n\n{}\n{}\n\
         # What the worker says it did\n\nThis is a claim, not evidence.\n\n{}\n\n\
         # Files the worker says it created or modified\n\n{}\n",
        task.title,
        task.id,
        task.description,
        plan_section(&settings.plan),
        summary,
        files,
    );

    tracing::info!("Reviewer [{}]: starting on {}", task.id, settings.review_model);
    let run = verify_items(llm, subject, &review_items(task), REVIEW_MAX_TURNS, settings).await;
    AgentRun {
        outcome: run.outcome.map(|findings| verdict_from(&findings)),
        usage: run.usage,
        calls: run.calls,
        reply: run.reply,
    }
}

/// The items the finished project is checked against.
pub fn design_check_items(requirements: &[Requirement]) -> Vec<CheckItem> {
    baseline(PROJECT_BASELINE)
        .into_iter()
        .chain(requirements.iter().map(|r| CheckItem {
            text: r.text.clone(),
            // Behaviour has to be established; only appearance may be left to the human.
            decidable: !r.appearance,
            needs_quote: true,
        }))
        .collect()
}

/// Check the whole project in the output directory against the design's requirements.
pub async fn check_project(
    llm: &dyn Llm,
    requirements: &[Requirement],
    settings: &AgentSettings,
) -> AgentRun<Result<Vec<Finding>, LlmError>> {
    let subject = "# Whole-project check\n\n\
        The project in the project directory is supposed to implement the design document. \
        Check it against every item below.\n"
        .to_string();
    let items = design_check_items(requirements);
    tracing::info!(
        "Design check: verifying {} items ({} from the design, {} that apply to every project)",
        items.len(),
        requirements.len(),
        items.len() - requirements.len()
    );
    verify_items(llm, subject, &items, DESIGN_CHECK_MAX_TURNS, settings).await
}

/// Turn a design document into a checklist of concrete requirements. This is
/// done without looking at any code, so the list cannot be shaped by what
/// happens to have been built.
pub async fn extract_requirements(
    llm: &dyn Llm,
    settings: &AgentSettings,
) -> Result<(Vec<Requirement>, TokenUsage), LlmError> {
    let request = AgentRequest {
        model: settings.review_model.clone(),
        preamble: prompts::QA_SYSTEM_PROMPT.to_string(),
        prompt: format!("{}\nDesign document:\n\n{}", prompts::REQUIREMENTS_INSTRUCTIONS, settings.design_doc),
        tools: ToolAccess::None,
        sandbox: settings.sandbox.clone(),
        max_turns: 1,
    };
    let reply = run_with_backoff(llm, request, settings.retry_delay).await?;
    let unusable = |why: &str| LlmError::Other(anyhow::anyhow!("the requirement list was unusable: {why}"));
    let value = extract_json(&reply.text, '[', ']').ok_or_else(|| unusable("no JSON list in the reply"))?;
    let requirements: Vec<Requirement> = value
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|item| {
            let text = item.as_str().or_else(|| item.get("requirement").and_then(Value::as_str))?.trim();
            let kind = item.get("kind").and_then(Value::as_str).unwrap_or("behaviour");
            (!text.is_empty()).then(|| Requirement {
                text: text.to_string(),
                appearance: kind.trim().eq_ignore_ascii_case("appearance"),
            })
        })
        .take(MAX_REQUIREMENTS)
        .collect();
    if requirements.is_empty() {
        return Err(unusable("it is empty"));
    }
    Ok((requirements, reply.usage))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::AgentRole;
    use serde_json::json;

    #[test]
    fn extracts_bare_fenced_and_embedded_json() {
        assert_eq!(extract_json(r#"{"a":1}"#, '{', '}'), Some(json!({"a":1})));
        assert_eq!(
            extract_json("Here you go:\n```json\n{\"a\": 1}\n```\nDone.", '{', '}'),
            Some(json!({"a":1}))
        );
        assert_eq!(extract_json("```\n[1, 2]\n```", '[', ']'), Some(json!([1, 2])));
        assert_eq!(extract_json("The plan is [ {\"id\": \"T\"} ] as requested", '[', ']'), Some(json!([{"id":"T"}])));
        // A code sample earlier in the reply must not shadow the final report.
        assert_eq!(
            extract_json("```rust\nfn main() {}\n```\n```json\n{\"summary\":\"ok\"}\n```", '{', '}'),
            Some(json!({"summary":"ok"}))
        );
    }

    #[test]
    fn hostile_replies_do_not_panic() {
        // Seen live: a reviewer answered, wrote "Wait - I must correct the
        // ordering", and answered again. The second answer is the answer.
        assert_eq!(
            extract_json("[{\"id\":1,\"status\":\"met\"}]\n\nWait, the correct answer is:\n\n[{\"id\":1,\"status\":\"unmet\"}]", '[', ']'),
            Some(json!([{"id":1,"status":"unmet"}]))
        );
        assert_eq!(
            extract_json("{\n  \"a\": [1]\n}\nNo, rather:\n{\n  \"a\": [2]\n}\nThat is all.", '{', '}'),
            Some(json!({"a":[2]}))
        );
        // A fragment of a broken answer is not an answer.
        assert_eq!(extract_json("I ran it:\nfn main() { println!(\"{}\", x) }\nand it works", '{', '}'), None);
        assert_eq!(
            extract_json("[\n  {\"id\": \"T1\", \"dependencies\": []},\n  {\"id\": \"T2\", \"dependencies\": [\"T1\"]},\n  {\"id\":", '[', ']'),
            None
        );
        assert_eq!(
            extract_json("{\"status\": \"completed\", \"new_tasks\": [\n{\"title\": \"x\", \"dependencies\": []}\n], \"summary\": \"cut off", '{', '}'),
            None
        );
        assert_eq!(extract_json("", '{', '}'), None);
        assert_eq!(extract_json("no json here", '{', '}'), None);
        assert_eq!(extract_json("} backwards {", '{', '}'), None);
        assert_eq!(extract_json("{", '{', '}'), None);
        assert_eq!(extract_json("```json", '{', '}'), None);
        assert_eq!(extract_json("ünïcödé } { ünïcödé", '{', '}'), None);
    }

    #[test]
    fn worker_reports_are_parsed() {
        let outcome = parse_worker_reply(
            r#"{"status":"completed","summary":"Built it","files_created":["src/main.rs"],
                "new_tasks":[{"title":"Docs","description":"Write docs","assigned_role":"frontend","dependencies":[]},
                             {"title":"broken"}]}"#,
        );
        let WorkerOutcome::Completed { summary, files, new_tasks } = outcome else {
            panic!("expected completion");
        };
        assert_eq!(summary, "Built it");
        assert_eq!(files, vec!["src/main.rs"]);
        assert_eq!(new_tasks.len(), 1, "the malformed follow-up task is dropped");
        assert_eq!(new_tasks[0].assigned_role, AgentRole::Frontend);
    }

    #[test]
    fn worker_can_report_failure() {
        assert_eq!(
            parse_worker_reply(r#"{"status":"failed","summary":"cargo is not installed"}"#),
            WorkerOutcome::Failed { reason: "cargo is not installed".into() }
        );
        assert!(matches!(parse_worker_reply("   "), WorkerOutcome::Failed { .. }));
    }

    #[test]
    fn prose_reply_goes_to_review_instead_of_being_discarded() {
        let outcome = parse_worker_reply("I created the project and it builds.");
        assert_eq!(
            outcome,
            WorkerOutcome::Completed {
                summary: "I created the project and it builds.".into(),
                files: vec![],
                new_tasks: vec![]
            }
        );
    }

    fn items(n: usize) -> Vec<CheckItem> {
        (1..=n).map(|i| CheckItem::requirement(&format!("requirement {i}"))).collect()
    }

    fn sandbox_with(file: &str, content: &str) -> (tempfile::TempDir, Sandbox) {
        let dir = tempfile::tempdir().unwrap();
        let sandbox = Sandbox::new(dir.path().join("out"), Duration::from_secs(5)).unwrap();
        let path = sandbox.root().join(file);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
        (dir, sandbox)
    }

    #[test]
    fn findings_are_parsed_per_item() {
        let reply = r#"Here is my review.
```json
[
  {"id": 1, "status": "met", "file": "src/lib.rs", "quote": "pub fn sum(", "evidence": "test_sum passes"},
  {"id": "2", "status": "UNMET", "evidence": "'(' is not accepted: src/app.rs:52 only pushes digits and + - * /"},
  {"id": 3, "status": "cannot_verify", "evidence": "colours need a terminal"}
]
```"#;
        let answers = parse_findings(reply, &items(3)).unwrap();
        assert_eq!(
            answers.iter().map(|a| a.finding.status).collect::<Vec<_>>(),
            [FindingStatus::Met, FindingStatus::Unmet, FindingStatus::CannotVerify]
        );
        assert_eq!(answers[0].file.as_deref(), Some("src/lib.rs"));
        assert_eq!(answers[0].quote.as_deref(), Some("pub fn sum("));
        assert_eq!(answers[1].finding.requirement, "requirement 2");
        assert!(answers[1].finding.evidence.contains("src/app.rs:52"));
    }

    #[test]
    fn incomplete_or_malformed_answers_are_errors_not_passes() {
        let two = items(2);
        assert!(parse_findings("Everything looks great. PASSED", &two).unwrap_err().contains("JSON list"));
        assert!(parse_findings(r#"[{"id":1,"status":"met","evidence":"x"}]"#, &two).unwrap_err().contains("items 2"));
        assert!(
            parse_findings(r#"[{"id":1,"status":"met","evidence":"x"},{"id":2,"status":"fine i guess"}]"#, &two)
                .unwrap_err()
                .contains("unknown status")
        );
        // Answers for items that were never asked about are ignored.
        assert!(parse_findings(r#"[{"id":7,"status":"met","evidence":"x"}]"#, &two).is_err());
        assert!(parse_findings("[]", &two).is_err());
    }

    const APP_RS: &str = "fn handle(key: KeyCode) {\n    match key {\n        KeyCode::Char('=') => self.evaluate_expression(),\n        KeyCode::Esc => return,\n    }\n}\n";

    fn audited(reply: &str, items: &[CheckItem]) -> (Vec<Finding>, Vec<(usize, String)>) {
        let (_dir, sandbox) = sandbox_with("src/app.rs", APP_RS);
        audit(parse_findings(reply, items).unwrap(), items, &sandbox)
    }

    #[test]
    fn met_counts_only_with_a_quote_that_is_really_in_the_file() {
        let item = items(1);
        // An honest quote, with different indentation and a line number on the file name.
        let (findings, problems) = audited(
            r#"[{"id":1,"status":"met","file":"src/app.rs:3","quote":"KeyCode::Char('=')   =>   self.evaluate_expression(),","evidence":"= evaluates"}]"#,
            &item,
        );
        assert_eq!(findings[0].status, FindingStatus::Met);
        assert!(problems.is_empty());

        // What the first real design check did: a confident description, no code.
        for reply in [
            r#"[{"id":1,"status":"met","evidence":"Status/footer component with keybindings in draw function, src/app.rs."}]"#,
            r#"[{"id":1,"status":"met","file":"src/app.rs","quote":"KeyCode::Enter => self.evaluate_expression(),","evidence":"Enter evaluates"}]"#,
            r#"[{"id":1,"status":"met","file":"src/nope.rs","quote":"KeyCode::Esc => return,","evidence":"x"}]"#,
            r#"[{"id":1,"status":"met","file":"../../etc/passwd","quote":"root:x:0:0:root","evidence":"x"}]"#,
            r#"[{"id":1,"status":"met","file":"src/app.rs","quote":"}","evidence":"too short to mean anything"}]"#,
            // A fragment that could be anywhere, and no quote from a file that has code to quote.
            r#"[{"id":1,"status":"met","file":"src/app.rs","quote":"KeyCode","evidence":"keys are handled"}]"#,
            r#"[{"id":1,"status":"met","file":"src/app.rs","quote":"","evidence":"it is all there"}]"#,
        ] {
            let (findings, problems) = audited(reply, &item);
            assert_eq!(findings[0].status, FindingStatus::Unmet, "{reply}");
            assert_eq!(problems.len(), 1, "{reply}");
            assert!(findings[0].evidence.contains("could not quote code"), "{}", findings[0].evidence);
        }
    }

    /// Seen live: a planner asked for a project skeleton ("src/lib.rs exists
    /// (can be empty)", "src/main.rs exists"), the worker built it, the
    /// reviewer said so, and the quote check failed the task four times.
    #[test]
    fn a_correct_review_of_a_project_skeleton_is_accepted() {
        let dir = tempfile::tempdir().unwrap();
        let sandbox = Sandbox::new(dir.path().join("out"), Duration::from_secs(5)).unwrap();
        let write = |file: &str, content: &str| {
            let path = sandbox.root().join(file);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, content).unwrap();
        };
        write("src/lib.rs", "");
        write("src/main.rs", "fn main() {\n    // Placeholder main for now\n}");
        write("Cargo.toml", "[package]\nname = \"sumcli\"\nversion = \"0.1.0\"\nedition = \"2021\"\n");

        let item = items(1);
        let check = |file: &str, quote: &str| {
            let reply = json!([{"id": 1, "status": "met", "file": file, "quote": quote, "evidence": "it exists"}]);
            let (findings, problems) = audit(parse_findings(&reply.to_string(), &item).unwrap(), &item, &sandbox);
            (findings[0].status, problems.into_iter().map(|(_, text)| text).collect::<String>())
        };
        // A whole line counts however short it is.
        assert_eq!(check("src/main.rs", "fn main() {").0, FindingStatus::Met);
        // An empty file has nothing to quote; that it exists is what was checked.
        assert_eq!(check("src/lib.rs", "").0, FindingStatus::Met);
        assert_eq!(check("Cargo.toml", "name = \"sumcli\"").0, FindingStatus::Met);

        // The reviewer is told what is actually wrong, so that it can correct it.
        let (status, problem) = check("src/main.rs", "main()");
        assert_eq!(status, FindingStatus::Unmet);
        assert!(problem.contains("only a fragment"), "{problem}");
        let (status, problem) = check("src/missing.rs", "");
        assert_eq!(status, FindingStatus::Unmet);
        assert!(problem.contains("is not a file in the project"), "{problem}");
        // A quote that is not in an empty file is still not in it.
        assert_eq!(check("src/lib.rs", "pub fn sum(numbers: &[i64]) -> i64 {").0, FindingStatus::Unmet);
        // Closing punctuation is a whole line too, and still says nothing.
        assert_eq!(check("src/main.rs", "}").0, FindingStatus::Unmet);
    }

    #[test]
    fn items_decidable_from_the_code_cannot_be_dodged() {
        let requirement = |text: &str, appearance| Requirement { text: text.into(), appearance };
        let project = design_check_items(&[
            requirement("Typing '15 + 3 * (10 / 2)' and pressing Enter shows 30", false),
            requirement("Borders are cyan", true),
        ]);
        let example = project.len() - 1;
        let entry_point = project.iter().position(|i| i.text.contains("entry point")).unwrap() + 1;
        let reply = serde_json::to_string(
            &project
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    if i + 1 == entry_point {
                        // The answer the first real check gave for a TUI that never enters raw mode.
                        json!({"id": i + 1, "status": "cannot_verify", "evidence": "main initializes and runs CalculatorApp"})
                    } else if i + 1 == example {
                        // The answer the second real check gave for every example in the design.
                        json!({"id": i + 1, "status": "cannot_verify", "evidence": "Cannot verify without running the program interactively."})
                    } else if i + 1 == project.len() {
                        json!({"id": i + 1, "status": "cannot_verify", "evidence": "needs a terminal"})
                    } else {
                        json!({"id": i + 1, "status": "met", "evidence": "Finished `dev` profile"})
                    }
                })
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let (findings, problems) = audited(&reply, &project);
        assert_eq!(findings[entry_point - 1].status, FindingStatus::Unmet);
        assert_eq!(findings[example - 1].status, FindingStatus::Unmet, "an input and its result is behaviour");
        assert!(findings[example - 1].evidence.starts_with("Not established"));
        assert_eq!(problems.len(), 2);
        assert_eq!(problems[1].0, example - 1);
        assert!(problems[1].1.contains("a test that runs exactly that input"));
        // Only appearance may be left to the human.
        assert_eq!(findings.last().unwrap().status, FindingStatus::CannotVerify);
        // Build and test items are evidenced by command output, not by a quote.
        assert_eq!(findings[0].status, FindingStatus::Met);

        let (findings, problems) = audited(r#"[{"id":1,"status":"met"}]"#, &baseline(TASK_BASELINE)[..1]);
        assert_eq!(findings[0].status, FindingStatus::Unmet, "met with no evidence at all");
        assert_eq!(problems.len(), 1);
    }

    #[test]
    fn the_code_not_the_model_decides_the_verdict() {
        let finding = |status, evidence: &str| Finding {
            requirement: "Enter evaluates the expression".into(),
            status,
            evidence: evidence.into(),
            overruled: false,
        };
        assert_eq!(verdict_from(&[finding(FindingStatus::Met, "src/app.rs:40")]), ReviewVerdict::Passed { notes: vec![] });

        let ReviewVerdict::Passed { notes } =
            verdict_from(&[finding(FindingStatus::Met, "ok"), finding(FindingStatus::CannotVerify, "needs a terminal")])
        else {
            panic!("expected a pass");
        };
        assert_eq!(notes.len(), 1);
        assert!(notes[0].contains("needs a terminal"));

        let ReviewVerdict::Rejected(feedback) =
            verdict_from(&[finding(FindingStatus::Met, "ok"), finding(FindingStatus::Unmet, "only '=' is handled")])
        else {
            panic!("one unmet item must reject");
        };
        assert!(feedback.contains("Enter evaluates the expression"));
        assert!(feedback.contains("only '=' is handled"));
    }

    #[test]
    fn every_task_is_reviewed_against_the_baseline() {
        let mut task = crate::context::tests::task("TASK-1", &[]);
        let fallback = review_items(&task);
        assert_eq!(fallback.len(), TASK_BASELINE.len() + 1);
        assert!(fallback.last().unwrap().text.contains("Description of TASK-1"));

        task.acceptance_criteria = vec!["Typing `15 + 3 * (10 / 2)` and Enter shows 30".into()];
        let with_criteria = review_items(&task);
        assert!(with_criteria[0].text.contains("builds") && with_criteria[0].decidable && !with_criteria[0].needs_quote);
        assert!(with_criteria.iter().any(|i| i.text.contains("automated tests")));
        let own = with_criteria.last().unwrap();
        assert_eq!(own.text, "Typing `15 + 3 * (10 / 2)` and Enter shows 30");
        assert!(own.needs_quote && !own.decidable);
        assert!(design_check_items(&[]).iter().any(|i| i.text.contains("entry point") && i.needs_quote));
    }

    #[test]
    fn only_transient_errors_are_worth_waiting_for() {
        assert!(is_permanent("CompletionError: HTTP 401 Unauthorized"));
        assert!(is_permanent("Incorrect API key provided (invalid_api_key)"));
        assert!(is_permanent("OPENAI_API_KEY is not set"));
        assert!(is_permanent(crate::llm::FREE_ALLOWANCE_USED_UP));
        assert!(!crate::llm::is_transient(crate::llm::FREE_ALLOWANCE_USED_UP));
        assert!(is_permanent("Invalid status code 402 Payment Required"));
        assert!(!is_permanent("HTTP 429 Too Many Requests: rate limit reached"));
        assert!(is_permanent("HttpError: Invalid status code 403 Forbidden"));
        // Seen in a test of this very thing: digits inside a rate-limit message.
        assert!(!is_permanent("HttpError: Invalid status code 429 Too Many Requests with message: Requested 40312 tokens"));
        assert!(!is_permanent("ProviderError: limit of 4010 requests reached, retry shortly"));
        assert!(!is_permanent("503 service unavailable"));
        assert!(!is_permanent("connection reset by peer"));
    }
}
