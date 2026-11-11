pub mod prompts;

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use tokio::sync::{Mutex, Semaphore};
use tokio::task::JoinSet;

use crate::agents::{self, AgentSettings, ReviewVerdict, WorkerOutcome};
use crate::context::{
    AgentRole, Finding, MAX_CONSULTS, MAX_RETRIES, Requirement, ProjectContext, ProjectTask, RunPhase, STATE_DIR, TaskDef,
    TaskStatus, validate_task_graph, STEP_UP_AFTER,
};
use crate::llm::{AgentRequest, Llm, TokenUsage};
use crate::tools::{Sandbox, ToolAccess};

const MAX_CONCURRENT_REQUESTS: usize = 4;
/// Planning gets a second try if the first plan is unusable.
const PLANNING_ATTEMPTS: usize = 2;
/// Ceiling on the task list, so workers reporting follow-up tasks cannot grow it without bound.
const MAX_TASKS: usize = 40;
/// How often the run loop looks at the context while it waits for a human.
const HUMAN_POLL_INTERVAL: Duration = Duration::from_millis(200);
const PRIOR_SUMMARY_CHARS: usize = 1_500;
/// The design check gets a second try if the reviewer's answer is unusable.
const DESIGN_CHECK_ATTEMPTS: usize = 2;
/// Base delay before a failed provider call is tried again.
pub const PROVIDER_RETRY_DELAY: Duration = Duration::from_secs(15);

pub type SharedContext = Arc<Mutex<ProjectContext>>;

/// Where human approvals come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hitl {
    /// Ask on stdin (headless mode).
    Stdin,
    /// Something else (the TUI) approves or rejects by editing the shared context.
    External,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunEnd {
    /// Every task is Completed or Failed.
    Finished,
    /// The step limit was reached with work remaining.
    StepLimit,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct IterationReport {
    pub executed: usize,
    pub reviewed: usize,
    pub auto_approved: usize,
    /// Tasks failed without an attempt, because no worker model is left.
    pub given_up: usize,
}

pub struct Orchestrator {
    llm: Arc<dyn Llm>,
    design_doc: String,
    sandbox: Sandbox,
    semaphore: Arc<Semaphore>,
    retry_delay: Duration,
}

/// Turn the planner's reply into a validated task list.
pub fn parse_plan(reply: &str) -> Result<Vec<ProjectTask>, String> {
    let value = agents::extract_json(reply, '[', ']')
        .ok_or_else(|| "the reply did not contain a JSON list".to_string())?;
    let defs: Vec<TaskDef> =
        serde_json::from_value(value).map_err(|e| format!("the task list is malformed: {e}"))?;
    let tasks: Vec<ProjectTask> = defs
        .into_iter()
        .enumerate()
        .map(|(i, mut def)| {
            let id = def
                .id
                .take()
                .filter(|id| !id.trim().is_empty())
                .unwrap_or_else(|| format!("TASK-{}", i + 1));
            ProjectTask::new(id.trim().to_string(), def)
        })
        .collect();
    if tasks.len() > MAX_TASKS {
        return Err(format!("the plan has {} tasks; the maximum is {MAX_TASKS}", tasks.len()));
    }
    validate_task_graph(&tasks)?;
    Ok(tasks)
}

/// What the harness knows about each worker model, for a model that has to
/// choose between them. A model's name alone may mean nothing to it.
pub fn describe_pool(ctx: &ProjectContext) -> String {
    let free_left = ctx.free_requests_at_start.map(|left| left.saturating_sub(crate::llm::free_model_requests()));
    ctx.pool()
        .iter()
        .map(|model| {
            let mut facts: Vec<String> = Vec::new();
            if crate::llm::uses_free_tier(&[model]) {
                facts.push(match free_left {
                    Some(left) => format!(
                        "free of charge, but all free models together have about {left} requests left today"
                    ),
                    None => "free of charge, but limited to a small number of requests a day".to_string(),
                });
                facts.push("one attempt at a small task takes about 10 requests".to_string());
            } else {
                facts.push("paid".to_string());
            }
            if let Some(used) = ctx.model_usage.get(*model)
                && used.passed + used.rejected > 0
            {
                facts.push(format!(
                    "so far in this run {} of its task attempts passed review and {} failed",
                    used.passed, used.rejected
                ));
            }
            if let Some((_, reason)) = ctx.retired_models.iter().find(|(retired, _)| retired == model) {
                facts.push(format!("CANNOT BE USED in this run ({})", truncate(reason, 120)));
            }
            format!("- {model}: {}\n", facts.join("; "))
        })
        .collect()
}

/// The worker models a planner may assign, and what is known about them.
#[derive(Debug, Clone, Default)]
pub struct WorkerChoice {
    /// Names the plan may use. Empty: tasks are not assigned by the planner.
    pub models: Vec<String>,
    pub facts: String,
}

impl WorkerChoice {
    /// Assignment is only asked for when there is something to choose between.
    pub fn of(ctx: &ProjectContext) -> Self {
        let models: Vec<String> = ctx.usable_pool().iter().map(|m| m.to_string()).collect();
        if models.len() < 2 { Self::default() } else { Self { models, facts: describe_pool(ctx) } }
    }
}

/// Ask the model to decompose a design document into tasks.
pub async fn plan_tasks(
    llm: &dyn Llm,
    model: &str,
    design: &str,
    sandbox: &Sandbox,
    workers: &WorkerChoice,
) -> Result<(Vec<ProjectTask>, TokenUsage, u64)> {
    let assignment = if workers.models.is_empty() {
        String::new()
    } else {
        format!("{}{}", prompts::WORKER_MODELS_INSTRUCTIONS, workers.facts)
    };
    let base_prompt =
        format!("{}{assignment}\nDesign document:\n\n{}", prompts::PLANNING_INSTRUCTIONS, design);
    let mut usage = TokenUsage::default();
    let mut calls = 0;
    let mut problem = String::new();

    for attempt in 1..=PLANNING_ATTEMPTS {
        let prompt = if problem.is_empty() {
            base_prompt.clone()
        } else {
            format!(
                "{base_prompt}\n\nYour previous answer could not be used: {problem}.\n\
                 Answer again with a corrected JSON list only."
            )
        };
        let reply = llm
            .run(AgentRequest {
                model: model.to_string(),
                preamble: prompts::PLANNER_SYSTEM_PROMPT.to_string(),
                prompt,
                tools: ToolAccess::None,
                sandbox: sandbox.clone(),
                max_turns: 1,
            })
            .await
            .map_err(|e| anyhow!("planning failed: {e}"))?;
        calls += 1;
        usage += reply.usage;

        match parse_plan(&reply.text) {
            Ok(mut tasks) => {
                tracing::info!("Planner: produced {} tasks", tasks.len());
                for task in &mut tasks {
                    match &task.model {
                        Some(chosen) if workers.models.contains(chosen) => {
                            tracing::info!("Planner: {} ({}) goes to {chosen}", task.id, task.title);
                        }
                        Some(unknown) => {
                            // Not worth another planning call: the default order applies.
                            tracing::warn!(
                                "Planner: {} was given to {unknown:?}, which is not one of the worker models; \
                                 it starts on the first usable one instead",
                                task.id
                            );
                            task.model = None;
                        }
                        None => {}
                    }
                }
                return Ok((tasks, usage, calls));
            }
            Err(e) => {
                tracing::warn!("Planner: attempt {attempt} was unusable: {e}");
                tracing::debug!("Planner reply was: {}", reply.text);
                problem = e;
            }
        }
    }
    bail!("the model did not produce a usable task plan: {problem}")
}

fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        text.to_string()
    } else {
        let cut: String = text.chars().take(max_chars).collect();
        format!("{cut} [...]")
    }
}

fn describe_prior_work(ctx: &ProjectContext) -> String {
    ctx.tasks
        .iter()
        .filter_map(|t| match &t.status {
            TaskStatus::Completed { output_summary } => {
                let files = if t.files.is_empty() {
                    String::new()
                } else {
                    format!("\n  Files: {}", t.files.join(", "))
                };
                Some(format!(
                    "- {} {}: {}{}\n",
                    t.id,
                    t.title,
                    truncate(output_summary, PRIOR_SUMMARY_CHARS),
                    files
                ))
            }
            _ => None,
        })
        .collect()
}

fn describe_plan(ctx: &ProjectContext) -> String {
    ctx.tasks
        .iter()
        .map(|t| format!("- {} [{:?}] {} ({})\n", t.id, t.assigned_role, t.title, t.status.label()))
        .collect()
}

/// Keep a model's raw reply under `.clankistrator/transcripts/`, so that a run
/// can be audited afterwards (why did this review pass?).
fn save_transcript(root: &Path, name: &str, reply: Option<&str>) {
    let Some(reply) = reply else { return };
    let safe: String =
        name.chars().map(|c| if c.is_ascii_alphanumeric() || "-_.".contains(c) { c } else { '_' }).collect();
    let dir = root.join(STATE_DIR).join("transcripts");
    let written = std::fs::create_dir_all(&dir).and_then(|_| {
        let path = (1..)
            .map(|n| dir.join(format!("{safe}.{n}.md")))
            .find(|path| !path.exists())
            .expect("an unused file name exists");
        std::fs::write(path, reply)
    });
    if let Err(e) = written {
        tracing::warn!("Could not save the transcript for {name}: {e}");
    }
}

/// The outcome of checking a project directory against a design document.
pub struct DesignCheck {
    pub requirements: Vec<Requirement>,
    pub findings: Vec<Finding>,
    pub usage: TokenUsage,
    pub calls: u64,
    /// The reviewer's raw answer.
    pub reply: Option<String>,
}

/// Check the project in the sandbox against the design: extract the design's
/// requirements (unless given), then have the reviewer answer each one with
/// evidence. Changes nothing in the project and saves nothing.
pub async fn check_design(
    llm: &dyn Llm,
    known_requirements: &[Requirement],
    settings: &AgentSettings,
) -> Result<DesignCheck> {
    let mut usage = TokenUsage::default();
    let mut calls = 0;
    let mut add = |more: TokenUsage| usage += more;

    let requirements = if known_requirements.is_empty() {
        let (requirements, used) = agents::extract_requirements(llm, settings)
            .await
            .map_err(|e| anyhow!("could not list the design's requirements: {e}"))?;
        calls += 1;
        add(used);
        tracing::info!("Design check: the design has {} checkable requirements", requirements.len());
        requirements
    } else {
        known_requirements.to_vec()
    };

    let mut problem = String::new();
    for attempt in 1..=DESIGN_CHECK_ATTEMPTS {
        let run = agents::check_project(llm, &requirements, settings).await;
        calls += run.calls;
        add(run.usage);
        match run.outcome {
            Ok(findings) => {
                return Ok(DesignCheck { requirements, findings, usage, calls, reply: run.reply });
            }
            Err(e) => {
                tracing::warn!("Design check: attempt {attempt} did not produce a usable answer: {e}");
                problem = e.to_string();
            }
        }
    }
    bail!("the design check could not be completed: {problem}")
}

fn record_usage(ctx: &mut ProjectContext, model: &str, usage: TokenUsage, calls: u64) {
    ctx.input_tokens += usage.input_tokens;
    ctx.cached_input_tokens += usage.cached_input_tokens;
    ctx.output_tokens += usage.output_tokens;
    ctx.llm_calls += calls;
    let of_model = ctx.model_usage.entry(model.to_string()).or_default();
    of_model.calls += calls;
    of_model.input_tokens += usage.input_tokens;
    of_model.cached_input_tokens += usage.cached_input_tokens;
    of_model.output_tokens += usage.output_tokens;
}

/// A worker's attempt on `model` failed for a reason that lies in the work.
/// After `STEP_UP_AFTER` such failures the task moves to the next worker
/// model. The count of attempts (`retries`) is not reset by the move, so a
/// task cannot go round the pool for ever.
fn note_failed_attempt(ctx: &mut ProjectContext, task_id: &str, model: &str) {
    ctx.model_usage.entry(model.to_string()).or_default().rejected += 1;
    let has_advisor = ctx.advisor_model.is_some();
    let Some(failures) = ctx.get_task_mut(task_id).map(|t| {
        t.failures_on_model += 1;
        if has_advisor && t.consults < MAX_CONSULTS {
            t.advice_wanted_after = Some(model.to_string());
        }
        t.failures_on_model
    }) else {
        return;
    };
    if failures < STEP_UP_AFTER {
        return;
    }
    if let Some(next) = ctx.next_model_up(model) {
        tracing::warn!(
            "Orchestrator [{task_id}]: {failures} failed attempts on {model}; the next attempt goes to {next}"
        );
        if let Some(t) = ctx.get_task_mut(task_id) {
            t.model = Some(next);
            t.failures_on_model = 0;
        }
    }
}

/// Give the advisor's guidance to the worker, and the task to the model the advisor chose.
fn apply_advice(ctx: &mut ProjectContext, task_id: &str, next_model: &str, advice: agents::Advice) {
    let chosen = advice.model.filter(|model| model != next_model);
    let usable = chosen.as_ref().is_some_and(|model| ctx.usable_pool().contains(&model.as_str()));
    let Some(t) = ctx.get_task_mut(task_id) else { return };
    t.advice = Some(advice.guidance);
    match chosen {
        Some(model) if usable => {
            tracing::warn!("Orchestrator [{task_id}]: the advisor moved the task from {next_model} to {model}");
            t.model = Some(model);
            t.failures_on_model = 0;
        }
        Some(model) => tracing::warn!(
            "Orchestrator [{task_id}]: the advisor named {model:?}, which is not a usable worker model; \
             the task stays on {next_model}"
        ),
        None => tracing::info!("Orchestrator [{task_id}]: the advisor gave guidance; the next attempt runs on {next_model}"),
    }
}

fn apply_worker_outcome(ctx: &mut ProjectContext, task_id: &str, model: &str, outcome: WorkerOutcome) {
    match outcome {
        WorkerOutcome::ModelUnavailable { reason } => {
            ctx.retire_model(model, &reason);
            let replacement = ctx.usable_pool().first().map(|m| m.to_string());
            match replacement {
                Some(next) => {
                    // The task is not at fault: hand it on without counting an attempt.
                    tracing::warn!(
                        "Orchestrator [{task_id}]: {model} cannot be used for the rest of this run ({reason}); \
                         its tasks go to {next}"
                    );
                    if let Some(t) = ctx.get_task_mut(task_id) {
                        t.status = TaskStatus::Pending;
                        t.model = None;
                        t.failures_on_model = 0;
                    }
                }
                None => {
                    tracing::error!("Orchestrator [{task_id}]: {model} cannot be used and no other worker model is left: {reason}");
                    if let Some(t) = ctx.get_task_mut(task_id) {
                        t.status = TaskStatus::Failed { error: format!("The model call failed: {reason}") };
                    }
                }
            }
        }
        other => apply_attempt(ctx, task_id, model, other),
    }
}

fn save(ctx: &ProjectContext) {
    if let Err(e) = ctx.save() {
        tracing::warn!("Could not save state: {e:#}");
    }
}

fn apply_attempt(ctx: &mut ProjectContext, task_id: &str, model: &str, outcome: WorkerOutcome) {
    match outcome {
        WorkerOutcome::ModelUnavailable { .. } => {}
        WorkerOutcome::Completed { summary, files, new_tasks } => {
            let inherited_model = ctx.get_task(task_id).and_then(|t| t.model.clone());
            if let Some(t) = ctx.get_task_mut(task_id) {
                t.status = TaskStatus::ReadyForReview { output_summary: summary };
                t.files = files;
                tracing::info!("Orchestrator [{task_id}]: worker finished, ready for review");
            }
            for mut def in new_tasks {
                // A task that is sent back and redone reports its follow-ups again.
                let title = def.title.trim().to_lowercase();
                if ctx.tasks.iter().any(|t| t.title.trim().to_lowercase() == title) {
                    tracing::debug!("Orchestrator [{task_id}]: follow-up task {:?} already exists", def.title);
                    continue;
                }
                if ctx.tasks.len() >= MAX_TASKS {
                    tracing::warn!(
                        "Orchestrator [{task_id}]: ignoring follow-up task {:?}; the task limit ({MAX_TASKS}) is reached",
                        def.title
                    );
                    continue;
                }
                // Keep only dependencies that exist, and make the follow-up wait for
                // the task that asked for it.
                def.dependencies.retain(|dep| ctx.get_task(dep).is_some());
                if !def.dependencies.iter().any(|dep| dep == task_id) {
                    def.dependencies.push(task_id.to_string());
                }
                // A follow-up belongs with the work it came from, so it stays on that model.
                def.model = inherited_model.clone();
                let id = ctx.next_task_id();
                tracing::info!("Orchestrator [{task_id}]: worker added follow-up task {id}: {}", def.title);
                ctx.tasks.push(ProjectTask::new(id, def));
            }
        }
        WorkerOutcome::Failed { reason } => {
            let mut given_up = true;
            if let Some(t) = ctx.get_task_mut(task_id) {
                t.retries += 1;
                if t.retries > MAX_RETRIES {
                    tracing::error!("Orchestrator [{task_id}]: giving up after {} attempts: {reason}", t.retries);
                    t.status = TaskStatus::Failed { error: reason };
                } else {
                    tracing::warn!("Orchestrator [{task_id}]: attempt failed, will retry: {reason}");
                    t.status = TaskStatus::Pending;
                    t.feedback = Some(reason);
                    given_up = false;
                }
            }
            if given_up {
                ctx.model_usage.entry(model.to_string()).or_default().rejected += 1;
            } else {
                note_failed_attempt(ctx, task_id, model);
            }
        }
    }
}

fn apply_review(
    ctx: &mut ProjectContext,
    task_id: &str,
    summary: String,
    verdict: Result<ReviewVerdict, crate::llm::LlmError>,
) {
    // The model whose work is being judged: the one the attempt was dispatched to.
    let Some(model) = ctx.get_task(task_id).map(|t| t.attempt_model.clone().unwrap_or_else(|| ctx.model.clone()))
    else {
        return;
    };
    match &verdict {
        Ok(ReviewVerdict::Passed { .. }) => ctx.model_usage.entry(model.clone()).or_default().passed += 1,
        Ok(ReviewVerdict::Rejected(_)) => {
            // Counted as a failed attempt of the model only if the task goes back to it.
            if ctx.get_task(task_id).is_some_and(|t| t.retries < MAX_RETRIES) {
                note_failed_attempt(ctx, task_id, &model);
            } else {
                ctx.model_usage.entry(model.clone()).or_default().rejected += 1;
            }
        }
        _ => {}
    }
    let Some(t) = ctx.get_task_mut(task_id) else { return };
    match verdict {
        Ok(ReviewVerdict::Unproven(feedback)) => {
            // Nothing was found wrong with the work, so the model that did it
            // is not marked down and the task does not move to another one.
            t.retries += 1;
            if t.retries > MAX_RETRIES {
                tracing::error!("Orchestrator [{task_id}]: the review could not be substantiated, giving up");
                t.status = TaskStatus::Failed { error: format!("Failed review: {feedback}") };
            } else {
                tracing::warn!(
                    "Orchestrator [{task_id}]: the reviewer found nothing wrong but did not show its evidence; \
                     sending the task back to the worker"
                );
                t.status = TaskStatus::Pending;
                t.feedback = Some(format!("The QA reviewer rejected your work:\n{feedback}"));
            }
        }
        Ok(ReviewVerdict::Passed { notes }) => {
            if notes.is_empty() {
                tracing::info!("Orchestrator [{task_id}]: passed review");
            } else {
                tracing::info!(
                    "Orchestrator [{task_id}]: passed review; {} item(s) could not be verified and are left to you",
                    notes.len()
                );
            }
            t.status = TaskStatus::RequiresHumanApproval { output_summary: summary };
            t.feedback = None;
            t.retries = 0;
            t.failures_on_model = 0;
            t.advice = None;
            t.review_notes = notes;
        }
        Ok(ReviewVerdict::Rejected(feedback)) => {
            t.retries += 1;
            if t.retries > MAX_RETRIES {
                tracing::error!("Orchestrator [{task_id}]: failed review too many times, giving up");
                t.status = TaskStatus::Failed { error: format!("Failed review: {feedback}") };
            } else {
                tracing::warn!("Orchestrator [{task_id}]: failed review, sending back to the worker");
                t.status = TaskStatus::Pending;
                t.feedback = Some(format!("The QA reviewer rejected your work:\n{feedback}"));
            }
        }
        Err(e) => {
            // The review itself broke (network, turn limit). The work is not at
            // fault, so keep it and review again rather than redoing the task.
            t.retries += 1;
            if t.retries > MAX_RETRIES {
                tracing::error!("Orchestrator [{task_id}]: review could not be completed: {e}");
                t.status = TaskStatus::Failed { error: format!("Review could not be completed: {e}") };
            } else {
                tracing::warn!("Orchestrator [{task_id}]: review did not complete ({e}), will review again");
                t.status = TaskStatus::ReadyForReview { output_summary: summary };
            }
        }
    }
}

fn auto_approve(ctx: &mut ProjectContext) -> usize {
    if !ctx.brave_mode {
        return 0;
    }
    let ids: Vec<String> = ctx.get_tasks_for_approval().iter().map(|t| t.id.clone()).collect();
    for id in &ids {
        tracing::info!("Orchestrator [{id}]: brave mode, approving automatically");
        ctx.approve(id);
    }
    ids.len()
}

/// Read one line from stdin without blocking the runtime. `None` means EOF.
async fn read_stdin_line() -> Result<Option<String>> {
    tokio::task::spawn_blocking(|| {
        let mut line = String::new();
        let read = std::io::stdin().read_line(&mut line)?;
        Ok(if read == 0 { None } else { Some(line) })
    })
    .await?
}

impl Orchestrator {
    pub fn new(llm: Arc<dyn Llm>, design_doc: String, sandbox: Sandbox) -> Self {
        Self {
            llm,
            design_doc,
            sandbox,
            semaphore: Arc::new(Semaphore::new(MAX_CONCURRENT_REQUESTS)),
            retry_delay: PROVIDER_RETRY_DELAY,
        }
    }

    /// Plan (unless the context already has tasks) and then iterate until every
    /// task is Completed or Failed, or the step limit is reached. The final
    /// state is recorded in `context.phase` and saved to disk.
    pub async fn run(&self, context: &SharedContext, hitl: Hitl) -> Result<RunEnd> {
        let result = self.run_inner(context, hitl).await;
        let mut ctx = context.lock().await;
        ctx.phase = match &result {
            Ok(RunEnd::Finished) => RunPhase::Finished,
            Ok(RunEnd::StepLimit) => RunPhase::StepLimit,
            Err(e) => RunPhase::Error(format!("{e:#}")),
        };
        save(&ctx);
        result
    }

    async fn run_inner(&self, context: &SharedContext, hitl: Hitl) -> Result<RunEnd> {
        let needs_plan = context.lock().await.tasks.is_empty();
        if needs_plan {
            let (model, workers) = {
                let mut ctx = context.lock().await;
                ctx.phase = RunPhase::Planning;
                (ctx.planner_model().to_string(), WorkerChoice::of(&ctx))
            };
            tracing::info!("Orchestrator: planning tasks from the design document with {model}");
            let (tasks, usage, calls) =
                plan_tasks(self.llm.as_ref(), &model, &self.design_doc, &self.sandbox, &workers).await?;
            let mut ctx = context.lock().await;
            record_usage(&mut ctx, &model, usage, calls);
            ctx.tasks = tasks;
            save(&ctx);
        }

        let mut announced_limit = false;
        loop {
            {
                let mut ctx = context.lock().await;
                for id in ctx.fail_blocked() {
                    tracing::warn!("Orchestrator [{id}]: cannot run because a dependency failed");
                }
                if ctx.all_terminal() {
                    if !ctx.design_check_due() {
                        return Ok(RunEnd::Finished);
                    }
                    drop(ctx);
                    self.design_check(context).await;
                    continue;
                }
                // The limit only matters when there is something left to spend a step
                // on; approvals that are already waiting can still be answered.
                let has_work = !ctx.get_ready_tasks().is_empty() || !ctx.get_tasks_for_review().is_empty();
                if has_work && ctx.max_steps != 0 && ctx.steps_taken >= ctx.max_steps {
                    if hitl == Hitl::Stdin {
                        tracing::info!("Orchestrator: reached the step limit ({})", ctx.max_steps);
                        return Ok(RunEnd::StepLimit);
                    }
                    // In the TUI the limit can be raised, so wait instead of ending.
                    if !announced_limit {
                        tracing::warn!(
                            "Orchestrator: reached the step limit ({}). Raise Max Steps in the menu (m) to continue.",
                            ctx.max_steps
                        );
                        announced_limit = true;
                    }
                    ctx.phase = RunPhase::StepLimit;
                    drop(ctx);
                    tokio::time::sleep(HUMAN_POLL_INTERVAL).await;
                    continue;
                }
                announced_limit = false;
                if ctx.get_tasks_for_approval().is_empty() || ctx.brave_mode {
                    ctx.phase = RunPhase::Running;
                }
            }

            let report = self.run_iteration(context).await?;
            if report.executed + report.reviewed > 0 {
                let mut ctx = context.lock().await;
                ctx.steps_taken += 1;
                save(&ctx);
                continue;
            }
            if report.auto_approved + report.given_up > 0 {
                save(&*context.lock().await);
                continue;
            }

            // Nothing could be executed or reviewed.
            let waiting: Vec<String> = {
                let mut ctx = context.lock().await;
                let waiting: Vec<String> =
                    ctx.get_tasks_for_approval().iter().map(|t| t.id.clone()).collect();
                if !waiting.is_empty() {
                    ctx.phase = RunPhase::AwaitingApproval;
                }
                waiting
            };
            if waiting.is_empty() {
                let ctx = context.lock().await;
                if ctx.all_terminal() || !ctx.get_ready_tasks().is_empty() {
                    continue;
                }
                // Unreachable as long as the task graph is valid, but never spin.
                bail!("no task can make progress; the task graph is inconsistent");
            }
            match hitl {
                Hitl::External => tokio::time::sleep(HUMAN_POLL_INTERVAL).await,
                Hitl::Stdin => {
                    for task_id in waiting {
                        self.ask_on_stdin(context, &task_id).await?;
                    }
                    save(&*context.lock().await);
                }
            }
        }
    }

    fn settings(&self, ctx: &ProjectContext) -> AgentSettings {
        AgentSettings {
            model: ctx.model.clone(),
            review_model: ctx.reviewer_model().to_string(),
            design_doc: self.design_doc.clone(),
            sandbox: self.sandbox.clone(),
            max_turns: ctx.max_turns,
            plan: describe_plan(ctx),
            retry_delay: self.retry_delay,
        }
    }

    /// Check the finished project against the design document. Unmet
    /// requirements become one fix task, as long as fix rounds remain; the
    /// findings stay in the context either way, so the run cannot end looking
    /// successful while requirements are known to be unmet.
    async fn design_check(&self, context: &SharedContext) {
        let (settings, known_requirements) = {
            let mut ctx = context.lock().await;
            ctx.phase = RunPhase::Checking;
            (self.settings(&ctx), ctx.requirements.clone())
        };
        tracing::info!("Orchestrator: all tasks are done; checking the project against the design document");
        let result = check_design(self.llm.as_ref(), &known_requirements, &settings).await;

        let mut ctx = context.lock().await;
        ctx.steps_taken += 1;
        match result {
            Err(e) => {
                tracing::error!("Orchestrator: {e:#}");
                ctx.design_check_error = Some(format!("{e:#}"));
            }
            Ok(check) => {
                save_transcript(self.sandbox.root(), "design-check", check.reply.as_deref());
                record_usage(&mut ctx, &settings.review_model, check.usage, check.calls);
                ctx.requirements = check.requirements;
                ctx.design_checked_task_count = ctx.tasks.len();
                ctx.design_check = Some(check.findings);

                let unmet: Vec<Finding> = ctx.unmet_requirements().into_iter().cloned().collect();
                let unverified = ctx.unverified_requirements().len();
                if unmet.is_empty() {
                    tracing::info!(
                        "Orchestrator: design check found no unmet requirements ({unverified} could not be verified)"
                    );
                } else if ctx.fix_rounds_done < ctx.fix_rounds && ctx.tasks.len() < MAX_TASKS {
                    ctx.fix_rounds_done += 1;
                    let id = ctx.next_task_id();
                    tracing::warn!(
                        "Orchestrator: design check found {} unmet requirement(s); adding fix task {id} (round {} of {})",
                        unmet.len(),
                        ctx.fix_rounds_done,
                        ctx.fix_rounds
                    );
                    let details: Vec<String> = unmet.iter().map(Finding::line).collect();
                    let def = TaskDef {
                        id: None,
                        title: format!("Fix unmet design requirements (round {})", ctx.fix_rounds_done),
                        description: format!(
                            "A check of the finished project against the design document found the requirements \
                             below unmet. Fix every one of them. Each entry is a requirement followed by what the \
                             checker found.\n\n{}",
                            details.join("\n")
                        ),
                        assigned_role: AgentRole::Backend,
                        dependencies: Vec::new(),
                        acceptance_criteria: unmet.iter().map(|f| f.requirement.clone()).collect(),
                        // These are what the earlier work missed, so they go to the
                        // strongest worker model there is.
                        model: if ctx.pool().len() > 1 { ctx.strongest_model() } else { None },
                    };
                    ctx.tasks.push(ProjectTask::new(id, def));
                } else {
                    tracing::error!(
                        "Orchestrator: design check found {} unmet requirement(s) and no fix rounds are left",
                        unmet.len()
                    );
                }
            }
        }
        save(&ctx);
    }

    /// One pass: run every ready task, then review everything that is ready for
    /// review. A failure of one task never aborts the pass.
    pub async fn run_iteration(&self, context: &SharedContext) -> Result<IterationReport> {
        let mut report = IterationReport::default();

        // Settings are read from the context every pass, so changes made in the
        // TUI (model, brave mode) take effect on the next one.
        let (settings, ready, prior_work) = {
            let mut ctx = context.lock().await;
            report.auto_approved += auto_approve(&mut ctx);
            let settings = self.settings(&ctx);
            let prior_work = describe_prior_work(&ctx);
            let candidates: Vec<(ProjectTask, Option<String>)> = ctx
                .get_ready_tasks()
                .into_iter()
                .map(|task| (task.clone(), ctx.worker_model_for(task)))
                .collect();
            let mut ready: Vec<(ProjectTask, String)> = Vec::new();
            for (task, model) in candidates {
                let Some(model) = model else {
                    // Every worker model has dropped out. The run does not reach for a
                    // model that was not offered as a worker.
                    let reasons: Vec<String> =
                        ctx.retired_models.iter().map(|(model, why)| format!("{model}: {why}")).collect();
                    tracing::error!("Orchestrator [{}]: no worker model can be used any more", task.id);
                    if let Some(t) = ctx.get_task_mut(&task.id) {
                        t.status = TaskStatus::Failed {
                            error: format!("No worker model can be used any more. {}", reasons.join("; ")),
                        };
                    }
                    report.given_up += 1;
                    continue;
                };
                tracing::info!(
                    "Orchestrator [{}]: dispatching to {:?} worker on {model}",
                    task.id,
                    task.assigned_role
                );
                if let Some(t) = ctx.get_task_mut(&task.id) {
                    t.status = TaskStatus::InProgress;
                    t.attempt_model = Some(model.clone());
                }
                ready.push((task, model));
            }
            (settings, ready, prior_work)
        };

        // 1. Execution
        let mut running = JoinSet::new();
        for (task, model) in ready {
            let llm = Arc::clone(&self.llm);
            let semaphore = Arc::clone(&self.semaphore);
            let settings = AgentSettings { model, ..settings.clone() };
            let prior_work = prior_work.clone();
            report.executed += 1;
            running.spawn(async move {
                let _permit = semaphore.acquire().await.expect("semaphore is never closed");
                let run = agents::execute_task(llm.as_ref(), &task, &prior_work, &settings).await;
                (task.id, settings.model, run)
            });
        }
        while let Some(joined) = running.join_next().await {
            let mut ctx = context.lock().await;
            match joined {
                Ok((task_id, model, run)) => {
                    save_transcript(self.sandbox.root(), &format!("{task_id}.worker"), run.reply.as_deref());
                    record_usage(&mut ctx, &model, run.usage, run.calls);
                    apply_worker_outcome(&mut ctx, &task_id, &model, run.outcome);
                }
                Err(e) => tracing::error!("Orchestrator: a worker crashed: {e}"),
            }
            save(&ctx);
        }

        // 2. Review
        let to_review: Vec<(ProjectTask, String)> = {
            let mut ctx = context.lock().await;
            // A crashed worker leaves its task InProgress; put it back in the queue.
            for t in ctx.tasks.iter_mut().filter(|t| t.status == TaskStatus::InProgress) {
                t.retries += 1;
                t.status = if t.retries > MAX_RETRIES {
                    TaskStatus::Failed { error: "The worker crashed repeatedly.".into() }
                } else {
                    TaskStatus::Pending
                };
            }
            let mut to_review = Vec::new();
            for t in ctx.tasks.iter_mut() {
                if let TaskStatus::ReadyForReview { output_summary } = &t.status {
                    let summary = output_summary.clone();
                    t.status = TaskStatus::Reviewing { output_summary: summary.clone() };
                    to_review.push((t.clone(), summary));
                }
            }
            to_review
        };

        let mut reviewing = JoinSet::new();
        for (task, summary) in to_review {
            let llm = Arc::clone(&self.llm);
            let semaphore = Arc::clone(&self.semaphore);
            let settings = settings.clone();
            report.reviewed += 1;
            reviewing.spawn(async move {
                let _permit = semaphore.acquire().await.expect("semaphore is never closed");
                let run = agents::review_task(llm.as_ref(), &task, &summary, &settings).await;
                (task.id, summary, run)
            });
        }
        while let Some(joined) = reviewing.join_next().await {
            let mut ctx = context.lock().await;
            match joined {
                Ok((task_id, summary, run)) => {
                    save_transcript(self.sandbox.root(), &format!("{task_id}.review"), run.reply.as_deref());
                    record_usage(&mut ctx, &settings.review_model, run.usage, run.calls);
                    apply_review(&mut ctx, &task_id, summary, run.outcome);
                }
                Err(e) => tracing::error!("Orchestrator: a reviewer crashed: {e}"),
            }
            save(&ctx);
        }

        // 3. Advice. A task whose attempt failed is looked at by the advisor
        // before it is tried again. The step-up rule has already been applied,
        // so if the advisor cannot be reached the task still moves on sensibly.
        let (advisor, to_advise, pool_facts) = {
            let mut ctx = context.lock().await;
            let advisor = ctx.advisor_model.clone();
            let mut to_advise: Vec<(ProjectTask, String, String)> = Vec::new();
            for index in 0..ctx.tasks.len() {
                let Some(failed_on) = ctx.tasks[index].advice_wanted_after.take() else { continue };
                if advisor.is_none() || ctx.tasks[index].status != TaskStatus::Pending {
                    continue;
                }
                // With no worker model left there will be no next attempt to advise.
                let Some(next) = ctx.worker_model_for(&ctx.tasks[index]) else { continue };
                ctx.tasks[index].consults += 1;
                to_advise.push((ctx.tasks[index].clone(), failed_on, next));
            }
            (advisor, to_advise, describe_pool(&ctx))
        };
        if let Some(advisor) = advisor {
            let mut advising = JoinSet::new();
            for (task, failed_on, next) in to_advise {
                let llm = Arc::clone(&self.llm);
                let semaphore = Arc::clone(&self.semaphore);
                let settings = settings.clone();
                let advisor = advisor.clone();
                let pool_facts = pool_facts.clone();
                advising.spawn(async move {
                    let _permit = semaphore.acquire().await.expect("semaphore is never closed");
                    let attempt = agents::FailedAttempt {
                        task: &task,
                        model: &failed_on,
                        next_model: &next,
                        pool_facts: &pool_facts,
                    };
                    let run = agents::advise(llm.as_ref(), &advisor, &attempt, &settings).await;
                    (task.id, next, run)
                });
            }
            while let Some(joined) = advising.join_next().await {
                let mut ctx = context.lock().await;
                match joined {
                    Ok((task_id, next, run)) => {
                        save_transcript(self.sandbox.root(), &format!("{task_id}.advice"), run.reply.as_deref());
                        record_usage(&mut ctx, &advisor, run.usage, run.calls);
                        match run.outcome {
                            Ok(advice) => apply_advice(&mut ctx, &task_id, &next, advice),
                            Err(e) => {
                                tracing::warn!(
                                    "Orchestrator [{task_id}]: no advice could be had ({e}); the next attempt goes ahead without it"
                                );
                                // Earlier advice was written about files the failed attempt has since changed.
                                if let Some(t) = ctx.get_task_mut(&task_id) {
                                    t.advice = None;
                                }
                            }
                        }
                    }
                    Err(e) => tracing::error!("Orchestrator: the advisor crashed: {e}"),
                }
                save(&ctx);
            }
        }

        {
            let mut ctx = context.lock().await;
            // A crashed reviewer leaves its task Reviewing; queue it for review again.
            for t in ctx.tasks.iter_mut() {
                if let TaskStatus::Reviewing { output_summary } = &t.status {
                    t.status = TaskStatus::ReadyForReview { output_summary: output_summary.clone() };
                }
            }
            report.auto_approved += auto_approve(&mut ctx);
        }

        Ok(report)
    }

    async fn ask_on_stdin(&self, context: &SharedContext, task_id: &str) -> Result<()> {
        let (title, summary, files, notes) = {
            let ctx = context.lock().await;
            let Some(t) = ctx.get_task(task_id) else { return Ok(()) };
            let TaskStatus::RequiresHumanApproval { output_summary } = &t.status else {
                return Ok(());
            };
            (t.title.clone(), output_summary.clone(), t.files.clone(), t.review_notes.clone())
        };

        let closed = || {
            anyhow!(
                "stdin closed while waiting for approval of {task_id}. Progress is saved: \
                 rerun with --resume, and add --brave for unattended runs."
            )
        };

        println!("\n--- HUMAN APPROVAL REQUIRED ---");
        println!("Task:    {task_id} - {title}");
        println!("Summary: {summary}");
        if !files.is_empty() {
            println!("Files:   {}", files.join(", "));
        }
        if !notes.is_empty() {
            println!("The reviewer could not verify these; they are yours to check:");
            for note in &notes {
                println!("{note}");
            }
        }
        println!("Approve? (y/n): ");
        let answer = read_stdin_line().await?.ok_or_else(closed)?;
        if matches!(answer.trim().to_lowercase().as_str(), "y" | "yes") {
            context.lock().await.approve(task_id);
            tracing::info!("Orchestrator [{task_id}]: approved");
        } else {
            println!("Feedback for the agent: ");
            let feedback = read_stdin_line().await?.ok_or_else(closed)?;
            context.lock().await.reject(task_id, &feedback);
            tracing::info!("Orchestrator [{task_id}]: rejected, sent back to the worker");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::FindingStatus;
    use crate::llm::{LlmError, LlmFuture, LlmReply};
    use std::sync::Mutex as StdMutex;

    /// What a request is for, as far as the scripted model can tell.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Kind {
        Plan,
        Requirements,
        Worker,
        Review,
        DesignCheck,
        Advice,
    }

    fn kind(request: &AgentRequest) -> Kind {
        match request.tools {
            ToolAccess::Full => Kind::Worker,
            ToolAccess::Inspect if request.prompt.starts_with("# Advice wanted") => Kind::Advice,
            ToolAccess::Inspect if request.prompt.contains("# Whole-project check") => Kind::DesignCheck,
            ToolAccess::Inspect => Kind::Review,
            ToolAccess::None if request.prompt.contains("checklist for verifying") => Kind::Requirements,
            ToolAccess::None | ToolAccess::Probe(_) => Kind::Plan,
        }
    }

    type Script = dyn Fn(Kind, &AgentRequest, usize) -> Result<String, LlmError> + Send + Sync;

    /// An `Llm` that answers from a closure and records every request. The
    /// closure gets the request, its kind, and how many requests of that kind came before.
    struct ScriptedLlm {
        script: Box<Script>,
        requests: StdMutex<Vec<AgentRequest>>,
    }

    impl ScriptedLlm {
        fn new(
            script: impl Fn(Kind, &AgentRequest, usize) -> Result<String, LlmError> + Send + Sync + 'static,
        ) -> Arc<Self> {
            Arc::new(Self { script: Box::new(script), requests: StdMutex::new(Vec::new()) })
        }

        fn requests(&self, wanted: Kind) -> Vec<AgentRequest> {
            self.requests.lock().unwrap().iter().filter(|r| kind(r) == wanted).cloned().collect()
        }
    }

    impl Llm for ScriptedLlm {
        fn run<'a>(&'a self, request: AgentRequest) -> LlmFuture<'a> {
            Box::pin(async move {
                let this = kind(&request);
                let earlier = {
                    let mut requests = self.requests.lock().unwrap();
                    let earlier = requests.iter().filter(|r| kind(r) == this).count();
                    requests.push(request.clone());
                    earlier
                };
                (self.script)(this, &request, earlier).map(|text| LlmReply {
                    text,
                    usage: TokenUsage { input_tokens: 10, cached_input_tokens: 4, output_tokens: 5 },
                })
            })
        }
    }

    const TWO_TASK_PLAN: &str = r#"[
        {"id":"TASK-1","title":"Set up","description":"Create the project","dependencies":[],"assigned_role":"Infrastructure",
         "acceptance_criteria":["Cargo.toml exists in the project root"]},
        {"id":"TASK-2","title":"Logic","description":"Implement the logic","dependencies":["TASK-1"],"assigned_role":"backend",
         "acceptance_criteria":["`sum(&[])` returns Ok(0.0)"]}
    ]"#;

    const DONE: &str = r#"{"status":"completed","summary":"did the work","files_created":["out.txt"]}"#;
    /// A line that exists in the test project, for reviewers to quote.
    const QUOTABLE: &str = "pub fn sum(args: &[String]) -> Result<f64, String> {";
    const REQUIREMENTS: &str = r#"[{"requirement": "Enter evaluates the expression", "kind": "behaviour"},
                                    {"requirement": "The history panel scrolls", "kind": "behaviour"},
                                    {"requirement": "Borders are cyan", "kind": "appearance"}]"#;

    /// The numbered items a reviewer was asked about.
    fn items_in(prompt: &str) -> Vec<String> {
        let list = prompt.split("# Items to verify").nth(1).expect("the prompt lists items");
        list.lines()
            .skip_while(|l| l.trim().is_empty())
            .take_while(|l| !l.trim().is_empty())
            .map(|l| l.split_once(". ").expect("items are numbered").1.to_string())
            .collect()
    }

    /// A reviewer answer in which `judge` decides each item.
    fn answer(prompt: &str, judge: impl Fn(&str) -> (&'static str, &'static str)) -> String {
        let findings: Vec<serde_json::Value> = items_in(prompt)
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let (status, evidence) = judge(item);
                serde_json::json!({ "id": i + 1, "status": status, "evidence": evidence, "file": "src/lib.rs", "quote": QUOTABLE })
            })
            .collect();
        serde_json::to_string(&findings).unwrap()
    }

    fn all_met(prompt: &str) -> String {
        answer(prompt, |_| ("met", "src/lib.rs:1"))
    }

    /// `unmet_when` picks the items that are unmet; everything else is met.
    fn unmet_if(prompt: &str, needle: &'static str, evidence: &'static str) -> String {
        answer(prompt, move |item| if item.contains(needle) { ("unmet", evidence) } else { ("met", "src/lib.rs:1") })
    }

    /// Planner, worker and reviewer all cooperate.
    fn agreeable(kind: Kind, request: &AgentRequest, _earlier: usize) -> Result<String, LlmError> {
        Ok(match kind {
            Kind::Plan => TWO_TASK_PLAN.to_string(),
            Kind::Requirements => REQUIREMENTS.to_string(),
            Kind::Worker => DONE.to_string(),
            Kind::Advice => r#"{"diagnosis": "the import is missing", "guidance": "add `use std::fmt;` to src/lib.rs"}"#.to_string(),
            Kind::Review | Kind::DesignCheck => all_met(&request.prompt),
        })
    }

    fn setup(
        llm: Arc<ScriptedLlm>,
        brave: bool,
        max_steps: usize,
    ) -> (tempfile::TempDir, Orchestrator, SharedContext) {
        let dir = tempfile::tempdir().unwrap();
        let sandbox = Sandbox::new(dir.path().join("out"), Duration::from_secs(5)).unwrap();
        std::fs::create_dir_all(sandbox.root().join("src")).unwrap();
        std::fs::write(sandbox.root().join("src/lib.rs"), format!("{QUOTABLE}\n    todo!()\n}}\n")).unwrap();
        let context = Arc::new(Mutex::new(ProjectContext::new(
            "test".into(),
            sandbox.root().to_path_buf(),
            max_steps,
            40,
            brave,
            "test-model".into(),
        )));
        let mut orchestrator = Orchestrator::new(llm, "THE DESIGN DOCUMENT".into(), sandbox);
        orchestrator.retry_delay = Duration::ZERO;
        (dir, orchestrator, context)
    }

    fn status_of(ctx: &ProjectContext, id: &str) -> TaskStatus {
        ctx.get_task(id).unwrap().status.clone()
    }

    #[tokio::test]
    async fn brave_run_completes_every_task_in_dependency_order() {
        let llm = ScriptedLlm::new(agreeable);
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 50);

        let end = orchestrator.run(&context, Hitl::Stdin).await.unwrap();
        assert_eq!(end, RunEnd::Finished);

        let ctx = context.lock().await;
        assert_eq!(ctx.count_completed(), 2);
        assert_eq!(ctx.phase, RunPhase::Finished);
        assert!(ctx.succeeded());
        // 1 plan + 2 workers + 2 reviews + 1 requirement list + 1 design check, 10 in / 5 out each.
        assert_eq!(ctx.llm_calls, 7);
        assert_eq!((ctx.input_tokens, ctx.cached_input_tokens, ctx.output_tokens), (70, 28, 35));
        assert_eq!(ctx.token_summary(), "70 in (28 from cache) / 35 out");
        assert_eq!(ctx.get_task("TASK-1").unwrap().files, vec!["out.txt"]);

        let workers = llm.requests(Kind::Worker);
        assert_eq!(workers.len(), 2);
        assert!(workers[0].prompt.contains("Set up"));
        assert!(workers[0].prompt.contains("Cargo.toml exists in the project root"), "workers see their criteria");
        assert!(
            workers[0].prompt.contains("- TASK-2 [Backend] Logic (Pending)"),
            "workers see the whole plan:\n{}",
            workers[0].prompt
        );
        assert!(workers[0].preamble.contains("THE DESIGN DOCUMENT"), "workers get the whole design");
        assert!(!workers[0].prompt.contains("already completed"));
        assert!(workers[1].prompt.contains("Implement the logic"));
        assert!(workers[1].prompt.contains("TASK-1 Set up: did the work"), "later tasks see earlier results");
        assert!(workers.iter().all(|r| r.model == "test-model" && r.max_turns == 40));

        let reviews = llm.requests(Kind::Review);
        assert!(reviews[0].prompt.contains("out.txt"), "the reviewer is told which files to inspect");
        assert!(reviews[0].prompt.contains("- TASK-2 [Backend] Logic"), "the reviewer sees the plan too");
        let items = items_in(&reviews[1].prompt);
        assert!(items[0].contains("builds"), "{items:?}");
        assert_eq!(items.last().unwrap(), "`sum(&[])` returns Ok(0.0)", "the task's own criteria are checked");

        let saved = ProjectContext::load(&ctx.working_directory).unwrap();
        assert_eq!(saved.count_completed(), 2);
        assert_eq!(saved.phase, RunPhase::Finished);

        let transcripts = ctx.working_directory.join(".clankistrator/transcripts");
        for name in ["TASK-1.worker.1.md", "TASK-1.review.1.md", "TASK-2.review.1.md", "design-check.1.md"] {
            assert!(transcripts.join(name).is_file(), "{name} should have been kept");
        }
        assert_eq!(std::fs::read_to_string(transcripts.join("TASK-1.worker.1.md")).unwrap(), DONE);
    }

    #[tokio::test]
    async fn one_unmet_item_sends_the_task_back_with_the_evidence() {
        let llm = ScriptedLlm::new(|kind, request, earlier| {
            Ok(match kind {
                Kind::Review if earlier == 0 => {
                    unmet_if(&request.prompt, "builds", "cargo build: error: expected `;` at src/main.rs:3")
                }
                _ => agreeable(kind, request, earlier)?,
            })
        });
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 50);

        assert_eq!(orchestrator.run(&context, Hitl::Stdin).await.unwrap(), RunEnd::Finished);
        assert_eq!(context.lock().await.count_completed(), 2);

        let workers = llm.requests(Kind::Worker);
        assert_eq!(workers.len(), 3, "TASK-1 twice, TASK-2 once");
        assert!(!workers[0].prompt.contains("expected `;`"));
        assert!(workers[1].prompt.contains("Set up"));
        assert!(workers[1].prompt.contains("expected `;` at src/main.rs:3"));
        assert!(workers[1].prompt.contains("The project builds without errors."));
    }

    #[tokio::test]
    async fn a_reviewer_that_says_everything_is_fine_without_itemised_evidence_does_not_pass_a_task() {
        // The old gate accepted any reply starting with PASSED.
        let llm = ScriptedLlm::new(|kind, request, earlier| match kind {
            Kind::Review => Ok("PASSED. Everything looks great!\nVERDICT: PASSED".to_string()),
            _ => agreeable(kind, request, earlier),
        });
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 0);

        assert_eq!(orchestrator.run(&context, Hitl::Stdin).await.unwrap(), RunEnd::Finished);
        let ctx = context.lock().await;
        assert_eq!(ctx.count_completed(), 0);
        assert!(matches!(status_of(&ctx, "TASK-1"), TaskStatus::Failed { error } if error.contains("Review could not be completed")));
        assert!(!ctx.succeeded());
    }

    #[tokio::test]
    async fn what_the_reviewer_cannot_verify_is_passed_on_to_the_human() {
        let llm = ScriptedLlm::new(|kind, request, earlier| {
            Ok(match kind {
                Kind::Review => answer(&request.prompt, |item| {
                    if item.contains("Cargo.toml") { ("cannot_verify", "needs a terminal") } else { ("met", "src/lib.rs:1") }
                }),
                _ => agreeable(kind, request, earlier)?,
            })
        });
        let (_dir, orchestrator, context) = setup(llm, false, 50);
        context.lock().await.tasks = parse_plan(TWO_TASK_PLAN).unwrap();

        orchestrator.run_iteration(&context).await.unwrap();
        let ctx = context.lock().await;
        let task = ctx.get_task("TASK-1").unwrap();
        assert!(matches!(task.status, TaskStatus::RequiresHumanApproval { .. }));
        assert_eq!(task.review_notes.len(), 1);
        assert!(task.review_notes[0].contains("Cargo.toml exists") && task.review_notes[0].contains("needs a terminal"));
    }

    #[tokio::test]
    async fn a_task_that_keeps_failing_is_given_up_and_blocks_its_dependents() {
        let llm = ScriptedLlm::new(|kind, request, earlier| match kind {
            Kind::Worker => Err(LlmError::Other(anyhow!("503 service unavailable"))),
            _ => agreeable(kind, request, earlier),
        });
        // Unlimited steps: the run must still end on its own.
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 0);

        let end = tokio::time::timeout(Duration::from_secs(20), orchestrator.run(&context, Hitl::Stdin))
            .await
            .expect("the run must terminate")
            .unwrap();
        assert_eq!(end, RunEnd::Finished);

        let ctx = context.lock().await;
        // Each of the task's attempts tries the provider three times before giving up.
        assert_eq!(llm.requests(Kind::Worker).len(), (MAX_RETRIES as usize + 1) * 3);
        assert!(matches!(status_of(&ctx, "TASK-1"), TaskStatus::Failed { error } if error.contains("503")));
        assert!(matches!(status_of(&ctx, "TASK-2"), TaskStatus::Failed { error } if error.starts_with("Blocked")));
        assert!(llm.requests(Kind::Review).is_empty());
        assert!(llm.requests(Kind::DesignCheck).is_empty(), "a failed run is not design-checked");
        assert!(!ctx.succeeded());
    }

    #[tokio::test]
    async fn a_review_error_does_not_abort_the_run_or_redo_the_work() {
        let llm = ScriptedLlm::new(|kind, request, earlier| match kind {
            Kind::Review if earlier == 0 => Err(LlmError::MaxTurns(20)),
            _ => agreeable(kind, request, earlier),
        });
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 50);

        assert_eq!(orchestrator.run(&context, Hitl::Stdin).await.unwrap(), RunEnd::Finished);
        assert_eq!(context.lock().await.count_completed(), 2);
        assert_eq!(llm.requests(Kind::Worker).len(), 2, "no task was executed twice");
        assert_eq!(llm.requests(Kind::Review).len(), 3);
    }

    #[tokio::test]
    async fn worker_reported_failure_is_retried_with_its_reason() {
        let llm = ScriptedLlm::new(|kind, request, earlier| match kind {
            Kind::Worker if earlier == 0 => {
                Ok(r#"{"status":"failed","summary":"the registry was unreachable"}"#.to_string())
            }
            _ => agreeable(kind, request, earlier),
        });
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 50);

        assert_eq!(orchestrator.run(&context, Hitl::Stdin).await.unwrap(), RunEnd::Finished);
        assert_eq!(context.lock().await.count_completed(), 2);
        assert!(llm.requests(Kind::Worker)[1].prompt.contains("registry was unreachable"));
    }

    #[tokio::test]
    async fn an_unusable_plan_is_retried_once_then_reported() {
        let cyclic = r#"[{"id":"A","title":"a","description":"a","dependencies":["B"],"assigned_role":"QA"},
                         {"id":"B","title":"b","description":"b","dependencies":["A"],"assigned_role":"QA"}]"#;

        let llm = ScriptedLlm::new(move |kind, request, earlier| match kind {
            Kind::Plan if earlier == 0 => Ok(cyclic.to_string()),
            _ => agreeable(kind, request, earlier),
        });
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 50);
        assert_eq!(orchestrator.run(&context, Hitl::Stdin).await.unwrap(), RunEnd::Finished);
        let plans = llm.requests(Kind::Plan);
        assert_eq!(plans.len(), 2);
        assert!(plans[1].prompt.contains("dependency cycle"), "the model is told what was wrong");

        let llm = ScriptedLlm::new(|_, _, _| Ok("I'd rather not.".to_string()));
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 50);
        let error = orchestrator.run(&context, Hitl::Stdin).await.unwrap_err();
        assert!(error.to_string().contains("usable task plan"), "{error}");
        assert!(matches!(context.lock().await.phase, RunPhase::Error(_)));
        assert!(llm.requests(Kind::Worker).is_empty());
    }

    #[tokio::test]
    async fn the_step_limit_ends_a_headless_run() {
        let llm = ScriptedLlm::new(agreeable);
        let (_dir, orchestrator, context) = setup(llm, true, 1);

        assert_eq!(orchestrator.run(&context, Hitl::Stdin).await.unwrap(), RunEnd::StepLimit);
        let ctx = context.lock().await;
        assert_eq!(ctx.steps_taken, 1);
        assert_eq!(ctx.count_completed(), 1);
        assert_eq!(status_of(&ctx, "TASK-2"), TaskStatus::Pending);
        assert_eq!(ctx.phase, RunPhase::StepLimit);
        assert!(!ctx.succeeded());
    }

    #[tokio::test]
    async fn waiting_for_a_human_costs_no_steps_and_resumes_on_approval() {
        let llm = ScriptedLlm::new(agreeable);
        // A step limit of 2 is exactly enough only if waiting is free.
        let (_dir, orchestrator, context) = setup(llm, false, 2);
        let orchestrator = Arc::new(orchestrator);

        let run = {
            let (orchestrator, context) = (Arc::clone(&orchestrator), Arc::clone(&context));
            tokio::spawn(async move { orchestrator.run(&context, Hitl::External).await })
        };

        let wait_for_approval_request = |id: &'static str| {
            let context = Arc::clone(&context);
            async move {
                for _ in 0..200 {
                    {
                        let ctx = context.lock().await;
                        let waiting = ctx
                            .get_task(id)
                            .is_some_and(|t| matches!(t.status, TaskStatus::RequiresHumanApproval { .. }));
                        if waiting && ctx.phase == RunPhase::AwaitingApproval {
                            return;
                        }
                    }
                    tokio::time::sleep(Duration::from_millis(25)).await;
                }
                panic!("{id} never asked for approval");
            }
        };

        wait_for_approval_request("TASK-1").await;
        // Many poll intervals pass while the "human" thinks.
        tokio::time::sleep(HUMAN_POLL_INTERVAL * 5).await;
        {
            let mut ctx = context.lock().await;
            assert_eq!(ctx.steps_taken, 1);
            assert_eq!(status_of(&ctx, "TASK-2"), TaskStatus::Pending);
            assert!(ctx.reject("TASK-1", "use tabs"));
        }

        wait_for_approval_request("TASK-1").await;
        assert!(context.lock().await.approve("TASK-1"));
        // Step limit (2) is now used up; raise it like the TUI menu would.
        tokio::time::sleep(HUMAN_POLL_INTERVAL * 3).await;
        {
            let mut ctx = context.lock().await;
            assert_eq!(ctx.phase, RunPhase::StepLimit);
            ctx.max_steps += 5;
        }

        wait_for_approval_request("TASK-2").await;
        assert!(context.lock().await.approve("TASK-2"));

        let end = tokio::time::timeout(Duration::from_secs(10), run).await.unwrap().unwrap().unwrap();
        assert_eq!(end, RunEnd::Finished);
        let ctx = context.lock().await;
        assert_eq!(ctx.count_completed(), 2);
        assert!(ctx.succeeded());
    }

    #[tokio::test]
    async fn settings_changed_mid_run_reach_later_agents() {
        let llm = ScriptedLlm::new(agreeable);
        let (_dir, orchestrator, context) = setup(llm.clone(), false, 50);
        context.lock().await.tasks = parse_plan(TWO_TASK_PLAN).unwrap();

        orchestrator.run_iteration(&context).await.unwrap();
        {
            let mut ctx = context.lock().await;
            assert!(ctx.approve("TASK-1"));
            ctx.model = "other-model".into();
        }
        orchestrator.run_iteration(&context).await.unwrap();

        let workers = llm.requests(Kind::Worker);
        assert_eq!(workers[0].model, "test-model");
        assert_eq!(workers[1].model, "other-model");
        assert_eq!(llm.requests(Kind::Review)[1].model, "other-model");
    }

    #[tokio::test]
    async fn a_separate_review_model_judges_but_does_not_build() {
        let llm = ScriptedLlm::new(agreeable);
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 50);
        context.lock().await.review_model = Some("judge-model".into());

        assert_eq!(orchestrator.run(&context, Hitl::Stdin).await.unwrap(), RunEnd::Finished);
        assert!(llm.requests(Kind::Worker).iter().all(|r| r.model == "test-model"));
        assert!(llm.requests(Kind::Plan).iter().all(|r| r.model == "test-model"));
        for judged in [Kind::Review, Kind::Requirements, Kind::DesignCheck] {
            let requests = llm.requests(judged);
            assert!(!requests.is_empty() && requests.iter().all(|r| r.model == "judge-model"), "{judged:?}");
        }
    }

    #[tokio::test]
    async fn follow_up_tasks_from_workers_are_scheduled_after_their_origin() {
        let llm = ScriptedLlm::new(|kind, request, earlier| match kind {
            Kind::Worker if earlier == 0 => Ok(r#"{"status":"completed","summary":"done","files_created":[],
                    "new_tasks":[{"title":"Add CI","description":"Set up CI","assigned_role":"Infrastructure","dependencies":["NOPE"]}]}"#
                .to_string()),
            _ => agreeable(kind, request, earlier),
        });
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 50);

        assert_eq!(orchestrator.run(&context, Hitl::Stdin).await.unwrap(), RunEnd::Finished);
        let ctx = context.lock().await;
        assert_eq!(ctx.tasks.len(), 3);
        let added = ctx.get_task("TASK-3").unwrap();
        assert_eq!(added.title, "Add CI");
        assert_eq!(added.dependencies, vec!["TASK-1"], "unknown dependency dropped, origin added");
        assert_eq!(ctx.count_completed(), 3);
    }

    #[tokio::test]
    async fn resuming_skips_planning_and_completed_work() {
        let llm = ScriptedLlm::new(agreeable);
        let (_dir, orchestrator, context) = setup(llm, true, 1);
        assert_eq!(orchestrator.run(&context, Hitl::Stdin).await.unwrap(), RunEnd::StepLimit);
        let output_dir = context.lock().await.working_directory.clone();

        let mut loaded = ProjectContext::load(Path::new(&output_dir)).unwrap();
        loaded.prepare_resume();
        let context = Arc::new(Mutex::new(loaded));
        let llm = ScriptedLlm::new(agreeable);
        let sandbox = Sandbox::new(&output_dir, Duration::from_secs(5)).unwrap();
        let mut orchestrator = Orchestrator::new(llm.clone(), "THE DESIGN DOCUMENT".into(), sandbox);
        orchestrator.retry_delay = Duration::ZERO;

        assert_eq!(orchestrator.run(&context, Hitl::Stdin).await.unwrap(), RunEnd::Finished);
        assert!(llm.requests(Kind::Plan).is_empty(), "no second planning call");
        let workers = llm.requests(Kind::Worker);
        assert_eq!(workers.len(), 1);
        assert!(workers[0].prompt.contains("Implement the logic"));
        assert!(context.lock().await.succeeded());
    }

    #[tokio::test]
    async fn a_transient_provider_error_is_retried_without_costing_the_task_a_retry() {
        let llm = ScriptedLlm::new(|kind, request, earlier| match kind {
            Kind::Worker if earlier == 0 => Err(LlmError::Other(anyhow!("429 rate limit reached"))),
            _ => agreeable(kind, request, earlier),
        });
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 50);

        assert_eq!(orchestrator.run(&context, Hitl::Stdin).await.unwrap(), RunEnd::Finished);
        let ctx = context.lock().await;
        assert_eq!(ctx.count_completed(), 2);
        assert_eq!(llm.requests(Kind::Worker).len(), 3);
        assert!(ctx.tasks.iter().all(|t| t.retries == 0));
    }

    #[tokio::test]
    async fn an_authentication_error_is_not_retried_at_the_call_level() {
        let llm = ScriptedLlm::new(|kind, request, earlier| match kind {
            Kind::Worker => Err(LlmError::Other(anyhow!("HttpError: Invalid status code 401 Unauthorized"))),
            _ => agreeable(kind, request, earlier),
        });
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 50);

        assert_eq!(orchestrator.run(&context, Hitl::Stdin).await.unwrap(), RunEnd::Finished);
        // A dead key does not get better: one request, and the task fails without further attempts.
        assert_eq!(llm.requests(Kind::Worker).len(), 1);
        let ctx = context.lock().await;
        assert_eq!(ctx.count_failed(), 2);
        assert!(ctx.is_retired("test-model"));
    }

    /// A worker that says which model it ran on, so that a scripted reviewer can judge by it.
    fn signed(request: &AgentRequest) -> String {
        format!(r#"{{"status": "completed", "summary": "built by {}", "files_created": ["src/lib.rs"]}}"#, request.model)
    }

    fn worker_models(llm: &ScriptedLlm) -> Vec<String> {
        llm.requests(Kind::Worker).into_iter().map(|r| r.model).collect()
    }

    #[tokio::test]
    async fn a_task_moves_to_the_next_worker_model_after_two_failed_attempts() {
        let llm = ScriptedLlm::new(|kind, request, earlier| {
            Ok(match kind {
                Kind::Worker => signed(request),
                Kind::Review if request.prompt.contains("built by cheap") => {
                    unmet_if(&request.prompt, "builds", "cargo build: error[E0425]")
                }
                _ => agreeable(kind, request, earlier)?,
            })
        });
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 50);
        context.lock().await.worker_models = vec!["cheap".into(), "strong".into()];

        assert_eq!(orchestrator.run(&context, Hitl::Stdin).await.unwrap(), RunEnd::Finished);
        // Every task starts on the cheapest model, whatever happened to the task before it.
        assert_eq!(worker_models(&llm), ["cheap", "cheap", "strong", "cheap", "cheap", "strong"]);
        let ctx = context.lock().await;
        assert_eq!(ctx.count_completed(), 2);
        assert_eq!((ctx.model_usage["cheap"].passed, ctx.model_usage["cheap"].rejected), (0, 4));
        assert_eq!((ctx.model_usage["strong"].passed, ctx.model_usage["strong"].rejected), (2, 0));
        assert_eq!(ctx.model_usage["cheap"].calls, 4);
        // The planner ran on the default model and is accounted for separately.
        assert_eq!(ctx.model_usage["test-model"].calls, ctx.llm_calls - 6);
    }

    #[tokio::test]
    async fn changing_model_does_not_give_a_task_more_attempts() {
        let llm = ScriptedLlm::new(|kind, request, earlier| {
            Ok(match kind {
                Kind::Worker => signed(request),
                Kind::Review => unmet_if(&request.prompt, "builds", "cargo build: error[E0425]"),
                _ => agreeable(kind, request, earlier)?,
            })
        });
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 0);
        context.lock().await.worker_models = vec!["cheap".into(), "middle".into(), "strong".into()];

        let end = tokio::time::timeout(Duration::from_secs(20), orchestrator.run(&context, Hitl::Stdin))
            .await
            .expect("the run must terminate")
            .unwrap();
        assert_eq!(end, RunEnd::Finished);
        assert_eq!(worker_models(&llm), ["cheap", "cheap", "middle", "middle"]);
        assert!(matches!(status_of(&*context.lock().await, "TASK-1"), TaskStatus::Failed { .. }));
    }

    #[tokio::test]
    async fn a_model_that_becomes_unusable_is_retired_and_its_work_moves_on_without_losing_an_attempt() {
        let llm = ScriptedLlm::new(|kind, request, earlier| match kind {
            Kind::Worker if request.model == "free" => {
                Err(LlmError::Other(anyhow!("ProviderError: {}", crate::llm::FREE_ALLOWANCE_USED_UP)))
            }
            _ => agreeable(kind, request, earlier),
        });
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 50);
        context.lock().await.worker_models = vec!["free".into(), "paid".into()];

        assert_eq!(orchestrator.run(&context, Hitl::Stdin).await.unwrap(), RunEnd::Finished);
        // Asked once, then left alone: the second task goes straight to the model that works.
        assert_eq!(worker_models(&llm), ["free", "paid", "paid"]);
        let ctx = context.lock().await;
        assert_eq!(ctx.count_completed(), 2);
        assert!(ctx.tasks.iter().all(|t| t.retries == 0));
        assert!(ctx.is_retired("free") && !ctx.is_retired("paid"));
        assert!(ctx.retired_models[0].1.contains("allowance for today is used up"));
    }

    const ASSIGNED_PLAN: &str = r#"[
        {"id":"TASK-1","title":"Set up","description":"Create the project","dependencies":[],"assigned_role":"Infrastructure",
         "acceptance_criteria":["Cargo.toml exists in the project root"],"model":"cheap"},
        {"id":"TASK-2","title":"Parser","description":"Implement the parser","dependencies":["TASK-1"],"assigned_role":"Backend",
         "acceptance_criteria":["`sum(&[])` returns Ok(0.0)"],"model":"strong"},
        {"id":"TASK-3","title":"Docs","description":"Write the README","dependencies":["TASK-2"],"assigned_role":"Backend",
         "acceptance_criteria":["README.md exists"],"model":"a-model-nobody-offered"}
    ]"#;

    #[tokio::test]
    async fn the_advisor_plans_assigns_each_task_to_a_worker_model_and_reviews() {
        let llm = ScriptedLlm::new(|kind, request, earlier| match kind {
            Kind::Plan => Ok(ASSIGNED_PLAN.to_string()),
            Kind::DesignCheck if earlier == 0 => {
                Ok(unmet_if(&request.prompt, "Enter evaluates", "src/app.rs:52 only handles '='"))
            }
            _ => agreeable(kind, request, earlier),
        });
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 50);
        {
            let mut ctx = context.lock().await;
            ctx.worker_models = vec!["cheap".into(), "strong".into()];
            ctx.advisor_model = Some("lead".into());
            ctx.model_usage.entry("cheap".into()).or_default().passed = 3;
        }

        assert_eq!(orchestrator.run(&context, Hitl::Stdin).await.unwrap(), RunEnd::Finished);

        // The advisor plans, knowing what the harness knows about the models.
        let plan = &llm.requests(Kind::Plan)[0];
        assert_eq!(plan.model, "lead");
        assert!(plan.prompt.contains("# Worker models"));
        assert!(plan.prompt.contains("- cheap: paid; so far in this run 3 of its task attempts passed review and 0 failed"));
        assert!(plan.prompt.contains("- strong: paid\n"), "{}", plan.prompt);

        // Tasks run where the plan put them; a model nobody offered falls back to the
        // first worker model; the fix task goes to the strongest.
        assert_eq!(worker_models(&llm), ["cheap", "strong", "cheap", "strong"]);
        let ctx = context.lock().await;
        assert!(ctx.get_task("TASK-4").unwrap().title.starts_with("Fix unmet design requirements"));
        assert!(ctx.succeeded());

        // With no review model named, the advisor is the judge. It never builds.
        for judged in [Kind::Review, Kind::Requirements, Kind::DesignCheck] {
            assert!(llm.requests(judged).iter().all(|r| r.model == "lead"), "{judged:?}");
        }
        assert_eq!(ctx.model_usage["lead"].passed + ctx.model_usage["lead"].rejected, 0);
    }

    #[tokio::test]
    async fn a_single_worker_model_is_not_offered_as_a_choice() {
        let llm = ScriptedLlm::new(agreeable);
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 50);
        context.lock().await.advisor_model = Some("lead".into());

        assert_eq!(orchestrator.run(&context, Hitl::Stdin).await.unwrap(), RunEnd::Finished);
        let plan = &llm.requests(Kind::Plan)[0];
        assert_eq!(plan.model, "lead");
        assert!(!plan.prompt.contains("# Worker models"));
        assert_eq!(worker_models(&llm), ["test-model", "test-model"]);
    }

    #[tokio::test]
    async fn the_advisor_looks_at_a_failed_attempt_and_its_guidance_reaches_the_worker() {
        let llm = ScriptedLlm::new(|kind, request, earlier| {
            Ok(match kind {
                Kind::Worker => signed(request),
                // The first attempt is rejected; with the advice followed, the second passes.
                Kind::Review if earlier == 0 => unmet_if(&request.prompt, "builds", "error[E0425]: cannot find `fmt`"),
                _ => agreeable(kind, request, earlier)?,
            })
        });
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 50);
        {
            let mut ctx = context.lock().await;
            ctx.worker_models = vec!["cheap".into(), "strong".into()];
            ctx.advisor_model = Some("lead".into());
        }

        assert_eq!(orchestrator.run(&context, Hitl::Stdin).await.unwrap(), RunEnd::Finished);

        let advice = llm.requests(Kind::Advice);
        assert_eq!(advice.len(), 1);
        assert_eq!(advice[0].model, "lead");
        assert_eq!(advice[0].tools, ToolAccess::Inspect, "the advisor can look but not write");
        assert!(advice[0].prompt.contains("error[E0425]: cannot find `fmt`"));
        assert!(advice[0].prompt.contains("failed attempt 1 of at most 4, made by cheap"));
        assert!(advice[0].prompt.contains("- cheap: paid; so far in this run 0 of its task attempts passed review and 1 failed"));
        assert!(advice[0].prompt.contains("The next attempt is set to run on cheap."));
        assert!(!advice[0].prompt.contains("Advice you gave after an earlier failed attempt"));

        let workers = llm.requests(Kind::Worker);
        assert_eq!(worker_models(&llm), ["cheap", "cheap", "cheap"], "the advisor left the model alone");
        assert!(!workers[0].prompt.contains("lead engineer"));
        assert!(workers[1].prompt.contains("error[E0425]"), "the reviewer's finding is still there");
        assert!(workers[1].prompt.contains("# Advice from the lead engineer"));
        assert!(workers[1].prompt.contains("What went wrong: the import is missing"));
        assert!(workers[1].prompt.contains("What to do: add `use std::fmt;` to src/lib.rs"));
        assert!(!workers[2].prompt.contains("lead engineer"), "the next task starts clean");

        let ctx = context.lock().await;
        assert!(ctx.succeeded());
        assert!(ctx.model_usage["lead"].calls >= 2, "planning and advice are both the advisor's");
    }

    #[tokio::test]
    async fn the_advisor_can_move_a_task_to_another_model_but_not_buy_it_more_attempts() {
        let llm = ScriptedLlm::new(|kind, request, earlier| {
            Ok(match kind {
                Kind::Worker => signed(request),
                Kind::Review => unmet_if(&request.prompt, "builds", "cargo build: error[E0425]"),
                // First it sends the task up, then it names a model that does not exist.
                Kind::Advice if earlier == 0 => r#"{"guidance": "rewrite the parser", "model": "strong"}"#.to_string(),
                Kind::Advice => r#"{"guidance": "try again", "model": "imaginary"}"#.to_string(),
                _ => agreeable(kind, request, earlier)?,
            })
        });
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 0);
        {
            let mut ctx = context.lock().await;
            ctx.worker_models = vec!["cheap".into(), "middle".into(), "strong".into()];
            ctx.advisor_model = Some("lead".into());
        }

        let end = tokio::time::timeout(Duration::from_secs(20), orchestrator.run(&context, Hitl::Stdin))
            .await
            .expect("the run must terminate")
            .unwrap();
        assert_eq!(end, RunEnd::Finished);
        // Straight to the model the advisor chose, skipping the one in between; four attempts in all.
        assert_eq!(worker_models(&llm), ["cheap", "strong", "strong", "strong"]);
        // The advisor is asked before every retry, but not about an attempt that will not be repeated.
        let consults = llm.requests(Kind::Advice);
        assert_eq!(consults.len(), MAX_RETRIES as usize);
        assert_eq!(MAX_CONSULTS, MAX_RETRIES);
        // Asked a second time, it is shown what it said the first time.
        assert!(consults[1].prompt.contains("# Advice you gave after an earlier failed attempt"));
        assert!(consults[1].prompt.contains("rewrite the parser"));
        // Each attempt has the latest advice in front of it, and only that.
        let workers = llm.requests(Kind::Worker);
        assert!(workers[1].prompt.contains("rewrite the parser"));
        assert!(workers[3].prompt.contains("try again") && !workers[3].prompt.contains("rewrite the parser"));
        let ctx = context.lock().await;
        assert!(matches!(status_of(&ctx, "TASK-1"), TaskStatus::Failed { .. }));
        assert_eq!(ctx.get_task("TASK-1").unwrap().consults, MAX_CONSULTS);
    }

    #[tokio::test]
    async fn a_run_goes_on_when_the_advisor_gives_nothing_usable() {
        let llm = ScriptedLlm::new(|kind, request, earlier| match kind {
            Kind::Review if earlier == 0 => Ok(unmet_if(&request.prompt, "builds", "error[E0425]")),
            Kind::Advice if earlier == 0 => Ok("I would look at the imports.".to_string()),
            Kind::Advice => Err(LlmError::Other(anyhow!("HTTP 401 Unauthorized"))),
            _ => agreeable(kind, request, earlier),
        });
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 50);
        context.lock().await.advisor_model = Some("lead".into());

        assert_eq!(orchestrator.run(&context, Hitl::Stdin).await.unwrap(), RunEnd::Finished);
        let ctx = context.lock().await;
        assert_eq!(ctx.count_completed(), 2);
        let retry = &llm.requests(Kind::Worker)[1];
        assert!(retry.prompt.contains("error[E0425]") && !retry.prompt.contains("lead engineer"));
    }

    /// Advice describes the files as they were before the attempt that has now
    /// failed, so it is not handed on when the advisor cannot be asked again.
    #[tokio::test]
    async fn advice_from_before_the_latest_failure_is_not_passed_on_when_no_new_advice_comes() {
        let llm = ScriptedLlm::new(|kind, request, earlier| match kind {
            Kind::Review if earlier < 2 => Ok(unmet_if(&request.prompt, "builds", "error[E0425]")),
            Kind::Advice if earlier == 0 => Ok(r#"{"guidance": "add `use std::fmt;` to src/lib.rs"}"#.to_string()),
            Kind::Advice => Err(LlmError::Other(anyhow!("HTTP 401 Unauthorized"))),
            _ => agreeable(kind, request, earlier),
        });
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 50);
        context.lock().await.advisor_model = Some("lead".into());

        assert_eq!(orchestrator.run(&context, Hitl::Stdin).await.unwrap(), RunEnd::Finished);
        let workers = llm.requests(Kind::Worker);
        assert!(workers[1].prompt.contains("add `use std::fmt;`"));
        assert!(!workers[2].prompt.contains("lead engineer"), "{}", workers[2].prompt);
        assert!(workers[2].prompt.contains("error[E0425]"), "the reviewer's findings are still passed on");
    }

    const INDEPENDENT_PLAN: &str = r#"[
        {"id":"TASK-1","title":"One","description":"First","dependencies":[],"assigned_role":"Backend",
         "acceptance_criteria":["Cargo.toml exists in the project root"]},
        {"id":"TASK-2","title":"Two","description":"Second","dependencies":[],"assigned_role":"Backend",
         "acceptance_criteria":["`sum(&[])` returns Ok(0.0)"]},
        {"id":"TASK-3","title":"Three","description":"Third","dependencies":["TASK-1"],"assigned_role":"Backend",
         "acceptance_criteria":["README.md exists"]}
    ]"#;

    /// `--model paid --workers free`: the paid model plans and reviews. When the
    /// free allowance runs out, building stops; it does not move to the paid
    /// model, which was never offered as a worker.
    #[tokio::test]
    async fn when_every_worker_model_has_dropped_out_nothing_else_is_used_to_build() {
        let llm = ScriptedLlm::new(|kind, request, earlier| match kind {
            Kind::Plan => Ok(INDEPENDENT_PLAN.to_string()),
            Kind::Worker => Err(LlmError::Other(anyhow!("ProviderError: {}", crate::llm::FREE_ALLOWANCE_USED_UP))),
            _ => agreeable(kind, request, earlier),
        });
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 0);
        context.lock().await.worker_models = vec!["free".into()];

        let end = tokio::time::timeout(Duration::from_secs(20), orchestrator.run(&context, Hitl::Stdin))
            .await
            .expect("the run must terminate")
            .unwrap();
        assert_eq!(end, RunEnd::Finished);
        let workers = llm.requests(Kind::Worker);
        assert!(!workers.is_empty() && workers.iter().all(|r| r.model == "free"), "{:?}", worker_models(&llm));
        assert!(workers.len() <= 2, "only the tasks already under way found out the hard way");
        let ctx = context.lock().await;
        assert_eq!(ctx.count_failed(), 3);
        assert!(matches!(status_of(&ctx, "TASK-3"), TaskStatus::Failed { .. }));
    }

    /// A rate limit is not a reason to drop a model, even when its text
    /// contains the digits of an authentication error.
    #[tokio::test]
    async fn a_rate_limited_model_stays_in_the_pool() {
        let llm = ScriptedLlm::new(|kind, request, earlier| match kind {
            // Two refusals are absorbed by the call-level retries of one attempt.
            Kind::Worker if earlier < 2 => Err(LlmError::Other(anyhow!(
                "HttpError: Invalid status code 429 Too Many Requests with message: Requested 40312 tokens"
            ))),
            _ => agreeable(kind, request, earlier),
        });
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 50);
        context.lock().await.worker_models = vec!["cheap".into(), "strong".into()];

        assert_eq!(orchestrator.run(&context, Hitl::Stdin).await.unwrap(), RunEnd::Finished);
        let ctx = context.lock().await;
        assert!(ctx.retired_models.is_empty());
        assert_eq!(ctx.count_completed(), 2);
        assert!(ctx.tasks.iter().all(|t| t.retries == 0));
        assert!(worker_models(&llm).iter().all(|model| model == "cheap"), "{:?}", worker_models(&llm));
    }

    /// Seen live: a correct review was overruled by the quote check four times.
    /// Nothing was wrong with the work, so the model that did it must not be
    /// marked down for it or have the task taken away.
    #[tokio::test]
    async fn a_review_that_is_merely_unproven_does_not_move_the_task_to_another_model() {
        let invented = |prompt: &str| {
            let findings: Vec<serde_json::Value> = items_in(prompt)
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    serde_json::json!({ "id": i + 1, "status": "met", "evidence": "it is all there",
                                        "file": "src/lib.rs", "quote": "this line is not in the file at all" })
                })
                .collect();
            serde_json::to_string(&findings).unwrap()
        };
        // Three reviews of two requests each (answer and correction) are unproven, then one is sound.
        let llm = ScriptedLlm::new(move |kind, request, earlier| match kind {
            Kind::Review if earlier < 6 => Ok(invented(&request.prompt)),
            _ => agreeable(kind, request, earlier),
        });
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 50);
        {
            let mut ctx = context.lock().await;
            ctx.worker_models = vec!["cheap".into(), "strong".into()];
            ctx.advisor_model = Some("lead".into());
        }

        assert_eq!(orchestrator.run(&context, Hitl::Stdin).await.unwrap(), RunEnd::Finished);
        assert_eq!(worker_models(&llm), ["cheap", "cheap", "cheap", "cheap", "cheap"]);
        assert!(llm.requests(Kind::Advice).is_empty(), "there is no failure for the advisor to explain");
        let ctx = context.lock().await;
        assert_eq!(ctx.count_completed(), 2);
        assert_eq!((ctx.model_usage["cheap"].passed, ctx.model_usage["cheap"].rejected), (2, 0));
    }

    #[tokio::test]
    async fn an_unusable_review_is_repeated_not_treated_as_a_rejection() {
        let llm = ScriptedLlm::new(|kind, request, earlier| match kind {
            Kind::Review if earlier == 0 => Ok("I had a look around the project.".to_string()),
            _ => agreeable(kind, request, earlier),
        });
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 50);

        assert_eq!(orchestrator.run(&context, Hitl::Stdin).await.unwrap(), RunEnd::Finished);
        assert_eq!(context.lock().await.count_completed(), 2);
        assert_eq!(llm.requests(Kind::Worker).len(), 2, "the worker was not sent back");
    }

    #[tokio::test]
    async fn a_redone_task_does_not_duplicate_its_follow_up_tasks() {
        let llm = ScriptedLlm::new(|kind, request, earlier| match kind {
            // TASK-1 is executed twice (first review rejects) and reports the same follow-up both times.
            Kind::Worker if earlier < 2 => Ok(r#"{"status":"completed","summary":"done","files_created":[],
                    "new_tasks":[{"title":"Add CI","description":"Set up CI","assigned_role":"Infrastructure"}]}"#
                .to_string()),
            Kind::Review if earlier == 0 => Ok(unmet_if(&request.prompt, "builds", "it does not build")),
            _ => agreeable(kind, request, earlier),
        });
        let (_dir, orchestrator, context) = setup(llm, true, 50);

        assert_eq!(orchestrator.run(&context, Hitl::Stdin).await.unwrap(), RunEnd::Finished);
        let ctx = context.lock().await;
        assert_eq!(ctx.tasks.iter().filter(|t| t.title == "Add CI").count(), 1);
        assert_eq!(ctx.count_completed(), 3);
    }

    #[tokio::test]
    async fn unmet_design_requirements_become_a_fix_task_and_the_project_is_checked_again() {
        let llm = ScriptedLlm::new(|kind, request, earlier| match kind {
            Kind::DesignCheck if earlier == 0 => {
                Ok(unmet_if(&request.prompt, "Enter evaluates", "src/app.rs:52 only handles '='"))
            }
            _ => agreeable(kind, request, earlier),
        });
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 50);

        assert_eq!(orchestrator.run(&context, Hitl::Stdin).await.unwrap(), RunEnd::Finished);
        let ctx = context.lock().await;
        assert_eq!(ctx.tasks.len(), 3);
        let fix = ctx.get_task("TASK-3").unwrap();
        assert!(fix.title.starts_with("Fix unmet design requirements"));
        assert!(fix.description.contains("src/app.rs:52 only handles '='"));
        assert_eq!(fix.acceptance_criteria, vec!["Enter evaluates the expression"]);
        assert_eq!(ctx.fix_rounds_done, 1);

        assert_eq!(llm.requests(Kind::Requirements).len(), 1, "the checklist is extracted once");
        let checks = llm.requests(Kind::DesignCheck);
        assert_eq!(checks.len(), 2);
        let items = items_in(&checks[0].prompt);
        assert!(items.iter().any(|i| i.contains("entry point")), "baseline items are always checked");
        assert!(items.contains(&"The history panel scrolls".to_string()));

        // The fix task's reviewer checks the unmet requirement itself.
        let fix_review = llm.requests(Kind::Review).pop().unwrap();
        assert!(items_in(&fix_review.prompt).contains(&"Enter evaluates the expression".to_string()));

        assert!(ctx.unmet_requirements().is_empty());
        assert!(ctx.succeeded());
    }

    #[tokio::test]
    async fn a_run_that_ends_with_unmet_requirements_is_not_a_success() {
        let llm = ScriptedLlm::new(|kind, request, earlier| match kind {
            Kind::DesignCheck => Ok(unmet_if(&request.prompt, "history panel", "there is no scrolling code")),
            _ => agreeable(kind, request, earlier),
        });
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 0);
        context.lock().await.fix_rounds = 1;

        let end = tokio::time::timeout(Duration::from_secs(20), orchestrator.run(&context, Hitl::Stdin))
            .await
            .expect("fix rounds are bounded")
            .unwrap();
        assert_eq!(end, RunEnd::Finished);

        let ctx = context.lock().await;
        assert_eq!(ctx.tasks.len(), 3, "one fix round, then stop");
        assert_eq!(ctx.count_completed(), 3);
        assert_eq!(llm.requests(Kind::DesignCheck).len(), 2);
        let unmet = ctx.unmet_requirements();
        assert_eq!(unmet.len(), 1);
        assert_eq!(unmet[0].requirement, "The history panel scrolls");
        assert_eq!(unmet[0].status, FindingStatus::Unmet);
        assert!(!ctx.succeeded(), "every task completed, but the design is not met");

        let saved = ProjectContext::load(&ctx.working_directory).unwrap();
        assert_eq!(saved.unmet_requirements().len(), 1, "the findings are kept in the saved state");
    }

    #[tokio::test]
    async fn a_design_check_that_cannot_be_completed_is_reported_not_assumed_fine() {
        let llm = ScriptedLlm::new(|kind, request, earlier| match kind {
            Kind::DesignCheck => Ok("Looks good to me!".to_string()),
            _ => agreeable(kind, request, earlier),
        });
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 50);

        assert_eq!(orchestrator.run(&context, Hitl::Stdin).await.unwrap(), RunEnd::Finished);
        let ctx = context.lock().await;
        assert_eq!(llm.requests(Kind::DesignCheck).len(), DESIGN_CHECK_ATTEMPTS);
        assert!(ctx.design_check_error.as_deref().unwrap().contains("could not be completed"));
        assert!(ctx.design_check.is_none());
        assert!(!ctx.succeeded());
    }

    #[tokio::test]
    async fn a_reviewer_whose_quote_is_not_in_the_file_is_asked_again_and_then_overruled() {
        let invented = |prompt: &str| {
            let findings: Vec<serde_json::Value> = items_in(prompt)
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    serde_json::json!({ "id": i + 1, "status": "met", "evidence": "implemented in the draw function",
                                        "file": "src/lib.rs", "quote": "KeyCode::Enter => self.evaluate_expression()" })
                })
                .collect();
            serde_json::to_string(&findings).unwrap()
        };
        let llm = ScriptedLlm::new(move |kind, request, earlier| match kind {
            Kind::DesignCheck => Ok(invented(&request.prompt)),
            _ => agreeable(kind, request, earlier),
        });
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 50);
        context.lock().await.fix_rounds = 0;

        assert_eq!(orchestrator.run(&context, Hitl::Stdin).await.unwrap(), RunEnd::Finished);
        let checks = llm.requests(Kind::DesignCheck);
        assert_eq!(checks.len(), 2, "one correction round");
        assert!(!checks[0].prompt.contains("Problems with your previous answer"));
        assert!(checks[1].prompt.contains("the quote was not found in src/lib.rs"), "{}", checks[1].prompt);

        let ctx = context.lock().await;
        // Build/test/layout items are evidenced by command output; everything that needed a quote is overruled.
        let unmet: Vec<&str> = ctx.unmet_requirements().iter().map(|f| f.requirement.as_str()).collect();
        assert!(unmet.contains(&"Enter evaluates the expression"), "{unmet:?}");
        assert!(unmet.contains(&"The history panel scrolls"));
        assert!(unmet.iter().any(|r| r.contains("entry point")));
        assert_eq!(unmet.len(), 4);
        assert!(!ctx.succeeded());
        assert_eq!(ctx.llm_calls, 1 + 2 + 2 + 1 + 2, "both check attempts are counted");
    }

    #[tokio::test]
    async fn behaviour_that_nothing_demonstrates_is_unmet_not_left_to_the_human() {
        // What gpt-4o answered for every example in the calculator design.
        let llm = ScriptedLlm::new(|kind, request, earlier| match kind {
            Kind::DesignCheck => Ok(answer(&request.prompt, |item| {
                if item.contains("Enter evaluates") || item.contains("Borders") {
                    ("cannot_verify", "Cannot verify without running the program interactively.")
                } else {
                    ("met", "src/lib.rs:1")
                }
            })),
            _ => agreeable(kind, request, earlier),
        });
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 50);
        context.lock().await.fix_rounds = 0;

        assert_eq!(orchestrator.run(&context, Hitl::Stdin).await.unwrap(), RunEnd::Finished);
        let ctx = context.lock().await;
        let unmet: Vec<&str> = ctx.unmet_requirements().iter().map(|f| f.requirement.as_str()).collect();
        assert_eq!(unmet, ["Enter evaluates the expression"]);
        let unverified: Vec<&str> = ctx.unverified_requirements().iter().map(|f| f.requirement.as_str()).collect();
        assert_eq!(unverified, ["Borders are cyan"], "only appearance is left to the human");
        assert!(llm.requests(Kind::DesignCheck)[1].prompt.contains("a test that runs exactly that input"));
        assert!(!ctx.succeeded());
    }

    #[tokio::test]
    async fn a_correction_round_cannot_undo_answers_that_were_already_accepted() {
        // Seen live: asked to fix one bad quote, the reviewer withdrew two good answers as well.
        let llm = ScriptedLlm::new(|kind, request, earlier| match kind {
            Kind::DesignCheck if earlier == 0 => {
                let mut findings: Vec<serde_json::Value> = serde_json::from_str(&all_met(&request.prompt)).unwrap();
                let bad = items_in(&request.prompt).iter().position(|i| i.contains("Enter evaluates")).unwrap();
                findings[bad]["quote"] = "this line is not in the file at all".into();
                Ok(serde_json::to_string(&findings).unwrap())
            }
            // Second answer: the bad one is now a proper "unmet", but everything else is withdrawn too.
            Kind::DesignCheck => Ok(answer(&request.prompt, |item| {
                if item.contains("Enter evaluates") { ("unmet", "only '=' is handled") } else { ("unmet", "changed my mind") }
            })),
            _ => agreeable(kind, request, earlier),
        });
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 50);
        context.lock().await.fix_rounds = 0;

        assert_eq!(orchestrator.run(&context, Hitl::Stdin).await.unwrap(), RunEnd::Finished);
        let correction = &llm.requests(Kind::DesignCheck)[1].prompt;
        assert_eq!(correction.matches("- Item ").count(), 1, "only the failed item is raised:\n{correction}");
        let ctx = context.lock().await;
        let unmet: Vec<(&str, &str)> =
            ctx.unmet_requirements().iter().map(|f| (f.requirement.as_str(), f.evidence.as_str())).collect();
        assert_eq!(unmet, [("Enter evaluates the expression", "only '=' is handled")]);
    }

    #[tokio::test]
    async fn long_checklists_are_verified_in_batches() {
        let llm = ScriptedLlm::new(|kind, request, earlier| match kind {
            Kind::Requirements => {
                // Plain strings are accepted too, and count as behaviour.
                Ok(serde_json::to_string(&(1..=17).map(|i| format!("Requirement number {i}")).collect::<Vec<_>>()).unwrap())
            }
            _ => agreeable(kind, request, earlier),
        });
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 50);

        assert_eq!(orchestrator.run(&context, Hitl::Stdin).await.unwrap(), RunEnd::Finished);
        let checks = llm.requests(Kind::DesignCheck);
        // 4 baseline items + 17 requirements, ten at a time.
        assert_eq!(checks.iter().map(|r| items_in(&r.prompt).len()).collect::<Vec<_>>(), [10, 10, 1]);
        let ctx = context.lock().await;
        assert_eq!(ctx.design_check.as_ref().unwrap().len(), 21);
        assert!(ctx.succeeded());
    }

    #[tokio::test]
    async fn the_design_check_can_be_switched_off() {
        let llm = ScriptedLlm::new(agreeable);
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 50);
        context.lock().await.design_check_enabled = false;

        assert_eq!(orchestrator.run(&context, Hitl::Stdin).await.unwrap(), RunEnd::Finished);
        assert!(llm.requests(Kind::Requirements).is_empty());
        assert!(llm.requests(Kind::DesignCheck).is_empty());
        assert!(context.lock().await.succeeded());
    }

    #[tokio::test]
    async fn the_standalone_check_reports_findings_and_writes_nothing() {
        let llm = ScriptedLlm::new(|kind, request, earlier| match kind {
            Kind::DesignCheck => Ok(answer(&request.prompt, |item| {
                if item.contains("Enter evaluates") {
                    ("unmet", "only '=' is handled")
                } else if item.contains("Borders are cyan") {
                    ("cannot_verify", "needs a terminal")
                } else {
                    ("met", "src/main.rs:1")
                }
            })),
            _ => agreeable(kind, request, earlier),
        });
        let (_dir, orchestrator, context) = setup(llm.clone(), true, 50);
        let settings = orchestrator.settings(&*context.lock().await);

        let check = check_design(llm.as_ref(), &[], &settings).await.unwrap();
        assert_eq!(check.requirements.len(), 3);
        assert!(check.requirements[2].appearance && !check.requirements[0].appearance);
        assert_eq!(check.calls, 2);
        let count = |status| check.findings.iter().filter(|f| f.status == status).count();
        assert_eq!((count(FindingStatus::Unmet), count(FindingStatus::CannotVerify)), (1, 1));
        assert!(count(FindingStatus::Met) >= 4, "the baseline items were answered too");
        assert!(llm.requests(Kind::Worker).is_empty());
        assert!(!orchestrator.sandbox.root().join(".clankistrator").exists());
    }

    #[test]
    fn plans_are_validated() {
        let plan = parse_plan(TWO_TASK_PLAN).unwrap();
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].acceptance_criteria, vec!["Cargo.toml exists in the project root"]);
        assert!(parse_plan("[]").unwrap_err().contains("empty"));
        assert!(parse_plan("nothing").unwrap_err().contains("JSON list"));
        assert!(parse_plan(r#"[{"title":"x"}]"#).unwrap_err().contains("malformed"));
        assert!(
            parse_plan(r#"[{"id":"A","title":"t","description":"d","assigned_role":"Wizard"}]"#)
                .unwrap_err()
                .contains("unknown role")
        );
        // Ids are filled in when the model leaves them out; criteria are optional.
        let tasks = parse_plan(r#"```json
            [{"title":"t","description":"d","assigned_role":"QA"}]
        ```"#)
        .unwrap();
        assert_eq!(tasks[0].id, "TASK-1");
        assert!(tasks[0].acceptance_criteria.is_empty());
    }

    #[test]
    fn transcript_names_cannot_escape_the_transcript_directory() {
        let dir = tempfile::tempdir().unwrap();
        save_transcript(dir.path(), "../../evil/TASK 1.worker", Some("first"));
        save_transcript(dir.path(), "../../evil/TASK 1.worker", Some("second"));
        let kept = dir.path().join(".clankistrator/transcripts");
        let mut names: Vec<String> =
            std::fs::read_dir(&kept).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        names.sort();
        assert_eq!(names, [".._.._evil_TASK_1.worker.1.md", ".._.._evil_TASK_1.worker.2.md"]);
        assert!(!dir.path().join("evil").exists());
    }
}
