use anyhow::{Context as _, Result};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

/// Directory inside the output directory where Clankistrator keeps its own state.
pub const STATE_DIR: &str = ".clankistrator";
const STATE_FILE: &str = "state.json";

/// How many times a task may fail execution or review before it is marked Failed.
pub const MAX_RETRIES: u32 = 3;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub enum TaskStatus {
    Pending,
    InProgress,
    ReadyForReview { output_summary: String },
    Reviewing { output_summary: String },
    RequiresHumanApproval { output_summary: String },
    Completed { output_summary: String },
    Failed { error: String },
}

impl TaskStatus {
    pub fn is_terminal(&self) -> bool {
        matches!(self, TaskStatus::Completed { .. } | TaskStatus::Failed { .. })
    }

    pub fn label(&self) -> &'static str {
        match self {
            TaskStatus::Pending => "Pending",
            TaskStatus::InProgress => "In progress",
            TaskStatus::ReadyForReview { .. } => "Ready for review",
            TaskStatus::Reviewing { .. } => "Reviewing",
            TaskStatus::RequiresHumanApproval { .. } => "Waiting for approval",
            TaskStatus::Completed { .. } => "Completed",
            TaskStatus::Failed { .. } => "Failed",
        }
    }
}

#[derive(Debug, Serialize, Clone, Copy, PartialEq, Eq)]
pub enum AgentRole {
    Infrastructure,
    Backend,
    Frontend,
    QA,
}

impl AgentRole {
    /// Case-insensitive, because models are not consistent about capitalisation.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "infrastructure" | "infra" => Some(Self::Infrastructure),
            "backend" => Some(Self::Backend),
            "frontend" => Some(Self::Frontend),
            "qa" => Some(Self::QA),
            _ => None,
        }
    }
}

impl<'de> Deserialize<'de> for AgentRole {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        AgentRole::parse(&s).ok_or_else(|| {
            serde::de::Error::custom(format!(
                "unknown role {s:?} (expected Infrastructure, Backend, Frontend or QA)"
            ))
        })
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ProjectTask {
    pub id: String,
    pub title: String,
    pub description: String,
    pub dependencies: Vec<String>,
    pub status: TaskStatus,
    pub assigned_role: AgentRole,
    /// Feedback from a failed attempt, a failed review or a human rejection.
    pub feedback: Option<String>,
    #[serde(default)]
    pub retries: u32,
    /// Files the worker reported creating or modifying (relative to the output directory).
    #[serde(default)]
    pub files: Vec<String>,
    /// Concrete, checkable conditions the reviewer verifies one by one.
    #[serde(default)]
    pub acceptance_criteria: Vec<String>,
    /// What the reviewer could not verify; shown to the human who approves.
    #[serde(default)]
    pub review_notes: Vec<String>,
    /// The model chosen for this task, if one was: by the advisor, or by
    /// stepping up after failures. Otherwise the first usable worker model.
    #[serde(default)]
    pub model: Option<String>,
    /// Failed attempts since this task was last given a different model.
    #[serde(default)]
    pub failures_on_model: u32,
}

impl ProjectTask {
    pub fn new(id: String, def: TaskDef) -> Self {
        Self {
            id,
            title: def.title,
            description: def.description,
            dependencies: def.dependencies,
            status: TaskStatus::Pending,
            assigned_role: def.assigned_role,
            feedback: None,
            retries: 0,
            files: Vec::new(),
            acceptance_criteria: def.acceptance_criteria,
            review_notes: Vec::new(),
            model: def.model.map(|m| m.trim().to_string()).filter(|m| !m.is_empty()),
            failures_on_model: 0,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum FindingStatus {
    Met,
    Unmet,
    /// Only checkable by running the program interactively; left to the human.
    CannotVerify,
}

/// A reviewer's answer for one requirement. The reviewer supplies status and
/// evidence per item; whether the whole thing passes is decided in code.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct Finding {
    pub requirement: String,
    pub status: FindingStatus,
    pub evidence: String,
    /// The reviewer called this met and a mechanical check turned it into
    /// unmet, because the reviewer did not show its evidence. That is a doubt
    /// about the review, not a defect anyone found in the work.
    #[serde(default)]
    pub overruled: bool,
}

impl Finding {
    pub fn line(&self) -> String {
        if self.evidence.trim().is_empty() {
            format!("- {}", self.requirement)
        } else {
            format!("- {}\n  {}", self.requirement, self.evidence.trim())
        }
    }
}

/// A task as described by a model: either in the initial plan or as a follow-up
/// task reported by a worker.
#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct TaskDef {
    #[serde(default)]
    pub id: Option<String>,
    pub title: String,
    pub description: String,
    pub assigned_role: AgentRole,
    #[serde(default)]
    pub dependencies: Vec<String>,
    #[serde(default)]
    pub acceptance_criteria: Vec<String>,
    /// The worker model this task should go to, when the planner chose one.
    #[serde(default)]
    pub model: Option<String>,
}

/// What one model did in a run: what it cost, and how its work fared.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Default)]
pub struct ModelUsage {
    pub calls: u64,
    pub input_tokens: u64,
    pub cached_input_tokens: u64,
    pub output_tokens: u64,
    /// Task attempts by this model that passed review.
    pub passed: u32,
    /// Task attempts by this model that failed or that the reviewer rejected.
    pub rejected: u32,
}

/// After this many failed attempts on one model, a task moves to the next
/// worker model, if there is one.
pub const STEP_UP_AFTER: u32 = 2;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Default)]
pub enum RunPhase {
    #[default]
    Starting,
    Planning,
    Running,
    AwaitingApproval,
    /// Checking the finished project against the design document.
    Checking,
    StepLimit,
    Finished,
    Error(String),
}

impl RunPhase {
    pub fn label(&self) -> String {
        match self {
            RunPhase::Starting => "Starting".into(),
            RunPhase::Planning => "Planning tasks".into(),
            RunPhase::Running => "Running".into(),
            RunPhase::AwaitingApproval => "Waiting for your approval".into(),
            RunPhase::Checking => "Checking the project against the design".into(),
            RunPhase::StepLimit => "Step limit reached".into(),
            RunPhase::Finished => "Finished".into(),
            RunPhase::Error(e) => format!("Stopped: {e}"),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ProjectContext {
    pub project_name: String,
    pub tasks: Vec<ProjectTask>,
    pub working_directory: PathBuf,
    /// Token counts as reported by the provider.
    #[serde(default)]
    pub input_tokens: u64,
    /// The part of `input_tokens` read from the provider's prompt cache.
    #[serde(default)]
    pub cached_input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default)]
    pub llm_calls: u64,
    /// Orchestration iterations that executed or reviewed at least one task.
    #[serde(default)]
    pub steps_taken: usize,
    pub max_steps: usize,
    pub max_turns: usize,
    pub brave_mode: bool,
    pub model: String,
    /// Model for reviews and the design check; the worker model if unset.
    #[serde(default)]
    pub review_model: Option<String>,
    #[serde(default)]
    pub phase: RunPhase,
    /// Whether the finished project is checked against the design document.
    #[serde(default = "enabled")]
    pub design_check_enabled: bool,
    /// How many times unmet requirements may be turned into a fix task.
    #[serde(default = "default_fix_rounds")]
    pub fix_rounds: usize,
    #[serde(default)]
    pub fix_rounds_done: usize,
    /// Requirements extracted from the design document for the design check.
    #[serde(default)]
    pub requirements: Vec<Requirement>,
    /// Result of the latest design check, if one has run.
    #[serde(default)]
    pub design_check: Option<Vec<Finding>>,
    /// How many tasks existed when that check ran. Tasks added since (a fix
    /// task) mean the project has to be checked again.
    #[serde(default)]
    pub design_checked_task_count: usize,
    /// Why the design check could not be completed, if it failed.
    #[serde(default)]
    pub design_check_error: Option<String>,
    /// The models that may build, cheapest first. Empty means only `model`.
    #[serde(default)]
    pub worker_models: Vec<String>,
    /// Worker models found unusable during this run (a dead key, a used-up
    /// allowance), with the reason. Not saved: tomorrow they may work again.
    #[serde(skip)]
    pub retired_models: Vec<(String, String)>,
    /// Calls, tokens and results per model.
    #[serde(default)]
    pub model_usage: BTreeMap<String, ModelUsage>,
    /// A stronger model that plans, assigns tasks to worker models and, unless
    /// a review model is set, reviews.
    #[serde(default)]
    pub advisor_model: Option<String>,
    /// OpenRouter's count of free-model requests left when the run started.
    #[serde(skip)]
    pub free_requests_at_start: Option<u64>,
}

fn enabled() -> bool {
    true
}

pub const DEFAULT_FIX_ROUNDS: usize = 2;

fn default_fix_rounds() -> usize {
    DEFAULT_FIX_ROUNDS
}

/// One checkable requirement taken from the design document.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct Requirement {
    pub text: String,
    /// True only for what can be judged solely by looking at the running
    /// program (colours, alignment). Everything else is behaviour, which the
    /// code or a test either demonstrates or does not.
    #[serde(default)]
    pub appearance: bool,
}

impl ProjectContext {
    pub fn new(
        name: String,
        output_dir: PathBuf,
        max_steps: usize,
        max_turns: usize,
        brave_mode: bool,
        model: String,
    ) -> Self {
        Self {
            project_name: name,
            tasks: Vec::new(),
            working_directory: output_dir,
            input_tokens: 0,
            cached_input_tokens: 0,
            output_tokens: 0,
            llm_calls: 0,
            steps_taken: 0,
            max_steps,
            max_turns,
            brave_mode,
            model,
            review_model: None,
            phase: RunPhase::Starting,
            design_check_enabled: true,
            fix_rounds: DEFAULT_FIX_ROUNDS,
            fix_rounds_done: 0,
            requirements: Vec::new(),
            design_check: None,
            design_checked_task_count: 0,
            design_check_error: None,
            worker_models: Vec::new(),
            retired_models: Vec::new(),
            model_usage: BTreeMap::new(),
            advisor_model: None,
            free_requests_at_start: None,
        }
    }

    /// The model that plans: the advisor if there is one.
    pub fn planner_model(&self) -> &str {
        self.advisor_model.as_deref().unwrap_or(&self.model)
    }

    /// The worker models in the order given (cheapest first).
    pub fn pool(&self) -> Vec<&str> {
        if self.worker_models.is_empty() {
            vec![self.model.as_str()]
        } else {
            self.worker_models.iter().map(String::as_str).collect()
        }
    }

    pub fn is_retired(&self, model: &str) -> bool {
        self.retired_models.iter().any(|(retired, _)| retired == model)
    }

    /// The worker models that can still be used in this run.
    pub fn usable_pool(&self) -> Vec<&str> {
        self.pool().into_iter().filter(|model| !self.is_retired(model)).collect()
    }

    /// Stop using a worker model for the rest of this run.
    pub fn retire_model(&mut self, model: &str, reason: &str) {
        if !self.is_retired(model) {
            self.retired_models.push((model.to_string(), reason.to_string()));
        }
    }

    /// The model a task's next attempt runs on: the one chosen for it if that
    /// is still usable, otherwise the first usable worker model.
    pub fn worker_model_for(&self, task: &ProjectTask) -> String {
        let usable = self.usable_pool();
        match &task.model {
            Some(model) if usable.contains(&model.as_str()) => model.clone(),
            _ => usable.first().copied().unwrap_or(&self.model).to_string(),
        }
    }

    /// The next usable worker model after `current` in the pool's order.
    pub fn next_model_up(&self, current: &str) -> Option<String> {
        let pool = self.pool();
        let at = pool.iter().position(|model| *model == current)?;
        pool[at + 1..].iter().find(|model| !self.is_retired(model)).map(|model| model.to_string())
    }

    /// The strongest usable worker model: the last one in the pool's order.
    pub fn strongest_model(&self) -> String {
        self.usable_pool().last().copied().unwrap_or(&self.model).to_string()
    }

    /// Whether the finished project still has to be checked against the design.
    /// A run with failed tasks is not checked: it has already not succeeded.
    pub fn design_check_due(&self) -> bool {
        self.design_check_enabled
            && self.all_terminal()
            && self.count_failed() == 0
            && self.design_check_error.is_none()
            && (self.design_check.is_none() || self.tasks.len() > self.design_checked_task_count)
    }

    /// "1234 in (1000 from cache) / 56 out", for summaries.
    pub fn token_summary(&self) -> String {
        format_tokens(self.input_tokens, self.cached_input_tokens, self.output_tokens)
    }

    /// The model used for reviews and the design check.
    pub fn reviewer_model(&self) -> &str {
        self.review_model.as_deref().or(self.advisor_model.as_deref()).unwrap_or(&self.model)
    }

    /// Requirements the latest design check found unmet.
    pub fn unmet_requirements(&self) -> Vec<&Finding> {
        self.design_check
            .iter()
            .flatten()
            .filter(|f| f.status == FindingStatus::Unmet)
            .collect()
    }

    /// Requirements the latest design check could not verify.
    pub fn unverified_requirements(&self) -> Vec<&Finding> {
        self.design_check
            .iter()
            .flatten()
            .filter(|f| f.status == FindingStatus::CannotVerify)
            .collect()
    }

    /// True when the run ended with nothing left to do and nothing known to be wrong.
    pub fn succeeded(&self) -> bool {
        self.all_terminal()
            && self.count_failed() == 0
            && self.unmet_requirements().is_empty()
            && self.design_check_error.is_none()
            && !self.design_check_due()
    }

    pub fn get_task_mut(&mut self, task_id: &str) -> Option<&mut ProjectTask> {
        self.tasks.iter_mut().find(|t| t.id == task_id)
    }

    pub fn get_task(&self, task_id: &str) -> Option<&ProjectTask> {
        self.tasks.iter().find(|t| t.id == task_id)
    }

    pub fn is_task_completed(&self, task_id: &str) -> bool {
        self.get_task(task_id)
            .map(|t| matches!(t.status, TaskStatus::Completed { .. }))
            .unwrap_or(false)
    }

    pub fn get_ready_tasks(&self) -> Vec<&ProjectTask> {
        self.tasks
            .iter()
            .filter(|task| {
                task.status == TaskStatus::Pending
                    && task
                        .dependencies
                        .iter()
                        .all(|dep_id| self.is_task_completed(dep_id))
            })
            .collect()
    }

    pub fn get_tasks_for_review(&self) -> Vec<&ProjectTask> {
        self.tasks
            .iter()
            .filter(|task| matches!(task.status, TaskStatus::ReadyForReview { .. }))
            .collect()
    }

    pub fn get_tasks_for_approval(&self) -> Vec<&ProjectTask> {
        self.tasks
            .iter()
            .filter(|task| matches!(task.status, TaskStatus::RequiresHumanApproval { .. }))
            .collect()
    }

    pub fn all_terminal(&self) -> bool {
        !self.tasks.is_empty() && self.tasks.iter().all(|t| t.status.is_terminal())
    }

    pub fn count_completed(&self) -> usize {
        self.tasks
            .iter()
            .filter(|t| matches!(t.status, TaskStatus::Completed { .. }))
            .count()
    }

    pub fn count_failed(&self) -> usize {
        self.tasks
            .iter()
            .filter(|t| matches!(t.status, TaskStatus::Failed { .. }))
            .count()
    }

    /// Fail every pending task that can never run because a dependency failed or
    /// does not exist. Returns the ids that were failed. Without this, a failed
    /// task would leave its dependents pending forever.
    pub fn fail_blocked(&mut self) -> Vec<String> {
        let mut newly_failed = Vec::new();
        loop {
            let blocked: Vec<(String, String)> = self
                .tasks
                .iter()
                .filter(|t| t.status == TaskStatus::Pending)
                .filter_map(|t| {
                    t.dependencies.iter().find_map(|dep| match self.get_task(dep) {
                        None => Some((t.id.clone(), format!("dependency {dep} does not exist"))),
                        Some(d) if matches!(d.status, TaskStatus::Failed { .. }) => {
                            Some((t.id.clone(), format!("dependency {dep} failed")))
                        }
                        Some(_) => None,
                    })
                })
                .collect();
            if blocked.is_empty() {
                return newly_failed;
            }
            for (id, why) in blocked {
                if let Some(t) = self.get_task_mut(&id) {
                    t.status = TaskStatus::Failed { error: format!("Blocked: {why}") };
                }
                newly_failed.push(id);
            }
        }
    }

    /// Human approval: RequiresHumanApproval -> Completed. Returns false if the
    /// task was not waiting for approval.
    pub fn approve(&mut self, task_id: &str) -> bool {
        let Some(t) = self.get_task_mut(task_id) else { return false };
        let TaskStatus::RequiresHumanApproval { output_summary } = &t.status else { return false };
        t.status = TaskStatus::Completed { output_summary: output_summary.clone() };
        t.feedback = None;
        true
    }

    /// Human rejection: the task goes back to Pending with the given feedback.
    pub fn reject(&mut self, task_id: &str, feedback: &str) -> bool {
        let Some(t) = self.get_task_mut(task_id) else { return false };
        if !matches!(t.status, TaskStatus::RequiresHumanApproval { .. }) {
            return false;
        }
        t.status = TaskStatus::Pending;
        t.retries = 0;
        t.feedback = Some(if feedback.trim().is_empty() {
            "A human reviewer rejected the previous attempt without giving details. Re-check the task against the design document.".to_string()
        } else {
            format!("A human reviewer rejected the previous attempt: {}", feedback.trim())
        });
        true
    }

    /// Pick an id that no existing task uses.
    pub fn next_task_id(&self) -> String {
        let mut n = self.tasks.len() + 1;
        loop {
            let id = format!("TASK-{n}");
            if self.get_task(&id).is_none() {
                return id;
            }
            n += 1;
        }
    }

    pub fn state_path(output_dir: &Path) -> PathBuf {
        output_dir.join(STATE_DIR).join(STATE_FILE)
    }

    /// Persist the context to `<output>/.clankistrator/state.json` (write + rename).
    pub fn save(&self) -> Result<()> {
        let path = Self::state_path(&self.working_directory);
        let dir = path.parent().expect("state path has a parent");
        std::fs::create_dir_all(dir)
            .with_context(|| format!("creating {}", dir.display()))?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(self)?)
            .with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, &path).with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }

    pub fn load(output_dir: &Path) -> Result<Self> {
        let path = Self::state_path(output_dir);
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("reading saved state {}", path.display()))?;
        serde_json::from_str(&raw).with_context(|| format!("parsing saved state {}", path.display()))
    }

    /// Make a loaded context runnable again: work that was interrupted mid-flight is
    /// re-queued, and failed tasks get a fresh set of retries.
    pub fn prepare_resume(&mut self) {
        for t in &mut self.tasks {
            match &t.status {
                TaskStatus::InProgress => t.status = TaskStatus::Pending,
                TaskStatus::Reviewing { output_summary } => {
                    t.status = TaskStatus::ReadyForReview { output_summary: output_summary.clone() };
                }
                TaskStatus::Failed { error } => {
                    if !error.starts_with("Blocked:") {
                        t.feedback = Some(format!("An earlier run gave up on this task: {error}"));
                    }
                    t.status = TaskStatus::Pending;
                    t.retries = 0;
                }
                _ => {}
            }
        }
        self.steps_taken = 0;
        self.phase = RunPhase::Starting;
        // The project is checked afresh at the end of the resumed run.
        self.fix_rounds_done = 0;
        self.design_check = None;
        self.design_check_error = None;
    }
}

pub fn format_tokens(input: u64, cached: u64, output: u64) -> String {
    if cached > 0 {
        format!("{input} in ({cached} from cache) / {output} out")
    } else {
        format!("{input} in / {output} out")
    }
}

/// Check that a task list forms a usable graph: unique non-empty ids, every
/// dependency refers to a task in the list, and there are no cycles.
pub fn validate_task_graph(tasks: &[ProjectTask]) -> Result<(), String> {
    if tasks.is_empty() {
        return Err("the task list is empty".into());
    }
    let mut ids = HashSet::new();
    for t in tasks {
        if t.id.trim().is_empty() {
            return Err(format!("task {:?} has an empty id", t.title));
        }
        if !ids.insert(t.id.as_str()) {
            return Err(format!("duplicate task id {}", t.id));
        }
    }
    for t in tasks {
        for dep in &t.dependencies {
            if dep == &t.id {
                return Err(format!("task {} depends on itself", t.id));
            }
            if !ids.contains(dep.as_str()) {
                return Err(format!("task {} depends on unknown task {}", t.id, dep));
            }
        }
    }

    // Kahn's algorithm: whatever cannot be scheduled is part of a cycle.
    let mut remaining: HashMap<&str, HashSet<&str>> = tasks
        .iter()
        .map(|t| (t.id.as_str(), t.dependencies.iter().map(String::as_str).collect()))
        .collect();
    loop {
        let ready: Vec<&str> = remaining
            .iter()
            .filter(|(_, deps)| deps.is_empty())
            .map(|(id, _)| *id)
            .collect();
        if ready.is_empty() {
            break;
        }
        for id in &ready {
            remaining.remove(id);
        }
        for deps in remaining.values_mut() {
            for id in &ready {
                deps.remove(id);
            }
        }
    }
    if !remaining.is_empty() {
        let mut cyclic: Vec<&str> = remaining.keys().copied().collect();
        cyclic.sort_unstable();
        return Err(format!("dependency cycle among tasks: {}", cyclic.join(", ")));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn task(id: &str, deps: &[&str]) -> ProjectTask {
        ProjectTask {
            id: id.into(),
            title: format!("Title of {id}"),
            description: format!("Description of {id}"),
            dependencies: deps.iter().map(|d| d.to_string()).collect(),
            status: TaskStatus::Pending,
            assigned_role: AgentRole::Backend,
            feedback: None,
            retries: 0,
            files: vec![],
            acceptance_criteria: vec![],
            review_notes: vec![],
            model: None,
            failures_on_model: 0,
        }
    }

    fn context(dir: &Path) -> ProjectContext {
        ProjectContext::new("Test".into(), dir.to_path_buf(), 50, 40, false, "test-model".into())
    }

    #[test]
    fn ready_tasks_follow_dependencies() {
        let mut ctx = context(Path::new("."));
        ctx.tasks = vec![task("TASK-1", &[]), task("TASK-2", &["TASK-1"])];

        let ready = ctx.get_ready_tasks();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, "TASK-1");

        ctx.get_task_mut("TASK-1").unwrap().status =
            TaskStatus::Completed { output_summary: "Done".into() };

        let ready = ctx.get_ready_tasks();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, "TASK-2");
    }

    #[test]
    fn failed_dependency_fails_the_whole_chain() {
        let mut ctx = context(Path::new("."));
        ctx.tasks = vec![
            task("A", &[]),
            task("B", &["A"]),
            task("C", &["B"]),
            task("D", &[]),
        ];
        ctx.get_task_mut("A").unwrap().status = TaskStatus::Failed { error: "boom".into() };

        let mut failed = ctx.fail_blocked();
        failed.sort();
        assert_eq!(failed, vec!["B", "C"]);
        assert_eq!(ctx.get_task("D").unwrap().status, TaskStatus::Pending);
        assert!(!ctx.all_terminal());
    }

    #[test]
    fn graph_validation_rejects_bad_graphs() {
        assert!(validate_task_graph(&[]).is_err());
        assert!(validate_task_graph(&[task("A", &[]), task("B", &["A"])]).is_ok());
        assert!(validate_task_graph(&[task("A", &[]), task("A", &[])])
            .unwrap_err()
            .contains("duplicate"));
        assert!(validate_task_graph(&[task("A", &["Z"])]).unwrap_err().contains("unknown"));
        assert!(validate_task_graph(&[task("A", &["A"])]).unwrap_err().contains("itself"));
        assert!(validate_task_graph(&[task("A", &["B"]), task("B", &["A"]), task("C", &[])])
            .unwrap_err()
            .contains("cycle"));
    }

    #[test]
    fn roles_parse_case_insensitively() {
        let def: TaskDef = serde_json::from_str(
            r#"{"title":"t","description":"d","assigned_role":"backend"}"#,
        )
        .unwrap();
        assert_eq!(def.assigned_role, AgentRole::Backend);
        assert!(def.dependencies.is_empty());
        assert!(serde_json::from_str::<AgentRole>(r#""Critic""#).is_err());
        assert_eq!(serde_json::from_str::<AgentRole>(r#""QA""#).unwrap(), AgentRole::QA);
    }

    #[test]
    fn approve_and_reject() {
        let mut ctx = context(Path::new("."));
        ctx.tasks = vec![task("A", &[])];
        assert!(!ctx.approve("A"), "pending tasks cannot be approved");

        ctx.get_task_mut("A").unwrap().status =
            TaskStatus::RequiresHumanApproval { output_summary: "did it".into() };
        assert!(ctx.reject("A", "wrong colour"));
        let t = ctx.get_task("A").unwrap();
        assert_eq!(t.status, TaskStatus::Pending);
        assert!(t.feedback.as_deref().unwrap().contains("wrong colour"));

        ctx.get_task_mut("A").unwrap().status =
            TaskStatus::RequiresHumanApproval { output_summary: "did it".into() };
        assert!(ctx.approve("A"));
        assert!(ctx.all_terminal());
    }

    #[test]
    fn state_round_trips_and_resume_requeues_interrupted_work() {
        let dir = tempfile::tempdir().unwrap();
        let mut ctx = context(dir.path());
        ctx.tasks = vec![task("A", &[]), task("B", &[]), task("C", &[]), task("D", &["C"])];
        ctx.get_task_mut("A").unwrap().status =
            TaskStatus::Completed { output_summary: "ok".into() };
        ctx.get_task_mut("B").unwrap().status = TaskStatus::InProgress;
        ctx.get_task_mut("C").unwrap().status = TaskStatus::Failed { error: "nope".into() };
        ctx.get_task_mut("C").unwrap().retries = 4;
        ctx.fail_blocked();
        ctx.steps_taken = 7;
        ctx.input_tokens = 123;
        ctx.save().unwrap();

        let mut loaded = ProjectContext::load(dir.path()).unwrap();
        assert_eq!(loaded.input_tokens, 123);
        assert_eq!(loaded.get_task("B").unwrap().status, TaskStatus::InProgress);

        loaded.prepare_resume();
        assert!(loaded.is_task_completed("A"));
        assert_eq!(loaded.get_task("B").unwrap().status, TaskStatus::Pending);
        let c = loaded.get_task("C").unwrap();
        assert_eq!(c.status, TaskStatus::Pending);
        assert_eq!(c.retries, 0);
        assert!(c.feedback.as_deref().unwrap().contains("nope"));
        let d = loaded.get_task("D").unwrap();
        assert_eq!(d.status, TaskStatus::Pending);
        assert!(d.feedback.is_none());
        assert_eq!(loaded.steps_taken, 0);
    }

    #[test]
    fn next_task_id_skips_ids_in_use() {
        let mut ctx = context(Path::new("."));
        ctx.tasks = vec![task("TASK-2", &[])];
        assert_eq!(ctx.next_task_id(), "TASK-3");
        ctx.tasks = vec![task("TASK-2", &[]), task("TASK-3", &[])];
        assert_eq!(ctx.next_task_id(), "TASK-4");
    }
}
