pub mod agents;
pub mod context;
pub mod delegate;
pub mod llm;
pub mod mcp;
pub mod orchestrator;
pub mod parser;
pub mod probe;
pub mod tools;
pub mod tui;

use std::path::PathBuf;
use std::process::ExitCode;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use clap::Parser;
use tokio::sync::Mutex;
use tokio_stream::StreamExt;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::agents::AgentSettings;
use crate::context::{DEFAULT_FIX_ROUNDS, Finding, FindingStatus, ProjectContext, TaskStatus, format_tokens};
use crate::llm::{RigLlm, check_credentials};
use crate::orchestrator::{Hitl, Orchestrator, PROVIDER_RETRY_DELAY, RunEnd, SharedContext, check_design};
use crate::tools::Sandbox;
use crate::tui::App;

/// Used when neither --model nor $DEFAULT_MODEL names a model. A model name
/// in source code goes stale (this was `gpt-4o` until OpenAI scheduled it for
/// shutdown), so the startup request to the model is what really decides
/// whether it can be used.
const FALLBACK_MODEL: &str = "gpt-6.1-sol";
const DEFAULT_MAX_STEPS: usize = 50;
const DEFAULT_MAX_TURNS: usize = 40;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Markdown design document to implement
    #[arg(short, long, default_value = "design.md")]
    design: PathBuf,

    /// Directory the agents work in; their file tools cannot reach outside it
    #[arg(short, long, env = "WORKING_DIRECTORY", default_value = "output")]
    output: PathBuf,

    /// Model, optionally with a provider prefix (openai:, anthropic:, gemini:, ollama:).
    /// Defaults to $DEFAULT_MODEL, then gpt-6.1-sol
    #[arg(short, long)]
    model: Option<String>,

    /// Plain log output instead of the TUI; approvals are asked on stdin
    #[arg(long)]
    headless: bool,

    /// Approve tasks automatically once they pass review
    #[arg(short, long)]
    brave: bool,

    /// Maximum orchestration steps, counting only passes that ran or reviewed a task (0 = unlimited) [default: 50]
    #[arg(short = 'S', long)]
    max_steps: Option<usize>,

    /// Maximum model turns (tool-call rounds) a worker gets per task attempt [default: 40]
    #[arg(long)]
    max_turns: Option<usize>,

    /// Seconds a command started by an agent may run before it is killed
    #[arg(long, default_value_t = 300)]
    command_timeout: u64,

    /// Model for reviews and the design check, if it should differ from --model.
    /// Several, comma-separated: the first does the design check, and each task
    /// is reviewed by the first one from another provider than its builder
    #[arg(long, value_delimiter = ',', value_name = "MODEL,MODEL")]
    review_model: Vec<String>,

    /// A stronger model that leads the run: it plans, assigns each task to one of the --workers
    /// models, and reviews unless --review-model names another model
    #[arg(long)]
    advisor_model: Option<String>,

    /// Models that may build, cheapest first, separated by commas. A task starts on the first
    /// and moves to the next after two failed attempts, or at once if a model becomes unusable
    /// (for example when a free allowance runs out). Without this, only --model builds
    #[arg(long, value_delimiter = ',', value_name = "MODEL,MODEL")]
    workers: Vec<String>,

    /// How many times unmet design requirements are turned into a fix task [default: 2]
    #[arg(long)]
    fix_rounds: Option<usize>,

    /// Do not check the finished project against the design document
    #[arg(long)]
    skip_design_check: bool,

    /// Only check the project already in --output against --design, and report. Runs no tasks
    #[arg(long, conflicts_with_all = ["resume", "mcp_server"])]
    check: bool,

    /// Do not ask Anthropic to cache prompts between an agent's turns
    #[arg(long)]
    no_prompt_cache: bool,

    /// Continue the run saved in the output directory instead of starting over
    #[arg(long)]
    resume: bool,

    /// Run as an MCP server on stdio instead of orchestrating
    #[arg(long)]
    mcp_server: bool,

    /// With --mcp-server: also offer the run_agent tool, which lets the MCP client run an agent
    /// that reads files, runs commands and, if asked, changes files in a project directory.
    /// Agents run only on the models named with --model, --workers, --review-model and
    /// --advisor-model. Settings are then not read from .env files in the current directory
    #[arg(long, requires = "mcp_server")]
    delegate: bool,

    /// Read settings (API keys, DEFAULT_MODEL and so on) from this file instead of .env.local
    /// and .env in the current directory
    #[arg(long, value_name = "FILE")]
    env_file: Option<PathBuf>,

    /// With --delegate: run agents' commands as ordinary processes instead of in a bubblewrap
    /// jail. Without this, the server does not start where it cannot build a jail
    #[arg(long, requires = "delegate")]
    no_jail: bool,

    /// With --delegate: let jailed commands use the network
    #[arg(long, requires = "delegate", conflicts_with = "no_jail")]
    allow_network: bool,

    /// With --delegate: paths outside the project that jailed commands may read, separated by
    /// commas (a toolchain under your home directory, say). The rest of your home directory is
    /// empty in the jail
    #[arg(long, requires = "delegate", conflicts_with = "no_jail", value_delimiter = ',', value_name = "PATH,PATH")]
    allow_read: Vec<PathBuf>,

    /// With --delegate: paths in the project, relative to its root, that agents may not see,
    /// separated by commas
    #[arg(long, requires = "delegate", value_delimiter = ',', value_name = "PATH,PATH")]
    hide: Vec<PathBuf>,

    /// Test the given models instead of orchestrating: one plain request and one
    /// tool call each. Uses no files and runs no commands
    #[arg(long, num_args = 1.., value_name = "MODEL", conflicts_with_all = ["resume", "mcp_server", "check"])]
    probe: Vec<String>,
}

fn default_model() -> String {
    std::env::var("DEFAULT_MODEL")
        .ok()
        .filter(|m| !m.trim().is_empty())
        .unwrap_or_else(|| FALLBACK_MODEL.to_string())
}

fn log_level() -> String {
    std::env::var("LOG_LEVEL").ok().filter(|l| !l.trim().is_empty()).unwrap_or_else(|| "info".to_string())
}

/// `RUST_LOG` wins if set; otherwise `LOG_LEVEL` (default: info). rig logs every
/// tool call with its full arguments and result at info level, which buries
/// everything else, so it is held at warn unless `RUST_LOG` asks for more.
fn env_filter() -> tracing_subscriber::EnvFilter {
    tracing_subscriber::EnvFilter::try_from_default_env()
        .or_else(|_| tracing_subscriber::EnvFilter::try_new(format!("{},rig=warn", log_level())))
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,rig=warn"))
}

fn main() -> ExitCode {
    let runtime = match tokio::runtime::Runtime::new() {
        Ok(runtime) => runtime,
        Err(e) => {
            eprintln!("Error: could not start the async runtime: {e}");
            return ExitCode::FAILURE;
        }
    };
    let code = match runtime.block_on(run()) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("Error: {e:#}");
            ExitCode::FAILURE
        }
    };
    // Leave promptly: do not wait for a blocked stdin read or for agents that
    // were cut short, but do not leave their commands running either.
    tools::command_executor::kill_running_commands();
    runtime.shutdown_background();
    code
}

/// Load settings from dotenv files. Variables already in the environment win.
///
/// A delegating MCP server is started by its client, usually inside the
/// project the agents will work on, so `.env.local` in the current directory
/// is that project's and not ours: reading it would bill model calls to the
/// project's own provider key and hand every one of its settings to the
/// commands agents run. Such a server takes its settings from the environment
/// and from `--env-file` only.
fn load_settings(env_file: Option<&std::path::Path>, delegate: bool) -> Result<()> {
    match env_file {
        Some(file) => {
            // dotenvy's error repeats its cause, so it is not kept as one.
            dotenvy::from_path(file).map_err(|e| anyhow::anyhow!("could not read --env-file {}: {e}", file.display()))?;
        }
        None if delegate => {}
        None => {
            dotenvy::from_filename(".env.local").ok();
            dotenvy::dotenv().ok();
        }
    }
    Ok(())
}

/// The jail for the commands of delegated agents, from the command line.
/// `None` means the owner switched it off. A jail that cannot be built is an
/// error: the server must not quietly run commands without one.
fn jail_policy(args: &Args) -> Result<Option<tools::JailPolicy>> {
    if args.no_jail {
        tracing::warn!("--no-jail: agents' commands run as ordinary processes and can read whatever you can");
        return Ok(None);
    }
    tools::jail::probe().map_err(|e| {
        anyhow::anyhow!(
            "{e}. run_agent is not offered without a jail for its commands; \
             --no-jail runs them as ordinary processes instead"
        )
    })?;
    let mut policy = tools::JailPolicy::for_current_user();
    // Without a home directory to empty, the jail would leave the real one in place.
    let Some(home) = policy.home.take() else {
        bail!("$HOME is not set to an absolute path, so the jail cannot tell which directory to hide from commands");
    };
    // The jail is built from real paths: a symlinked home would not be covered.
    policy.home = Some(home.canonicalize().with_context(|| format!("$HOME ({}) cannot be resolved", home.display()))?);
    policy.network = args.allow_network;
    for path in &args.allow_read {
        if !path.is_absolute() {
            bail!("--allow-read {} must be an absolute path", path.display());
        }
        match path.canonicalize() {
            Ok(real) => policy.readable.push(real),
            Err(e) => tracing::warn!("--allow-read {} is ignored: {e}", path.display()),
        }
    }
    Ok(Some(policy))
}

/// The models named on the command line, each once, in the order given.
fn named_models(args: &Args) -> Vec<String> {
    let mut models: Vec<String> = Vec::new();
    let named = args.model.iter().chain(&args.workers).chain(&args.review_model).chain(&args.advisor_model);
    for model in named.map(|model| model.trim()).filter(|model| !model.is_empty()) {
        if !models.iter().any(|known| known == model) {
            models.push(model.to_string());
        }
    }
    models
}

async fn run() -> Result<ExitCode> {
    // Some options default to settings from the dotenv files, and which files
    // to read is itself an option: so the command line is read twice.
    let early = Args::parse();
    load_settings(early.env_file.as_deref(), early.delegate)?;
    let args = Args::parse();

    if args.mcp_server {
        // stdout carries the protocol, so logs go to stderr.
        tracing_subscriber::fmt().with_env_filter(env_filter()).with_writer(std::io::stderr).init();
        let llm = RigLlm { prompt_caching: !args.no_prompt_cache };
        let mut server = mcp::McpServer::new(Arc::new(llm), args.model.clone().unwrap_or_else(default_model));
        if args.delegate {
            // $DEFAULT_MODEL and the built-in fallback do not count: an agent
            // sends a project's code to its provider, and that has to be asked for.
            let models = named_models(&args);
            if models.is_empty() {
                bail!(
                    "--delegate needs to be told which models agents may run on: name them with --model \
                     or --workers (the first is the default)"
                );
            }
            models.iter().try_for_each(|model| check_credentials(model)).context(
                "with --delegate, settings come from the environment and --env-file, \
                 not from .env files in the current directory",
            )?;
            let jail = jail_policy(&args)?;
            // Checked now, so that a mistyped --hide stops the server and not every call.
            Sandbox::new(std::env::temp_dir(), Duration::from_secs(1))?
                .hiding(&args.hide)
                .map_err(|e| anyhow::anyhow!("--hide: {e}"))?;
            tracing::info!(
                "run_agent is offered, on: {}. Commands are {}",
                models.join(", "),
                match &jail {
                    Some(jail) if jail.network => "jailed, with the network",
                    Some(_) => "jailed, without a network",
                    None => "NOT jailed",
                }
            );
            server = server.delegating(mcp::Delegating {
                models,
                max_turns: args.max_turns.unwrap_or(DEFAULT_MAX_TURNS).max(1),
                command_timeout: Duration::from_secs(args.command_timeout.max(1)),
                jail,
                hidden: args.hide.clone(),
            });
        }
        server.run().await?;
        return Ok(ExitCode::SUCCESS);
    }

    if !args.probe.is_empty() {
        return run_probe(&args).await;
    }

    if !args.design.is_file() {
        bail!("design file not found: {}", args.design.display());
    }
    let design_content = std::fs::read_to_string(&args.design)
        .with_context(|| format!("reading {}", args.design.display()))?;
    if design_content.trim().is_empty() {
        bail!("design file is empty: {}", args.design.display());
    }

    if args.check {
        return run_check(&args, design_content).await;
    }

    // Fail before touching the disk if the provider cannot be used at all.
    // (A resumed run takes its model from the saved state and is checked below.)
    if !args.resume {
        match worker_pool(&args) {
            Some(pool) => pool.iter().try_for_each(|model| check_credentials(model))?,
            None => check_credentials(&args.model.clone().unwrap_or_else(default_model))?,
        }
    }

    let sandbox = Sandbox::new(&args.output, Duration::from_secs(args.command_timeout.max(1)))
        .with_context(|| format!("preparing output directory {}", args.output.display()))?;
    let output_dir = sandbox.root().to_path_buf();
    let state_path = ProjectContext::state_path(&output_dir);

    let context = if args.resume {
        let mut ctx = ProjectContext::load(&output_dir)
            .context("--resume needs the state of an earlier run in the output directory")?;
        ctx.prepare_resume();
        ctx.working_directory = output_dir.clone();
        if let Some(model) = &args.model {
            ctx.model = model.clone();
        }
        if let Some(workers) = worker_pool(&args) {
            ctx.worker_models = workers;
        }
        if args.advisor_model.is_some() {
            ctx.advisor_model = args.advisor_model.clone();
        }
        if let Some(max_steps) = args.max_steps {
            ctx.max_steps = max_steps;
        }
        if let Some(max_turns) = args.max_turns {
            ctx.max_turns = max_turns;
        }
        ctx.brave_mode |= args.brave;
        if let Some((first, others)) = args.review_model.split_first() {
            ctx.review_model = Some(first.clone());
            ctx.other_review_models = others.to_vec();
        }
        if let Some(fix_rounds) = args.fix_rounds {
            ctx.fix_rounds = fix_rounds;
        }
        ctx.design_check_enabled &= !args.skip_design_check;
        ctx
    } else {
        let project_name =
            args.design.file_stem().and_then(|s| s.to_str()).unwrap_or("project").to_string();
        let mut ctx = ProjectContext::new(
            project_name,
            output_dir.clone(),
            args.max_steps.unwrap_or(DEFAULT_MAX_STEPS),
            args.max_turns.unwrap_or(DEFAULT_MAX_TURNS).max(1),
            args.brave,
            // Without --model, planning falls to the first worker model.
            args.model
                .clone()
                .or_else(|| worker_pool(&args).and_then(|pool| pool.first().cloned()))
                .unwrap_or_else(default_model),
        );
        ctx.worker_models = worker_pool(&args).unwrap_or_default();
        ctx.advisor_model = args.advisor_model.clone();
        ctx.review_model = args.review_model.first().cloned();
        ctx.other_review_models = args.review_model.iter().skip(1).cloned().collect();
        ctx.fix_rounds = args.fix_rounds.unwrap_or(DEFAULT_FIX_ROUNDS);
        ctx.design_check_enabled = !args.skip_design_check;
        ctx
    };

    check_credentials(context.planner_model())?;
    for model in context.reviewers() {
        check_credentials(model)?;
    }
    for model in context.pool() {
        check_credentials(model)?;
    }

    let use_tui = !args.headless;
    if use_tui {
        let level = log::LevelFilter::from_str(&log_level()).unwrap_or(log::LevelFilter::Info);
        let _ = tui_logger::init_logger(level);
        let _ = tracing_subscriber::registry()
            .with(tui_logger::TuiTracingSubscriberLayer)
            .with(env_filter())
            .try_init();
        tui_logger::set_default_level(level);
    } else {
        tracing_subscriber::fmt().with_env_filter(env_filter()).init();
    }

    if args.resume {
        tracing::info!(
            "Clankistrator: resuming {} ({} of {} tasks completed)",
            context.project_name,
            context.count_completed(),
            context.tasks.len()
        );
    } else if state_path.exists() {
        tracing::warn!(
            "Clankistrator: {} holds an earlier run, which this run replaces. Use --resume to continue it instead.",
            output_dir.display()
        );
    }
    tracing::info!("Clankistrator: model {}, output directory {}", context.model, output_dir.display());
    if context.pool().len() > 1 || context.advisor_model.is_some() {
        tracing::info!(
            "Clankistrator: worker models, cheapest first: {}. Planning by {}, reviews by {}",
            context.pool().join(", "),
            context.planner_model(),
            context.reviewers().join(", ")
        );
    }

    let enclosing = sandbox.enclosing_projects();
    if !enclosing.is_empty() {
        let found: Vec<String> = enclosing.iter().map(|p| p.display().to_string()).collect();
        tracing::warn!(
            "Clankistrator: the output directory is inside another project ({}). Build tools and version control \
             search parent directories, so an agent's command can end up acting on that project. cargo and git are \
             guarded against this; other tools are not. An output directory outside any project is safer.",
            found.join(", ")
        );
    }

    let llm = RigLlm { prompt_caching: !args.no_prompt_cache };
    let mut context = context;
    let pool: Vec<String> = context.pool().iter().map(|m| m.to_string()).collect();
    // Models the run cannot do without: the planner and the reviewers.
    let mut required = vec![context.planner_model().to_string()];
    required.extend(context.reviewers().iter().map(|model| model.to_string()));
    let is_free = |model: &str| llm::uses_free_tier(&[model]);

    if pool.iter().chain(&required).any(|model| is_free(model)) {
        let left = match llm::free_allowance().await {
            Some(llm::FreeAllowance { remaining: 0, .. }) => {
                // The free models are out for today. Go on without them if the run can.
                let has_other_workers = pool.iter().any(|model| !is_free(model));
                // The reviewer is always needed; the planner only while there is no plan yet.
                let needed = context.reviewers().iter().any(|model| is_free(model))
                    || (context.tasks.is_empty() && is_free(context.planner_model()));
                if !has_other_workers || needed {
                    bail!("{}", llm::FREE_ALLOWANCE_USED_UP);
                }
                for model in pool.iter().filter(|model| is_free(model)) {
                    context.retire_model(model, llm::FREE_ALLOWANCE_USED_UP);
                }
                "OpenRouter reports none of today's free-model requests left, so the free models sit this run out"
                    .to_string()
            }
            Some(allowance) => {
                context.free_requests_at_start = Some(allowance.remaining);
                format!(
                    "OpenRouter reports {} of today's {} free-model requests left",
                    allowance.remaining, allowance.limit
                )
            }
            None => "OpenRouter did not say how many free-model requests are left today".to_string(),
        };
        tracing::warn!(
            "Clankistrator: using an OpenRouter free model. {left}. Every turn of an agent is one request; \
             requests are spaced out to stay under 20 a minute. If the day's allowance runs out, work moves to the \
             next worker model if there is one; otherwise the remaining tasks fail and the run can be continued \
             later with --resume."
        );
    }

    // One small request per model, so a mistyped model name stops the run here
    // instead of after a worker has spent tokens on something nobody can review.
    let mut asked: Vec<&str> = Vec::new();
    for model in required.iter().chain(&pool) {
        if asked.contains(&model.as_str()) || context.is_retired(model) {
            continue;
        }
        asked.push(model);
        match probe::preflight(&llm, &[model.as_str()]).await {
            Ok(used) => {
                context.input_tokens += used.input_tokens;
                context.cached_input_tokens += used.cached_input_tokens;
                context.output_tokens += used.output_tokens;
            }
            // A worker model that does not answer is left out, as long as another one does.
            Err(e) if !required.contains(model) => {
                tracing::warn!("Clankistrator: leaving {model} out of this run: {e:#}");
                context.retire_model(model, &format!("{e:#}"));
            }
            Err(e) => return Err(e),
        }
    }
    if context.usable_pool().is_empty() {
        bail!("none of the worker models can be used");
    }

    let context: SharedContext = Arc::new(Mutex::new(context));
    let orchestrator = Orchestrator::new(Arc::new(llm), design_content, sandbox);

    if use_tui {
        run_tui(context, orchestrator).await?;
        Ok(ExitCode::SUCCESS)
    } else {
        run_cli(context, orchestrator).await
    }
}

async fn run_cli(context: SharedContext, orchestrator: Orchestrator) -> Result<ExitCode> {
    let outcome = tokio::select! {
        outcome = orchestrator.run(&context, Hitl::Stdin) => outcome,
        _ = tokio::signal::ctrl_c() => {
            tools::command_executor::kill_running_commands();
            let ctx = context.lock().await;
            let _ = ctx.save();
            eprintln!("\nInterrupted. Progress is saved; rerun with --resume to continue.");
            return Ok(ExitCode::from(130));
        }
    };

    let ctx = context.lock().await;
    println!("\n--- ORCHESTRATION SUMMARY ---");
    println!("Project: {}", ctx.project_name);
    println!("Output:  {}", ctx.working_directory.display());
    for task in &ctx.tasks {
        let status = match &task.status {
            TaskStatus::Completed { .. } => "✅ Completed".to_string(),
            TaskStatus::Failed { error } => {
                format!("❌ Failed: {}", error.lines().next().unwrap_or_default())
            }
            other => format!("⏳ {}", other.label()),
        };
        println!("{}: {} - {}", task.id, task.title, status);
    }
    println!("Model calls: {}, tokens: {}, steps: {}", ctx.llm_calls, ctx.token_summary(), ctx.steps_taken);
    if ctx.model_usage.len() > 1 {
        for (model, used) in &ctx.model_usage {
            println!(
                "  {model}: {} calls, tokens: {}; task attempts passed {}, failed {}",
                used.calls,
                format_tokens(used.input_tokens, used.cached_input_tokens, used.output_tokens),
                used.passed,
                used.rejected
            );
        }
    }
    for (model, reason) in &ctx.retired_models {
        println!("  {model} was left out during this run: {reason}");
    }
    print_free_tier_use().await;
    if let Some(findings) = &ctx.design_check {
        print_design_report(findings, false);
        if ctx.tasks.len() > ctx.design_checked_task_count {
            println!(
                "\nNote: this check ran before the last fix task. That task changed the project and did not \
                 complete, so the project may now be in a different state (it may not even build)."
            );
        }
    }

    match outcome {
        Err(e) => Err(e),
        Ok(RunEnd::StepLimit) => {
            println!("Stopped at the step limit ({}). Rerun with --resume to continue.", ctx.max_steps);
            Ok(ExitCode::from(2))
        }
        Ok(RunEnd::Finished) if ctx.count_failed() > 0 => {
            println!(
                "{} of {} tasks failed. Rerun with --resume to retry them.",
                ctx.count_failed(),
                ctx.tasks.len()
            );
            Ok(ExitCode::FAILURE)
        }
        Ok(RunEnd::Finished) if !ctx.unmet_requirements().is_empty() => {
            println!(
                "Every task completed, but {} design requirement(s) are still unmet after {} fix round(s). \
                 The project does not implement the design yet. Rerun with --resume to try further fix rounds.",
                ctx.unmet_requirements().len(),
                ctx.fix_rounds_done
            );
            Ok(ExitCode::FAILURE)
        }
        Ok(RunEnd::Finished) if ctx.design_check_error.is_some() => {
            println!(
                "Every task completed, but the project could not be checked against the design: {}",
                ctx.design_check_error.as_deref().unwrap_or_default()
            );
            Ok(ExitCode::FAILURE)
        }
        Ok(RunEnd::Finished) if !ctx.design_check_enabled => {
            println!("All tasks completed. The project was not checked against the design (--skip-design-check).");
            Ok(ExitCode::SUCCESS)
        }
        Ok(RunEnd::Finished) => {
            println!("All tasks completed, and the design check found nothing unmet.");
            Ok(ExitCode::SUCCESS)
        }
    }
}

/// Print the result of a design check. `list_met` also lists what was found met.
/// The models named with --workers, or `None` if the option was not given.
fn worker_pool(args: &Args) -> Option<Vec<String>> {
    let mut pool: Vec<String> = Vec::new();
    for model in args.workers.iter().map(|m| m.trim()).filter(|m| !m.is_empty()) {
        if !pool.iter().any(|known| known == model) {
            pool.push(model.to_string());
        }
    }
    (!pool.is_empty()).then_some(pool)
}

/// The free tier counts requests, not tokens, so say how many this run used.
async fn print_free_tier_use() {
    let requests = llm::free_model_requests();
    if requests == 0 {
        return;
    }
    match llm::free_allowance().await {
        Some(allowance) => println!(
            "OpenRouter free-model requests: {requests} sent; OpenRouter reports {} of today's {} left \
             (its counter can lag by a few minutes)",
            allowance.remaining, allowance.limit
        ),
        None => println!("OpenRouter free-model requests: {requests} sent"),
    }
}

fn print_design_report(findings: &[Finding], list_met: bool) {
    let of = |status: FindingStatus| findings.iter().filter(move |f| f.status == status).collect::<Vec<_>>();
    let (met, unmet, unverified) =
        (of(FindingStatus::Met), of(FindingStatus::Unmet), of(FindingStatus::CannotVerify));
    println!(
        "\n--- DESIGN CHECK: {} met, {} unmet, {} could not be verified ---",
        met.len(),
        unmet.len(),
        unverified.len()
    );
    let section = |title: &str, items: &[&Finding]| {
        if !items.is_empty() {
            println!("\n{title}:");
            for finding in items {
                println!("{}", finding.line());
            }
        }
    };
    section("UNMET", &unmet);
    section("COULD NOT BE VERIFIED (check these yourself)", &unverified);
    if list_met {
        section("MET", &met);
    }
}

/// `--probe`: find out whether each model works here, and say exactly what failed.
async fn run_probe(args: &Args) -> Result<ExitCode> {
    tracing_subscriber::fmt().with_env_filter(env_filter()).with_writer(std::io::stderr).init();
    let llm = RigLlm { prompt_caching: !args.no_prompt_cache };
    let mut all_passed = true;
    for model in &args.probe {
        let report = probe::probe_model(&llm, model).await;
        all_passed &= report.passed();
        println!("\n{} (provider: {})", report.model, report.provider);
        let line = |label: &str, result: &Result<llm::TokenUsage, String>| match result {
            Ok(usage) => println!(
                "  {label:<12} ok      tokens: {}",
                format_tokens(usage.input_tokens, usage.cached_input_tokens, usage.output_tokens)
            ),
            Err(why) => println!("  {label:<12} FAILED  {why}"),
        };
        line("completion", &report.completion);
        match &report.tool_call {
            Some(result) => line("tool call", result),
            None => println!("  {:<12} skipped", "tool call"),
        }
        if let Ok(usage) = &report.completion
            && usage.input_tokens == 0
        {
            println!("  note: the provider reported no token usage, so run summaries will show 0 for this model");
        }
    }
    print_free_tier_use().await;
    println!(
        "\n{}",
        if all_passed {
            "Every model passed. Models that pass can be used with --model and --review-model."
        } else {
            "Not every model passed. A model that fails the tool call cannot be used: workers and reviewers depend on tools."
        }
    );
    Ok(if all_passed { ExitCode::SUCCESS } else { ExitCode::FAILURE })
}

/// `--check`: judge the project already in the output directory against the
/// design, print the findings, and change nothing.
async fn run_check(args: &Args, design_content: String) -> Result<ExitCode> {
    if !args.output.is_dir() {
        bail!("--check needs an existing project directory, but {} is not one", args.output.display());
    }
    let model = args.model.clone().unwrap_or_else(default_model);
    let review_model = args.review_model.first().cloned().unwrap_or_else(|| model.clone());
    check_credentials(&review_model)?;
    tracing_subscriber::fmt().with_env_filter(env_filter()).init();

    let sandbox = Sandbox::new(&args.output, Duration::from_secs(args.command_timeout.max(1)))?;
    tracing::info!("Clankistrator: checking {} with model {review_model}", sandbox.root().display());
    let settings = AgentSettings {
        model,
        review_model,
        design_doc: design_content,
        sandbox,
        max_turns: args.max_turns.unwrap_or(DEFAULT_MAX_TURNS),
        plan: String::new(),
        retry_delay: PROVIDER_RETRY_DELAY,
    };
    let llm = RigLlm { prompt_caching: !args.no_prompt_cache };
    probe::preflight(&llm, &[settings.review_model.as_str()]).await?;
    let check = tokio::select! {
        check = check_design(&llm, &[], &settings) => check?,
        _ = tokio::signal::ctrl_c() => return Ok(ExitCode::from(130)),
    };

    print_design_report(&check.findings, true);
    println!(
        "\nModel calls: {}, tokens: {}",
        check.calls,
        format_tokens(check.usage.input_tokens, check.usage.cached_input_tokens, check.usage.output_tokens)
    );
    print_free_tier_use().await;
    let unmet = check.findings.iter().filter(|f| f.status == FindingStatus::Unmet).count();
    if unmet == 0 {
        println!("Nothing was found unmet.");
        Ok(ExitCode::SUCCESS)
    } else {
        println!("{unmet} requirement(s) are unmet.");
        Ok(ExitCode::FAILURE)
    }
}

async fn run_tui(context: SharedContext, orchestrator: Orchestrator) -> Result<()> {
    struct Cleanup;
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = crossterm::terminal::disable_raw_mode();
            let _ = crossterm::execute!(
                std::io::stdout(),
                crossterm::terminal::LeaveAlternateScreen,
                crossterm::cursor::Show
            );
        }
    }

    crossterm::terminal::enable_raw_mode()?;
    let _cleanup = Cleanup;
    let mut stdout = std::io::stdout();
    crossterm::execute!(stdout, crossterm::terminal::EnterAlternateScreen)?;
    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let mut terminal = ratatui::Terminal::new(backend)?;

    let mut app = App::new(Arc::clone(&context));

    // The orchestrator runs alongside the UI and talks to it through the shared
    // context. Aborting it on exit kills any command an agent has running.
    let run_context = Arc::clone(&context);
    let orchestration = tokio::spawn(async move {
        match orchestrator.run(&run_context, Hitl::External).await {
            Ok(RunEnd::Finished) => {
                let ctx = run_context.lock().await;
                tracing::info!(
                    "Orchestrator: finished, {} completed and {} failed. Press q to quit.",
                    ctx.count_completed(),
                    ctx.count_failed()
                );
                if !ctx.unmet_requirements().is_empty() {
                    tracing::error!(
                        "Orchestrator: {} design requirement(s) are still unmet; see the Stats tab.",
                        ctx.unmet_requirements().len()
                    );
                } else if let Some(error) = &ctx.design_check_error {
                    tracing::error!("Orchestrator: the project could not be checked against the design: {error}");
                }
            }
            Ok(RunEnd::StepLimit) => tracing::warn!("Orchestrator: stopped at the step limit."),
            Err(e) => tracing::error!("Orchestrator stopped: {e:#}"),
        }
    });

    let mut events = crossterm::event::EventStream::new();
    let mut redraw = tokio::time::interval(Duration::from_millis(250));
    while app.running {
        {
            let ctx = context.lock().await;
            terminal.draw(|f| crate::tui::views::draw(f, &mut app, &ctx))?;
        }

        tokio::select! {
            event = events.next() => match event {
                Some(Ok(event)) => app.handle_event(event).await?,
                Some(Err(e)) => bail!("terminal input failed: {e}"),
                None => break,
            },
            _ = redraw.tick() => {}
        }
    }

    orchestration.abort();
    let _ = orchestration.await;
    tools::command_executor::kill_running_commands();
    let unfinished = {
        let mut ctx = context.lock().await;
        let unfinished = !ctx.all_terminal();
        if unfinished {
            ctx.phase = context::RunPhase::Starting;
        }
        let _ = ctx.save();
        unfinished
    };
    drop(terminal);
    drop(_cleanup);
    if unfinished {
        println!("Stopped before all tasks finished. Progress is saved; rerun with --resume to continue.");
    }
    Ok(())
}
