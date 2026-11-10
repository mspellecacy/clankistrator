pub mod agents;
pub mod context;
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

const FALLBACK_MODEL: &str = "gpt-4o";
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
    /// Defaults to $DEFAULT_MODEL, then gpt-4o
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

    /// Model for reviews and the design check, if it should differ from --model
    #[arg(long)]
    review_model: Option<String>,

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

async fn run() -> Result<ExitCode> {
    dotenvy::from_filename(".env.local").ok();
    dotenvy::dotenv().ok();

    let args = Args::parse();

    if args.mcp_server {
        // stdout carries the protocol, so logs go to stderr.
        tracing_subscriber::fmt().with_env_filter(env_filter()).with_writer(std::io::stderr).init();
        let llm = RigLlm { prompt_caching: !args.no_prompt_cache };
        let server = mcp::McpServer::new(Arc::new(llm), args.model.unwrap_or_else(default_model));
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
        check_credentials(&args.model.clone().unwrap_or_else(default_model))?;
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
        if let Some(max_steps) = args.max_steps {
            ctx.max_steps = max_steps;
        }
        if let Some(max_turns) = args.max_turns {
            ctx.max_turns = max_turns;
        }
        ctx.brave_mode |= args.brave;
        if args.review_model.is_some() {
            ctx.review_model = args.review_model.clone();
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
            args.model.clone().unwrap_or_else(default_model),
        );
        ctx.review_model = args.review_model.clone();
        ctx.fix_rounds = args.fix_rounds.unwrap_or(DEFAULT_FIX_ROUNDS);
        ctx.design_check_enabled = !args.skip_design_check;
        ctx
    };

    check_credentials(&context.model)?;
    check_credentials(context.reviewer_model())?;

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
    // One small request per model, so a mistyped model name stops the run here
    // instead of after a worker has spent tokens on something nobody can review.
    let mut context = context;
    if llm::uses_free_tier(&[context.model.as_str(), context.reviewer_model()]) {
        tracing::warn!(
            "Clankistrator: using an OpenRouter free model. Requests to it are spaced out to stay under 20 a \
             minute, and the free tier allows 50 a day (1,000 once the account has bought 10 credits). When the \
             day's allowance runs out the run stops with failed tasks; continue it later with --resume."
        );
    }
    let used = probe::preflight(&llm, &[context.model.as_str(), context.reviewer_model()]).await?;
    context.input_tokens += used.input_tokens;
    context.cached_input_tokens += used.cached_input_tokens;
    context.output_tokens += used.output_tokens;

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
    print_free_tier_use();
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
/// The free tier counts requests, not tokens, so say how many this run used.
fn print_free_tier_use() {
    let requests = llm::free_model_requests();
    if requests > 0 {
        println!(
            "OpenRouter free-model requests: {requests} (the free tier allows 50 a day, \
             or 1,000 once the account has bought 10 credits)"
        );
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
    print_free_tier_use();
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
    let review_model = args.review_model.clone().unwrap_or_else(|| model.clone());
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
    print_free_tier_use();
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
