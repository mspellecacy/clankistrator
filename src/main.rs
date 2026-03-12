pub mod context;
pub mod agents;
pub mod orchestrator;
pub mod tools;
pub mod parser;
pub mod tui;

use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use std::time::Duration;
use clap::Parser;
use crate::context::{ProjectContext, TaskStatus};
use crate::orchestrator::Orchestrator;
use crate::tui::App;
use anyhow::Result;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use tokio_stream::StreamExt;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "design.md")]
    design: PathBuf,

    #[arg(short, long, default_value = "output")]
    output: PathBuf,

    #[arg(short, long, default_value = "gpt-4o")]
    model: String,

    #[arg(long, default_value_t = false)]
    headless: bool,

    #[arg(short, long, default_value_t = false)]
    brave: bool,

    #[arg(short = 'S', long, default_value_t = 50, help = "Maximum orchestration steps (0 for unlimited)")]
    max_steps: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::from_filename(".env.local").ok();
    dotenvy::dotenv().ok();

    let args = Args::parse();
    let use_tui = !args.headless;
    
    let env_filter = tracing_subscriber::EnvFilter::from_default_env()
        .add_directive(tracing_subscriber::filter::LevelFilter::INFO.into());

    if use_tui {
        let _ = tui_logger::init_logger(log::LevelFilter::Info);
        let _ = tracing_subscriber::registry()
            .with(tui_logger::TuiTracingSubscriberLayer)
            .with(env_filter)
            .try_init();
        tui_logger::set_default_level(log::LevelFilter::Info);
        tracing::info!("TUI Mode initialized");
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .init();
    }

    if !args.design.exists() {
        anyhow::bail!("Design file not found: {:?}", args.design);
    }

    if !args.output.exists() {
        std::fs::create_dir_all(&args.output)?;
    }

    let design_content = std::fs::read_to_string(&args.design)?;
    let project_name = args.design
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("New Project")
        .to_string();

    let context = Arc::new(Mutex::new(ProjectContext::new(project_name, args.output.clone(), args.max_steps, args.brave, args.model.clone())));
    let orchestrator = Orchestrator::new(args.model, design_content.clone(), use_tui, args.max_steps);

    if use_tui {
        run_tui(context, orchestrator, design_content, args.max_steps).await?;
    } else {
        run_cli(context, orchestrator, design_content, args.max_steps).await?;
    }

    Ok(())
}

async fn run_cli(context: Arc<Mutex<ProjectContext>>, orchestrator: Orchestrator, design_content: String, max_steps: usize) -> Result<()> {
    tracing::info!("Clankistrator: Starting orchestration for project {:?}", context.lock().await.project_name);

    // 1. Bootstrap Phase
    let initial_tasks = orchestrator.bootstrap_project(&design_content).await?;
    {
        let mut ctx = context.lock().await;
        ctx.tasks = initial_tasks;
        ctx.token_usage += 1000;
        ctx.total_cost += 0.02;
    }

    if context.lock().await.tasks.is_empty() {
        tracing::warn!("Clankistrator: No tasks were generated from the design document. Bailing out.");
        return Ok(());
    }

    tracing::info!("Clankistrator: Bootstrapped with {} initial tasks", context.lock().await.tasks.len());

    // 2. Execution Phase
    let mut i = 1;
    loop {
        if max_steps != 0 && i > max_steps {
            tracing::info!("Clankistrator: Reached maximum steps ({})", max_steps);
            break;
        }
        tracing::info!("Clankistrator: Starting iteration {}", i);
        orchestrator.run_iteration(Arc::clone(&context)).await?;
        
        let ctx = context.lock().await;
        let completed_count = ctx.tasks.iter().filter(|t| matches!(t.status, TaskStatus::Completed { .. })).count();
        if completed_count == ctx.tasks.len() && !ctx.tasks.is_empty() {
            tracing::info!("Clankistrator: All tasks completed!");
            break;
        }
        i += 1;
    }

    let ctx = context.lock().await;
    println!("\n--- ORCHESTRATION SUMMARY ---");
    println!("Project: {}", ctx.project_name);
    for task in &ctx.tasks {
        let status = match &task.status {
            TaskStatus::Completed { .. } => "✅ Completed",
            TaskStatus::Failed { error } => &format!("❌ Failed: {}", error),
            TaskStatus::Pending => "⏳ Pending",
            TaskStatus::InProgress(_) => "🔄 In Progress",
            TaskStatus::ReadyForReview { .. } => "👀 Ready for Review",
            TaskStatus::Reviewing(_) => "🔍 Reviewing",
            TaskStatus::RequiresHumanApproval { .. } => "👤 Waiting for Approval",
        };
        println!("Task {}: {} - {}", task.id, task.title, status);
    }

    Ok(())
}

async fn run_tui(context: Arc<Mutex<ProjectContext>>, orchestrator: Orchestrator, design_content: String, max_steps: usize) -> Result<()> {
    // Setup terminal
    struct Cleanup;
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = crossterm::terminal::disable_raw_mode();
            let _ = crossterm::execute!(
                std::io::stdout(),
                crossterm::terminal::LeaveAlternateScreen,
                crossterm::event::DisableMouseCapture,
                crossterm::cursor::Show
            );
        }
    }

    let _cleanup = Cleanup;

    crossterm::terminal::enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    crossterm::execute!(stdout, crossterm::terminal::EnterAlternateScreen, crossterm::event::EnableMouseCapture)?;
    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let mut terminal = ratatui::Terminal::new(backend)?;

    let mut app = App::new(Arc::clone(&context));

    // Spawn orchestrator thread
    let ctx_clone = Arc::clone(&context);
    tokio::spawn(async move {
        // Bootstrap
        if let Ok(tasks) = orchestrator.bootstrap_project(&design_content).await {
            let mut ctx = ctx_clone.lock().await;
            ctx.tasks = tasks;
            ctx.token_usage += 1000;
            ctx.total_cost += 0.02;
        }

        // Execution loop
        let mut i = 1;
        loop {
            if max_steps != 0 && i > max_steps {
                tracing::info!("Orchestrator: Reached maximum steps ({}). Execution loop ending.", max_steps);
                break;
            }

            // Check if all completed
            {
                let ctx = ctx_clone.lock().await;
                if !ctx.tasks.is_empty() && ctx.tasks.iter().all(|t| matches!(t.status, TaskStatus::Completed { .. })) {
                    tracing::info!("Orchestrator: All tasks completed. Execution loop ending.");
                    break;
                }
            }

            // Run iteration
            if let Err(e) = orchestrator.run_iteration(Arc::clone(&ctx_clone)).await {
                tracing::error!("Orchestrator error: {}", e);
            }
            
            i += 1;
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    });

    // TUI Event loop
    let mut event_reader = crossterm::event::EventStream::new();
    let mut heartbeat_timer = tokio::time::interval(Duration::from_secs(5));
    while app.running {
        // Draw the current state
        {
            let mut ctx = context.lock().await;
            terminal.draw(|f| {
                crate::tui::views::draw(f, &mut app, &mut ctx);
            })?;
        }

        tokio::select! {
            Some(Ok(event)) = event_reader.next() => {
                app.handle_event(event).await?;
            }
            _ = heartbeat_timer.tick() => {
                tracing::info!("TUI Loop heartbeat");
            }
        }
    }

    Ok(())
}