pub mod context;
pub mod agents;
pub mod orchestrator;
pub mod tools;
pub mod parser;

use std::path::PathBuf;
use clap::Parser;
use dotenvy::dotenv;
use crate::context::ProjectContext;
use crate::orchestrator::Orchestrator;
use anyhow::Result;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    design: PathBuf,

    #[arg(short, long)]
    output: PathBuf,

    #[arg(short, long, default_value = "gpt-4o")]
    model: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::from_filename(".env.local").ok();
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt::init();

    let args = Args::parse();

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

    let mut context = ProjectContext::new(project_name, args.output.clone());
    let orchestrator = Orchestrator::new(args.model, design_content.clone());

    tracing::info!("Clankistrator: Starting orchestration for project {:?}", context.project_name);

    // 1. Bootstrap Phase
    let initial_tasks = orchestrator.bootstrap_project(&design_content).await?;
    context.tasks = initial_tasks;

    tracing::info!("Clankistrator: Bootstrapped with {} initial tasks", context.tasks.len());

    // 2. Execution Phase
    // For now, let's run a few iterations to demonstrate the process.
    // In a real scenario, this would loop until all tasks are completed.
    for i in 1..=15 {
        tracing::info!("Clankistrator: Starting iteration {}", i);
        orchestrator.run_iteration(&mut context).await?;
        
        let completed_count = context.tasks.iter().filter(|t| matches!(t.status, crate::context::TaskStatus::Completed { .. })).count();
        if completed_count == context.tasks.len() {
            tracing::info!("Clankistrator: All tasks completed!");
            break;
        }
    }

    println!("\n--- ORCHESTRATION SUMMARY ---");
    println!("Project: {}", context.project_name);
    for task in &context.tasks {
        let status = match &task.status {
            crate::context::TaskStatus::Completed { .. } => "✅ Completed",
            crate::context::TaskStatus::Failed { error } => &format!("❌ Failed: {}", error),
            crate::context::TaskStatus::Pending => "⏳ Pending",
            crate::context::TaskStatus::InProgress(_) => "🔄 In Progress",
            crate::context::TaskStatus::ReadyForReview { .. } => "👀 Ready for Review",
            crate::context::TaskStatus::Reviewing(_) => "🔍 Reviewing",
            crate::context::TaskStatus::RequiresHumanApproval { .. } => "👤 Waiting for Approval",
        };
        println!("Task {}: {} - {}", task.id, task.title, status);
    }

    Ok(())
}