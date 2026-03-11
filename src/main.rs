mod context;
mod orchestrator;
mod agents;
mod tools;
mod parser;

use clap::Parser;
use anyhow::Result;
use dotenvy::dotenv;
use std::env;
use parser::MarkdownParser;
use rig::completion::Prompt;
use crate::context::ProjectContext;
use crate::orchestrator::Orchestrator;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the markdown design document
    #[arg(short, long)]
    design: String,

    /// Working directory for the project implementation
    #[arg(short, long, default_value = ".")]
    output: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load environment variables from .env.local if it exists
    dotenvy::from_filename(".env.local").ok();
    // Load .env file
    dotenv().ok();

    // Initialize logging
    tracing_subscriber::fmt::init();
    
    // Parse CLI arguments
    let args = Args::parse();
    
    println!("Clankistrator: Initializing AI Orchestration");
    println!("Design Doc: {}", args.design);
    println!("Output Path: {}", args.output);

    // Load design doc
    let design_content = std::fs::read_to_string(&args.design)?;
    
    // Parse markdown sections
    let sections = MarkdownParser::parse_sections(&design_content);
    println!("Found {} sections in design document.", sections.len());

    // Summarize document using LLM (Hello World task)
    summarize_document(&design_content).await?;

    // Initialize ProjectContext
    let mut context = ProjectContext::new(
        "Clankistrator Project".into(),
        PathBuf::from(&args.output)
    );

    // Initialize Orchestrator
    let orchestrator = Orchestrator::new("gpt-4o".into());

    println!("Orchestrator: Bootstrapping project from design doc...");
    let initial_tasks = orchestrator.bootstrap_project(&design_content).await?;
    context.tasks = initial_tasks;
    println!("Found {} initial tasks.", context.tasks.len());

    // Run first iteration
    println!("Orchestrator: Running first iteration...");
    orchestrator.run_iteration(&mut context).await?;

    println!("Phase 2 verification completed successfully.");

    Ok(())
}

async fn summarize_document(content: &str) -> Result<()> {
    println!("--- Document Summary (via LLM) ---");
    
    let openai_key = env::var("OPENAI_API_KEY").unwrap_or_else(|_| "stub".to_string());
    
    if openai_key == "stub" {
        println!("(No OPENAI_API_KEY found, skipping real LLM call)");
        println!("Content length: {} bytes", content.len());
        return Ok(());
    }

    // Attempting to use rig-core API (0.7.0)
    let client = rig::providers::openai::Client::new(&openai_key);
    let agent = client.agent("gpt-4o")
        .preamble("Summarize the following design document concisely.")
        .build();
    let response = agent.prompt(content).await?;
    println!("{}", response);

    Ok(())
}
