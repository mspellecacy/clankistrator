pub mod delegate;
pub mod llm;
pub mod mcp;
pub mod probe;
pub mod tools;

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use clap::Parser;

use crate::llm::{RigLlm, check_credentials};
use crate::tools::Sandbox;

const DEFAULT_MAX_TURNS: usize = 40;

/// An MCP server (JSON-RPC on stdio) with one tool, run_agent: it runs an agent on one of the
/// models named here, in a project the client names, with the agent's commands in a jail.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The only models an agent may run on, separated by commas, the default first. Each may
    /// carry a provider prefix (openai:, anthropic:, gemini:, ollama:). What an agent reads is
    /// sent to its model's provider, so the list is fixed here and a call cannot add to it
    #[arg(long, visible_alias = "workers", value_delimiter = ',', value_name = "MODEL,MODEL")]
    models: Vec<String>,

    /// One more model, placed before --models
    #[arg(short, long)]
    model: Option<String>,

    /// Read settings (API keys, LOG_LEVEL) from this file. Without it they come from the
    /// environment only: .env files in the current directory belong to the client's project
    /// and are never read
    #[arg(long, value_name = "FILE")]
    env_file: Option<PathBuf>,

    /// Maximum model turns (tool-call rounds) an agent gets, unless a call asks for another
    /// number
    #[arg(long, default_value_t = DEFAULT_MAX_TURNS)]
    max_turns: usize,

    /// Seconds a command started by an agent may run before it is killed
    #[arg(long, default_value_t = 300)]
    command_timeout: u64,

    /// Minutes one agent run (all of its turns) may take before it is given up on
    #[arg(long, value_name = "MINUTES", default_value_t = llm::DEFAULT_AGENT_RUN_TIMEOUT.as_secs() / 60)]
    run_timeout: u64,

    /// Do not ask Anthropic to cache prompts between an agent's turns
    #[arg(long)]
    no_prompt_cache: bool,

    /// Run agents' commands as ordinary processes instead of in a bubblewrap jail. Without
    /// this, the server does not start where it cannot build a jail
    #[arg(long)]
    no_jail: bool,

    /// Let jailed commands use the network
    #[arg(long, conflicts_with = "no_jail")]
    allow_network: bool,

    /// Paths outside the project that jailed commands may read, separated by commas (a
    /// toolchain under your home directory, say). The rest of your home directory is empty in
    /// the jail
    #[arg(long, conflicts_with = "no_jail", value_delimiter = ',', value_name = "PATH,PATH")]
    allow_read: Vec<PathBuf>,

    /// Paths in the project, relative to its root, that agents may not see, separated by commas
    #[arg(long, value_delimiter = ',', value_name = "PATH,PATH")]
    hide: Vec<PathBuf>,

    /// Append one line per agent run to this file (JSON: time, directory, agent, model, access,
    /// outcome, seconds, tokens). It must be outside the projects agents work in
    #[arg(long, value_name = "FILE")]
    usage_log: Option<PathBuf>,

    /// Test the given models instead of serving: one plain request and one tool call each.
    /// Uses no files and runs no commands
    #[arg(long, num_args = 1.., value_name = "MODEL")]
    probe: Vec<String>,

    /// From when the server had other modes. Accepted together with --delegate and ignored;
    /// refused without it (see `run`)
    #[arg(long, hide = true)]
    mcp_server: bool,

    /// From when the server had other modes: it was the opt-in for run_agent. Accepted and
    /// ignored
    #[arg(long, hide = true)]
    delegate: bool,
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

/// Load settings from `--env-file`. Variables already in the environment win.
///
/// The server is started by its client, usually inside the project the agents
/// will work on, so `.env.local` in the current directory is that project's
/// and not ours: reading it would bill model calls to the project's own
/// provider key and hand every one of its settings to the commands agents run.
fn load_settings(env_file: Option<&std::path::Path>) -> Result<()> {
    if let Some(file) = env_file {
        // dotenvy's error repeats its cause, so it is not kept as one.
        dotenvy::from_path(file).map_err(|e| anyhow::anyhow!("could not read --env-file {}: {e}", file.display()))?;
    }
    Ok(())
}

/// The jail for the commands of agents, from the command line. `None` means
/// the owner switched it off. A jail that cannot be built is an error: the
/// server must not quietly run commands without one.
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
    for model in args.model.iter().chain(&args.models).map(|model| model.trim()).filter(|model| !model.is_empty()) {
        if !models.iter().any(|known| known == model) {
            models.push(model.to_string());
        }
    }
    models
}

async fn run() -> Result<ExitCode> {
    let args = Args::parse();
    load_settings(args.env_file.as_deref())?;
    // stdout carries the protocol (or the probe's report), so logs go to stderr.
    tracing_subscriber::fmt().with_env_filter(env_filter()).with_writer(std::io::stderr).init();
    llm::set_agent_run_timeout(Duration::from_secs(args.run_timeout.max(1) * 60));
    let llm = RigLlm { prompt_caching: !args.no_prompt_cache };

    if !args.probe.is_empty() {
        return run_probe(&llm, &args.probe).await;
    }

    // That command line used to start a server that could not run a command.
    // It must not quietly become one that can.
    if args.mcp_server && !args.delegate {
        bail!(
            "--mcp-server without --delegate used to offer planning tools only, and those are gone. This server \
             has one tool, run_agent, which runs commands and can change files: add --delegate to say you want \
             that, or drop --mcp-server"
        );
    }

    // No built-in default: an agent sends a project's code to its model's
    // provider, and that has to be asked for.
    let models = named_models(&args);
    if models.is_empty() {
        bail!("name the models agents may run on with --models (the first is the default)");
    }
    models
        .iter()
        .try_for_each(|model| check_credentials(model))
        .context("settings come from the environment and --env-file, not from .env files in the current directory")?;
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
    let server = mcp::McpServer::new(
        Arc::new(llm),
        mcp::Delegating {
            models,
            max_turns: args.max_turns.max(1),
            command_timeout: Duration::from_secs(args.command_timeout.max(1)),
            jail,
            hidden: args.hide.clone(),
            usage_log: match &args.usage_log {
                Some(log) if !log.is_absolute() => bail!("--usage-log {} must be an absolute path", log.display()),
                log => log.clone(),
            },
        },
    );
    server.run().await?;
    Ok(ExitCode::SUCCESS)
}

fn format_tokens(input: u64, cached: u64, output: u64) -> String {
    if cached > 0 {
        format!("{input} in ({cached} from cache) / {output} out")
    } else {
        format!("{input} in / {output} out")
    }
}

/// `--probe`: find out whether each model works here, and say exactly what failed.
async fn run_probe(llm: &RigLlm, models: &[String]) -> Result<ExitCode> {
    let mut all_passed = true;
    for model in models {
        let report = probe::probe_model(llm, model).await;
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
            println!("  note: the provider reported no token usage, so the usage log will show 0 for this model");
        }
    }
    println!(
        "\n{}",
        if all_passed {
            "Every model passed. Models that pass can be named with --models."
        } else {
            "Not every model passed. A model that fails the tool call cannot be used: agents depend on tools."
        }
    );
    Ok(if all_passed { ExitCode::SUCCESS } else { ExitCode::FAILURE })
}
