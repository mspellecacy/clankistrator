//! The single place where Clankistrator talks to a model provider.
//!
//! Everything else goes through the [`Llm`] trait, which keeps the orchestration
//! logic testable with scripted replies.

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use rig::client::{CompletionClient, Nothing};
use rig::completion::{Prompt, PromptError};
use rig::providers::{anthropic, gemini, ollama, openai};

use crate::tools::{Sandbox, ToolAccess};

/// Upper bound on one agent run (all of its turns), so a stalled connection
/// cannot hang the orchestrator forever.
const AGENT_RUN_TIMEOUT: Duration = Duration::from_secs(30 * 60);

/// rig only knows output limits for older Claude models and refuses to send a
/// request without one, so newer models get this value.
const ANTHROPIC_FALLBACK_MAX_TOKENS: u64 = 16_000;

#[derive(Debug, Clone)]
pub struct AgentRequest {
    pub model: String,
    pub preamble: String,
    pub prompt: String,
    pub tools: ToolAccess,
    pub sandbox: Sandbox,
    pub max_turns: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Debug, Clone)]
pub struct LlmReply {
    pub text: String,
    pub usage: TokenUsage,
}

#[derive(Debug)]
pub enum LlmError {
    /// The agent used all of its turns without producing a final answer.
    MaxTurns(usize),
    Other(anyhow::Error),
}

impl std::fmt::Display for LlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LlmError::MaxTurns(n) => write!(f, "the agent used all {n} turns without finishing"),
            LlmError::Other(e) => write!(f, "{e:#}"),
        }
    }
}

pub type LlmFuture<'a> = Pin<Box<dyn Future<Output = Result<LlmReply, LlmError>> + Send + 'a>>;

/// Runs one agent (preamble + prompt + tools) to completion.
pub trait Llm: Send + Sync {
    fn run<'a>(&'a self, request: AgentRequest) -> LlmFuture<'a>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    OpenAi,
    Anthropic,
    Gemini,
    Ollama,
}

impl Provider {
    /// The environment variable holding this provider's API key, if it needs one.
    pub fn key_var(self) -> Option<&'static str> {
        match self {
            Provider::OpenAi => Some("OPENAI_API_KEY"),
            Provider::Anthropic => Some("ANTHROPIC_API_KEY"),
            Provider::Gemini => Some("GEMINI_API_KEY"),
            Provider::Ollama => None,
        }
    }
}

/// Split a model spec into provider and model name.
///
/// An explicit `provider:model` prefix always wins (`ollama:llama3.2`,
/// `anthropic:claude-...`). Without one, names starting with `claude` go to
/// Anthropic, names starting with `gemini` go to Gemini, and everything else
/// goes to OpenAI.
pub fn resolve_model(spec: &str) -> (Provider, &str) {
    let spec = spec.trim();
    if let Some((prefix, rest)) = spec.split_once(':') {
        let provider = match prefix.to_ascii_lowercase().as_str() {
            "openai" => Some(Provider::OpenAi),
            "anthropic" => Some(Provider::Anthropic),
            "gemini" | "google" => Some(Provider::Gemini),
            "ollama" => Some(Provider::Ollama),
            _ => None,
        };
        if let Some(provider) = provider {
            return (provider, rest.trim());
        }
    }
    let lower = spec.to_ascii_lowercase();
    if lower.starts_with("claude") {
        (Provider::Anthropic, spec)
    } else if lower.starts_with("gemini") {
        (Provider::Gemini, spec)
    } else {
        (Provider::OpenAi, spec)
    }
}

fn looks_like_placeholder(key: &str) -> bool {
    let key = key.trim();
    key.is_empty() || key.starts_with("your_") || key == "stub"
}

fn api_key(provider: Provider) -> Result<String> {
    let Some(var) = provider.key_var() else {
        return Ok(String::new());
    };
    match std::env::var(var) {
        Ok(key) if !looks_like_placeholder(&key) => Ok(key),
        _ => bail!(
            "{var} is not set (or still holds the placeholder from .env.example). \
             Put a real key in .env or .env.local, or pick a model from another provider with --model."
        ),
    }
}

/// Fail early, before any work starts, if the model's provider has no credentials.
pub fn check_credentials(model_spec: &str) -> Result<()> {
    let (provider, model) = resolve_model(model_spec);
    if model.is_empty() {
        bail!("model name is empty");
    }
    api_key(provider).map(|_| ())
}

fn ollama_base_url() -> Option<String> {
    std::env::var("OLLAMA_BASE_URL")
        .or_else(|_| std::env::var("OLLAMA_API_BASE_URL"))
        .ok()
        .filter(|url| !url.trim().is_empty())
}

/// The real implementation, backed by rig.
#[derive(Debug, Default, Clone, Copy)]
pub struct RigLlm;

/// Builds and runs an agent on a concrete rig client. A macro rather than a
/// generic function because each provider has its own client and model types.
macro_rules! run_agent {
    ($client:expr, $model:expr, $request:expr, $max_tokens:expr) => {{
        let request: &AgentRequest = $request;
        let mut builder = $client.agent($model).preamble(&request.preamble);
        if let Some(max_tokens) = $max_tokens {
            builder = builder.max_tokens(max_tokens);
        }
        let tools = request.tools.build(&request.sandbox);
        let agent = if tools.is_empty() {
            builder.build()
        } else {
            builder.tools(tools).build()
        };
        agent
            .prompt(request.prompt.as_str())
            .max_turns(request.max_turns)
            .extended_details()
            .await
    }};
}

impl RigLlm {
    async fn dispatch(&self, request: &AgentRequest) -> Result<LlmReply, LlmError> {
        let (provider, model) = resolve_model(&request.model);
        let key = api_key(provider).map_err(LlmError::Other)?;
        let client_error = |e| LlmError::Other(anyhow!("could not create the {provider:?} client: {e}"));
        let no_limit: Option<u64> = None;

        let result = match provider {
            Provider::OpenAi => {
                let client = openai::Client::new(&key).map_err(client_error)?;
                run_agent!(client, model, request, no_limit)
            }
            Provider::Anthropic => {
                let client = anthropic::Client::new(&key).map_err(client_error)?;
                // rig supplies the limit itself for the model families it knows.
                let max_tokens = if model.starts_with("claude-3")
                    || model.starts_with("claude-sonnet-4")
                    || model.starts_with("claude-opus-4")
                {
                    None
                } else {
                    Some(ANTHROPIC_FALLBACK_MAX_TOKENS)
                };
                run_agent!(client, model, request, max_tokens)
            }
            Provider::Gemini => {
                let client = gemini::Client::new(&key).map_err(client_error)?;
                run_agent!(client, model, request, no_limit)
            }
            Provider::Ollama => {
                let mut builder = ollama::Client::builder().api_key(Nothing);
                if let Some(url) = ollama_base_url() {
                    builder = builder.base_url(url);
                }
                let client = builder.build().map_err(client_error)?;
                run_agent!(client, model, request, no_limit)
            }
        };

        match result {
            Ok(response) => Ok(LlmReply {
                text: response.output,
                usage: TokenUsage {
                    input_tokens: response.usage.input_tokens,
                    output_tokens: response.usage.output_tokens,
                },
            }),
            Err(PromptError::MaxTurnsError { max_turns, .. }) => Err(LlmError::MaxTurns(max_turns)),
            Err(e) => Err(LlmError::Other(anyhow!("{e}"))),
        }
    }
}

impl Llm for RigLlm {
    fn run<'a>(&'a self, request: AgentRequest) -> LlmFuture<'a> {
        Box::pin(async move {
            match tokio::time::timeout(AGENT_RUN_TIMEOUT, self.dispatch(&request)).await {
                Ok(result) => result,
                Err(_) => Err(LlmError::Other(anyhow!(
                    "the model did not finish within {} minutes",
                    AGENT_RUN_TIMEOUT.as_secs() / 60
                ))),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_is_inferred_from_the_model_name() {
        assert_eq!(resolve_model("gpt-4o"), (Provider::OpenAi, "gpt-4o"));
        assert_eq!(resolve_model("o3-mini"), (Provider::OpenAi, "o3-mini"));
        assert_eq!(resolve_model("claude-sonnet-4-5"), (Provider::Anthropic, "claude-sonnet-4-5"));
        assert_eq!(resolve_model("gemini-2.5-pro"), (Provider::Gemini, "gemini-2.5-pro"));
    }

    #[test]
    fn explicit_prefix_wins() {
        assert_eq!(resolve_model("ollama:llama3.2"), (Provider::Ollama, "llama3.2"));
        // Ollama tags contain colons themselves.
        assert_eq!(resolve_model("ollama:qwen2.5-coder:14b"), (Provider::Ollama, "qwen2.5-coder:14b"));
        assert_eq!(resolve_model("openai:my-finetune"), (Provider::OpenAi, "my-finetune"));
        assert_eq!(resolve_model("anthropic:some-model"), (Provider::Anthropic, "some-model"));
        // An unknown prefix is part of the model name.
        assert_eq!(resolve_model("ft:gpt-4o:acme"), (Provider::OpenAi, "ft:gpt-4o:acme"));
    }

    #[test]
    fn placeholders_do_not_count_as_keys() {
        assert!(looks_like_placeholder(""));
        assert!(looks_like_placeholder("your_openai_key_here"));
        assert!(looks_like_placeholder("stub"));
        assert!(!looks_like_placeholder("sk-abc123"));
    }

    #[test]
    fn ollama_needs_no_key() {
        assert!(check_credentials("ollama:llama3.2").is_ok());
        assert!(check_credentials("ollama:").is_err());
    }
}
