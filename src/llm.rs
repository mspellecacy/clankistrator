//! The single place where Clankistrator talks to a model provider.
//!
//! Everything else goes through the [`Llm`] trait, which keeps the orchestration
//! logic testable with scripted replies.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use rig::agent::{AgentBuilder, HookAction, PromptHook};
use rig::client::{CompletionClient, Nothing};
use rig::completion::{Prompt, PromptError};
use rig::providers::{anthropic, gemini, ollama, openai, openrouter};

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
    /// All input tokens the provider reported, including those read from its prompt cache.
    pub input_tokens: u64,
    /// The part of `input_tokens` that was read from the provider's prompt
    /// cache, which providers bill at a fraction of the normal price.
    pub cached_input_tokens: u64,
    pub output_tokens: u64,
}

impl std::ops::AddAssign for TokenUsage {
    fn add_assign(&mut self, other: Self) {
        self.input_tokens += other.input_tokens;
        self.cached_input_tokens += other.cached_input_tokens;
        self.output_tokens += other.output_tokens;
    }
}

impl TokenUsage {
    /// Put a provider's numbers on one footing. OpenAI counts cached tokens
    /// inside its input total. Anthropic reports three separate numbers:
    /// uncached input, input read from the cache, and input written to the
    /// cache (`cache_writes`, which rig's own usage figures leave out).
    fn normalised(provider: Provider, input: u64, cached: u64, cache_writes: u64, output: u64) -> Self {
        let input_tokens = match provider {
            Provider::Anthropic => input + cached + cache_writes,
            _ => input.max(cached),
        };
        Self { input_tokens, cached_input_tokens: cached, output_tokens: output }
    }
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
    /// openrouter.ai: one key, models from many vendors.
    OpenRouter,
    /// Any server that speaks the OpenAI chat-completions protocol
    /// (LM Studio, vLLM, llama.cpp, a gateway), at a URL you supply.
    OpenAiCompatible,
}

/// Where an OpenAI-compatible server is, and its key if it wants one.
pub const COMPATIBLE_URL_VAR: &str = "OPENAI_COMPATIBLE_BASE_URL";
pub const COMPATIBLE_KEY_VAR: &str = "OPENAI_COMPATIBLE_API_KEY";

impl Provider {
    pub const ALL: [Provider; 6] = [
        Provider::OpenAi,
        Provider::Anthropic,
        Provider::Gemini,
        Provider::Ollama,
        Provider::OpenRouter,
        Provider::OpenAiCompatible,
    ];

    /// The prefix that selects this provider in a model spec.
    pub fn prefix(self) -> &'static str {
        match self {
            Provider::OpenAi => "openai",
            Provider::Anthropic => "anthropic",
            Provider::Gemini => "gemini",
            Provider::Ollama => "ollama",
            Provider::OpenRouter => "openrouter",
            Provider::OpenAiCompatible => "compatible",
        }
    }

    /// The environment variable holding this provider's API key, if it must have one.
    pub fn key_var(self) -> Option<&'static str> {
        match self {
            Provider::OpenAi => Some("OPENAI_API_KEY"),
            Provider::Anthropic => Some("ANTHROPIC_API_KEY"),
            Provider::Gemini => Some("GEMINI_API_KEY"),
            Provider::OpenRouter => Some("OPENROUTER_API_KEY"),
            Provider::Ollama | Provider::OpenAiCompatible => None,
        }
    }
}

/// Split a model spec into provider and model name.
///
/// An explicit `provider:model` prefix always wins (`ollama:llama3.2`,
/// `anthropic:claude-...`). Without one, names starting with `claude` go to
/// Anthropic, names starting with `gemini` go to Gemini, and everything else
/// goes to OpenAI. Ollama, OpenRouter and OpenAI-compatible servers are only
/// ever chosen by prefix.
pub fn resolve_model(spec: &str) -> (Provider, &str) {
    let spec = spec.trim();
    if let Some((prefix, rest)) = spec.split_once(':') {
        let provider = match prefix.to_ascii_lowercase().as_str() {
            "openai" => Some(Provider::OpenAi),
            "anthropic" => Some(Provider::Anthropic),
            "gemini" | "google" => Some(Provider::Gemini),
            "ollama" => Some(Provider::Ollama),
            "openrouter" => Some(Provider::OpenRouter),
            "compatible" | "openai-compatible" => Some(Provider::OpenAiCompatible),
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
    if provider == Provider::OpenAiCompatible && compatible_base_url().is_none() {
        bail!(
            "{COMPATIBLE_URL_VAR} is not set. Models with the compatible: prefix need the URL of an \
             OpenAI-compatible server, for example http://localhost:1234/v1."
        );
    }
    api_key(provider).map(|_| ())
}

fn compatible_base_url() -> Option<String> {
    std::env::var(COMPATIBLE_URL_VAR).ok().filter(|url| !url.trim().is_empty())
}

fn ollama_base_url() -> Option<String> {
    std::env::var("OLLAMA_BASE_URL")
        .or_else(|_| std::env::var("OLLAMA_API_BASE_URL"))
        .ok()
        .filter(|url| !url.trim().is_empty())
}

/// The real implementation, backed by rig.
#[derive(Debug, Clone, Copy)]
pub struct RigLlm {
    /// Ask Anthropic to cache the prompt prefix between an agent's turns.
    /// (OpenAI caches on its own; rig has no switch for the other providers.)
    pub prompt_caching: bool,
}

impl Default for RigLlm {
    fn default() -> Self {
        Self { prompt_caching: true }
    }
}

/// Adds up the tokens Anthropic says were written to its prompt cache. rig
/// drops this number when it converts a response, so without the hook a
/// request that fills the cache would look almost free.
#[derive(Clone, Default)]
struct CacheWriteCounter(Arc<AtomicU64>);

impl PromptHook<anthropic::completion::CompletionModel> for CacheWriteCounter {
    fn on_completion_response(
        &self,
        _prompt: &rig::completion::Message,
        response: &rig::completion::CompletionResponse<anthropic::completion::CompletionResponse>,
    ) -> impl Future<Output = HookAction> + Send {
        let written = response.raw_response.usage.cache_creation_input_tokens.unwrap_or(0);
        self.0.fetch_add(written, Ordering::Relaxed);
        async { HookAction::cont() }
    }
}

/// Builds and runs an agent on a concrete rig client. A macro rather than a
/// generic function because each provider has its own client and model types.
macro_rules! run_agent {
    ($builder:expr, $request:expr, $max_tokens:expr) => {
        run_agent!($builder, $request, $max_tokens, ())
    };
    ($builder:expr, $request:expr, $max_tokens:expr, $hook:expr) => {{
        let request: &AgentRequest = $request;
        let mut builder = $builder.preamble(&request.preamble);
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
            .with_hook($hook)
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
        let cache_writes = CacheWriteCounter::default();

        let result = match provider {
            Provider::OpenAi => {
                let client = openai::Client::new(&key).map_err(client_error)?;
                run_agent!(client.agent(model), request, no_limit)
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
                // An agent re-sends its whole conversation on every turn. With
                // caching, the part already sent is read back from the cache at a
                // fraction of the input price instead of being billed in full again.
                let mut completion_model = client.completion_model(model);
                if self.prompt_caching {
                    completion_model = completion_model.with_prompt_caching();
                }
                run_agent!(AgentBuilder::new(completion_model), request, max_tokens, cache_writes.clone())
            }
            Provider::Gemini => {
                let client = gemini::Client::new(&key).map_err(client_error)?;
                run_agent!(client.agent(model), request, no_limit)
            }
            Provider::OpenRouter => {
                let client = openrouter::Client::new(&key).map_err(client_error)?;
                run_agent!(client.agent(model), request, no_limit)
            }
            Provider::OpenAiCompatible => {
                let url = compatible_base_url()
                    .ok_or_else(|| LlmError::Other(anyhow!("{COMPATIBLE_URL_VAR} is not set")))?;
                // Local servers usually ignore the key, but the header must not be empty.
                let key = std::env::var(COMPATIBLE_KEY_VAR)
                    .ok()
                    .filter(|key| !looks_like_placeholder(key))
                    .unwrap_or_else(|| "not-needed".to_string());
                // Compatible servers implement chat completions, not OpenAI's newer Responses API.
                let client = openai::Client::builder()
                    .api_key(key)
                    .base_url(url)
                    .build()
                    .map_err(client_error)?
                    .completions_api();
                run_agent!(client.agent(model), request, no_limit)
            }
            Provider::Ollama => {
                let mut builder = ollama::Client::builder().api_key(Nothing);
                if let Some(url) = ollama_base_url() {
                    builder = builder.base_url(url);
                }
                let client = builder.build().map_err(client_error)?;
                run_agent!(client.agent(model), request, no_limit)
            }
        };

        match result {
            Ok(response) => Ok(LlmReply {
                text: response.output,
                usage: TokenUsage::normalised(
                    provider,
                    response.usage.input_tokens,
                    response.usage.cached_input_tokens,
                    cache_writes.0.load(Ordering::Relaxed),
                    response.usage.output_tokens,
                ),
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
        // OpenRouter names carry a vendor and sometimes a variant after a colon.
        assert_eq!(
            resolve_model("openrouter:meta-llama/llama-3.3-70b-instruct:free"),
            (Provider::OpenRouter, "meta-llama/llama-3.3-70b-instruct:free")
        );
        assert_eq!(resolve_model("compatible:qwen2.5-coder"), (Provider::OpenAiCompatible, "qwen2.5-coder"));
        assert_eq!(resolve_model("openai-compatible:x"), (Provider::OpenAiCompatible, "x"));
        // Without the prefix, a vendor-style name is NOT sent to OpenRouter.
        assert_eq!(resolve_model("anthropic/claude-x").0, Provider::OpenAi);
        for provider in Provider::ALL {
            assert_eq!(resolve_model(&format!("{}:m", provider.prefix())), (provider, "m"));
        }
    }

    #[test]
    fn placeholders_do_not_count_as_keys() {
        assert!(looks_like_placeholder(""));
        assert!(looks_like_placeholder("your_openai_key_here"));
        assert!(looks_like_placeholder("stub"));
        assert!(!looks_like_placeholder("sk-abc123"));
    }

    #[test]
    fn cached_tokens_are_part_of_the_input_total_whatever_the_provider_reports() {
        // Anthropic: 200 uncached + 9,000 read from cache.
        let anthropic = TokenUsage::normalised(Provider::Anthropic, 200, 9_000, 0, 50);
        assert_eq!((anthropic.input_tokens, anthropic.cached_input_tokens), (9_200, 9_000));
        // OpenAI: 9,200 in total, of which 9,000 were cached.
        let openai = TokenUsage::normalised(Provider::OpenAi, 9_200, 9_000, 0, 50);
        assert_eq!(openai, anthropic);
        // Seen live: the request that fills the cache reported 4 input tokens
        // through rig, with 7,201 written to the cache and counted nowhere.
        let first = TokenUsage::normalised(Provider::Anthropic, 4, 0, 7_201, 50);
        assert_eq!((first.input_tokens, first.cached_input_tokens), (7_205, 0));

        let mut total = TokenUsage::default();
        total += anthropic;
        total += openai;
        assert_eq!((total.input_tokens, total.cached_input_tokens, total.output_tokens), (18_400, 18_000, 100));
    }

    #[test]
    fn ollama_needs_no_key() {
        assert!(check_credentials("ollama:llama3.2").is_ok());
        assert!(check_credentials("ollama:").is_err());
    }
}
