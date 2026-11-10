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
use rig::completion::{CompletionError, CompletionModel, CompletionRequest, CompletionResponse, Prompt, PromptError};
use rig::providers::{anthropic, gemini, ollama, openai, openrouter};
use rig::streaming::StreamingCompletionResponse;
use tokio::time::Instant;

use crate::tools::{Sandbox, ToolAccess};

/// Upper bound on one agent run (all of its turns), so a stalled connection
/// cannot hang the orchestrator forever.
const AGENT_RUN_TIMEOUT: Duration = Duration::from_secs(30 * 60);

/// rig only knows output limits for older Claude models and refuses to send a
/// request without one, so newer models get this value.
const ANTHROPIC_FALLBACK_MAX_TOKENS: u64 = 16_000;

/// How long to wait before sending a request again after the provider turned
/// it away for a reason that passes (rate limit, overload, a dropped
/// connection). The last wait is long enough for a per-minute limit to reset.
const REQUEST_RETRY_DELAYS: [Duration; 3] =
    [Duration::from_secs(5), Duration::from_secs(20), Duration::from_secs(65)];

/// OpenRouter allows 20 requests a minute to its free models, across the
/// whole account. Spacing requests this far apart stays just under that.
const FREE_MODEL_REQUEST_INTERVAL: Duration = Duration::from_millis(3_200);

/// Set to a true value to refuse OpenRouter models that cost money.
pub const OPENROUTER_FREE_ONLY_VAR: &str = "OPENROUTER_FREE_ONLY";

static FREE_MODEL_PACER: Pacer = Pacer::new(FREE_MODEL_REQUEST_INTERVAL);

/// Hands out send times at a fixed spacing, to every agent in the process,
/// and counts the requests it let through.
struct Pacer {
    interval: Duration,
    next: std::sync::Mutex<Option<Instant>>,
    sent: AtomicU64,
}

impl Pacer {
    const fn new(interval: Duration) -> Self {
        Self { interval, next: std::sync::Mutex::new(None), sent: AtomicU64::new(0) }
    }

    /// Wait for this caller's turn. Turns are reserved in arrival order.
    async fn wait(&self) {
        let turn = {
            let mut next = self.next.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            let now = Instant::now();
            let turn = next.map_or(now, |at| at.max(now));
            *next = Some(turn + self.interval);
            turn
        };
        tokio::time::sleep_until(turn).await;
        self.sent.fetch_add(1, Ordering::Relaxed);
    }
}

/// Requests sent to OpenRouter's free models by this process, including
/// failed ones: the free tier's daily allowance counts requests, not tokens.
pub fn free_model_requests() -> u64 {
    FREE_MODEL_PACER.sent.load(Ordering::Relaxed)
}

/// OpenRouter marks the variants it does not charge for with this suffix.
pub fn is_free_openrouter_model(model: &str) -> bool {
    model.trim().ends_with(":free")
}

/// Is any of these model specs an OpenRouter free model?
pub fn uses_free_tier(model_specs: &[&str]) -> bool {
    model_specs.iter().any(|spec| {
        let (provider, model) = resolve_model(spec);
        provider == Provider::OpenRouter && is_free_openrouter_model(model)
    })
}

/// The HTTP status in an error from a provider, if it carries one.
pub fn http_status(error: &str) -> Option<u16> {
    let lower = error.to_ascii_lowercase();
    lower
        .split_once("status code ")
        .and_then(|(_, rest)| rest.split(|c: char| !c.is_ascii_digit()).next())
        .and_then(|digits| digits.parse::<u16>().ok())
}

/// Does this error mean the model cannot be used at all for now, whatever is
/// asked of it? A worker model is dropped from a run on the strength of this,
/// so it goes by the status the provider sent or by this program's own
/// messages, never by a word that happens to occur in an error text.
pub fn model_unusable(error: &str) -> bool {
    if let Some(status) = http_status(error) {
        return matches!(status, 401..=404);
    }
    [FREE_ALLOWANCE_USED_UP, "is not a free OpenRouter model", "is not set", "could not create the"]
        .iter()
        .any(|own| error.contains(own))
}

/// Would sending the same request again later plausibly work?
///
/// Deliberately a list of what is known to pass rather than "anything but a
/// bad key": a request that is wrong stays wrong, and resending it only
/// spends time and allowance.
pub fn is_transient(error: &str) -> bool {
    let lower = error.to_ascii_lowercase();
    if let Some(status) = http_status(error) {
        return matches!(status, 408 | 425 | 429) || status >= 500;
    }
    [
        "429",
        "too many requests",
        "rate limit",
        "rate-limit",
        "overloaded",
        "temporarily",
        "timed out",
        "timeout",
        "connection",
        "error sending request",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

/// What OpenRouter reports about the account's free-model requests for the current UTC day.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FreeAllowance {
    pub limit: u64,
    pub remaining: u64,
}

/// The error text for a used-up daily allowance. Not transient, and permanent for an agent.
pub const FREE_ALLOWANCE_USED_UP: &str = "OpenRouter's free-model allowance for today is used up. It is counted \
     per UTC day; continue with --resume after it resets, or use a model that is not on the free tier.";

fn parse_free_allowance(body: &serde_json::Value) -> Option<FreeAllowance> {
    let counter = body.get("data")?.get("free_model_daily_requests")?;
    Some(FreeAllowance { limit: counter.get("limit")?.as_u64()?, remaining: counter.get("remaining")?.as_u64()? })
}

/// Ask OpenRouter how much of the day's free-model allowance is left. `None`
/// if there is no key or the question could not be answered; the counter can
/// also lag a little behind requests just made.
pub async fn free_allowance() -> Option<FreeAllowance> {
    let key = api_key(Provider::OpenRouter).ok()?;
    let client = reqwest::Client::builder().timeout(Duration::from_secs(15)).build().ok()?;
    let response = client.get("https://openrouter.ai/api/v1/key").bearer_auth(key).send().await.ok()?;
    parse_free_allowance(&response.json::<serde_json::Value>().await.ok()?)
}

/// OpenRouter answers a used-up day and a busy minute with the same 429, and
/// only its own counter tells them apart.
async fn free_allowance_used_up() -> bool {
    matches!(free_allowance().await, Some(FreeAllowance { remaining: 0, .. }))
}

/// Why a request that failed for a passing reason should not be sent again after all.
enum GiveUp {
    No,
    Because(&'static str),
}

/// Make one request, waiting for the pacer first and trying again after a
/// passing failure. `give_up` is asked before each wait.
async fn send_with_retries<T, F, Fut, G, GFut>(
    pacer: Option<&Pacer>,
    delays: &[Duration],
    mut send: F,
    give_up: G,
) -> Result<T, CompletionError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, CompletionError>>,
    G: Fn() -> GFut,
    GFut: Future<Output = GiveUp>,
{
    let mut delays = delays.iter();
    loop {
        if let Some(pacer) = pacer {
            pacer.wait().await;
        }
        match send().await {
            Err(e) if is_transient(&e.to_string()) => {
                let Some(delay) = delays.next() else { return Err(e) };
                if let GiveUp::Because(reason) = give_up().await {
                    return Err(CompletionError::ProviderError(reason.to_string()));
                }
                tracing::warn!("Model request failed, sending it again in {}s: {e}", delay.as_secs());
                tokio::time::sleep(*delay).await;
            }
            other => return other,
        }
    }
}

/// A provider's model, made patient: a request that is turned away for a
/// passing reason is sent again instead of ending the agent, which would throw
/// away every turn the agent had already made (and paid for).
#[derive(Clone)]
struct Steady<M> {
    inner: M,
    pacer: Option<&'static Pacer>,
}

impl<M> Steady<M> {
    fn new(inner: M) -> Self {
        Self { inner, pacer: None }
    }

    fn paced(inner: M, pacer: &'static Pacer) -> Self {
        Self { inner, pacer: Some(pacer) }
    }
}

impl<M: CompletionModel> CompletionModel for Steady<M> {
    type Response = M::Response;
    type StreamingResponse = M::StreamingResponse;
    type Client = M::Client;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(M::make(client, model))
    }

    fn completion(
        &self,
        request: CompletionRequest,
    ) -> impl Future<Output = Result<CompletionResponse<Self::Response>, CompletionError>> + Send {
        // Only the free tier is paced, and only it has a daily allowance to run out of.
        let on_free_tier = self.pacer.is_some();
        send_with_retries(
            self.pacer,
            &REQUEST_RETRY_DELAYS,
            move || self.inner.completion(request.clone()),
            move || async move {
                if on_free_tier && free_allowance_used_up().await {
                    GiveUp::Because(FREE_ALLOWANCE_USED_UP)
                } else {
                    GiveUp::No
                }
            },
        )
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl Future<Output = Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError>> + Send
    {
        self.inner.stream(request)
    }
}

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

    /// The environment variables that may hold this provider's API key, the
    /// documented name first. Empty if the provider does not require a key.
    pub fn key_vars(self) -> &'static [&'static str] {
        match self {
            Provider::OpenAi => &["OPENAI_API_KEY"],
            Provider::Anthropic => &["ANTHROPIC_API_KEY"],
            Provider::Gemini => &["GEMINI_API_KEY"],
            Provider::OpenRouter => &["OPENROUTER_API_KEY", "OPENROUTER_KEY"],
            Provider::Ollama | Provider::OpenAiCompatible => &[],
        }
    }
}

/// Every environment variable that can hold a provider credential. Commands
/// run by agents must not see any of them: their output goes back to a model.
pub fn secret_env_vars() -> Vec<&'static str> {
    let mut vars: Vec<&'static str> =
        Provider::ALL.iter().flat_map(|provider| provider.key_vars().iter().copied()).collect();
    vars.push(COMPATIBLE_KEY_VAR);
    vars
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
    let vars = provider.key_vars();
    let Some(var) = vars.first() else {
        return Ok(String::new());
    };
    let found = vars.iter().filter_map(|var| std::env::var(var).ok()).find(|key| !looks_like_placeholder(key));
    match found {
        Some(key) => Ok(key.trim().to_string()),
        None => bail!(
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
    if provider == Provider::OpenRouter && free_only() && !is_free_openrouter_model(model) {
        bail!(
            "{model} is not a free OpenRouter model, and {OPENROUTER_FREE_ONLY_VAR} is set. \
             Free models have names ending in :free."
        );
    }
    api_key(provider).map(|_| ())
}

fn free_only() -> bool {
    std::env::var(OPENROUTER_FREE_ONLY_VAR)
        .is_ok_and(|value| matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
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

impl PromptHook<Steady<anthropic::completion::CompletionModel>> for CacheWriteCounter {
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
        // Checked here as well as at startup: the TUI and the MCP server can name a model later.
        check_credentials(&request.model).map_err(LlmError::Other)?;
        let key = api_key(provider).map_err(LlmError::Other)?;
        let client_error = |e| LlmError::Other(anyhow!("could not create the {provider:?} client: {e}"));
        let no_limit: Option<u64> = None;
        let cache_writes = CacheWriteCounter::default();

        let result = match provider {
            Provider::OpenAi => {
                let client = openai::Client::new(&key).map_err(client_error)?;
                run_agent!(AgentBuilder::new(Steady::new(client.completion_model(model))), request, no_limit)
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
                run_agent!(AgentBuilder::new(Steady::new(completion_model)), request, max_tokens, cache_writes.clone())
            }
            Provider::Gemini => {
                let client = gemini::Client::new(&key).map_err(client_error)?;
                run_agent!(AgentBuilder::new(Steady::new(client.completion_model(model))), request, no_limit)
            }
            Provider::OpenRouter => {
                let client = openrouter::Client::new(&key).map_err(client_error)?;
                let completion_model = client.completion_model(model);
                let completion_model = if is_free_openrouter_model(model) {
                    Steady::paced(completion_model, &FREE_MODEL_PACER)
                } else {
                    Steady::new(completion_model)
                };
                run_agent!(AgentBuilder::new(completion_model), request, no_limit)
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
                run_agent!(AgentBuilder::new(Steady::new(client.completion_model(model))), request, no_limit)
            }
            Provider::Ollama => {
                let mut builder = ollama::Client::builder().api_key(Nothing);
                if let Some(url) = ollama_base_url() {
                    builder = builder.base_url(url);
                }
                let client = builder.build().map_err(client_error)?;
                run_agent!(AgentBuilder::new(Steady::new(client.completion_model(model))), request, no_limit)
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
            resolve_model("openrouter:cohere/north-mini-code:free"),
            (Provider::OpenRouter, "cohere/north-mini-code:free")
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
    fn only_passing_failures_are_worth_sending_again() {
        // Seen live from OpenRouter for a free model.
        assert!(is_transient(
            "HttpError: Invalid status code 429 Too Many Requests with message: {\"error\":{\"message\":\"Provider returned error\",\"code\":429,\"metadata\":{\"raw\":\"google/gemma-4-31b-it:free is temporarily rate-limited upstream. Please retry shortly\"}}}"
        ));
        assert!(is_transient("HttpError: Invalid status code 503 Service Unavailable"));
        assert!(is_transient("HttpError: Invalid status code 529 with message: overloaded_error"));
        assert!(is_transient("ProviderError: Rate limit exceeded"));
        assert!(is_transient("HttpError: error sending request for url"));
        // What this program says once OpenRouter's counter shows the day is used up.
        assert!(!is_transient(&CompletionError::ProviderError(FREE_ALLOWANCE_USED_UP.to_string()).to_string()));
        // The request itself is wrong.
        assert!(!is_transient("HttpError: Invalid status code 404 Not Found with message: model not found"));
        assert!(!is_transient("HttpError: Invalid status code 401 Unauthorized"));
        assert!(!is_transient("HttpError: Invalid status code 402 Payment Required"));
        // A status decides; a 400 whose text mentions a timeout setting is still a 400.
        assert!(!is_transient("HttpError: Invalid status code 400 with message: timeout must be positive"));
        assert!(!is_transient("JsonError: missing field `choices`"));
    }

    fn failure(text: &str) -> CompletionError {
        CompletionError::ProviderError(text.to_string())
    }

    async fn carry_on() -> GiveUp {
        GiveUp::No
    }

    #[tokio::test(start_paused = true)]
    async fn a_request_turned_away_for_a_passing_reason_is_sent_again() {
        let delays = [Duration::from_secs(5), Duration::from_secs(20)];
        let started = Instant::now();

        let mut sent = 0;
        let reply = send_with_retries(
            None,
            &delays,
            || {
                sent += 1;
                let outcome = if sent < 3 { Err(failure("status code 429 Too Many Requests")) } else { Ok("done") };
                async move { outcome }
            },
            carry_on,
        )
        .await;
        assert_eq!((reply.unwrap(), sent), ("done", 3));
        assert_eq!(started.elapsed(), Duration::from_secs(25));

        // It gives up when the waits are used up...
        let mut sent = 0;
        let reply: Result<(), _> = send_with_retries(
            None,
            &delays,
            || {
                sent += 1;
                async { Err(failure("status code 503")) }
            },
            carry_on,
        )
        .await;
        assert_eq!((reply.is_err(), sent), (true, 3));

        // ...at once when sending again cannot help...
        let mut sent = 0;
        let reply: Result<(), _> = send_with_retries(
            None,
            &delays,
            || {
                sent += 1;
                async { Err(failure("status code 404 model not found")) }
            },
            carry_on,
        )
        .await;
        assert_eq!((reply.is_err(), sent), (true, 1));

        // ...and without waiting when the day's allowance turns out to be gone.
        let started = Instant::now();
        let mut sent = 0;
        let reply: Result<(), _> = send_with_retries(
            None,
            &delays,
            || {
                sent += 1;
                async { Err(failure("status code 429 Rate limit exceeded")) }
            },
            || async { GiveUp::Because(FREE_ALLOWANCE_USED_UP) },
        )
        .await;
        assert!(reply.unwrap_err().to_string().contains("allowance for today is used up"));
        assert_eq!((sent, started.elapsed()), (1, Duration::ZERO));
    }

    #[tokio::test(start_paused = true)]
    async fn paced_requests_keep_their_distance_even_from_parallel_agents() {
        let pacer = Pacer::new(Duration::from_secs(3));
        let started = Instant::now();
        // Three agents ask at the same moment.
        tokio::join!(pacer.wait(), pacer.wait(), pacer.wait());
        assert_eq!(started.elapsed(), Duration::from_secs(6));
        assert_eq!(pacer.sent.load(Ordering::Relaxed), 3);

        // After a quiet spell the next request goes out at once...
        tokio::time::sleep(Duration::from_secs(60)).await;
        let resumed = Instant::now();
        pacer.wait().await;
        assert_eq!(resumed.elapsed(), Duration::ZERO);
        // ...and a retry is paced like any other request.
        let reply: Result<(), _> = send_with_retries(
            Some(&pacer),
            &[Duration::from_secs(1)],
            || async { Err(failure("status code 429")) },
            carry_on,
        )
        .await;
        assert!(reply.is_err());
        assert_eq!(pacer.sent.load(Ordering::Relaxed), 6);
    }

    #[test]
    fn the_daily_counter_is_read_from_openrouters_answer() {
        // The shape OpenRouter returned for this account on 2026-11-10.
        let body = serde_json::json!({"data": {"is_free_tier": true,
            "free_model_daily_requests": {"used": 9, "limit": 50, "remaining": 41}}});
        assert_eq!(parse_free_allowance(&body), Some(FreeAllowance { limit: 50, remaining: 41 }));
        assert_eq!(parse_free_allowance(&serde_json::json!({"error": {"code": 401}})), None);
    }

    #[test]
    fn a_model_is_only_given_up_on_for_a_reason_that_is_certain() {
        assert!(model_unusable("HttpError: Invalid status code 401 Unauthorized"));
        assert!(model_unusable("HttpError: Invalid status code 402 Payment Required"));
        assert!(model_unusable("HttpError: Invalid status code 404 Not Found with message: no such model"));
        assert!(model_unusable(&CompletionError::ProviderError(FREE_ALLOWANCE_USED_UP.to_string()).to_string()));
        assert!(model_unusable("OPENROUTER_API_KEY is not set (or still holds the placeholder from .env.example)."));
        // A rate limit whose text happens to contain the digits of an auth error.
        assert!(!model_unusable(
            "HttpError: Invalid status code 429 Too Many Requests with message: Requested 40312 tokens, limit 40100"
        ));
        assert!(!model_unusable("ProviderError: you do not have permission to exceed 4030 requests per minute"));
        assert!(!model_unusable("HttpError: Invalid status code 503 Service Unavailable"));
        assert!(!model_unusable("HttpError: error sending request for url"));
    }

    #[test]
    fn free_openrouter_models_are_recognised_by_their_suffix() {
        assert!(is_free_openrouter_model("cohere/north-mini-code:free"));
        assert!(!is_free_openrouter_model("cohere/north-mini-code"));
        assert!(uses_free_tier(&["gpt-4o", "openrouter:cohere/north-mini-code:free"]));
        assert!(!uses_free_tier(&["gpt-4o", "openrouter:cohere/north-mini-code"]));
        // The suffix means nothing on another provider.
        assert!(!uses_free_tier(&["ollama:some-model:free"]));
    }

    #[test]
    fn every_credential_variable_is_on_the_secret_list() {
        let secrets = secret_env_vars();
        for provider in Provider::ALL {
            for var in provider.key_vars() {
                assert!(secrets.contains(var), "{var} is missing");
            }
        }
        for var in ["OPENROUTER_API_KEY", "OPENROUTER_KEY", COMPATIBLE_KEY_VAR] {
            assert!(secrets.contains(&var), "{var} is missing");
        }
    }

    #[test]
    fn ollama_needs_no_key() {
        assert!(check_credentials("ollama:llama3.2").is_ok());
        assert!(check_credentials("ollama:").is_err());
    }
}
