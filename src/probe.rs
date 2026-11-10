//! Find out whether a model actually works here before relying on it.
//!
//! A provider "being supported" means little until a request has gone through
//! and a tool call has come back, so the probe does exactly those two things
//! and reports what happened.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Result, bail};

use crate::llm::{AgentRequest, Llm, TokenUsage, check_credentials, resolve_model};
use crate::tools::probe::GetCode;
use crate::tools::{Sandbox, ToolAccess};

#[derive(Debug)]
pub struct ProbeReport {
    pub model: String,
    pub provider: String,
    /// A plain request and its reply.
    pub completion: Result<TokenUsage, String>,
    /// A tool-call round trip. `None` if the plain request already failed.
    pub tool_call: Option<Result<TokenUsage, String>>,
}

impl ProbeReport {
    pub fn passed(&self) -> bool {
        self.completion.is_ok() && matches!(self.tool_call, Some(Ok(_)))
    }
}

/// The probe uses no files and runs no commands; the sandbox only satisfies the request type.
fn idle_sandbox() -> Result<Sandbox> {
    Ok(Sandbox::new(std::env::temp_dir(), Duration::from_secs(1))?)
}

fn shorten(text: &str) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() > 300 { format!("{}...", flat.chars().take(300).collect::<String>()) } else { flat }
}

fn request(model: &str, preamble: &str, prompt: String, tools: ToolAccess, max_turns: usize) -> Result<AgentRequest> {
    Ok(AgentRequest {
        model: model.to_string(),
        preamble: preamble.to_string(),
        prompt,
        tools,
        sandbox: idle_sandbox()?,
        max_turns,
    })
}

/// One plain request. `Ok` carries the usage the provider reported.
async fn completion(llm: &dyn Llm, model: &str) -> Result<TokenUsage, String> {
    check_credentials(model).map_err(|e| e.to_string())?;
    let request = request(
        model,
        "You are a connectivity check.",
        "Reply with the single word READY.".to_string(),
        ToolAccess::None,
        1,
    )
    .map_err(|e| e.to_string())?;
    let reply = llm.run(request).await.map_err(|e| shorten(&e.to_string()))?;
    if reply.text.trim().is_empty() {
        return Err("the model returned an empty reply".to_string());
    }
    Ok(reply.usage)
}

/// One tool call: the model must call `get_code` with a list argument and
/// repeat the code it gets back, which it has no other way of knowing.
async fn tool_call(llm: &dyn Llm, model: &str) -> Result<TokenUsage, String> {
    let token = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.subsec_nanos()).unwrap_or(7)
        ^ std::process::id();
    let code = GetCode::code(token);
    let request = request(
        model,
        "You are a tool-calling check. Use the tool you are given.",
        "Call the tool `get_code` with `words` set to the list [\"alpha\", \"beta\"]. \
         Then reply with exactly the code the tool returned and nothing else."
            .to_string(),
        ToolAccess::Probe(token),
        4,
    )
    .map_err(|e| e.to_string())?;
    let reply = llm.run(request).await.map_err(|e| shorten(&e.to_string()))?;
    if reply.text.contains(&code) {
        Ok(reply.usage)
    } else {
        Err(format!(
            "the model did not return the tool's result, so the tool call did not complete. It replied: {}",
            shorten(&reply.text)
        ))
    }
}

/// Run both checks against one model.
pub async fn probe_model(llm: &dyn Llm, model: &str) -> ProbeReport {
    let (provider, _) = resolve_model(model);
    let completion = completion(llm, model).await;
    let tool_call = match &completion {
        Ok(_) => Some(tool_call(llm, model).await),
        Err(_) => None,
    };
    ProbeReport { model: model.to_string(), provider: format!("{provider:?}"), completion, tool_call }
}

/// Before a run: make one small request to every distinct model, so that a
/// mistyped model name or a dead key stops the run now rather than after a
/// worker has already spent tokens.
pub async fn preflight(llm: &dyn Llm, models: &[&str]) -> Result<TokenUsage> {
    let mut seen: Vec<&str> = Vec::new();
    let mut usage = TokenUsage::default();
    for model in models {
        if seen.contains(model) {
            continue;
        }
        seen.push(model);
        match completion(llm, model).await {
            Ok(used) => {
                tracing::info!("Preflight: {model} answered a test request");
                usage += used;
            }
            Err(e) => bail!("model {model} is not usable: {e}"),
        }
    }
    Ok(usage)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{LlmError, LlmFuture, LlmReply};
    use std::sync::Mutex;

    /// Answers from a closure and remembers which models were asked.
    struct Fake<F>(F, Mutex<Vec<String>>);

    impl<F> Fake<F>
    where
        F: Fn(&AgentRequest) -> Result<String, LlmError> + Send + Sync,
    {
        fn new(answer: F) -> Self {
            Self(answer, Mutex::new(Vec::new()))
        }
    }

    impl<F> Llm for Fake<F>
    where
        F: Fn(&AgentRequest) -> Result<String, LlmError> + Send + Sync,
    {
        fn run<'a>(&'a self, request: AgentRequest) -> LlmFuture<'a> {
            Box::pin(async move {
                self.1.lock().unwrap().push(request.model.clone());
                (self.0)(&request).map(|text| LlmReply {
                    text,
                    usage: TokenUsage { input_tokens: 9, cached_input_tokens: 0, output_tokens: 3 },
                })
            })
        }
    }

    /// A model that really calls the tool: it answers with the code for the request's token.
    fn working(request: &AgentRequest) -> Result<String, LlmError> {
        Ok(match request.tools {
            ToolAccess::Probe(token) => GetCode::code(token),
            _ => "READY".to_string(),
        })
    }

    // `ollama:` models need no key, which keeps these tests independent of the environment.

    #[tokio::test]
    async fn a_working_model_passes_both_checks() {
        let report = probe_model(&Fake::new(working), "ollama:some-model").await;
        assert!(report.passed(), "{report:?}");
        assert_eq!(report.provider, "Ollama");
        assert_eq!(report.completion.unwrap().input_tokens, 9);
    }

    #[tokio::test]
    async fn a_model_that_talks_but_cannot_call_tools_fails() {
        // It claims to have done it, with a made-up code.
        let llm = Fake::new(|request: &AgentRequest| {
            Ok(match request.tools {
                ToolAccess::Probe(_) => "The code is PROBE-12345".to_string(),
                _ => "READY".to_string(),
            })
        });
        let report = probe_model(&llm, "ollama:chatty").await;
        assert!(report.completion.is_ok());
        let failure = report.tool_call.as_ref().unwrap().as_ref().unwrap_err();
        assert!(failure.contains("did not return the tool's result"), "{failure}");
        assert!(!report.passed());
    }

    #[tokio::test]
    async fn an_unreachable_model_fails_without_a_second_request() {
        let llm = Fake::new(|_: &AgentRequest| Err(LlmError::Other(anyhow::anyhow!("404 model not found"))));
        let report = probe_model(&llm, "ollama:typo").await;
        assert!(report.completion.as_ref().unwrap_err().contains("404"));
        assert!(report.tool_call.is_none());
        assert!(!report.passed());
        assert_eq!(llm.1.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn preflight_asks_each_distinct_model_once_and_stops_at_the_first_failure() {
        let llm = Fake::new(working);
        let usage = preflight(&llm, &["ollama:a", "ollama:b", "ollama:a"]).await.unwrap();
        assert_eq!(*llm.1.lock().unwrap(), ["ollama:a", "ollama:b"]);
        assert_eq!(usage.input_tokens, 18);

        let llm = Fake::new(|request: &AgentRequest| {
            if request.model == "ollama:judge-typo" {
                Err(LlmError::Other(anyhow::anyhow!("404 model not found")))
            } else {
                Ok("READY".to_string())
            }
        });
        let error = preflight(&llm, &["ollama:worker", "ollama:judge-typo"]).await.unwrap_err().to_string();
        assert!(error.contains("ollama:judge-typo") && error.contains("404"), "{error}");
    }

    #[tokio::test]
    async fn a_missing_key_is_reported_without_calling_the_provider() {
        let llm = Fake::new(working);
        // An OpenAI-compatible model with no server URL configured cannot be probed.
        // SAFETY: test-only; no other test reads this variable.
        unsafe { std::env::remove_var(crate::llm::COMPATIBLE_URL_VAR) };
        let report = probe_model(&llm, "compatible:local-model").await;
        assert!(report.completion.unwrap_err().contains("OPENAI_COMPATIBLE_BASE_URL"));
        assert!(llm.1.lock().unwrap().is_empty());
    }
}
