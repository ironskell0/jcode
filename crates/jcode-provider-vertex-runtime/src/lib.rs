use anyhow::{Context, Result};
use async_trait::async_trait;
use jcode_message_types::{ConnectionPhase, Message, StreamEvent, ToolDefinition};
use jcode_provider_core::{EventStream, ModelRoute, Provider};
use jcode_provider_gemini::{
    CodeAssistGenerateResponse, GeminiToolConfig, VertexGenerateContentRequest,
};
use jcode_provider_vertex::{
    AVAILABLE_MODELS, DEFAULT_MODEL, PersistedCatalog, generate_content_api_url,
    generative_language_models_url, list_project_models_api_url,
};
use serde_json::{Value, json};
use std::sync::{Arc, RwLock};
use tokio::sync::mpsc;
use uuid::Uuid;

pub struct VertexProvider {
    client: reqwest::Client,
    model: Arc<RwLock<String>>,
    fetched_models: Arc<RwLock<Vec<String>>>,
}

impl Clone for VertexProvider {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            model: self.model.clone(),
            fetched_models: self.fetched_models.clone(),
        }
    }
}

impl Default for VertexProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl VertexProvider {
    pub fn new() -> Self {
        let model = std::env::var("JCODE_VERTEX_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
        let provider = Self {
            client: jcode_base::provider::shared_http_client(),
            model: Arc::new(RwLock::new(model)),
            fetched_models: Arc::new(RwLock::new(Vec::new())),
        };
        provider.seed_cached_catalog();
        provider
    }

    fn seed_cached_catalog(&self) {
        if let Some(catalog) = jcode_base::provider::vertex::load_persisted_catalog()
            && let Ok(mut models) = self.fetched_models.write()
        {
            *models = catalog.models;
        }
    }

    pub async fn refresh_available_models(&self) -> Result<Vec<String>> {
        let tokens = jcode_base::auth::vertex::load_or_refresh_tokens().await?;
        let location = tokens
            .location
            .clone()
            .unwrap_or_else(|| jcode_base::auth::vertex::fetch_location());
        let project_id = match tokens.project_id.as_deref() {
            Some(project) if !project.trim().is_empty() => project.to_string(),
            _ => jcode_base::auth::vertex::fetch_project_id(&tokens.access_token)
                .await
                .unwrap_or_else(|_| "default".to_string()),
        };

        let mut discovered = Vec::new();

        // 1. Discover models from generativelanguage API
        let genlang_url = generative_language_models_url();
        if let Ok(resp) = self
            .client
            .get(genlang_url)
            .bearer_auth(&tokens.access_token)
            .header(reqwest::header::USER_AGENT, "jcode")
            .send()
            .await
            && resp.status().is_success()
            && let Ok(body) = resp.json::<Value>().await
            && let Some(models) = body.get("models").and_then(Value::as_array)
        {
            for m in models {
                let is_chat = m.get("supportedGenerationMethods")
                    .and_then(Value::as_array)
                    .map(|methods| methods.iter().any(|method| method == "generateContent"))
                    .unwrap_or(true);
                if is_chat
                    && let Some(name) = m.get("name").and_then(Value::as_str)
                {
                    let clean = name.trim_start_matches("models/").to_string();
                    if !discovered.contains(&clean) {
                        discovered.push(clean);
                    }
                }
            }
        }

        // 2. Discover models from Vertex AI project endpoint
        let project_models_url = list_project_models_api_url(&location, &project_id);
        if let Ok(resp) = self
            .client
            .get(&project_models_url)
            .bearer_auth(&tokens.access_token)
            .header(reqwest::header::USER_AGENT, "jcode")
            .send()
            .await
            && resp.status().is_success()
            && let Ok(body) = resp.json::<Value>().await
            && let Some(models) = body.get("models").and_then(Value::as_array)
        {
            for m in models {
                if let Some(name) = m.get("name").and_then(Value::as_str) {
                    let clean = name
                        .rsplit('/')
                        .next()
                        .unwrap_or(name)
                        .to_string();
                    if !discovered.contains(&clean) {
                        discovered.push(clean);
                    }
                }
            }
        }

        // 3. Merge with standard Vertex foundation models
        for &model in AVAILABLE_MODELS {
            let m = model.to_string();
            if !discovered.contains(&m) {
                discovered.push(m);
            }
        }

        if !discovered.is_empty() {
            jcode_base::logging::info(&format!(
                "Discovered {} Vertex AI models: {}",
                discovered.len(),
                discovered.join(", ")
            ));
            if let Ok(mut guard) = self.fetched_models.write() {
                *guard = discovered.clone();
            }
            jcode_base::provider::vertex::persist_catalog(&PersistedCatalog {
                models: discovered.clone(),
                fetched_at_rfc3339: chrono::Utc::now().to_rfc3339(),
            });
        }

        Ok(discovered)
    }

    async fn send_generate_content(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        system: &str,
    ) -> Result<CodeAssistGenerateResponse> {
        let mut tokens = jcode_base::auth::vertex::load_or_refresh_tokens().await?;
        let location = tokens
            .location
            .clone()
            .unwrap_or_else(|| jcode_base::auth::vertex::fetch_location());
        let project_id = match tokens
            .project_id
            .as_deref()
            .filter(|value| !value.trim().is_empty())
        {
            Some(project) => project.to_string(),
            None => {
                let resolved = jcode_base::auth::vertex::fetch_project_id(&tokens.access_token).await?;
                tokens.project_id = Some(resolved.clone());
                let _ = jcode_base::auth::vertex::save_tokens(&tokens);
                resolved
            }
        };

        let bare_model = model
            .strip_prefix("vertex:")
            .or_else(|| model.strip_prefix("vertexai:"))
            .unwrap_or(model);

        let tools_is_empty = tools.is_empty();
        let tools_payload = jcode_provider_gemini::build_tools(tools);
        let request = VertexGenerateContentRequest {
            contents: jcode_provider_gemini::build_contents(messages),
            system_instruction: jcode_provider_gemini::build_system_instruction_with_tool_guard(
                system,
                !tools_is_empty,
            ),
            tools: tools_payload,
            tool_config: if tools_is_empty {
                None
            } else {
                Some(GeminiToolConfig {
                    function_calling_config: json!({ "mode": "AUTO" }),
                })
            },
            session_id: None,
        };

        let url = generate_content_api_url(&location, &project_id, bare_model);
        let response = self
            .client
            .post(&url)
            .bearer_auth(&tokens.access_token)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .header(reqwest::header::USER_AGENT, "jcode")
            .json(&request)
            .send()
            .await
            .with_context(|| format!("Failed to send Vertex AI request to {url}"))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = jcode_base::util::http_error_body(response, "HTTP error").await;
            anyhow::bail!("Vertex AI request failed (HTTP {}): {}", status, body.trim());
        }

        // The response on Vertex AI is either a direct GeminiResponse or array/object
        let val: Value = response
            .json()
            .await
            .context("Failed to decode Vertex AI JSON response")?;

        let resp: CodeAssistGenerateResponse = if val.get("candidates").is_some() {
            serde_json::from_value(json!({ "response": val }))
                .context("Failed to wrap Vertex AI response")?
        } else if let Some(arr) = val.as_array() && let Some(first) = arr.first() {
            serde_json::from_value(json!({ "response": first }))
                .context("Failed to parse Vertex AI array response")?
        } else {
            serde_json::from_value(val).context("Failed to parse Vertex AI response")?
        };

        Ok(resp)
    }
}

#[async_trait]
impl Provider for VertexProvider {
    fn name(&self) -> &str {
        "Vertex AI"
    }

    fn model(&self) -> String {
        self.model
            .read()
            .ok()
            .map(|m| m.clone())
            .unwrap_or_else(|| DEFAULT_MODEL.to_string())
    }

    fn set_model(&self, model: &str) -> Result<()> {
        let bare = model
            .strip_prefix("vertex:")
            .or_else(|| model.strip_prefix("vertexai:"))
            .unwrap_or(model)
            .trim();
        if bare.is_empty() {
            anyhow::bail!("Vertex model cannot be empty");
        }
        if let Ok(mut guard) = self.model.write() {
            *guard = bare.to_string();
        }
        Ok(())
    }

    fn available_models_display(&self) -> Vec<String> {
        let fetched = self
            .fetched_models
            .read()
            .ok()
            .map(|guard| guard.clone())
            .unwrap_or_default();
        if !fetched.is_empty() {
            fetched
        } else {
            AVAILABLE_MODELS.iter().map(|&s| s.to_string()).collect()
        }
    }

    fn model_routes(&self) -> Vec<ModelRoute> {
        let models = self.available_models_display();
        models
            .into_iter()
            .map(|model| {
                let context_window = if model.contains("gemini-2.5") || model.contains("gemini-3") || model.contains("gemini-4") {
                    Some(1_000_000)
                } else if model.contains("claude") {
                    Some(200_000)
                } else {
                    Some(128_000)
                };
                ModelRoute {
                    provider: "Vertex".to_string(),
                    api_method: "vertex".to_string(),
                    model: model.clone(),
                    display_name: format!("Vertex {model}"),
                    available: true,
                    detail: "Google Cloud Vertex AI".to_string(),
                    usage: None,
                    cheapness: None,
                    is_default: model == DEFAULT_MODEL,
                    context_window,
                    max_output_tokens: Some(65_536),
                    tags: vec!["google-cloud".to_string(), "vertex".to_string()],
                    supports_fast: None,
                }
            })
            .collect()
    }

    async fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        system: &str,
        _resume_session_id: Option<&str>,
    ) -> Result<EventStream> {
        let model = self
            .model
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let messages = messages.to_vec();
        let tools = tools.to_vec();
        let system = system.to_string();
        let provider = self.clone();
        let (tx, rx) = mpsc::channel::<Result<StreamEvent>>(100);

        tokio::spawn(async move {
            let _ = tx
                .send(Ok(StreamEvent::ConnectionType {
                    connection: "https".to_string(),
                }))
                .await;
            let _ = tx
                .send(Ok(StreamEvent::ConnectionPhase {
                    phase: ConnectionPhase::Authenticating,
                }))
                .await;
            let _ = tx
                .send(Ok(StreamEvent::ConnectionPhase {
                    phase: ConnectionPhase::WaitingForResponse,
                }))
                .await;

            let response = match provider
                .send_generate_content(&model, &messages, &tools, &system)
                .await
            {
                Ok(resp) => resp,
                Err(err) => {
                    let _ = tx.send(Err(err)).await;
                    return;
                }
            };

            let _ = tx
                .send(Ok(StreamEvent::ConnectionPhase {
                    phase: ConnectionPhase::Streaming,
                }))
                .await;

            if let Some(usage) = response
                .response
                .as_ref()
                .and_then(|r| r.usage_metadata.as_ref())
            {
                let _ = tx
                    .send(Ok(StreamEvent::TokenUsage {
                        input_tokens: usage.input_token_count(),
                        output_tokens: usage.output_token_count(),
                        cache_read_input_tokens: usage.cached_content_token_count,
                        cache_creation_input_tokens: None,
                    }))
                    .await;
            }

            let Some(candidate) = response
                .response
                .and_then(|r| r.candidates)
                .and_then(|mut c| c.drain(..).next())
            else {
                let _ = tx
                    .send(Err(anyhow::anyhow!("Vertex AI returned no candidates")))
                    .await;
                return;
            };

            let mut produced_output = false;
            if let Some(content) = candidate.content {
                let mut pending_signature: Option<String> = None;
                for part in content.parts {
                    let part_signature = part
                        .thought_signature
                        .as_ref()
                        .filter(|sig| !sig.is_empty())
                        .cloned();
                    if let Some(text) = part.text.filter(|text| !text.is_empty()) {
                        produced_output = true;
                        let _ = tx.send(Ok(StreamEvent::TextDelta(text))).await;
                    }
                    if let Some(function_call) = part.function_call {
                        produced_output = true;
                        let signature = part_signature.clone().or_else(|| pending_signature.take());
                        let raw_call_id = function_call
                            .id
                            .clone()
                            .unwrap_or_else(|| Uuid::new_v4().to_string());
                        let call_id = jcode_message_types::sanitize_tool_id(&raw_call_id);
                        let _ = tx
                            .send(Ok(StreamEvent::ToolUseStart {
                                id: call_id,
                                name: function_call.name,
                            }))
                            .await;
                        let _ = tx
                            .send(Ok(StreamEvent::ToolInputDelta(
                                function_call.args.to_string(),
                            )))
                            .await;
                        let _ = tx.send(Ok(StreamEvent::ToolUseEnd)).await;
                        if let Some(sig) = signature {
                            let _ = tx.send(Ok(StreamEvent::ToolUseSignature(sig))).await;
                        }
                    } else if let Some(sig) = part_signature {
                        pending_signature = Some(sig);
                    }
                }
                if let Some(sig) = pending_signature.take() {
                    let _ = tx
                        .send(Ok(StreamEvent::ThinkingSignatureDelta(sig)))
                        .await;
                }
            }

            if !produced_output {
                let reason = candidate.finish_reason.as_deref().unwrap_or("unknown");
                let _ = tx
                    .send(Err(anyhow::anyhow!(
                        "Vertex AI returned no usable output (finish_reason={reason})"
                    )))
                    .await;
                return;
            }

            let _ = tx.send(Ok(StreamEvent::MessageEnd)).await;
        });

        Ok(Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertex_provider_defaults() {
        let provider = VertexProvider::new();
        assert_eq!(provider.name(), "Vertex AI");
        assert_eq!(provider.model(), DEFAULT_MODEL);
        let routes = provider.model_routes();
        assert!(!routes.is_empty());
        assert!(routes.iter().any(|r| r.model == DEFAULT_MODEL));
    }

    #[test]
    fn test_vertex_provider_set_model() {
        let provider = VertexProvider::new();
        assert!(provider.set_model("vertex:claude-3-7-sonnet@20250219").is_ok());
        assert_eq!(provider.model(), "claude-3-7-sonnet@20250219");
    }
}
