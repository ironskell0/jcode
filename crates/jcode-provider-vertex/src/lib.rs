use serde::{Deserialize, Serialize};

pub const DEFAULT_LOCATION: &str = "us-central1";
pub const DEFAULT_MODEL: &str = "gemini-2.5-flash";

pub const AVAILABLE_MODELS: &[&str] = &[
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-3.8-flash-cyber",
    "gemini-4-argon",
    "gemini-3.1-pro",
    "gemini-3-flash",
    "claude-3-7-sonnet@20250219",
    "claude-3-5-sonnet-v2@20241022",
    "claude-3-5-haiku@20241022",
    "claude-3-opus@20240229",
    "meta/llama-3.3-70b-instruct-maas",
    "meta/llama-3.1-405b-instruct-maas",
    "meta/llama-3.1-70b-instruct-maas",
    "meta/llama-3.1-8b-instruct-maas",
    "mistralai/mistral-large-2407",
];

pub fn stream_generate_content_api_url(location: &str, project_id: &str, model: &str) -> String {
    format!(
        "https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model}:streamGenerateContent?alt=sse"
    )
}

pub fn generate_content_api_url(location: &str, project_id: &str, model: &str) -> String {
    format!(
        "https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model}:generateContent"
    )
}

pub fn list_project_models_api_url(location: &str, project_id: &str) -> String {
    format!(
        "https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/models"
    )
}

pub fn generative_language_models_url() -> &'static str {
    "https://generativelanguage.googleapis.com/v1beta/models"
}

pub fn openapi_chat_completions_url(location: &str, project_id: &str) -> String {
    format!(
        "https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{location}/endpoints/openapi/chat/completions"
    )
}

pub fn is_known_model(model: &str) -> bool {
    let bare = model
        .strip_prefix("vertex:")
        .or_else(|| model.strip_prefix("vertexai:"))
        .unwrap_or(model);
    AVAILABLE_MODELS.contains(&bare)
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct PersistedCatalog {
    pub models: Vec<String>,
    pub fetched_at_rfc3339: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_urls_and_models() {
        assert!(is_known_model("gemini-2.5-flash"));
        assert!(is_known_model("vertex:gemini-2.5-flash"));
        assert_eq!(
            stream_generate_content_api_url("us-central1", "proj-123", "gemini-2.5-flash"),
            "https://us-central1-aiplatform.googleapis.com/v1/projects/proj-123/locations/us-central1/publishers/google/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
        );
    }
}
