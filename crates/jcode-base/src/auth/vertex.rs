use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::io::{self, Write};

const GOOGLE_AUTHORIZE_URL: &str = "https://accounts.google.com/o/oauth2/v2/auth";
const GOOGLE_USERINFO_URL: &str = "https://www.googleapis.com/oauth2/v1/userinfo?alt=json";
pub const DEFAULT_PORT: u16 = 51122;
const LOOPBACK_HOST: &str = "127.0.0.1";
const REDIRECT_PATH: &str = "/oauth-callback";
pub const DEFAULT_LOCATION: &str = "us-central1";

const VERTEX_CLIENT_ID: &str =
    "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com"; // gitleaks:allow
const VERTEX_CLIENT_SECRET: &str = "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"; // gitleaks:allow
const CLIENT_ID_ENV: &str = "VERTEX_CLIENT_ID";
const CLIENT_SECRET_ENV: &str = "VERTEX_CLIENT_SECRET";

const VERTEX_SCOPES: &[&str] = &[
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
];

fn vertex_client_id() -> String {
    std::env::var(CLIENT_ID_ENV)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| VERTEX_CLIENT_ID.to_string())
}

fn vertex_client_secret() -> String {
    std::env::var(CLIENT_SECRET_ENV)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| VERTEX_CLIENT_SECRET.to_string())
}

pub fn fetch_location() -> String {
    std::env::var("VERTEX_LOCATION")
        .or_else(|_| std::env::var("GOOGLE_CLOUD_LOCATION"))
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| DEFAULT_LOCATION.to_string())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexTokens {
    pub access_token: String,
    pub refresh_token: String,
    pub expires_at: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<String>,
}

impl VertexTokens {
    pub fn is_expired(&self) -> bool {
        let now_ms = chrono::Utc::now().timestamp_millis();
        self.expires_at <= now_ms + 60_000
    }
}

#[derive(Debug, Deserialize)]
struct GoogleUserInfo {
    email: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LoadCodeAssistResponse {
    #[serde(default)]
    cloudaicompanion_project: Option<serde_json::Value>,
}

pub fn tokens_path() -> Result<std::path::PathBuf> {
    Ok(crate::storage::jcode_dir()?.join("vertex_oauth.json"))
}

pub fn load_tokens() -> Result<VertexTokens> {
    let path = tokens_path()?;
    if path.exists() {
        crate::storage::harden_secret_file_permissions(&path);
        return crate::storage::read_json(&path).map_err(|_| {
            anyhow::anyhow!("No Vertex AI tokens found. Run `jcode login --provider vertex`.")
        });
    }

    anyhow::bail!("No Vertex AI tokens found. Run `jcode login --provider vertex`.");
}

pub fn save_tokens(tokens: &VertexTokens) -> Result<()> {
    let path = tokens_path()?;
    crate::storage::write_json_secret(&path, tokens)?;
    super::AuthStatus::invalidate_cache();
    Ok(())
}

pub async fn load_or_refresh_tokens() -> Result<VertexTokens> {
    let tokens = load_tokens()?;
    if tokens.is_expired() {
        return refresh_tokens(&tokens).await;
    }
    Ok(tokens)
}

pub async fn refresh_tokens(tokens: &VertexTokens) -> Result<VertexTokens> {
    crate::auth::refresh_coordinator::single_flight(
        "vertex".to_string(),
        || load_tokens().ok(),
        |stored: &VertexTokens| !stored.is_expired(),
        {
            let observed = tokens.clone();
            move |stored: Option<VertexTokens>| async move {
                let source = stored.unwrap_or(observed);
                refresh_tokens_uncoordinated(&source).await
            }
        },
    )
    .await
}

async fn refresh_tokens_uncoordinated(tokens: &VertexTokens) -> Result<VertexTokens> {
    let refreshed = super::google_oauth::refresh_access_token(
        "Vertex AI",
        &vertex_client_id(),
        &vertex_client_secret(),
        &tokens.refresh_token,
        Some("jcode"),
    )
    .await?;

    let mut new_tokens = VertexTokens {
        access_token: refreshed.access_token,
        refresh_token: refreshed.refresh_token,
        expires_at: refreshed.expires_at_ms,
        email: tokens.email.clone(),
        project_id: tokens.project_id.clone(),
        location: tokens.location.clone().or_else(|| Some(fetch_location())),
    };

    if new_tokens.project_id.is_none() {
        new_tokens.project_id = fetch_project_id(&new_tokens.access_token).await.ok();
    }

    save_tokens(&new_tokens)?;
    Ok(new_tokens)
}

pub fn redirect_uri(port: u16) -> String {
    format!("http://{LOOPBACK_HOST}:{port}{REDIRECT_PATH}")
}

pub fn build_auth_url(redirect_uri: &str, code_challenge: &str, state: &str) -> Result<String> {
    let mut url = reqwest::Url::parse(GOOGLE_AUTHORIZE_URL)?;
    url.query_pairs_mut()
        .append_pair("client_id", &vertex_client_id())
        .append_pair("redirect_uri", redirect_uri)
        .append_pair("response_type", "code")
        .append_pair("scope", &VERTEX_SCOPES.join(" "))
        .append_pair("code_challenge", code_challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("state", state)
        .append_pair("access_type", "offline")
        .append_pair("prompt", "consent");
    Ok(url.to_string())
}

pub async fn exchange_code(
    code: &str,
    code_verifier: &str,
    redirect_uri: &str,
) -> Result<VertexTokens> {
    let client = crate::provider::shared_http_client();
    let resp = client
        .post(super::google_oauth::GOOGLE_TOKEN_URL)
        .form(&[
            ("grant_type", "authorization_code"),
            ("client_id", &vertex_client_id()),
            ("client_secret", &vertex_client_secret()),
            ("code", code),
            ("code_verifier", code_verifier),
            ("redirect_uri", redirect_uri),
        ])
        .send()
        .await
        .context("Failed to exchange Vertex AI authorization code")?;

    if !resp.status().is_success() {
        let body = crate::util::http_error_body(resp, "HTTP error").await;
        anyhow::bail!("Vertex AI token exchange failed: {}", body.trim());
    }

    #[derive(Deserialize)]
    struct TokenResp {
        access_token: String,
        refresh_token: Option<String>,
        expires_in: i64,
    }

    let token_resp: TokenResp = resp
        .json()
        .await
        .context("Failed to parse Vertex AI token exchange response")?;

    let email = fetch_user_email(&token_resp.access_token).await.ok();
    let project_id = fetch_project_id(&token_resp.access_token).await.ok();

    let tokens = VertexTokens {
        access_token: token_resp.access_token,
        refresh_token: token_resp.refresh_token.unwrap_or_default(),
        expires_at: chrono::Utc::now().timestamp_millis() + (token_resp.expires_in * 1000),
        email,
        project_id,
        location: Some(fetch_location()),
    };

    save_tokens(&tokens)?;
    Ok(tokens)
}

pub async fn exchange_callback_input(
    verifier: &str,
    input: &str,
    expected_state: Option<&str>,
    redirect_uri: &str,
) -> Result<VertexTokens> {
    let code = if let Some(expected_state) = expected_state {
        let (code, callback_state) = crate::auth::oauth::parse_callback_input_with_state(input)?;
        if callback_state != expected_state {
            anyhow::bail!(
                "OAuth state mismatch. Start Vertex AI login again and use the latest callback URL."
            );
        }
        code
    } else {
        input.trim().to_string()
    };
    exchange_code(&code, verifier, redirect_uri).await
}

pub async fn exchange_callback_code(
    code: &str,
    verifier: &str,
    redirect_uri: &str,
) -> Result<VertexTokens> {
    exchange_code(code, verifier, redirect_uri).await
}

pub async fn fetch_user_email(access_token: &str) -> Result<String> {
    let client = crate::provider::shared_http_client();
    let resp = client
        .get(GOOGLE_USERINFO_URL)
        .bearer_auth(access_token)
        .send()
        .await
        .context("Failed to fetch Vertex AI Google profile")?;

    if !resp.status().is_success() {
        let body = crate::util::http_error_body(resp, "HTTP error").await;
        anyhow::bail!("Failed to fetch Vertex AI Google profile: {}", body.trim());
    }

    let info: GoogleUserInfo = resp.json().await?;
    info.email
        .filter(|e| !e.trim().is_empty())
        .context("No email in Google profile")
}

pub async fn fetch_project_id(access_token: &str) -> Result<String> {
    if let Ok(project) = std::env::var("VERTEX_PROJECT_ID").or_else(|_| std::env::var("GOOGLE_CLOUD_PROJECT"))
        && !project.trim().is_empty()
    {
        return Ok(project.trim().to_string());
    }

    // Try loadCodeAssist to resolve default project
    let client = crate::provider::shared_http_client();
    let body = serde_json::json!({
        "metadata": {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI"
        }
    });

    if let Ok(resp) = client
        .post("https://cloudcode-pa.googleapis.com/v1internal:loadCodeAssist")
        .bearer_auth(access_token)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .json(&body)
        .send()
        .await
        && resp.status().is_success()
        && let Ok(parsed) = resp.json::<LoadCodeAssistResponse>().await
        && let Some(proj_val) = parsed.cloudaicompanion_project
    {
        if let Some(s) = proj_val.as_str() && !s.trim().is_empty() {
            return Ok(s.trim().to_string());
        }
        if let Some(id) = proj_val.get("id").and_then(|v| v.as_str()) && !id.trim().is_empty() {
            return Ok(id.trim().to_string());
        }
    }

    Ok("default".to_string())
}

pub async fn login(no_browser: bool) -> Result<VertexTokens> {
    let (verifier, challenge) = crate::auth::oauth::generate_pkce_public();
    let expected_state = crate::auth::oauth::generate_state_public();
    let port = DEFAULT_PORT;
    let redirect = redirect_uri(port);
    let auth_url = build_auth_url(&redirect, &challenge, &expected_state)?;

    if no_browser {
        println!("\nVertex AI login URL:\n\n{auth_url}\n");
        print!("Paste callback URL or code: ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        return exchange_callback_input(&verifier, input.trim(), Some(&expected_state), &redirect).await;
    }

    let listener = crate::auth::oauth::bind_callback_listener(port)?;
    println!("\nOpening browser for Vertex AI login...\n");
    let _ = crate::companion_client::open_url(&auth_url);

    let code = crate::auth::oauth::wait_for_callback_async_on_listener(listener, &expected_state).await?;
    exchange_code(&code, &verifier, &redirect).await
}
