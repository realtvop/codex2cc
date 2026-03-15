use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{anyhow, Context};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;
use tokio::sync::Mutex;
use tracing::warn;

use crate::config::UpstreamConfig;

const OPENID_CONFIG_URL: &str = "https://auth.openai.com/.well-known/openid-configuration";
const TOKEN_REFRESH_SKEW: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CredentialSource {
    ConfigApiKey,
    LocalCodexApiKey,
    LocalCodexAccessToken,
}

impl CredentialSource {
    pub fn is_local(&self) -> bool {
        matches!(self, Self::LocalCodexApiKey | Self::LocalCodexAccessToken)
    }
}

#[derive(Debug, Clone)]
pub struct ResolvedCredential {
    pub token: String,
    pub source: CredentialSource,
}

pub struct UpstreamAuthManager {
    config: UpstreamConfig,
    client: Client,
    state: Mutex<AuthState>,
}

#[derive(Default)]
struct AuthState {
    cached: Option<CachedCredential>,
    discovered_token_endpoint: Option<String>,
}

#[derive(Clone)]
struct CachedCredential {
    token: String,
    source: CredentialSource,
    expires_at: Option<SystemTime>,
    modified: Option<SystemTime>,
}

#[derive(Debug, Clone)]
struct LocalCodexAuthData {
    raw: Value,
    openai_api_key: Option<String>,
    access_token: Option<String>,
    refresh_token: Option<String>,
    id_token: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenIdConfiguration {
    token_endpoint: String,
}

#[derive(Debug, Deserialize)]
struct RefreshTokenResponse {
    access_token: String,
    refresh_token: Option<String>,
    id_token: Option<String>,
}

#[derive(Debug, Deserialize)]
struct JwtClaims {
    #[serde(default)]
    exp: Option<u64>,
    #[serde(default)]
    aud: Option<AudienceClaim>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum AudienceClaim {
    One(String),
    Many(Vec<String>),
}

impl UpstreamAuthManager {
    pub fn new(config: UpstreamConfig, client: Client) -> Arc<Self> {
        Arc::new(Self {
            config,
            client,
            state: Mutex::new(AuthState::default()),
        })
    }

    pub async fn resolve_primary(&self) -> ResolvedCredential {
        let configured = self.configured_credential();
        if !self.config.prefer_local_codex_credentials {
            return configured;
        }

        match self.resolve_local(false).await {
            Ok(Some(credential)) => credential,
            Ok(None) => configured,
            Err(err) => {
                warn!(error = %err, "failed resolving local Codex credentials; falling back to configured API key");
                configured
            }
        }
    }

    pub async fn resolve_after_token_unauthorized(&self) -> Option<ResolvedCredential> {
        if !self.config.prefer_local_codex_credentials {
            return None;
        }

        self.invalidate_local_cache().await;
        match self.resolve_local(true).await {
            Ok(Some(credential))
                if credential.source == CredentialSource::LocalCodexAccessToken =>
            {
                Some(credential)
            }
            Ok(_) => None,
            Err(err) => {
                warn!(error = %err, "failed refreshing local Codex access token after upstream auth rejection");
                None
            }
        }
    }

    pub fn configured_api_key_fallback(
        &self,
        failed: &ResolvedCredential,
    ) -> Option<ResolvedCredential> {
        let configured = self.configured_credential();
        if configured.token.is_empty()
            || failed.source == CredentialSource::ConfigApiKey
            || configured.token == failed.token
        {
            None
        } else {
            Some(configured)
        }
    }

    async fn invalidate_local_cache(&self) {
        let mut state = self.state.lock().await;
        state.cached = None;
    }

    fn configured_credential(&self) -> ResolvedCredential {
        ResolvedCredential {
            token: self.config.api_key.clone(),
            source: CredentialSource::ConfigApiKey,
        }
    }

    async fn resolve_local(
        &self,
        force_refresh: bool,
    ) -> anyhow::Result<Option<ResolvedCredential>> {
        let path = expand_home(&self.config.local_codex_auth_path);
        let modified = file_modified_time(&path);

        if !force_refresh {
            if let Some(cached) = self.cached_credential(modified).await {
                return Ok(Some(cached));
            }
        }

        let mut auth = match read_local_auth_file(&path) {
            Ok(auth) => auth,
            Err(err) => {
                warn!(path = %path.display(), error = %err, "failed reading local Codex auth file");
                return Ok(None);
            }
        };

        if let Some(api_key) = auth
            .openai_api_key
            .clone()
            .filter(|value| !value.is_empty())
        {
            let credential = ResolvedCredential {
                token: api_key.clone(),
                source: CredentialSource::LocalCodexApiKey,
            };
            self.set_cached(Some(CachedCredential {
                token: api_key,
                source: CredentialSource::LocalCodexApiKey,
                expires_at: None,
                modified,
            }))
            .await;
            return Ok(Some(credential));
        }

        let access_token = auth.access_token.clone().filter(|value| !value.is_empty());
        let expires_at = access_token
            .as_deref()
            .and_then(jwt_expiry)
            .or_else(|| auth.id_token.as_deref().and_then(jwt_expiry));

        if !force_refresh {
            if let Some(token) = access_token.clone() {
                if !expires_soon(expires_at) {
                    let credential = ResolvedCredential {
                        token: token.clone(),
                        source: CredentialSource::LocalCodexAccessToken,
                    };
                    self.set_cached(Some(CachedCredential {
                        token,
                        source: CredentialSource::LocalCodexAccessToken,
                        expires_at,
                        modified,
                    }))
                    .await;
                    return Ok(Some(credential));
                }
            }
        }

        if self.config.refresh_local_codex_tokens {
            if let Some(refreshed) = self.refresh_and_persist(&path, &mut auth).await? {
                return Ok(Some(refreshed));
            }
        }

        Ok(None)
    }

    async fn cached_credential(&self, modified: Option<SystemTime>) -> Option<ResolvedCredential> {
        let state = self.state.lock().await;
        let cached = state.cached.as_ref()?;
        if cached.modified != modified {
            return None;
        }
        if expires_soon(cached.expires_at) {
            return None;
        }
        Some(ResolvedCredential {
            token: cached.token.clone(),
            source: cached.source.clone(),
        })
    }

    async fn set_cached(&self, cached: Option<CachedCredential>) {
        let mut state = self.state.lock().await;
        state.cached = cached;
    }

    async fn refresh_and_persist(
        &self,
        path: &Path,
        auth: &mut LocalCodexAuthData,
    ) -> anyhow::Result<Option<ResolvedCredential>> {
        let refresh_token = auth
            .refresh_token
            .clone()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow!("local Codex auth file does not contain refresh_token"))?;
        let client_id = self
            .config
            .local_codex_oauth_client_id
            .clone()
            .or_else(|| auth.id_token.as_deref().and_then(jwt_audience))
            .ok_or_else(|| anyhow!("unable to determine local Codex OAuth client_id"))?;
        let token_endpoint = self.resolve_token_endpoint().await?;

        let response = self
            .client
            .post(&token_endpoint)
            .form(&[
                ("grant_type", "refresh_token"),
                ("refresh_token", refresh_token.as_str()),
                ("client_id", client_id.as_str()),
            ])
            .send()
            .await
            .context("failed sending refresh token request")?;
        let status = response.status();
        if !status.is_success() {
            return Err(anyhow!("refresh token request returned status {status}"));
        }

        let refreshed = response
            .json::<RefreshTokenResponse>()
            .await
            .context("failed parsing refresh token response")?;
        update_local_auth_json(auth, &refreshed)?;
        atomic_write_json(path, &auth.raw)?;

        let modified = file_modified_time(path);
        let access_token = refreshed.access_token;
        let id_token = refreshed.id_token.or_else(|| auth.id_token.clone());
        let expires_at =
            jwt_expiry(&access_token).or_else(|| id_token.as_deref().and_then(jwt_expiry));
        let credential = ResolvedCredential {
            token: access_token.clone(),
            source: CredentialSource::LocalCodexAccessToken,
        };

        self.set_cached(Some(CachedCredential {
            token: access_token,
            source: CredentialSource::LocalCodexAccessToken,
            expires_at,
            modified,
        }))
        .await;

        Ok(Some(credential))
    }

    async fn resolve_token_endpoint(&self) -> anyhow::Result<String> {
        if let Some(endpoint) = self.config.local_codex_oauth_token_endpoint.clone() {
            return Ok(endpoint);
        }

        {
            let state = self.state.lock().await;
            if let Some(endpoint) = state.discovered_token_endpoint.clone() {
                return Ok(endpoint);
            }
        }

        let discovered = self
            .client
            .get(OPENID_CONFIG_URL)
            .send()
            .await
            .context("failed fetching OpenID configuration")?;
        let status = discovered.status();
        if !status.is_success() {
            return Err(anyhow!(
                "OpenID configuration request returned status {status}"
            ));
        }

        let config = discovered
            .json::<OpenIdConfiguration>()
            .await
            .context("failed parsing OpenID configuration")?;
        let endpoint = config.token_endpoint;

        let mut state = self.state.lock().await;
        state.discovered_token_endpoint = Some(endpoint.clone());
        Ok(endpoint)
    }
}

fn read_local_auth_file(path: &Path) -> anyhow::Result<LocalCodexAuthData> {
    let contents =
        fs::read_to_string(path).with_context(|| format!("failed reading {}", path.display()))?;
    let raw: Value = serde_json::from_str(&contents)
        .with_context(|| format!("failed parsing {}", path.display()))?;
    let (openai_api_key, access_token, refresh_token, id_token) = {
        let object = raw
            .as_object()
            .ok_or_else(|| anyhow!("local Codex auth file root must be a JSON object"))?;
        let tokens = object.get("tokens").and_then(Value::as_object);

        (
            object
                .get("OPENAI_API_KEY")
                .and_then(Value::as_str)
                .map(str::to_string),
            tokens
                .and_then(|tokens| tokens.get("access_token"))
                .and_then(Value::as_str)
                .map(str::to_string),
            tokens
                .and_then(|tokens| tokens.get("refresh_token"))
                .and_then(Value::as_str)
                .map(str::to_string),
            tokens
                .and_then(|tokens| tokens.get("id_token"))
                .and_then(Value::as_str)
                .map(str::to_string),
        )
    };

    Ok(LocalCodexAuthData {
        raw,
        openai_api_key,
        access_token,
        refresh_token,
        id_token,
    })
}

fn update_local_auth_json(
    auth: &mut LocalCodexAuthData,
    refreshed: &RefreshTokenResponse,
) -> anyhow::Result<()> {
    let object = auth
        .raw
        .as_object_mut()
        .ok_or_else(|| anyhow!("local Codex auth file root must be a JSON object"))?;
    let tokens = object
        .entry("tokens")
        .or_insert_with(|| Value::Object(Default::default()))
        .as_object_mut()
        .ok_or_else(|| anyhow!("local Codex auth file tokens field must be an object"))?;

    tokens.insert(
        "access_token".to_string(),
        Value::String(refreshed.access_token.clone()),
    );
    auth.access_token = Some(refreshed.access_token.clone());

    if let Some(refresh_token) = refreshed.refresh_token.clone() {
        tokens.insert(
            "refresh_token".to_string(),
            Value::String(refresh_token.clone()),
        );
        auth.refresh_token = Some(refresh_token);
    }

    if let Some(id_token) = refreshed.id_token.clone() {
        tokens.insert("id_token".to_string(), Value::String(id_token.clone()));
        auth.id_token = Some(id_token);
    }

    Ok(())
}

fn atomic_write_json(path: &Path, value: &Value) -> anyhow::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).with_context(|| format!("failed creating {}", parent.display()))?;

    let contents = serde_json::to_vec_pretty(value)
        .context("failed serializing updated local Codex auth file")?;
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("auth.json");
    let temp_path = parent.join(format!(".{file_name}.{unique}.tmp"));

    fs::write(&temp_path, contents)
        .with_context(|| format!("failed writing {}", temp_path.display()))?;
    if let Ok(metadata) = fs::metadata(path) {
        let _ = fs::set_permissions(&temp_path, metadata.permissions());
    }

    if let Err(err) = fs::rename(&temp_path, path) {
        let _ = fs::remove_file(&temp_path);
        return Err(err).with_context(|| format!("failed replacing {}", path.display()));
    }

    Ok(())
}

fn expand_home(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    PathBuf::from(path)
}

fn file_modified_time(path: &Path) -> Option<SystemTime> {
    fs::metadata(path)
        .ok()
        .and_then(|metadata| metadata.modified().ok())
}

fn jwt_expiry(token: &str) -> Option<SystemTime> {
    let claims = decode_jwt_claims(token).ok()?;
    claims
        .exp
        .map(|seconds| UNIX_EPOCH + Duration::from_secs(seconds))
}

fn jwt_audience(token: &str) -> Option<String> {
    let claims = decode_jwt_claims(token).ok()?;
    match claims.aud? {
        AudienceClaim::One(value) => Some(value),
        AudienceClaim::Many(values) => values.into_iter().next(),
    }
}

fn decode_jwt_claims(token: &str) -> anyhow::Result<JwtClaims> {
    let payload = token
        .split('.')
        .nth(1)
        .ok_or_else(|| anyhow!("JWT is missing payload segment"))?;
    let decoded = URL_SAFE_NO_PAD
        .decode(payload.as_bytes())
        .context("failed decoding JWT payload")?;
    serde_json::from_slice::<JwtClaims>(&decoded).context("failed parsing JWT payload JSON")
}

fn expires_soon(expires_at: Option<SystemTime>) -> bool {
    match expires_at {
        Some(expires_at) => {
            let threshold = SystemTime::now() + TOKEN_REFRESH_SKEW;
            expires_at <= threshold
        }
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn sample_config(api_key: &str) -> UpstreamConfig {
        UpstreamConfig {
            base_url: "https://api.openai.com/v1".to_string(),
            api_key: api_key.to_string(),
            prefer_local_codex_credentials: false,
            local_codex_auth_path: "~/.codex/auth.json".to_string(),
            refresh_local_codex_tokens: true,
            local_codex_oauth_client_id: None,
            local_codex_oauth_token_endpoint: None,
        }
    }

    fn temp_path(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "codextocc-{label}-{}-{unique}.json",
            std::process::id()
        ))
    }

    fn write_temp_file(label: &str, contents: &str) -> PathBuf {
        let path = temp_path(label);
        fs::write(&path, contents).expect("failed to write temp file");
        path
    }

    fn encode_jwt(claims: Value) -> String {
        let payload = URL_SAFE_NO_PAD.encode(claims.to_string().as_bytes());
        format!("header.{payload}.signature")
    }

    #[test]
    fn credential_source_identifies_local_variants() {
        assert!(!CredentialSource::ConfigApiKey.is_local());
        assert!(CredentialSource::LocalCodexApiKey.is_local());
        assert!(CredentialSource::LocalCodexAccessToken.is_local());
    }

    #[test]
    fn configured_api_key_fallback_only_returns_distinct_configured_token() {
        let manager = UpstreamAuthManager::new(sample_config("configured"), Client::new());

        let fallback = manager.configured_api_key_fallback(&ResolvedCredential {
            token: "local-token".to_string(),
            source: CredentialSource::LocalCodexAccessToken,
        });
        let fallback = fallback.expect("expected configured key fallback");
        assert_eq!(fallback.token, "configured");
        assert_eq!(fallback.source, CredentialSource::ConfigApiKey);

        assert!(manager
            .configured_api_key_fallback(&ResolvedCredential {
                token: "configured".to_string(),
                source: CredentialSource::LocalCodexApiKey,
            })
            .is_none());
        assert!(manager
            .configured_api_key_fallback(&ResolvedCredential {
                token: "configured".to_string(),
                source: CredentialSource::ConfigApiKey,
            })
            .is_none());

        let no_config = UpstreamAuthManager::new(sample_config(""), Client::new());
        assert!(no_config
            .configured_api_key_fallback(&ResolvedCredential {
                token: "local-token".to_string(),
                source: CredentialSource::LocalCodexAccessToken,
            })
            .is_none());
    }

    #[test]
    fn jwt_helpers_decode_expiry_and_audience() {
        let token = encode_jwt(json!({
            "exp": 1_700_000_000u64,
            "aud": ["client-id", "other-client"]
        }));

        assert_eq!(
            jwt_expiry(&token),
            Some(UNIX_EPOCH + Duration::from_secs(1_700_000_000))
        );
        assert_eq!(jwt_audience(&token), Some("client-id".to_string()));
    }

    #[test]
    fn jwt_helpers_reject_invalid_tokens() {
        assert!(jwt_expiry("missing-payload").is_none());
        assert!(jwt_audience("header.not-base64.signature").is_none());
        assert!(decode_jwt_claims("header.not-base64.signature").is_err());
    }

    #[test]
    fn read_local_auth_file_parses_expected_fields() {
        let path = write_temp_file(
            "read-local-auth",
            r#"{
                "OPENAI_API_KEY": "sk-local",
                "tokens": {
                    "access_token": "access",
                    "refresh_token": "refresh",
                    "id_token": "id-token"
                }
            }"#,
        );

        let auth = read_local_auth_file(&path).expect("expected auth file to parse");
        assert_eq!(auth.openai_api_key.as_deref(), Some("sk-local"));
        assert_eq!(auth.access_token.as_deref(), Some("access"));
        assert_eq!(auth.refresh_token.as_deref(), Some("refresh"));
        assert_eq!(auth.id_token.as_deref(), Some("id-token"));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn read_local_auth_file_rejects_invalid_json_and_root_shape() {
        let malformed = write_temp_file("malformed-auth", "{");
        let malformed_err = read_local_auth_file(&malformed).expect_err("expected parse failure");
        assert!(malformed_err.to_string().contains("failed parsing"));
        let _ = fs::remove_file(malformed);

        let wrong_root = write_temp_file("wrong-root-auth", "[]");
        let wrong_root_err =
            read_local_auth_file(&wrong_root).expect_err("expected object root failure");
        assert!(wrong_root_err
            .to_string()
            .contains("local Codex auth file root must be a JSON object"));
        let _ = fs::remove_file(wrong_root);
    }

    #[test]
    fn update_local_auth_json_preserves_missing_optional_tokens() {
        let mut auth = LocalCodexAuthData {
            raw: json!({
                "tokens": {
                    "access_token": "old-access",
                    "refresh_token": "old-refresh",
                    "id_token": "old-id"
                }
            }),
            openai_api_key: None,
            access_token: Some("old-access".to_string()),
            refresh_token: Some("old-refresh".to_string()),
            id_token: Some("old-id".to_string()),
        };

        update_local_auth_json(
            &mut auth,
            &RefreshTokenResponse {
                access_token: "new-access".to_string(),
                refresh_token: None,
                id_token: None,
            },
        )
        .expect("expected auth JSON update to succeed");

        assert_eq!(auth.access_token.as_deref(), Some("new-access"));
        assert_eq!(auth.refresh_token.as_deref(), Some("old-refresh"));
        assert_eq!(auth.id_token.as_deref(), Some("old-id"));
        assert_eq!(auth.raw["tokens"]["access_token"], "new-access");
        assert_eq!(auth.raw["tokens"]["refresh_token"], "old-refresh");
        assert_eq!(auth.raw["tokens"]["id_token"], "old-id");
    }

    #[test]
    fn update_local_auth_json_overwrites_optional_tokens_when_present() {
        let mut auth = LocalCodexAuthData {
            raw: json!({"tokens": {}}),
            openai_api_key: None,
            access_token: None,
            refresh_token: None,
            id_token: None,
        };

        update_local_auth_json(
            &mut auth,
            &RefreshTokenResponse {
                access_token: "new-access".to_string(),
                refresh_token: Some("new-refresh".to_string()),
                id_token: Some("new-id".to_string()),
            },
        )
        .expect("expected auth JSON update to succeed");

        assert_eq!(auth.access_token.as_deref(), Some("new-access"));
        assert_eq!(auth.refresh_token.as_deref(), Some("new-refresh"));
        assert_eq!(auth.id_token.as_deref(), Some("new-id"));
        assert_eq!(auth.raw["tokens"]["access_token"], "new-access");
        assert_eq!(auth.raw["tokens"]["refresh_token"], "new-refresh");
        assert_eq!(auth.raw["tokens"]["id_token"], "new-id");
    }
}
