use std::{
    collections::{HashMap, HashSet},
    env, fs,
    path::Path,
};

use anyhow::Context;
use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub api_keys: HashSet<String>,
    pub upstream: UpstreamConfig,
    pub model_map: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone)]
pub struct UpstreamConfig {
    pub base_url: String,
    pub api_key: String,
    pub prefer_local_codex_credentials: bool,
    pub local_codex_auth_path: String,
    pub refresh_local_codex_tokens: bool,
    pub local_codex_oauth_client_id: Option<String>,
    pub local_codex_oauth_token_endpoint: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct RawConfig {
    server: Option<RawServerConfig>,
    api_keys: Option<Vec<String>>,
    upstream: Option<RawUpstreamConfig>,
    model_map: Option<HashMap<String, String>>,
}

#[derive(Debug, Deserialize, Default)]
struct RawServerConfig {
    host: Option<String>,
    port: Option<u16>,
}

#[derive(Debug, Deserialize, Default)]
struct RawUpstreamConfig {
    base_url: Option<String>,
    api_key: Option<String>,
    prefer_local_codex_credentials: Option<bool>,
    local_codex_auth_path: Option<String>,
    refresh_local_codex_tokens: Option<bool>,
    local_codex_oauth_client_id: Option<String>,
    local_codex_oauth_token_endpoint: Option<String>,
}

impl AppConfig {
    pub fn load() -> anyhow::Result<Self> {
        let config_path = env::var("CONFIG_PATH").unwrap_or_else(|_| "config.yaml".to_string());
        let raw = if Path::new(&config_path).is_file() {
            let contents = fs::read_to_string(&config_path)
                .with_context(|| format!("failed reading config file: {config_path}"))?;
            serde_yaml::from_str::<RawConfig>(&contents)
                .with_context(|| format!("failed parsing config file: {config_path}"))?
        } else {
            RawConfig::default()
        };

        let server = raw.server.unwrap_or_default();
        let upstream = raw.upstream.unwrap_or_default();

        Ok(Self {
            server: ServerConfig {
                host: server.host.unwrap_or_else(|| "0.0.0.0".to_string()),
                port: server.port.unwrap_or(8082),
            },
            api_keys: raw.api_keys.unwrap_or_default().into_iter().collect(),
            upstream: UpstreamConfig {
                base_url: env::var("OPENAI_BASE_URL")
                    .ok()
                    .or(upstream.base_url)
                    .unwrap_or_else(|| "https://api.openai.com/v1".to_string()),
                api_key: env::var("OPENAI_API_KEY")
                    .ok()
                    .or(upstream.api_key)
                    .unwrap_or_default(),
                prefer_local_codex_credentials: env_bool("PREFER_LOCAL_CODEX_CREDENTIALS")
                    .or(upstream.prefer_local_codex_credentials)
                    .unwrap_or(false),
                local_codex_auth_path: env::var("LOCAL_CODEX_AUTH_PATH")
                    .ok()
                    .or(upstream.local_codex_auth_path)
                    .unwrap_or_else(|| "~/.codex/auth.json".to_string()),
                refresh_local_codex_tokens: env_bool("REFRESH_LOCAL_CODEX_TOKENS")
                    .or(upstream.refresh_local_codex_tokens)
                    .unwrap_or(true),
                local_codex_oauth_client_id: env::var("LOCAL_CODEX_OAUTH_CLIENT_ID")
                    .ok()
                    .or(upstream.local_codex_oauth_client_id)
                    .filter(|value| !value.is_empty()),
                local_codex_oauth_token_endpoint: env::var("LOCAL_CODEX_OAUTH_TOKEN_ENDPOINT")
                    .ok()
                    .or(upstream.local_codex_oauth_token_endpoint)
                    .filter(|value| !value.is_empty()),
            },
            model_map: raw.model_map.unwrap_or_default(),
        })
    }

    pub fn auth_enabled(&self) -> bool {
        !self.api_keys.is_empty()
    }
}

fn env_bool(name: &str) -> Option<bool> {
    env::var(name).ok().and_then(|value| parse_bool(&value))
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}
