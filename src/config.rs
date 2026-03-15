use std::{
    collections::{HashMap, HashSet},
    env, fs,
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

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

#[derive(Debug)]
pub enum ConfigLoad {
    Loaded(AppConfig),
    Generated { path: PathBuf },
}

#[derive(Debug, Deserialize, Serialize, Default)]
struct RawConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    server: Option<RawServerConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    api_keys: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    upstream: Option<RawUpstreamConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_map: Option<HashMap<String, String>>,
}

#[derive(Debug, Deserialize, Serialize, Default)]
struct RawServerConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    host: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    port: Option<u16>,
}

#[derive(Debug, Deserialize, Serialize, Default)]
struct RawUpstreamConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefer_local_codex_credentials: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    local_codex_auth_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    refresh_local_codex_tokens: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    local_codex_oauth_client_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    local_codex_oauth_token_endpoint: Option<String>,
}

impl AppConfig {
    pub fn load() -> anyhow::Result<ConfigLoad> {
        let config_path = env::var("CONFIG_PATH").unwrap_or_else(|_| "config.yaml".to_string());
        Self::load_from_path(Path::new(&config_path))
    }

    fn load_from_path(path: &Path) -> anyhow::Result<ConfigLoad> {
        if path.is_file() {
            return Ok(ConfigLoad::Loaded(Self::load_existing(path)?));
        }

        if path.exists() {
            bail!("config path exists but is not a file: {}", path.display());
        }

        generate_default_config(path)?;
        Ok(ConfigLoad::Generated {
            path: path.to_path_buf(),
        })
    }

    fn load_existing(path: &Path) -> anyhow::Result<Self> {
        let contents = fs::read_to_string(path)
            .with_context(|| format!("failed reading config file: {}", path.display()))?;
        let raw = serde_yaml::from_str::<RawConfig>(&contents)
            .with_context(|| format!("failed parsing config file: {}", path.display()))?;
        Ok(Self::from_raw(raw))
    }

    fn from_raw(raw: RawConfig) -> Self {
        let server = raw.server.unwrap_or_default();
        let upstream = raw.upstream.unwrap_or_default();

        Self {
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
        }
    }

    pub fn auth_enabled(&self) -> bool {
        !self.api_keys.is_empty()
    }
}

fn generate_default_config(path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = path.parent().filter(|dir| !dir.as_os_str().is_empty()) {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed creating parent directory for config file: {}",
                path.display()
            )
        })?;
    }

    let raw = RawConfig {
        server: Some(RawServerConfig {
            host: Some("0.0.0.0".to_string()),
            port: Some(8082),
        }),
        api_keys: Some(vec![generate_client_api_key()]),
        upstream: Some(RawUpstreamConfig {
            base_url: Some("https://api.openai.com/v1".to_string()),
            api_key: Some(String::new()),
            prefer_local_codex_credentials: Some(true),
            local_codex_auth_path: Some("~/.codex/auth.json".to_string()),
            refresh_local_codex_tokens: Some(true),
            local_codex_oauth_client_id: None,
            local_codex_oauth_token_endpoint: None,
        }),
        model_map: Some(HashMap::new()),
    };
    let yaml =
        serde_yaml::to_string(&raw).context("failed serializing generated default config")?;

    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("failed creating config file: {}", path.display()))?;
    file.write_all(yaml.as_bytes())
        .with_context(|| format!("failed writing config file: {}", path.display()))?;
    Ok(())
}

fn generate_client_api_key() -> String {
    Uuid::new_v4().simple().to_string()
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        collections::{HashMap, HashSet},
        fs,
    };

    fn sample_config(api_keys: &[&str]) -> AppConfig {
        AppConfig {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 8082,
            },
            api_keys: api_keys.iter().map(|key| (*key).to_string()).collect(),
            upstream: UpstreamConfig {
                base_url: "https://api.openai.com/v1".to_string(),
                api_key: "configured-key".to_string(),
                prefer_local_codex_credentials: false,
                local_codex_auth_path: "~/.codex/auth.json".to_string(),
                refresh_local_codex_tokens: true,
                local_codex_oauth_client_id: None,
                local_codex_oauth_token_endpoint: None,
            },
            model_map: HashMap::new(),
        }
    }

    struct TestWorkspace {
        root: PathBuf,
    }

    impl TestWorkspace {
        fn new() -> Self {
            let root =
                env::temp_dir().join(format!("codextocc-config-test-{}", Uuid::new_v4().simple()));
            fs::create_dir_all(&root).expect("failed to create test workspace");
            Self { root }
        }

        fn path(&self, relative: &str) -> PathBuf {
            self.root.join(relative)
        }
    }

    impl Drop for TestWorkspace {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn is_lower_hex_32(value: &str) -> bool {
        value.len() == 32
            && value
                .bytes()
                .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    }

    #[test]
    fn parse_bool_accepts_true_values() {
        for value in ["1", "true", "TRUE", " yes ", "On"] {
            assert_eq!(parse_bool(value), Some(true), "value: {value}");
        }
    }

    #[test]
    fn parse_bool_accepts_false_values() {
        for value in ["0", "false", "FALSE", " no ", "Off"] {
            assert_eq!(parse_bool(value), Some(false), "value: {value}");
        }
    }

    #[test]
    fn parse_bool_rejects_invalid_values() {
        for value in ["", "2", "truthy", "disabled"] {
            assert_eq!(parse_bool(value), None, "value: {value}");
        }
    }

    #[test]
    fn auth_enabled_reflects_api_keys_presence() {
        assert!(!sample_config(&[]).auth_enabled());
        assert!(sample_config(&["secret"]).auth_enabled());

        let config = sample_config(&["secret", "secret", "other"]);
        let expected = HashSet::from(["secret".to_string(), "other".to_string()]);
        assert_eq!(config.api_keys, expected);
    }

    #[test]
    fn load_from_path_generates_missing_config_and_creates_parent_dirs() {
        let workspace = TestWorkspace::new();
        let path = workspace.path("nested/config.yaml");

        let outcome = AppConfig::load_from_path(&path).expect("failed to generate config");
        let generated_path = match outcome {
            ConfigLoad::Generated { path } => path,
            ConfigLoad::Loaded(_) => panic!("expected generated config outcome"),
        };
        assert_eq!(generated_path, path);
        assert!(path.is_file());

        let contents = fs::read_to_string(&path).expect("failed reading generated config");
        let raw =
            serde_yaml::from_str::<RawConfig>(&contents).expect("failed parsing generated config");

        let api_keys = raw.api_keys.expect("missing api_keys");
        assert_eq!(api_keys.len(), 1);
        assert!(is_lower_hex_32(&api_keys[0]));

        let upstream = raw.upstream.expect("missing upstream");
        assert_eq!(
            upstream.base_url.as_deref(),
            Some("https://api.openai.com/v1")
        );
        assert_eq!(upstream.api_key.as_deref(), Some(""));
        assert_eq!(upstream.prefer_local_codex_credentials, Some(true));
        assert_eq!(
            upstream.local_codex_auth_path.as_deref(),
            Some("~/.codex/auth.json")
        );
        assert_eq!(upstream.refresh_local_codex_tokens, Some(true));

        assert!(raw.model_map.expect("missing model_map").is_empty());
    }

    #[test]
    fn load_from_path_reads_existing_config_without_rewriting() {
        let workspace = TestWorkspace::new();
        let path = workspace.path("config.yaml");
        let contents = r#"
server:
  host: "127.0.0.1"
  port: 9090
api_keys:
  - "secret"
upstream:
  base_url: "https://example.com/v1"
model_map:
  claude: "gpt"
"#;
        fs::write(&path, contents).expect("failed writing config fixture");

        let before = fs::read_to_string(&path).expect("failed reading config fixture");
        let outcome = AppConfig::load_from_path(&path).expect("failed loading config");
        let config = match outcome {
            ConfigLoad::Loaded(config) => config,
            ConfigLoad::Generated { .. } => panic!("expected loaded config outcome"),
        };

        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 9090);
        assert_eq!(config.api_keys, HashSet::from(["secret".to_string()]));
        assert_eq!(config.upstream.base_url, "https://example.com/v1");
        assert!(!config.upstream.prefer_local_codex_credentials);
        assert_eq!(
            config.model_map.get("claude").map(String::as_str),
            Some("gpt")
        );

        let after = fs::read_to_string(&path).expect("failed rereading config fixture");
        assert_eq!(before, after);
    }

    #[test]
    fn load_from_path_errors_for_non_file_paths() {
        let workspace = TestWorkspace::new();
        let path = workspace.path("config-dir");
        fs::create_dir_all(&path).expect("failed creating directory fixture");

        let err = AppConfig::load_from_path(&path).expect_err("expected non-file path error");
        assert!(
            err.to_string()
                .contains("config path exists but is not a file"),
            "unexpected error: {err}"
        );
    }
}
