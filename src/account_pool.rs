use std::{
    cmp::Ordering,
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{anyhow, bail, Context};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use reqwest::{header::HeaderMap, Client, StatusCode};
use serde::{Deserialize, Serialize};
use tokio::{sync::Mutex, time::sleep};
use uuid::Uuid;

use crate::config::{AccountPoolConfig, DEFAULT_CODEX_BASE_URL};

const TOKEN_REFRESH_SKEW: Duration = Duration::from_secs(60);
const DEVICE_CODE_GRANT_TYPE: &str = "urn:ietf:params:oauth:grant-type:device_code";
const DEFAULT_POLL_INTERVAL_SECS: u64 = 5;

#[derive(Debug, Clone)]
pub struct DeviceAuthorizationSession {
    pub device_code: String,
    pub user_code: String,
    pub verification_uri: String,
    pub verification_uri_complete: Option<String>,
    pub expires_at: SystemTime,
    pub interval_secs: u64,
}

#[derive(Debug, Clone)]
pub struct PoolCredential {
    pub id: String,
    pub access_token: String,
    pub account_id: Option<String>,
}

#[derive(Debug, Clone)]
pub enum AccountSelection {
    Selected(PoolCredential),
    Exhausted { reset_at: Option<SystemTime> },
}

#[derive(Debug, Clone)]
pub struct AccountSummary {
    pub id: String,
    pub alias: String,
    pub enabled: bool,
    pub remaining: Option<i64>,
    pub reset_at: Option<SystemTime>,
    pub expires_at: Option<SystemTime>,
    pub last_error: Option<String>,
    pub last_used_at: Option<SystemTime>,
}

#[derive(Debug, Clone)]
pub struct AccountPoolManager {
    config: AccountPoolConfig,
    client: Client,
    codex_base_url: String,
    gate: std::sync::Arc<Mutex<()>>,
}

#[derive(Debug, Deserialize)]
struct OpenIdConfiguration {
    token_endpoint: Option<String>,
    device_authorization_endpoint: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DeviceAuthorizationResponse {
    device_code: String,
    user_code: String,
    verification_uri: String,
    #[serde(default)]
    verification_uri_complete: Option<String>,
    expires_in: u64,
    #[serde(default)]
    interval: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct OAuthTokenResponse {
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default)]
    id_token: Option<String>,
    #[serde(default)]
    expires_in: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct OAuthErrorResponse {
    error: String,
    #[serde(default)]
    error_description: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct AccountPoolStore {
    #[serde(default)]
    accounts: Vec<AccountRecord>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct AccountRecord {
    id: String,
    alias: String,
    enabled: bool,
    tokens: AccountTokens,
    quota: AccountQuota,
    #[serde(default)]
    last_error: Option<String>,
    #[serde(default)]
    last_used_at: Option<u64>,
    created_at: u64,
    updated_at: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct AccountTokens {
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default)]
    id_token: Option<String>,
    #[serde(default)]
    account_id: Option<String>,
    #[serde(default)]
    expires_at: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct AccountQuota {
    #[serde(default)]
    remaining: Option<i64>,
    #[serde(default)]
    reset_at: Option<u64>,
    #[serde(default)]
    updated_at: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct JwtClaims {
    #[serde(default)]
    exp: Option<u64>,
    #[serde(default)]
    sub: Option<String>,
}

impl AccountPoolManager {
    pub fn new(
        config: AccountPoolConfig,
        codex_base_url: Option<String>,
        client: Client,
    ) -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self {
            config,
            client,
            codex_base_url: codex_base_url.unwrap_or_else(|| DEFAULT_CODEX_BASE_URL.to_string()),
            gate: std::sync::Arc::new(Mutex::new(())),
        })
    }

    pub fn quota_refresh_interval(&self) -> Duration {
        Duration::from_secs(self.config.quota_refresh_interval_secs.max(1))
    }

    pub async fn begin_device_login(&self) -> anyhow::Result<DeviceAuthorizationSession> {
        let _guard = self.gate.lock().await;
        let client_id = self
            .config
            .oauth_client_id
            .as_deref()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow!("upstream.account_pool.oauth.client_id is required"))?;
        let device_endpoint = self.resolve_device_authorization_endpoint().await?;
        let scopes = self.config.oauth_scopes.join(" ");

        let response = self
            .client
            .post(device_endpoint)
            .form(&[("client_id", client_id), ("scope", scopes.as_str())])
            .send()
            .await
            .context("failed sending device authorization request")?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            bail!(
                "device authorization request returned status {}: {}",
                status,
                truncate_error(&body)
            );
        }

        let payload = response
            .json::<DeviceAuthorizationResponse>()
            .await
            .context("failed parsing device authorization response")?;
        let now = SystemTime::now();
        Ok(DeviceAuthorizationSession {
            device_code: payload.device_code,
            user_code: payload.user_code,
            verification_uri: payload.verification_uri,
            verification_uri_complete: payload.verification_uri_complete,
            expires_at: now + Duration::from_secs(payload.expires_in),
            interval_secs: payload
                .interval
                .unwrap_or(DEFAULT_POLL_INTERVAL_SECS)
                .max(1),
        })
    }

    pub async fn complete_device_login(
        &self,
        session: &DeviceAuthorizationSession,
        alias: Option<String>,
    ) -> anyhow::Result<AccountSummary> {
        let _guard = self.gate.lock().await;
        let client_id = self
            .config
            .oauth_client_id
            .as_deref()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow!("upstream.account_pool.oauth.client_id is required"))?;
        let token_endpoint = self.resolve_token_endpoint().await?;
        let mut poll_interval = session.interval_secs.max(1);

        loop {
            if SystemTime::now() >= session.expires_at {
                bail!("device code authorization timed out");
            }

            let response = self
                .client
                .post(&token_endpoint)
                .form(&[
                    ("grant_type", DEVICE_CODE_GRANT_TYPE),
                    ("device_code", session.device_code.as_str()),
                    ("client_id", client_id),
                ])
                .send()
                .await
                .context("failed polling token endpoint for device login")?;

            if response.status().is_success() {
                let tokens = response
                    .json::<OAuthTokenResponse>()
                    .await
                    .context("failed parsing OAuth token response")?;
                let summary = self.add_account(alias.clone(), tokens)?;
                return Ok(summary);
            }

            let status = response.status();
            let body_text = response.text().await.unwrap_or_default();
            let error_payload = serde_json::from_str::<OAuthErrorResponse>(&body_text).ok();
            let code = error_payload
                .as_ref()
                .map(|payload| payload.error.as_str())
                .unwrap_or("");
            match code {
                "authorization_pending" => {
                    sleep(Duration::from_secs(poll_interval)).await;
                    continue;
                }
                "slow_down" => {
                    poll_interval += 5;
                    sleep(Duration::from_secs(poll_interval)).await;
                    continue;
                }
                "access_denied" => {
                    bail!("device login denied by user");
                }
                "expired_token" => {
                    bail!("device code has expired; run login again");
                }
                _ => {
                    let desc = error_payload
                        .and_then(|payload| payload.error_description)
                        .unwrap_or_else(|| truncate_error(&body_text));
                    bail!("device login polling failed with status {status}: {desc}");
                }
            }
        }
    }

    pub async fn list_accounts(&self) -> anyhow::Result<Vec<AccountSummary>> {
        let _guard = self.gate.lock().await;
        let store = self.read_store()?;
        Ok(store
            .accounts
            .iter()
            .map(account_summary_from_record)
            .collect())
    }

    pub async fn set_enabled(
        &self,
        account_ref: &str,
        enabled: bool,
    ) -> anyhow::Result<AccountSummary> {
        let _guard = self.gate.lock().await;
        let mut store = self.read_store()?;
        let now = now_epoch_seconds();
        let account = find_account_mut(&mut store.accounts, account_ref)?;
        account.enabled = enabled;
        account.updated_at = now;
        if !enabled {
            account.last_error = Some("disabled".to_string());
        } else {
            account.last_error = None;
        }
        let summary = account_summary_from_record(account);
        self.write_store(&store)?;
        Ok(summary)
    }

    pub async fn remove(&self, account_ref: &str) -> anyhow::Result<AccountSummary> {
        let _guard = self.gate.lock().await;
        let mut store = self.read_store()?;
        let index = find_account_index(&store.accounts, account_ref)?;
        let removed = store.accounts.remove(index);
        self.write_store(&store)?;
        Ok(account_summary_from_record(&removed))
    }

    pub async fn refresh_all_quotas(&self) -> anyhow::Result<()> {
        let _guard = self.gate.lock().await;
        let mut store = self.read_store()?;
        let mut changed = false;

        for account in store.accounts.iter_mut().filter(|record| record.enabled) {
            if self.ensure_access_token(account).await.is_err() {
                changed = true;
                continue;
            }
            match self.probe_account_quota(account).await {
                Ok(()) => changed = true,
                Err(err) => {
                    account.last_error = Some(err.to_string());
                    account.updated_at = now_epoch_seconds();
                    changed = true;
                }
            }
        }

        if changed {
            self.write_store(&store)?;
        }
        Ok(())
    }

    pub async fn select_account(
        &self,
        excluded_ids: &[String],
    ) -> anyhow::Result<AccountSelection> {
        let _guard = self.gate.lock().await;
        let mut store = self.read_store()?;
        let excluded: HashSet<&str> = excluded_ids.iter().map(String::as_str).collect();
        let now = SystemTime::now();
        let now_epoch = now_epoch_seconds();
        let mut changed = false;

        let mut candidate_indexes = Vec::new();
        for (index, account) in store.accounts.iter_mut().enumerate() {
            if !account.enabled || excluded.contains(account.id.as_str()) {
                continue;
            }
            if self.ensure_access_token(account).await.is_err() {
                changed = true;
                continue;
            }
            if account.tokens.access_token.is_empty() {
                continue;
            }
            candidate_indexes.push(index);
        }

        if candidate_indexes.is_empty() {
            if changed {
                self.write_store(&store)?;
            }
            return Ok(AccountSelection::Exhausted {
                reset_at: earliest_reset_at(&store.accounts),
            });
        }

        let mut active_indexes = Vec::new();
        for index in candidate_indexes {
            let class = account_quota_class(&store.accounts[index], now);
            if class <= 1 {
                active_indexes.push(index);
            }
        }
        if active_indexes.is_empty() {
            if changed {
                self.write_store(&store)?;
            }
            return Ok(AccountSelection::Exhausted {
                reset_at: earliest_reset_at(&store.accounts),
            });
        }

        active_indexes.sort_by(|left_index, right_index| {
            compare_accounts_for_selection(
                &store.accounts[*left_index],
                &store.accounts[*right_index],
                now,
            )
        });
        let selected_index = active_indexes[0];
        let selected = &mut store.accounts[selected_index];
        selected.last_used_at = Some(now_epoch);
        selected.last_error = None;
        selected.updated_at = now_epoch;
        changed = true;

        let credential = PoolCredential {
            id: selected.id.clone(),
            access_token: selected.tokens.access_token.clone(),
            account_id: selected.tokens.account_id.clone(),
        };

        if changed {
            self.write_store(&store)?;
        }
        Ok(AccountSelection::Selected(credential))
    }

    pub async fn update_quota_from_response(
        &self,
        account_id: &str,
        headers: &HeaderMap,
        status: StatusCode,
    ) -> anyhow::Result<()> {
        let _guard = self.gate.lock().await;
        let mut store = self.read_store()?;
        let account = match store
            .accounts
            .iter_mut()
            .find(|record| record.id == account_id)
        {
            Some(account) => account,
            None => return Ok(()),
        };

        let mut changed = false;
        if let Some(value) = header_value(headers, &self.config.quota_remaining_header)
            .and_then(parse_remaining_header)
        {
            account.quota.remaining = Some(value);
            changed = true;
        } else if status == StatusCode::TOO_MANY_REQUESTS {
            account.quota.remaining = Some(0);
            changed = true;
        }

        if let Some(reset_at) =
            header_value(headers, &self.config.quota_reset_header).and_then(parse_reset_header)
        {
            account.quota.reset_at = system_time_to_epoch_seconds(reset_at);
            changed = true;
        }

        if changed {
            account.quota.updated_at = Some(now_epoch_seconds());
        }

        if status.is_success() {
            account.last_error = None;
            changed = true;
        } else if status == StatusCode::UNAUTHORIZED
            || status == StatusCode::FORBIDDEN
            || status == StatusCode::TOO_MANY_REQUESTS
        {
            account.last_error = Some(format!("upstream returned status {}", status.as_u16()));
            account.updated_at = now_epoch_seconds();
            changed = true;
        }

        if changed {
            self.write_store(&store)?;
        }
        Ok(())
    }

    fn add_account(
        &self,
        alias: Option<String>,
        tokens: OAuthTokenResponse,
    ) -> anyhow::Result<AccountSummary> {
        let mut store = self.read_store()?;
        let now = now_epoch_seconds();
        let alias = resolve_alias(alias, &store.accounts);
        if store.accounts.iter().any(|account| account.alias == alias) {
            bail!("account alias already exists: {alias}");
        }

        let account_id = tokens
            .id_token
            .as_deref()
            .and_then(jwt_subject)
            .filter(|value| !value.is_empty());
        let expires_at = tokens
            .expires_in
            .map(|seconds| now.saturating_add(seconds))
            .or_else(|| jwt_expiry(&tokens.access_token).and_then(system_time_to_epoch_seconds))
            .or_else(|| {
                tokens
                    .id_token
                    .as_deref()
                    .and_then(jwt_expiry)
                    .and_then(system_time_to_epoch_seconds)
            });

        let account = AccountRecord {
            id: Uuid::new_v4().simple().to_string(),
            alias,
            enabled: true,
            tokens: AccountTokens {
                access_token: tokens.access_token,
                refresh_token: tokens.refresh_token,
                id_token: tokens.id_token,
                account_id,
                expires_at,
            },
            quota: AccountQuota::default(),
            last_error: None,
            last_used_at: None,
            created_at: now,
            updated_at: now,
        };
        let summary = account_summary_from_record(&account);
        store.accounts.push(account);
        self.write_store(&store)?;
        Ok(summary)
    }

    async fn ensure_access_token(&self, account: &mut AccountRecord) -> anyhow::Result<()> {
        if account.tokens.access_token.is_empty() {
            bail!("account token is empty");
        }

        if !expires_soon(
            account
                .tokens
                .expires_at
                .and_then(epoch_seconds_to_system_time),
        ) {
            return Ok(());
        }

        let refresh_token = account
            .tokens
            .refresh_token
            .clone()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow!("access token expired and refresh_token is unavailable"))?;
        let client_id = self
            .config
            .oauth_client_id
            .as_deref()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow!("upstream.account_pool.oauth.client_id is required"))?;
        let token_endpoint = self.resolve_token_endpoint().await?;

        let response = self
            .client
            .post(token_endpoint)
            .form(&[
                ("grant_type", "refresh_token"),
                ("refresh_token", refresh_token.as_str()),
                ("client_id", client_id),
            ])
            .send()
            .await
            .context("failed refreshing account pool token")?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            account.last_error = Some(format!(
                "refresh token request returned status {} ({})",
                status.as_u16(),
                truncate_error(&body)
            ));
            account.updated_at = now_epoch_seconds();
            bail!("failed refreshing account token");
        }

        let refreshed = response
            .json::<OAuthTokenResponse>()
            .await
            .context("failed parsing refresh token response")?;
        account.tokens.access_token = refreshed.access_token.clone();
        if let Some(refresh_token) = refreshed.refresh_token {
            account.tokens.refresh_token = Some(refresh_token);
        }
        if let Some(id_token) = refreshed.id_token.clone() {
            account.tokens.id_token = Some(id_token);
        }
        account.tokens.expires_at = refreshed
            .expires_in
            .map(|seconds| now_epoch_seconds().saturating_add(seconds))
            .or_else(|| jwt_expiry(&refreshed.access_token).and_then(system_time_to_epoch_seconds))
            .or_else(|| {
                account
                    .tokens
                    .id_token
                    .as_deref()
                    .and_then(jwt_expiry)
                    .and_then(system_time_to_epoch_seconds)
            });
        if account.tokens.account_id.is_none() {
            account.tokens.account_id = account.tokens.id_token.as_deref().and_then(jwt_subject);
        }
        account.last_error = None;
        account.updated_at = now_epoch_seconds();
        Ok(())
    }

    async fn probe_account_quota(&self, account: &mut AccountRecord) -> anyhow::Result<()> {
        let url = join_upstream_url(&self.codex_base_url, "/responses/input_tokens");
        let response = self
            .client
            .post(url)
            .headers(account_pool_headers(
                &account.tokens.access_token,
                account.tokens.account_id.as_deref(),
            ))
            .json(&serde_json::json!({
                "model": "gpt-4.1-mini",
                "input": "quota probe",
            }))
            .send()
            .await
            .context("quota probe request failed")?;
        let status = response.status();
        let headers = response.headers().clone();

        if let Some(remaining) = header_value(&headers, &self.config.quota_remaining_header)
            .and_then(parse_remaining_header)
        {
            account.quota.remaining = Some(remaining);
        }
        if let Some(reset_at) =
            header_value(&headers, &self.config.quota_reset_header).and_then(parse_reset_header)
        {
            account.quota.reset_at = system_time_to_epoch_seconds(reset_at);
        }

        account.quota.updated_at = Some(now_epoch_seconds());
        account.updated_at = now_epoch_seconds();
        if status.is_success() {
            account.last_error = None;
        } else {
            account.last_error = Some(format!("quota probe returned status {}", status.as_u16()));
        }
        Ok(())
    }

    async fn resolve_token_endpoint(&self) -> anyhow::Result<String> {
        if let Some(endpoint) = self.config.oauth_token_endpoint.clone() {
            return Ok(endpoint);
        }
        let discovery = self.fetch_openid_configuration().await?;
        discovery
            .token_endpoint
            .ok_or_else(|| anyhow!("OpenID configuration missing token_endpoint"))
    }

    async fn resolve_device_authorization_endpoint(&self) -> anyhow::Result<String> {
        if let Some(endpoint) = self.config.oauth_device_authorization_endpoint.clone() {
            return Ok(endpoint);
        }
        let discovery = self.fetch_openid_configuration().await?;
        discovery
            .device_authorization_endpoint
            .ok_or_else(|| anyhow!("OpenID configuration missing device_authorization_endpoint"))
    }

    async fn fetch_openid_configuration(&self) -> anyhow::Result<OpenIdConfiguration> {
        let response = self
            .client
            .get(&self.config.openid_config_url)
            .send()
            .await
            .with_context(|| {
                format!(
                    "failed fetching OpenID configuration: {}",
                    self.config.openid_config_url
                )
            })?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            bail!(
                "OpenID configuration request returned status {}: {}",
                status.as_u16(),
                truncate_error(&body)
            );
        }
        response
            .json::<OpenIdConfiguration>()
            .await
            .context("failed parsing OpenID configuration")
    }

    fn read_store(&self) -> anyhow::Result<AccountPoolStore> {
        let path = expand_home(&self.config.store_path);
        if !path.exists() {
            return Ok(AccountPoolStore::default());
        }
        let contents = fs::read_to_string(&path)
            .with_context(|| format!("failed reading account pool store: {}", path.display()))?;
        serde_json::from_str::<AccountPoolStore>(&contents)
            .with_context(|| format!("failed parsing account pool store: {}", path.display()))
    }

    fn write_store(&self, store: &AccountPoolStore) -> anyhow::Result<()> {
        let path = expand_home(&self.config.store_path);
        atomic_write_json(&path, store)
    }
}

fn resolve_alias(alias: Option<String>, existing: &[AccountRecord]) -> String {
    if let Some(alias) = alias
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
    {
        return alias;
    }

    let mut index = existing.len() + 1;
    loop {
        let candidate = format!("account-{index}");
        if existing.iter().all(|account| account.alias != candidate) {
            return candidate;
        }
        index += 1;
    }
}

fn find_account_mut<'a>(
    accounts: &'a mut [AccountRecord],
    account_ref: &str,
) -> anyhow::Result<&'a mut AccountRecord> {
    let trimmed = account_ref.trim();
    if let Some(index) = accounts
        .iter()
        .position(|account| account.id == trimmed || account.alias == trimmed)
    {
        return Ok(&mut accounts[index]);
    }
    Err(anyhow!("account not found: {trimmed}"))
}

fn find_account_index(accounts: &[AccountRecord], account_ref: &str) -> anyhow::Result<usize> {
    let trimmed = account_ref.trim();
    accounts
        .iter()
        .position(|account| account.id == trimmed || account.alias == trimmed)
        .ok_or_else(|| anyhow!("account not found: {trimmed}"))
}

fn account_summary_from_record(record: &AccountRecord) -> AccountSummary {
    AccountSummary {
        id: record.id.clone(),
        alias: record.alias.clone(),
        enabled: record.enabled,
        remaining: record.quota.remaining,
        reset_at: record.quota.reset_at.and_then(epoch_seconds_to_system_time),
        expires_at: record
            .tokens
            .expires_at
            .and_then(epoch_seconds_to_system_time),
        last_error: record.last_error.clone(),
        last_used_at: record.last_used_at.and_then(epoch_seconds_to_system_time),
    }
}

fn account_quota_class(account: &AccountRecord, now: SystemTime) -> i32 {
    match account.quota.remaining {
        Some(remaining) if remaining > 0 => 0,
        Some(_) => {
            if account
                .quota
                .reset_at
                .and_then(epoch_seconds_to_system_time)
                .is_some_and(|reset_at| reset_at <= now)
            {
                1
            } else {
                2
            }
        }
        None => 1,
    }
}

fn compare_accounts_for_selection(
    left: &AccountRecord,
    right: &AccountRecord,
    now: SystemTime,
) -> Ordering {
    let left_class = account_quota_class(left, now);
    let right_class = account_quota_class(right, now);
    if left_class != right_class {
        return left_class.cmp(&right_class);
    }

    if left_class == 0 {
        let left_reset = left.quota.reset_at.unwrap_or(u64::MAX);
        let right_reset = right.quota.reset_at.unwrap_or(u64::MAX);
        if left_reset != right_reset {
            return left_reset.cmp(&right_reset);
        }

        let left_remaining = left.quota.remaining.unwrap_or(0);
        let right_remaining = right.quota.remaining.unwrap_or(0);
        if left_remaining != right_remaining {
            return right_remaining.cmp(&left_remaining);
        }
    }

    let left_last_used = left.last_used_at.unwrap_or(0);
    let right_last_used = right.last_used_at.unwrap_or(0);
    if left_last_used != right_last_used {
        return left_last_used.cmp(&right_last_used);
    }

    left.alias.cmp(&right.alias)
}

fn earliest_reset_at(accounts: &[AccountRecord]) -> Option<SystemTime> {
    accounts
        .iter()
        .filter(|account| account.enabled)
        .filter_map(|account| account.quota.reset_at)
        .min()
        .and_then(epoch_seconds_to_system_time)
}

fn account_pool_headers(access_token: &str, account_id: Option<&str>) -> HeaderMap {
    let mut headers = HeaderMap::new();
    let bearer = format!("Bearer {access_token}");
    if let Ok(value) = bearer.parse() {
        headers.insert("authorization", value);
    }
    headers.insert("openai-beta", "responses=experimental".parse().unwrap());
    if let Some(account_id) = account_id {
        if let Ok(value) = account_id.parse() {
            headers.insert("chatgpt-account-id", value);
        }
    }
    headers.insert("content-type", "application/json".parse().unwrap());
    headers
}

fn header_value<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name).and_then(|value| value.to_str().ok())
}

fn parse_remaining_header(value: &str) -> Option<i64> {
    value.trim().parse::<i64>().ok()
}

fn parse_reset_header(value: &str) -> Option<SystemTime> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }

    if let Ok(seconds) = trimmed.parse::<i64>() {
        if seconds >= 0 {
            let seconds = if seconds > 10_000_000_000 {
                (seconds as u64) / 1_000
            } else {
                seconds as u64
            };
            return Some(UNIX_EPOCH + Duration::from_secs(seconds));
        }
    }

    if let Some(timestamp) = parse_rfc3339_to_unix_seconds(trimmed) {
        if timestamp >= 0 {
            return Some(UNIX_EPOCH + Duration::from_secs(timestamp as u64));
        }
    }

    None
}

fn now_epoch_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn system_time_to_epoch_seconds(value: SystemTime) -> Option<u64> {
    value
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_secs())
}

fn epoch_seconds_to_system_time(value: u64) -> Option<SystemTime> {
    Some(UNIX_EPOCH + Duration::from_secs(value))
}

fn expand_home(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    PathBuf::from(path)
}

fn join_upstream_url(base_url: &str, endpoint: &str) -> String {
    let base = base_url.trim_end_matches('/');
    let suffix = endpoint.trim_start_matches('/');
    format!("{base}/{suffix}")
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

fn jwt_expiry(token: &str) -> Option<SystemTime> {
    decode_jwt_claims(token)
        .ok()
        .and_then(|claims| claims.exp)
        .map(|seconds| UNIX_EPOCH + Duration::from_secs(seconds))
}

fn jwt_subject(token: &str) -> Option<String> {
    decode_jwt_claims(token).ok()?.sub
}

fn expires_soon(expires_at: Option<SystemTime>) -> bool {
    match expires_at {
        Some(expires_at) => expires_at <= (SystemTime::now() + TOKEN_REFRESH_SKEW),
        None => false,
    }
}

fn truncate_error(value: &str) -> String {
    const LIMIT: usize = 256;
    if value.len() <= LIMIT {
        value.to_string()
    } else {
        format!("{}...", &value[..LIMIT])
    }
}

fn atomic_write_json(path: &Path, store: &AccountPoolStore) -> anyhow::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).with_context(|| format!("failed creating {}", parent.display()))?;

    let contents = serde_json::to_vec_pretty(store).context("failed serializing account pool")?;
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("account-pool.json");
    let temp_path = parent.join(format!(".{file_name}.{unique}.tmp"));

    fs::write(&temp_path, contents)
        .with_context(|| format!("failed writing {}", temp_path.display()))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let permissions = fs::Permissions::from_mode(0o600);
        let _ = fs::set_permissions(&temp_path, permissions);
    }

    if let Err(err) = fs::rename(&temp_path, path) {
        let _ = fs::remove_file(&temp_path);
        return Err(err).with_context(|| format!("failed replacing {}", path.display()));
    }

    Ok(())
}

pub fn format_reset_time_iso(value: SystemTime) -> String {
    let timestamp = value
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0);
    format_unix_seconds_as_rfc3339(timestamp)
}

fn parse_rfc3339_to_unix_seconds(value: &str) -> Option<i64> {
    let (date_time, offset_seconds) = if let Some(rest) = value.strip_suffix('Z') {
        (rest, 0i64)
    } else if let Some((prefix, sign, offset)) = split_rfc3339_offset(value) {
        let offset = parse_offset_seconds(offset)?;
        let offset = if sign == '-' { -offset } else { offset };
        (prefix, offset)
    } else {
        return None;
    };

    let (date, time) = date_time.split_once('T')?;
    let (year, month, day) = parse_date_parts(date)?;
    let (hour, minute, second) = parse_time_parts(time)?;
    let days = days_from_civil(year, month, day)?;
    let seconds = days
        .checked_mul(86_400)?
        .checked_add((hour as i64).checked_mul(3_600)?)?
        .checked_add((minute as i64).checked_mul(60)?)?
        .checked_add(second as i64)?;
    seconds.checked_sub(offset_seconds)
}

fn split_rfc3339_offset(value: &str) -> Option<(&str, char, &str)> {
    let bytes = value.as_bytes();
    for (index, byte) in bytes.iter().enumerate().rev() {
        if *byte == b'+' || *byte == b'-' {
            let sign = *byte as char;
            let prefix = &value[..index];
            let offset = &value[index + 1..];
            return Some((prefix, sign, offset));
        }
    }
    None
}

fn parse_offset_seconds(value: &str) -> Option<i64> {
    let (hours, minutes) = value.split_once(':')?;
    let hours = hours.parse::<i64>().ok()?;
    let minutes = minutes.parse::<i64>().ok()?;
    if !(0..=23).contains(&hours) || !(0..=59).contains(&minutes) {
        return None;
    }
    Some(hours * 3600 + minutes * 60)
}

fn parse_date_parts(value: &str) -> Option<(i64, u32, u32)> {
    let mut iter = value.split('-');
    let year = iter.next()?.parse::<i64>().ok()?;
    let month = iter.next()?.parse::<u32>().ok()?;
    let day = iter.next()?.parse::<u32>().ok()?;
    if iter.next().is_some() || month == 0 || month > 12 || day == 0 || day > 31 {
        return None;
    }
    Some((year, month, day))
}

fn parse_time_parts(value: &str) -> Option<(u32, u32, u32)> {
    let mut iter = value.split(':');
    let hour = iter.next()?.parse::<u32>().ok()?;
    let minute = iter.next()?.parse::<u32>().ok()?;
    let second_raw = iter.next()?;
    if iter.next().is_some() {
        return None;
    }
    let second = second_raw
        .split('.')
        .next()
        .and_then(|part| part.parse::<u32>().ok())?;
    if hour > 23 || minute > 59 || second > 60 {
        return None;
    }
    Some((hour, minute, second))
}

fn days_from_civil(year: i64, month: u32, day: u32) -> Option<i64> {
    if day == 0 || month == 0 || month > 12 {
        return None;
    }
    let year = year - if month <= 2 { 1 } else { 0 };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let yoe = year - era * 400;
    let month = month as i64;
    let day = day as i64;
    let doy = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some(era * 146_097 + doe - 719_468)
}

fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let days = days + 719_468;
    let era = if days >= 0 { days } else { days - 146_096 } / 146_097;
    let doe = days - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    let year = year + if month <= 2 { 1 } else { 0 };
    (year, month as u32, day as u32)
}

fn format_unix_seconds_as_rfc3339(timestamp: i64) -> String {
    let days = timestamp.div_euclid(86_400);
    let secs_of_day = timestamp.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let hour = secs_of_day / 3_600;
    let minute = (secs_of_day % 3_600) / 60;
    let second = secs_of_day % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample_account(alias: &str, remaining: Option<i64>, reset_at: Option<u64>) -> AccountRecord {
        AccountRecord {
            id: Uuid::new_v4().simple().to_string(),
            alias: alias.to_string(),
            enabled: true,
            tokens: AccountTokens {
                access_token: "token".to_string(),
                refresh_token: Some("refresh".to_string()),
                id_token: None,
                account_id: Some(format!("acct-{alias}")),
                expires_at: Some(now_epoch_seconds() + 3600),
            },
            quota: AccountQuota {
                remaining,
                reset_at,
                updated_at: Some(now_epoch_seconds()),
            },
            last_error: None,
            last_used_at: None,
            created_at: now_epoch_seconds(),
            updated_at: now_epoch_seconds(),
        }
    }

    fn encode_jwt(claims: serde_json::Value) -> String {
        let payload = URL_SAFE_NO_PAD.encode(claims.to_string().as_bytes());
        format!("header.{payload}.signature")
    }

    #[test]
    fn parse_reset_header_supports_unix_and_rfc3339() {
        let unix = parse_reset_header("1700000000").expect("expected unix reset");
        assert_eq!(system_time_to_epoch_seconds(unix), Some(1_700_000_000));

        let rfc = parse_reset_header("2025-01-01T00:00:00Z").expect("expected rfc3339 reset");
        assert_eq!(system_time_to_epoch_seconds(rfc), Some(1_735_689_600));
    }

    #[test]
    fn compare_accounts_prefers_earlier_reset_then_higher_remaining() {
        let now = SystemTime::now();
        let mut a = sample_account("a", Some(10), Some(now_epoch_seconds() + 60));
        let b = sample_account("b", Some(9), Some(now_epoch_seconds() + 120));
        assert_eq!(compare_accounts_for_selection(&a, &b, now), Ordering::Less);

        a.quota.reset_at = b.quota.reset_at;
        assert_eq!(compare_accounts_for_selection(&a, &b, now), Ordering::Less);
    }

    #[test]
    fn quota_class_treats_expired_reset_as_unknown_quota() {
        let now = SystemTime::now();
        let mut account = sample_account("a", Some(0), Some(now_epoch_seconds().saturating_sub(1)));
        assert_eq!(account_quota_class(&account, now), 1);

        account.quota.reset_at = Some(now_epoch_seconds() + 60);
        assert_eq!(account_quota_class(&account, now), 2);
    }

    #[test]
    fn jwt_helpers_decode_subject_and_expiry() {
        let token = encode_jwt(json!({
            "sub": "user-123",
            "exp": 1_700_000_000u64
        }));
        assert_eq!(jwt_subject(&token), Some("user-123".to_string()));
        assert_eq!(
            jwt_expiry(&token),
            Some(UNIX_EPOCH + Duration::from_secs(1_700_000_000))
        );
    }

    #[test]
    fn truncate_error_limits_output() {
        let long = "x".repeat(300);
        let truncated = truncate_error(&long);
        assert!(truncated.len() <= 259);
        assert!(truncated.ends_with("..."));
    }
}
