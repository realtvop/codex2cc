mod account_pool;
mod config;
mod converter;
mod metrics;
mod upstream_auth;

use std::{
    collections::HashSet,
    convert::Infallible,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    sync::Arc,
    time::{Duration, SystemTime},
};

use account_pool::{format_reset_time_iso, AccountPoolManager, AccountSelection};
use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, HeaderValue, Response, StatusCode},
    response::{IntoResponse, Json},
    routing::post,
    Router,
};
use bytes::Bytes;
use config::{AppConfig, AuthMode, ConfigLoad, DEFAULT_CODEX_BASE_URL};
use converter::{
    anthropic_to_openai_request, build_error_data, openai_to_anthropic_response, StreamConverter,
};
use futures::{stream, StreamExt, TryStreamExt};
use metrics::{MetricsRegistry, RequestMetricsHandle};
use reqwest::{Client, Url};
use serde_json::{json, Value};
use tower_http::trace::TraceLayer;
use tracing::{error, info, warn};
use upstream_auth::{CredentialSource, ResolvedCredential, UpstreamAuthManager};

#[derive(Clone)]
struct AppState {
    config: Arc<AppConfig>,
    client: Client,
    metrics: Arc<MetricsRegistry>,
    upstream_auth: Arc<UpstreamAuthManager>,
    account_pool: Arc<AccountPoolManager>,
}

enum CliCommand {
    Serve,
    Pool(PoolCommand),
}

enum PoolCommand {
    Login { alias: Option<String> },
    List,
    Enable { account: String },
    Disable { account: String },
    Remove { account: String },
    Refresh,
}

enum UpstreamRequestError {
    Request(reqwest::Error),
    AccountPoolExhausted { reset_at: Option<SystemTime> },
    Internal(anyhow::Error),
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();
    let command = parse_cli_command()?;

    let config = match AppConfig::load()? {
        ConfigLoad::Loaded(config) => Arc::new(config),
        ConfigLoad::Generated { path } => {
            info!(
                path = %path.display(),
                "configuration file was missing; generated default config and exiting"
            );
            return Ok(());
        }
    };
    let client = Client::builder()
        .timeout(Duration::from_secs(600))
        .build()?;
    let metrics = MetricsRegistry::new();
    let upstream_auth = UpstreamAuthManager::new(config.upstream.clone(), client.clone());
    let account_pool = AccountPoolManager::new(
        config.upstream.account_pool.clone(),
        config.upstream.codex_base_url.clone(),
        client.clone(),
    );

    if let CliCommand::Pool(pool_command) = command {
        run_pool_command(&account_pool, pool_command).await?;
        return Ok(());
    }

    info!(
        host = %config.server.host,
        port = config.server.port,
        upstream = %config.upstream.base_url,
        codex_upstream = %config
            .upstream
            .codex_base_url
            .as_deref()
            .unwrap_or(DEFAULT_CODEX_BASE_URL),
        auth_mode = config.upstream.auth_mode.as_str(),
        auth_enabled = config.auth_enabled(),
        prefer_local_codex_credentials = config.upstream.prefer_local_codex_credentials,
        "starting Rust proxy server"
    );

    let cc_switch_deeplink = build_cc_switch_import_deeplink_for_log(&config);
    info!(
        deep_link = %cc_switch_deeplink,
        "CC Switch import deep link"
    );
    if config.api_keys.is_empty() {
        warn!(
            "CC Switch import deep link generated without apiKey because client auth is disabled"
        );
    }

    let bind_host: IpAddr = config.server.host.parse()?;
    let bind_port = config.server.port;

    let state = AppState {
        config: Arc::clone(&config),
        client,
        metrics,
        upstream_auth,
        account_pool: Arc::clone(&account_pool),
    };

    if config.upstream.auth_mode == AuthMode::AccountPool {
        let refresh_manager = Arc::clone(&account_pool);
        tokio::spawn(async move {
            let interval = refresh_manager.quota_refresh_interval();
            loop {
                tokio::time::sleep(interval).await;
                if let Err(err) = refresh_manager.refresh_all_quotas().await {
                    warn!(error = %err, "background account pool quota refresh failed");
                }
            }
        });
    }

    let listener = tokio::net::TcpListener::bind((bind_host, bind_port)).await?;

    axum::serve(listener, build_app(state)).await?;
    Ok(())
}

fn parse_cli_command() -> anyhow::Result<CliCommand> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        return Ok(CliCommand::Serve);
    }

    if args[0] != "pool" {
        anyhow::bail!(
            "unknown command: {} (supported: pool login|list|enable|disable|remove|refresh)",
            args[0]
        );
    }

    let subcommand = args.get(1).map(String::as_str).unwrap_or("list");
    let pool_command = match subcommand {
        "login" => PoolCommand::Login {
            alias: args.get(2).cloned(),
        },
        "list" => PoolCommand::List,
        "enable" => PoolCommand::Enable {
            account: args
                .get(2)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("usage: pool enable <account-id-or-alias>"))?,
        },
        "disable" => PoolCommand::Disable {
            account: args
                .get(2)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("usage: pool disable <account-id-or-alias>"))?,
        },
        "remove" => PoolCommand::Remove {
            account: args
                .get(2)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("usage: pool remove <account-id-or-alias>"))?,
        },
        "refresh" => PoolCommand::Refresh,
        _ => {
            anyhow::bail!(
                "unknown pool subcommand: {subcommand} (supported: login|list|enable|disable|remove|refresh)"
            );
        }
    };

    Ok(CliCommand::Pool(pool_command))
}

async fn run_pool_command(
    manager: &Arc<AccountPoolManager>,
    command: PoolCommand,
) -> anyhow::Result<()> {
    match command {
        PoolCommand::Login { alias } => {
            let session = manager.begin_device_login().await?;
            if let Some(url) = session.verification_uri_complete.as_deref() {
                println!("Open this URL to authorize:\n{url}");
            } else {
                println!("Open this URL: {}", session.verification_uri);
            }
            println!("Then enter this code: {}", session.user_code);
            let summary = manager.complete_device_login(&session, alias).await?;
            println!(
                "Added account: alias={} id={} enabled={}",
                summary.alias, summary.id, summary.enabled
            );
        }
        PoolCommand::List => {
            let accounts = manager.list_accounts().await?;
            if accounts.is_empty() {
                println!("No account pool entries.");
            } else {
                for account in accounts {
                    println!(
                        "id={} alias={} enabled={} remaining={} reset_at={} expires_at={} last_used_at={} last_error={}",
                        account.id,
                        account.alias,
                        account.enabled,
                        account
                            .remaining
                            .map(|value| value.to_string())
                            .unwrap_or_else(|| "unknown".to_string()),
                        account
                            .reset_at
                            .map(format_reset_time_iso)
                            .unwrap_or_else(|| "unknown".to_string()),
                        account
                            .expires_at
                            .map(format_reset_time_iso)
                            .unwrap_or_else(|| "unknown".to_string()),
                        account
                            .last_used_at
                            .map(format_reset_time_iso)
                            .unwrap_or_else(|| "never".to_string()),
                        account.last_error.unwrap_or_else(|| "-".to_string()),
                    );
                }
            }
        }
        PoolCommand::Enable { account } => {
            let summary = manager.set_enabled(&account, true).await?;
            println!("Enabled account: alias={} id={}", summary.alias, summary.id);
        }
        PoolCommand::Disable { account } => {
            let summary = manager.set_enabled(&account, false).await?;
            println!(
                "Disabled account: alias={} id={}",
                summary.alias, summary.id
            );
        }
        PoolCommand::Remove { account } => {
            let summary = manager.remove(&account).await?;
            println!("Removed account: alias={} id={}", summary.alias, summary.id);
        }
        PoolCommand::Refresh => {
            manager.refresh_all_quotas().await?;
            println!("Account pool quota refresh completed.");
        }
    }
    Ok(())
}

fn build_app(state: AppState) -> Router {
    Router::new()
        .route("/v1/messages", post(create_message))
        .route("/v1/messages/count_tokens", post(count_tokens))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

fn init_tracing() {
    let filter =
        std::env::var("RUST_LOG").unwrap_or_else(|_| "codextocc=info,tower_http=warn".to_string());
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .compact()
        .init();
}

async fn create_message(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    if let Some(resp) = check_auth(&state.config, &headers) {
        return resp;
    }

    let request_model = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let thinking_enabled = body
        .get("thinking")
        .and_then(Value::as_object)
        .and_then(|thinking| thinking.get("type"))
        .and_then(Value::as_str)
        == Some("enabled");
    let is_stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);

    let metrics_handle = state.metrics.start_request(is_stream);

    info!(model = %request_model, "started /v1/messages request");

    let openai_body = anthropic_to_openai_request(&body, &state.config.model_map);

    if is_stream {
        return stream_response(
            state,
            openai_body,
            request_model,
            thinking_enabled,
            metrics_handle,
        )
        .await
        .into_response();
    }

    let resp = match execute_upstream_request(&state, "/responses", &openai_body, None).await {
        Ok(resp) => resp,
        Err(UpstreamRequestError::AccountPoolExhausted { reset_at }) => {
            metrics_handle.fail();
            return account_pool_exhausted_response(reset_at);
        }
        Err(UpstreamRequestError::Request(err)) => {
            metrics_handle.fail();
            error!(request_id = metrics_handle.id(), error = %err, "upstream request failed");
            return error_response(
                StatusCode::BAD_GATEWAY,
                json!({
                    "type": "error",
                    "error": {"type": "api_error", "message": err.to_string()}
                }),
            );
        }
        Err(UpstreamRequestError::Internal(err)) => {
            metrics_handle.fail();
            error!(request_id = metrics_handle.id(), error = %err, "internal upstream routing failed");
            return error_response(
                StatusCode::BAD_GATEWAY,
                json!({
                    "type": "error",
                    "error": {"type": "api_error", "message": err.to_string()}
                }),
            );
        }
    };

    let status = resp.status();
    let response_text = match resp.text().await {
        Ok(text) => text,
        Err(err) => {
            metrics_handle.fail();
            error!(request_id = metrics_handle.id(), error = %err, "failed reading upstream response body");
            return error_response(
                StatusCode::BAD_GATEWAY,
                json!({
                    "type": "error",
                    "error": {"type": "api_error", "message": err.to_string()}
                }),
            );
        }
    };

    if !status.is_success() {
        metrics_handle.fail();
        let upstream_json = serde_json::from_str::<Value>(&response_text)
            .unwrap_or_else(|_| json!({"error": {"message": response_text}}));
        return error_response(status, build_error_data(status.as_u16(), &upstream_json));
    }

    let openai_resp: Value = match serde_json::from_str(&response_text) {
        Ok(value) => value,
        Err(err) => {
            metrics_handle.fail();
            error!(request_id = metrics_handle.id(), error = %err, "failed to parse upstream JSON response");
            return error_response(
                StatusCode::BAD_GATEWAY,
                json!({
                    "type": "error",
                    "error": {"type": "api_error", "message": err.to_string()}
                }),
            );
        }
    };

    let input_tokens = openai_resp
        .get("usage")
        .and_then(|usage| usage.get("input_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output_tokens = openai_resp
        .get("usage")
        .and_then(|usage| usage.get("output_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    metrics_handle.finish(input_tokens, output_tokens);

    let anthropic_resp =
        openai_to_anthropic_response(&openai_resp, &request_model, thinking_enabled);
    info!(model = %request_model, "finished /v1/messages request");

    let mut response = Json(anthropic_resp).into_response();
    if let Some(request_id) = openai_resp.get("id").and_then(Value::as_str) {
        if let Ok(value) = HeaderValue::from_str(request_id) {
            response.headers_mut().insert("x-request-id", value);
        }
    }
    response
}

async fn stream_response(
    state: AppState,
    openai_body: Value,
    request_model: String,
    thinking_enabled: bool,
    metrics_handle: RequestMetricsHandle,
) -> Response<Body> {
    let resp = match execute_upstream_request(&state, "/responses", &openai_body, None).await {
        Ok(resp) => resp,
        Err(UpstreamRequestError::AccountPoolExhausted { reset_at }) => {
            metrics_handle.fail();
            return sse_error_response(account_pool_exhausted_body(reset_at));
        }
        Err(UpstreamRequestError::Request(err)) => {
            metrics_handle.fail();
            error!(request_id = metrics_handle.id(), error = %err, "failed to open upstream SSE stream");
            return sse_error_response(json!({
                "type": "error",
                "error": {"type": "api_error", "message": err.to_string()}
            }));
        }
        Err(UpstreamRequestError::Internal(err)) => {
            metrics_handle.fail();
            error!(request_id = metrics_handle.id(), error = %err, "internal upstream routing failed");
            return sse_error_response(json!({
                "type": "error",
                "error": {"type": "api_error", "message": err.to_string()}
            }));
        }
    };

    let status = resp.status();
    if !status.is_success() {
        metrics_handle.fail();
        let text = resp.text().await.unwrap_or_default();
        warn!(
            request_id = metrics_handle.id(),
            status = %status,
            body = %truncate_str(&text),
            "upstream SSE request returned error"
        );
        let upstream_json = serde_json::from_str::<Value>(&text)
            .unwrap_or_else(|_| json!({"error": {"message": text}}));
        return sse_error_response(build_error_data(status.as_u16(), &upstream_json));
    }

    let mut converter = StreamConverter::new(
        request_model.clone(),
        thinking_enabled,
        Some(metrics_handle.clone()),
    );
    let mut event_type = String::new();
    let mut data_buf = String::new();
    let stream_metrics_handle = metrics_handle.clone();

    let byte_stream = resp.bytes_stream();
    let stream = byte_stream
        .map_err(move |err| {
            stream_metrics_handle.fail();
            error!(request_id = stream_metrics_handle.id(), error = %err, "error while reading upstream SSE bytes");
            err
        })
        .into_stream()
        .flat_map(move |item| {
            let mut outputs: Vec<Result<Bytes, Infallible>> = Vec::new();
            match item {
                Ok(chunk) => {
                    let text = String::from_utf8_lossy(&chunk);
                    for raw_line in text.split('\n') {
                        let line = raw_line.trim_end_matches('\r');
                        if let Some(rest) = line.strip_prefix("event: ") {
                            event_type = rest.trim().to_string();
                        } else if let Some(rest) = line.strip_prefix("data: ") {
                            data_buf = rest.to_string();
                        } else if line.is_empty() && !event_type.is_empty() && !data_buf.is_empty() {
                            let parsed = serde_json::from_str::<Value>(&data_buf).unwrap_or_else(|_| json!({}));
                            let converted = converter.process_event(&event_type, &parsed);
                            if event_type == "response.completed" {
                                info!(model = %request_model, "finished /v1/messages request");
                            }
                            if !converted.is_empty() {
                                outputs.push(Ok(Bytes::from(converted)));
                            }
                            event_type.clear();
                            data_buf.clear();
                        }
                    }
                }
                Err(_) => outputs.push(Ok(Bytes::from_static(b""))),
            }
            stream::iter(outputs)
        });

    let mut response = Response::new(Body::from_stream(stream));
    *response.status_mut() = StatusCode::OK;
    response.headers_mut().insert(
        "content-type",
        HeaderValue::from_static("text/event-stream"),
    );
    response
        .headers_mut()
        .insert("cache-control", HeaderValue::from_static("no-cache"));
    response
        .headers_mut()
        .insert("connection", HeaderValue::from_static("keep-alive"));
    response
        .headers_mut()
        .insert("x-accel-buffering", HeaderValue::from_static("no"));
    response
}

async fn count_tokens(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    if let Some(resp) = check_auth(&state.config, &headers) {
        return resp;
    }

    info!("started /v1/messages/count_tokens request");

    let openai_body = anthropic_to_openai_request(&body, &state.config.model_map);
    let resp = execute_upstream_request(
        &state,
        "/responses/input_tokens",
        &openai_body,
        Some(Duration::from_secs(30)),
    )
    .await;

    match resp {
        Ok(resp) if resp.status().is_success() => {
            let status = resp.status();
            let value = match resp.json::<Value>().await {
                Ok(value) => value,
                Err(err) => {
                    warn!(error = %err, "failed to parse upstream input_tokens response; using fallback");
                    let estimated = estimate_tokens_from_body(&body);
                    info!(estimated, "count-tokens fallback used after parse failure");
                    return Json(json!({"input_tokens": estimated})).into_response();
                }
            };
            info!(status = %status, "finished /v1/messages/count_tokens request");
            Json(json!({"input_tokens": value.get("input_tokens").and_then(Value::as_i64).unwrap_or(0)})).into_response()
        }
        Ok(resp) => {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            warn!(
                status = %status,
                body = %truncate_str(&text),
                "upstream input_tokens request failed; using fallback"
            );
            let estimated = estimate_tokens_from_body(&body);
            info!(
                estimated,
                "count-tokens fallback used after upstream non-success status"
            );
            Json(json!({"input_tokens": estimated})).into_response()
        }
        Err(UpstreamRequestError::AccountPoolExhausted { reset_at }) => {
            warn!(
                reset_at = ?reset_at.map(format_reset_time_iso),
                "account pool exhausted for count_tokens"
            );
            account_pool_exhausted_response(reset_at)
        }
        Err(UpstreamRequestError::Request(err)) => {
            warn!(error = %err, "upstream input_tokens request errored; using fallback");
            let estimated = estimate_tokens_from_body(&body);
            info!(
                estimated,
                "count-tokens fallback used after upstream request error"
            );
            Json(json!({"input_tokens": estimated})).into_response()
        }
        Err(UpstreamRequestError::Internal(err)) => {
            warn!(error = %err, "internal upstream routing errored; using fallback");
            let estimated = estimate_tokens_from_body(&body);
            info!(estimated, "count-tokens fallback used after internal error");
            Json(json!({"input_tokens": estimated})).into_response()
        }
    }
}

async fn execute_upstream_request(
    state: &AppState,
    endpoint: &str,
    body: &Value,
    timeout: Option<Duration>,
) -> Result<reqwest::Response, UpstreamRequestError> {
    match state.config.upstream.auth_mode {
        AuthMode::AccountPool => execute_account_pool_request(state, endpoint, body, timeout).await,
        AuthMode::LocalCodex | AuthMode::ConfigApiKey => {
            execute_legacy_upstream_request(state, endpoint, body, timeout).await
        }
    }
}

async fn execute_legacy_upstream_request(
    state: &AppState,
    endpoint: &str,
    body: &Value,
    timeout: Option<Duration>,
) -> Result<reqwest::Response, UpstreamRequestError> {
    let base_body = body.clone();
    let mut credential = state.upstream_auth.resolve_primary().await;
    let mut request_body = prepare_upstream_body(&base_body, &credential);
    let mut url = join_upstream_url(
        upstream_base_url(&state.config.upstream, &credential),
        endpoint,
    );
    let mut response =
        send_upstream_request(&state.client, &url, &request_body, timeout, &credential)
            .await
            .map_err(UpstreamRequestError::Request)?;

    if is_auth_error(response.status())
        && credential.source == CredentialSource::LocalCodexAccessToken
    {
        if let Some(refreshed) = state.upstream_auth.resolve_after_token_unauthorized().await {
            info!("retrying upstream request with refreshed local Codex access token");
            request_body = prepare_upstream_body(&base_body, &refreshed);
            url = join_upstream_url(
                upstream_base_url(&state.config.upstream, &refreshed),
                endpoint,
            );
            response =
                send_upstream_request(&state.client, &url, &request_body, timeout, &refreshed)
                    .await
                    .map_err(UpstreamRequestError::Request)?;
            credential = refreshed;
        }
    }

    if is_auth_error(response.status()) && credential.source.is_local() {
        if let Some(configured) = state.upstream_auth.configured_api_key_fallback(&credential) {
            info!("retrying upstream request with configured API key fallback");
            request_body = prepare_upstream_body(&base_body, &configured);
            url = join_upstream_url(
                upstream_base_url(&state.config.upstream, &configured),
                endpoint,
            );
            response =
                send_upstream_request(&state.client, &url, &request_body, timeout, &configured)
                    .await
                    .map_err(UpstreamRequestError::Request)?;
        }
    }

    Ok(response)
}

async fn execute_account_pool_request(
    state: &AppState,
    endpoint: &str,
    body: &Value,
    timeout: Option<Duration>,
) -> Result<reqwest::Response, UpstreamRequestError> {
    let mut attempted_ids = HashSet::new();
    let first = select_account_pool_credential(state, &attempted_ids).await?;
    attempted_ids.insert(first.id.clone());

    let mut credential = ResolvedCredential {
        token: first.access_token.clone(),
        source: CredentialSource::AccountPoolAccessToken,
        account_id: first.account_id.clone(),
    };
    let base_body = body.clone();
    let mut request_body = prepare_upstream_body(&base_body, &credential);
    let mut url = join_upstream_url(
        upstream_base_url(&state.config.upstream, &credential),
        endpoint,
    );
    let mut response =
        send_upstream_request(&state.client, &url, &request_body, timeout, &credential)
            .await
            .map_err(UpstreamRequestError::Request)?;
    let first_status = response.status();
    let first_headers = response.headers().clone();
    if let Err(err) = state
        .account_pool
        .update_quota_from_response(&first.id, &first_headers, first_status)
        .await
    {
        warn!(error = %err, "failed updating account pool quota metadata");
    }

    if is_account_pool_retry_status(first_status) {
        match select_account_pool_credential(state, &attempted_ids).await {
            Ok(next) => {
                attempted_ids.insert(next.id.clone());
                credential = ResolvedCredential {
                    token: next.access_token.clone(),
                    source: CredentialSource::AccountPoolAccessToken,
                    account_id: next.account_id.clone(),
                };
                request_body = prepare_upstream_body(&base_body, &credential);
                url = join_upstream_url(
                    upstream_base_url(&state.config.upstream, &credential),
                    endpoint,
                );
                response =
                    send_upstream_request(&state.client, &url, &request_body, timeout, &credential)
                        .await
                        .map_err(UpstreamRequestError::Request)?;
                let second_status = response.status();
                let second_headers = response.headers().clone();
                if let Err(err) = state
                    .account_pool
                    .update_quota_from_response(&next.id, &second_headers, second_status)
                    .await
                {
                    warn!(error = %err, "failed updating account pool quota metadata");
                }
            }
            Err(UpstreamRequestError::AccountPoolExhausted { reset_at }) => {
                if first_status == StatusCode::TOO_MANY_REQUESTS {
                    return Err(UpstreamRequestError::AccountPoolExhausted { reset_at });
                }
            }
            Err(err) => return Err(err),
        }
    }

    Ok(response)
}

async fn select_account_pool_credential(
    state: &AppState,
    attempted_ids: &HashSet<String>,
) -> Result<account_pool::PoolCredential, UpstreamRequestError> {
    let excluded: Vec<String> = attempted_ids.iter().cloned().collect();
    match state.account_pool.select_account(&excluded).await {
        Ok(AccountSelection::Selected(credential)) => Ok(credential),
        Ok(AccountSelection::Exhausted { reset_at }) => {
            Err(UpstreamRequestError::AccountPoolExhausted { reset_at })
        }
        Err(err) => Err(UpstreamRequestError::Internal(err)),
    }
}

async fn send_upstream_request(
    client: &Client,
    url: &str,
    body: &Value,
    timeout: Option<Duration>,
    credential: &ResolvedCredential,
) -> Result<reqwest::Response, reqwest::Error> {
    let mut request = client
        .post(url)
        .headers(upstream_headers(credential))
        .json(body);
    if let Some(timeout) = timeout {
        request = request.timeout(timeout);
    }
    request.send().await
}

fn upstream_headers(credential: &ResolvedCredential) -> HeaderMap {
    let mut headers = HeaderMap::new();
    let bearer = format!("Bearer {}", credential.token);
    headers.insert(
        "authorization",
        HeaderValue::from_str(&bearer).unwrap_or_else(|_| HeaderValue::from_static("Bearer ")),
    );
    if credential.source.is_local() {
        headers.insert(
            "openai-beta",
            HeaderValue::from_static("responses=experimental"),
        );
        headers.insert("accept", HeaderValue::from_static("text/event-stream"));
        if let Some(account_id) = credential.account_id.as_deref() {
            if let Ok(value) = HeaderValue::from_str(account_id) {
                headers.insert("chatgpt-account-id", value);
            }
        }
    }
    headers.insert("content-type", HeaderValue::from_static("application/json"));
    headers
}

fn join_upstream_url(base_url: &str, endpoint: &str) -> String {
    let base = base_url.trim_end_matches('/');
    let suffix = endpoint.trim_start_matches('/');
    format!("{base}/{suffix}")
}

fn upstream_base_url<'a>(
    config: &'a config::UpstreamConfig,
    credential: &ResolvedCredential,
) -> &'a str {
    if credential.source.is_local() {
        config
            .codex_base_url
            .as_deref()
            .unwrap_or(DEFAULT_CODEX_BASE_URL)
    } else {
        &config.base_url
    }
}

fn prepare_upstream_body(body: &Value, credential: &ResolvedCredential) -> Value {
    if !credential.source.is_local() {
        return body.clone();
    }
    if let Some(object) = body.as_object() {
        let mut clone = object.clone();
        clone.remove("max_output_tokens");
        clone.remove("temperature");
        return Value::Object(clone);
    }
    body.clone()
}

fn is_auth_error(status: StatusCode) -> bool {
    status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN
}

fn is_account_pool_retry_status(status: StatusCode) -> bool {
    is_auth_error(status) || status == StatusCode::TOO_MANY_REQUESTS
}

fn check_auth(config: &AppConfig, headers: &HeaderMap) -> Option<Response<Body>> {
    if !config.auth_enabled() {
        return None;
    }

    let key = get_client_key(headers);
    if config.api_keys.contains(&key) {
        None
    } else {
        warn!(provided = !key.is_empty(), "client auth rejected");
        Some(error_response(
            StatusCode::UNAUTHORIZED,
            json!({
                "type": "error",
                "error": {"type": "authentication_error", "message": "Invalid API key"}
            }),
        ))
    }
}

fn get_client_key(headers: &HeaderMap) -> String {
    if let Some(value) = headers.get("x-api-key").and_then(|v| v.to_str().ok()) {
        if !value.is_empty() {
            return value.to_string();
        }
    }

    if let Some(value) = headers.get("authorization").and_then(|v| v.to_str().ok()) {
        let lower = value.to_ascii_lowercase();
        if lower.starts_with("bearer ") {
            return value[7..].to_string();
        }
    }

    String::new()
}

fn error_response(status: StatusCode, body: Value) -> Response<Body> {
    let mut response = Json(body).into_response();
    *response.status_mut() = status;
    response
}

fn account_pool_exhausted_body(reset_at: Option<SystemTime>) -> Value {
    let reset_at_iso = reset_at.map(format_reset_time_iso);
    json!({
        "type": "error",
        "error": {
            "type": "rate_limit_error",
            "message": "All account pool entries are exhausted",
            "reset_at": reset_at_iso,
        }
    })
}

fn account_pool_exhausted_response(reset_at: Option<SystemTime>) -> Response<Body> {
    error_response(
        StatusCode::TOO_MANY_REQUESTS,
        account_pool_exhausted_body(reset_at),
    )
}

fn sse_error_response(body: Value) -> Response<Body> {
    let payload = format!("event: error\ndata: {}\n\n", body);
    let mut response = Response::new(Body::from(payload));
    *response.status_mut() = StatusCode::OK;
    response.headers_mut().insert(
        "content-type",
        HeaderValue::from_static("text/event-stream"),
    );
    response
}

fn build_cc_switch_import_deeplink_for_log(config: &AppConfig) -> String {
    let mut url = Url::parse("ccswitch://v1/import").expect("cc switch import url should be valid");
    let endpoint = build_cc_switch_endpoint(&config.server.host, config.server.port);

    let mut query = url.query_pairs_mut();
    query.append_pair("resource", "provider");
    query.append_pair("app", "claude");
    query.append_pair("name", "codextocc");
    query.append_pair("endpoint", &endpoint);

    if let Some(api_key) = sorted_first_api_key(config) {
        query.append_pair("apiKey", &mask_api_key(api_key));
    }

    drop(query);
    url.into()
}

fn build_cc_switch_endpoint(host: &str, port: u16) -> String {
    let normalized_host = normalize_endpoint_host(host);
    if let Ok(ip) = normalized_host.parse::<IpAddr>() {
        SocketAddr::new(ip, port).to_string()
    } else {
        format!("{normalized_host}:{port}")
    }
}

fn normalize_endpoint_host(host: &str) -> String {
    if let Ok(ip) = host.parse::<IpAddr>() {
        if ip.is_unspecified() {
            return Ipv4Addr::LOCALHOST.to_string();
        }
        return ip.to_string();
    }

    host.to_string()
}

fn sorted_first_api_key(config: &AppConfig) -> Option<&str> {
    let mut keys: Vec<&str> = config.api_keys.iter().map(String::as_str).collect();
    keys.sort_unstable();
    keys.first().copied()
}

fn mask_api_key(api_key: &str) -> String {
    let char_count = api_key.chars().count();
    if char_count <= 8 {
        return "****".to_string();
    }

    let prefix: String = api_key.chars().take(4).collect();
    let suffix: String = api_key.chars().skip(char_count.saturating_sub(4)).collect();
    format!("{prefix}...{suffix}")
}

fn estimate_tokens_from_body(body: &Value) -> i64 {
    let mut total_chars = 0usize;

    match body.get("system") {
        Some(Value::Array(blocks)) => {
            total_chars += blocks
                .iter()
                .filter_map(|block| block.get("text").and_then(Value::as_str))
                .map(str::len)
                .sum::<usize>();
        }
        Some(Value::String(text)) => total_chars += text.len(),
        _ => {}
    }

    if let Some(messages) = body.get("messages").and_then(Value::as_array) {
        for msg in messages {
            match msg.get("content") {
                Some(Value::String(text)) => total_chars += text.len(),
                Some(Value::Array(blocks)) => {
                    for block in blocks {
                        if let Some(text) = block.get("text").and_then(Value::as_str) {
                            total_chars += text.len();
                        }
                        if let Some(content) = block.get("content").and_then(Value::as_str) {
                            total_chars += content.len();
                        }
                    }
                }
                _ => {}
            }
        }
    }

    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        for tool in tools {
            total_chars += serde_json::to_string(tool).map(|s| s.len()).unwrap_or(0);
        }
    }

    (total_chars / 4) as i64
}

fn truncate_str(value: &str) -> String {
    const LIMIT: usize = 2_000;
    if value.len() <= LIMIT {
        value.to_string()
    } else {
        format!(
            "{}...<truncated {} chars>",
            &value[..LIMIT],
            value.len() - LIMIT
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::{HashMap, HashSet};

    fn sample_config(api_keys: &[&str]) -> AppConfig {
        AppConfig {
            server: config::ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 8082,
            },
            api_keys: api_keys.iter().map(|key| (*key).to_string()).collect(),
            upstream: config::UpstreamConfig {
                base_url: "https://api.openai.com/v1".to_string(),
                api_key: "configured-key".to_string(),
                auth_mode: AuthMode::ConfigApiKey,
                prefer_local_codex_credentials: false,
                local_codex_auth_path: "~/.codex/auth.json".to_string(),
                refresh_local_codex_tokens: true,
                local_codex_oauth_client_id: None,
                local_codex_oauth_token_endpoint: None,
                codex_base_url: Some(DEFAULT_CODEX_BASE_URL.to_string()),
                account_pool: config::AccountPoolConfig {
                    store_path: config::DEFAULT_ACCOUNT_POOL_STORE_PATH.to_string(),
                    quota_refresh_interval_secs: config::DEFAULT_ACCOUNT_POOL_REFRESH_INTERVAL_SECS,
                    quota_remaining_header: config::DEFAULT_ACCOUNT_POOL_QUOTA_REMAINING_HEADER
                        .to_string(),
                    quota_reset_header: config::DEFAULT_ACCOUNT_POOL_QUOTA_RESET_HEADER.to_string(),
                    oauth_client_id: None,
                    oauth_scopes: vec![
                        "openid".to_string(),
                        "profile".to_string(),
                        "email".to_string(),
                        "offline_access".to_string(),
                    ],
                    openid_config_url: config::DEFAULT_OPENID_CONFIG_URL.to_string(),
                    oauth_token_endpoint: None,
                    oauth_device_authorization_endpoint: None,
                },
            },
            model_map: HashMap::new(),
        }
    }

    fn query_map(url: &str) -> HashMap<String, String> {
        Url::parse(url)
            .expect("deep link should be valid url")
            .query_pairs()
            .into_owned()
            .collect()
    }

    #[test]
    fn cc_switch_deeplink_has_expected_structure_and_required_params() {
        let config = sample_config(&["zzzzzzzz", "aaaaBBBBcccc"]);
        let deeplink = build_cc_switch_import_deeplink_for_log(&config);
        let parsed = Url::parse(&deeplink).expect("deep link should parse");
        let query = query_map(&deeplink);

        assert_eq!(parsed.scheme(), "ccswitch");
        assert_eq!(parsed.host_str(), Some("v1"));
        assert_eq!(parsed.path(), "/import");

        assert_eq!(query.get("resource").map(String::as_str), Some("provider"));
        assert_eq!(query.get("app").map(String::as_str), Some("claude"));
        assert_eq!(query.get("name").map(String::as_str), Some("codextocc"));
        assert_eq!(
            query.get("endpoint").map(String::as_str),
            Some("127.0.0.1:8082")
        );
        assert_eq!(query.get("apiKey").map(String::as_str), Some("aaaa...cccc"));
    }

    #[test]
    fn cc_switch_deeplink_url_encodes_special_characters() {
        let config = sample_config(&["a&? zzzzYYYY"]);
        let deeplink = build_cc_switch_import_deeplink_for_log(&config);

        assert!(deeplink.contains("apiKey=a%26%3F+...YYYY"));

        let query = query_map(&deeplink);
        assert_eq!(query.get("apiKey").map(String::as_str), Some("a&? ...YYYY"));
    }

    #[test]
    fn cc_switch_deeplink_masks_api_key_and_never_logs_plaintext() {
        let plain = "secret-token-1234";
        let config = sample_config(&[plain]);
        let deeplink = build_cc_switch_import_deeplink_for_log(&config);

        assert!(!deeplink.contains(plain));
        assert!(deeplink.contains("apiKey=secr...1234"));
    }

    #[test]
    fn cc_switch_deeplink_falls_back_unspecified_hosts_to_localhost() {
        let mut ipv4_config = sample_config(&["secrettoken"]);
        ipv4_config.server.host = "0.0.0.0".to_string();

        let mut ipv6_config = sample_config(&["secrettoken"]);
        ipv6_config.server.host = "::".to_string();

        let ipv4_query = query_map(&build_cc_switch_import_deeplink_for_log(&ipv4_config));
        let ipv6_query = query_map(&build_cc_switch_import_deeplink_for_log(&ipv6_config));

        assert_eq!(
            ipv4_query.get("endpoint").map(String::as_str),
            Some("127.0.0.1:8082")
        );
        assert_eq!(
            ipv6_query.get("endpoint").map(String::as_str),
            Some("127.0.0.1:8082")
        );
    }

    #[test]
    fn cc_switch_deeplink_omits_api_key_when_auth_is_disabled() {
        let config = sample_config(&[]);
        let deeplink = build_cc_switch_import_deeplink_for_log(&config);
        let query = query_map(&deeplink);

        assert!(!query.contains_key("apiKey"));
    }

    #[test]
    fn upstream_base_url_prefers_codex_for_local_credentials() {
        let mut config = sample_config(&[]);
        config.upstream.codex_base_url = Some("https://chatgpt.example".to_string());

        let local_credential = ResolvedCredential {
            token: "token".to_string(),
            source: CredentialSource::LocalCodexAccessToken,
            account_id: None,
        };
        let configured_credential = ResolvedCredential {
            token: "token".to_string(),
            source: CredentialSource::ConfigApiKey,
            account_id: None,
        };

        assert_eq!(
            upstream_base_url(&config.upstream, &local_credential),
            "https://chatgpt.example"
        );
        assert_eq!(
            upstream_base_url(&config.upstream, &configured_credential),
            config.upstream.base_url
        );
    }

    #[test]
    fn join_upstream_url_normalizes_slashes() {
        assert_eq!(
            join_upstream_url("https://api.openai.com/v1/", "/responses"),
            "https://api.openai.com/v1/responses"
        );
        assert_eq!(
            join_upstream_url("https://api.openai.com/v1", "responses"),
            "https://api.openai.com/v1/responses"
        );
    }

    #[test]
    fn upstream_headers_add_codex_metadata_for_local_credentials() {
        let headers = upstream_headers(&ResolvedCredential {
            token: "token".to_string(),
            source: CredentialSource::LocalCodexAccessToken,
            account_id: Some("acct-123".to_string()),
        });

        assert_eq!(
            headers
                .get("authorization")
                .and_then(|value| value.to_str().ok()),
            Some("Bearer token")
        );
        assert_eq!(
            headers
                .get("openai-beta")
                .and_then(|value| value.to_str().ok()),
            Some("responses=experimental")
        );
        assert_eq!(
            headers
                .get("chatgpt-account-id")
                .and_then(|value| value.to_str().ok()),
            Some("acct-123")
        );
    }

    #[test]
    fn prepare_upstream_body_strips_max_output_tokens_for_local_credential() {
        let body = json!({"max_output_tokens": 10, "temperature": 0.7, "other": 1});

        let local = prepare_upstream_body(
            &body,
            &ResolvedCredential {
                token: "t".to_string(),
                source: CredentialSource::LocalCodexAccessToken,
                account_id: None,
            },
        );
        assert_eq!(local, json!({"other": 1}));

        let configured = prepare_upstream_body(
            &body,
            &ResolvedCredential {
                token: "t".to_string(),
                source: CredentialSource::ConfigApiKey,
                account_id: None,
            },
        );
        assert_eq!(configured, body);
    }

    #[test]
    fn get_client_key_prefers_x_api_key_then_bearer() {
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", HeaderValue::from_static("header-key"));
        headers.insert(
            "authorization",
            HeaderValue::from_static("Bearer bearer-key"),
        );
        assert_eq!(get_client_key(&headers), "header-key");

        let mut bearer_only = HeaderMap::new();
        bearer_only.insert(
            "authorization",
            HeaderValue::from_static("bEaReR mixed-case"),
        );
        assert_eq!(get_client_key(&bearer_only), "mixed-case");

        assert_eq!(get_client_key(&HeaderMap::new()), "");
    }

    #[test]
    fn check_auth_respects_disabled_and_matching_keys() {
        assert!(check_auth(&sample_config(&[]), &HeaderMap::new()).is_none());

        let config = sample_config(&["secret"]);
        let mut valid_headers = HeaderMap::new();
        valid_headers.insert("x-api-key", HeaderValue::from_static("secret"));
        assert!(check_auth(&config, &valid_headers).is_none());

        let mut invalid_headers = HeaderMap::new();
        invalid_headers.insert("authorization", HeaderValue::from_static("Bearer wrong"));
        let response = check_auth(&config, &invalid_headers).expect("expected auth failure");
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn estimate_tokens_from_body_counts_relevant_text() {
        let tool = json!({"name": "lookup", "description": "desc"});
        let body = json!({
            "system": [{"text": "abcd"}, {"text": "ef"}],
            "messages": [
                {"content": "12345678"},
                {"content": [
                    {"type": "text", "text": "wxyz"},
                    {"type": "tool_result", "content": "tool-output"}
                ]}
            ],
            "tools": [tool.clone()]
        });

        let total_chars =
            4 + 2 + 8 + 4 + "tool-output".len() + serde_json::to_string(&tool).unwrap().len();
        assert_eq!(estimate_tokens_from_body(&body), (total_chars / 4) as i64);
    }

    #[test]
    fn sample_config_deduplicates_api_keys() {
        let config = sample_config(&["secret", "secret", "other"]);
        let expected = HashSet::from(["secret".to_string(), "other".to_string()]);
        assert_eq!(config.api_keys, expected);
    }
}
