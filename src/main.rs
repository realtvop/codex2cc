mod config;
mod converter;
mod metrics;

use std::{convert::Infallible, sync::Arc, time::Duration};

use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, HeaderValue, Response, StatusCode},
    response::{IntoResponse, Json},
    routing::post,
    Router,
};
use bytes::Bytes;
use config::AppConfig;
use converter::{anthropic_to_openai_request, build_error_data, openai_to_anthropic_response, StreamConverter};
use metrics::{MetricsRegistry, RequestMetricsHandle};
use futures::{stream, StreamExt, TryStreamExt};
use reqwest::Client;
use serde_json::{json, Value};
use tower_http::trace::TraceLayer;
use tracing::{error, info, warn};

#[derive(Clone)]
struct AppState {
    config: Arc<AppConfig>,
    client: Client,
    metrics: Arc<MetricsRegistry>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();

    let config = Arc::new(AppConfig::load()?);
    let client = Client::builder().timeout(Duration::from_secs(600)).build()?;
    let metrics = MetricsRegistry::new();

    info!(host = %config.server.host, port = config.server.port, upstream = %config.upstream.base_url, auth_enabled = config.auth_enabled(), "starting Rust proxy server");

    let bind_host: std::net::IpAddr = config.server.host.parse()?;
    let bind_port = config.server.port;

    let app = Router::new()
        .route("/v1/messages", post(create_message))
        .route("/v1/messages/count_tokens", post(count_tokens))
        .layer(TraceLayer::new_for_http())
        .with_state(AppState { config, client, metrics });

    let listener = tokio::net::TcpListener::bind((bind_host, bind_port)).await?;

    axum::serve(listener, app).await?;
    Ok(())
}

fn init_tracing() {
    let filter = std::env::var("RUST_LOG").unwrap_or_else(|_| "codextocc=info,tower_http=warn".to_string());
    tracing_subscriber::fmt().with_env_filter(filter).with_target(false).compact().init();
}

async fn create_message(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    if let Some(resp) = check_auth(&state.config, &headers) {
        return resp;
    }

    let request_model = body.get("model").and_then(Value::as_str).unwrap_or_default().to_string();
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
        return stream_response(state, openai_body, request_model, thinking_enabled, metrics_handle).await.into_response();
    }

    let url = format!("{}/responses", state.config.upstream.base_url);
    let request_builder = state.client.post(&url).headers(upstream_headers(&state.config)).json(&openai_body);

    let resp = match request_builder.send().await {
        Ok(resp) => resp,
        Err(err) => {
            metrics_handle.fail();
            error!(request_id = metrics_handle.id(), error = %err, "upstream request failed");
            return error_response(StatusCode::BAD_GATEWAY, json!({
                "type": "error",
                "error": {"type": "api_error", "message": err.to_string()}
            }));
        }
    };

    let status = resp.status();
    let response_text = match resp.text().await {
        Ok(text) => text,
        Err(err) => {
            metrics_handle.fail();
            error!(request_id = metrics_handle.id(), error = %err, "failed reading upstream response body");
            return error_response(StatusCode::BAD_GATEWAY, json!({
                "type": "error",
                "error": {"type": "api_error", "message": err.to_string()}
            }));
        }
    };

    if !status.is_success() {
        metrics_handle.fail();
        let upstream_json = serde_json::from_str::<Value>(&response_text).unwrap_or_else(|_| json!({"error": {"message": response_text}}));
        return error_response(status, build_error_data(status.as_u16(), &upstream_json));
    }

    let openai_resp: Value = match serde_json::from_str(&response_text) {
        Ok(value) => value,
        Err(err) => {
            metrics_handle.fail();
            error!(request_id = metrics_handle.id(), error = %err, "failed to parse upstream JSON response");
            return error_response(StatusCode::BAD_GATEWAY, json!({
                "type": "error",
                "error": {"type": "api_error", "message": err.to_string()}
            }));
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

    let anthropic_resp = openai_to_anthropic_response(&openai_resp, &request_model, thinking_enabled);
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
    let url = format!("{}/responses", state.config.upstream.base_url);

    let resp = match state
        .client
        .post(&url)
        .headers(upstream_headers(&state.config))
        .json(&openai_body)
        .send()
        .await
    {
        Ok(resp) => resp,
        Err(err) => {
            metrics_handle.fail();
            error!(request_id = metrics_handle.id(), error = %err, "failed to open upstream SSE stream");
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
        warn!(request_id = metrics_handle.id(), status = %status, body = %truncate_str(&text), "upstream SSE request returned error");
        let upstream_json = serde_json::from_str::<Value>(&text).unwrap_or_else(|_| json!({"error": {"message": text}}));
        return sse_error_response(build_error_data(status.as_u16(), &upstream_json));
    }

    let mut converter = StreamConverter::new(request_model.clone(), thinking_enabled, Some(metrics_handle.clone()));
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
                Err(_) => {
                    outputs.push(Ok(Bytes::from_static(b"")));
                }
            }
            stream::iter(outputs)
        });

    let mut response = Response::new(Body::from_stream(stream));
    *response.status_mut() = StatusCode::OK;
    response.headers_mut().insert("content-type", HeaderValue::from_static("text/event-stream"));
    response.headers_mut().insert("cache-control", HeaderValue::from_static("no-cache"));
    response.headers_mut().insert("connection", HeaderValue::from_static("keep-alive"));
    response.headers_mut().insert("x-accel-buffering", HeaderValue::from_static("no"));
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

    let url = format!("{}/responses/input_tokens", state.config.upstream.base_url);
    let resp = state
        .client
        .post(&url)
        .headers(upstream_headers(&state.config))
        .json(&openai_body)
        .timeout(Duration::from_secs(30))
        .send()
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
            warn!(status = %status, body = %truncate_str(&text), "upstream input_tokens request failed; using fallback");
            let estimated = estimate_tokens_from_body(&body);
            info!(estimated, "count-tokens fallback used after upstream non-success status");
            Json(json!({"input_tokens": estimated})).into_response()
        }
        Err(err) => {
            warn!(error = %err, "upstream input_tokens request errored; using fallback");
            let estimated = estimate_tokens_from_body(&body);
            info!(estimated, "count-tokens fallback used after upstream request error");
            Json(json!({"input_tokens": estimated})).into_response()
        }
    }
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

fn upstream_headers(config: &AppConfig) -> HeaderMap {
    let mut headers = HeaderMap::new();
    let bearer = format!("Bearer {}", config.upstream.api_key);
    headers.insert("authorization", HeaderValue::from_str(&bearer).unwrap_or_else(|_| HeaderValue::from_static("Bearer ")));
    headers.insert("content-type", HeaderValue::from_static("application/json"));
    headers
}

fn error_response(status: StatusCode, body: Value) -> Response<Body> {
    let mut response = Json(body).into_response();
    *response.status_mut() = status;
    response
}

fn sse_error_response(body: Value) -> Response<Body> {
    let payload = format!("event: error\ndata: {}\n\n", body);
    let mut response = Response::new(Body::from(payload));
    *response.status_mut() = StatusCode::OK;
    response.headers_mut().insert("content-type", HeaderValue::from_static("text/event-stream"));
    response
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
        format!("{}...<truncated {} chars>", &value[..LIMIT], value.len() - LIMIT)
    }
}
