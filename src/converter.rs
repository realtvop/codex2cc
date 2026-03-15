use std::collections::HashMap;

use serde_json::{json, Map, Value};
use uuid::Uuid;

use crate::metrics::RequestMetricsHandle;

pub fn anthropic_to_openai_request(body: &Value, model_map: &HashMap<String, String>) -> Value {
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .map(|model| model_map.get(model).cloned().unwrap_or_else(|| model.to_string()))
        .unwrap_or_default();

    let mut req = Map::new();
    req.insert("model".to_string(), Value::String(model));
    req.insert(
        "input".to_string(),
        Value::Array(convert_messages(
            body.get("messages").and_then(Value::as_array),
        )),
    );

    if let Some(system) = body.get("system") {
        let instructions = match system {
            Value::Array(blocks) => blocks
                .iter()
                .filter_map(|block| block.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n"),
            Value::String(text) => text.clone(),
            _ => String::new(),
        };
        req.insert("instructions".to_string(), Value::String(instructions));
    }

    copy_key(body, &mut req, "temperature");
    copy_key(body, &mut req, "top_p");
    copy_key(body, &mut req, "stream");

    if let Some(max_tokens) = body.get("max_tokens") {
        req.insert("max_output_tokens".to_string(), max_tokens.clone());
    }

    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        let converted = convert_tools(tools);
        if !converted.is_empty() {
            req.insert("tools".to_string(), Value::Array(converted));
        }
    }

    if let Some(tool_choice) = body.get("tool_choice") {
        if let Some(value) = convert_tool_choice(tool_choice) {
            req.insert("tool_choice".to_string(), value);
        }
    }

    if let Some(thinking) = body.get("thinking").and_then(Value::as_object) {
        if thinking.get("type").and_then(Value::as_str) == Some("enabled") {
            let budget = thinking.get("budget_tokens").and_then(Value::as_i64).unwrap_or(1024);
            let effort = if budget >= 8192 {
                "high"
            } else if budget >= 2048 {
                "medium"
            } else {
                "low"
            };
            req.insert("reasoning".to_string(), json!({"effort": effort, "summary": "auto"}));
        }
    }

    req.insert("store".to_string(), Value::Bool(false));
    Value::Object(req)
}

pub fn openai_to_anthropic_response(resp: &Value, request_model: &str, thinking_enabled: bool) -> Value {
    let usage = resp.get("usage").cloned().unwrap_or_else(|| json!({}));
    let input_tokens = usage.get("input_tokens").and_then(Value::as_i64).unwrap_or(0);
    let output_tokens = usage.get("output_tokens").and_then(Value::as_i64).unwrap_or(0);

    json!({
        "id": resp.get("id").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| format!("msg_{}", Uuid::new_v4().simple())),
        "type": "message",
        "role": "assistant",
        "model": request_model,
        "content": extract_content_blocks(resp.get("output").and_then(Value::as_array), thinking_enabled),
        "stop_reason": determine_stop_reason(resp),
        "stop_sequence": Value::Null,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
    })
}

pub fn build_error_data(status: u16, resp: &Value) -> Value {
    let error = resp.get("error").unwrap_or(resp);
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| resp.to_string());

    let error_type = match status {
        401 => "authentication_error",
        429 => "rate_limit_error",
        400 => "invalid_request_error",
        404 => "not_found_error",
        500..=599 => "api_error",
        _ => "api_error",
    };

    json!({
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        }
    })
}

pub struct StreamConverter {
    request_model: String,
    thinking_enabled: bool,
    block_index: usize,
    input_tokens: i64,
    output_tokens: i64,
    streamed_output_tokens: i64,
    message_started: bool,
    active_blocks: HashMap<String, usize>,
    function_call_buffers: HashMap<String, Value>,
    metrics_handle: Option<RequestMetricsHandle>,
}

impl StreamConverter {
    pub fn new(request_model: String, thinking_enabled: bool, metrics_handle: Option<RequestMetricsHandle>) -> Self {
        Self {
            request_model,
            thinking_enabled,
            block_index: 0,
            input_tokens: 0,
            output_tokens: 0,
            streamed_output_tokens: 0,
            message_started: false,
            active_blocks: HashMap::new(),
            function_call_buffers: HashMap::new(),
            metrics_handle,
        }
    }

    pub fn process_event(&mut self, event_type: &str, data: &Value) -> String {
        let mut out = String::new();

        match event_type {
            "response.created" | "response.in_progress" => {
                if !self.message_started {
                    out.push_str(&self.emit_message_start());
                }
            }
            "response.output_item.added" => {
                let item = data.get("item").unwrap_or(&Value::Null);
                if item.get("type").and_then(Value::as_str) == Some("function_call") {
                    let item_id = item.get("id").and_then(Value::as_str).unwrap_or_default().to_string();
                    let call_id = item.get("call_id").and_then(Value::as_str).unwrap_or(&item_id).to_string();
                    let idx = self.block_index;
                    self.active_blocks.insert(item_id.clone(), idx);
                    self.function_call_buffers.insert(item_id, json!({
                        "name": item.get("name").and_then(Value::as_str).unwrap_or_default(),
                        "call_id": call_id,
                        "arguments": "",
                    }));
                    out.push_str(&self.sse("content_block_start", json!({
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": {
                            "type": "tool_use",
                            "id": call_id,
                            "name": item.get("name").and_then(Value::as_str).unwrap_or_default(),
                            "input": {},
                        }
                    })));
                    self.block_index += 1;
                }
            }
            "response.content_part.added" => {
                let part = data.get("part").unwrap_or(&Value::Null);
                if part.get("type").and_then(Value::as_str) == Some("output_text") {
                    let key = format!(
                        "text_{}_{}",
                        data.get("output_index").and_then(Value::as_i64).unwrap_or(0),
                        data.get("content_index").and_then(Value::as_i64).unwrap_or(0)
                    );
                    let idx = self.block_index;
                    self.active_blocks.insert(key, idx);
                    out.push_str(&self.sse("content_block_start", json!({
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": {"type": "text", "text": ""},
                    })));
                    self.block_index += 1;
                }
            }
            "response.output_text.delta" => {
                let key = format!(
                    "text_{}_{}",
                    data.get("output_index").and_then(Value::as_i64).unwrap_or(0),
                    data.get("content_index").and_then(Value::as_i64).unwrap_or(0)
                );
                let delta_text = data.get("delta").and_then(Value::as_str).unwrap_or_default();
                self.streamed_output_tokens += 1;
                let idx = self.active_blocks.get(&key).copied().unwrap_or(0);
                out.push_str(&self.sse("content_block_delta", json!({
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "text_delta", "text": delta_text},
                })));
            }
            "response.output_text.done" => {
                let key = format!(
                    "text_{}_{}",
                    data.get("output_index").and_then(Value::as_i64).unwrap_or(0),
                    data.get("content_index").and_then(Value::as_i64).unwrap_or(0)
                );
                let idx = self.active_blocks.get(&key).copied().unwrap_or(0);
                out.push_str(&self.sse("content_block_stop", json!({
                    "type": "content_block_stop",
                    "index": idx,
                })));
            }
            "response.function_call_arguments.delta" => {
                let item_id = data.get("item_id").and_then(Value::as_str).unwrap_or_default();
                let delta = data.get("delta").and_then(Value::as_str).unwrap_or_default();
                if let Some(buffer) = self.function_call_buffers.get_mut(item_id) {
                    let current = buffer.get("arguments").and_then(Value::as_str).unwrap_or_default().to_string();
                    *buffer = json!({
                        "name": buffer.get("name").and_then(Value::as_str).unwrap_or_default(),
                        "call_id": buffer.get("call_id").and_then(Value::as_str).unwrap_or_default(),
                        "arguments": format!("{}{}", current, delta),
                    });
                }
                let idx = self.active_blocks.get(item_id).copied().unwrap_or(0);
                out.push_str(&self.sse("content_block_delta", json!({
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "input_json_delta", "partial_json": delta},
                })));
            }
            "response.function_call_arguments.done" => {
                let item_id = data.get("item_id").and_then(Value::as_str).unwrap_or_default();
                let idx = self.active_blocks.get(item_id).copied().unwrap_or(0);
                out.push_str(&self.sse("content_block_stop", json!({
                    "type": "content_block_stop",
                    "index": idx,
                })));
            }
            "response.reasoning_summary_text.delta" if self.thinking_enabled => {
                let key = format!(
                    "thinking_{}_{}",
                    data.get("output_index").and_then(Value::as_i64).unwrap_or(0),
                    data.get("summary_index").and_then(Value::as_i64).unwrap_or(0)
                );
                let idx = if let Some(idx) = self.active_blocks.get(&key).copied() {
                    idx
                } else {
                    let idx = self.block_index;
                    self.active_blocks.insert(key.clone(), idx);
                    out.push_str(&self.sse("content_block_start", json!({
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": {"type": "thinking", "thinking": ""},
                    })));
                    self.block_index += 1;
                    idx
                };
                out.push_str(&self.sse("content_block_delta", json!({
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "thinking_delta", "thinking": data.get("delta").and_then(Value::as_str).unwrap_or_default()},
                })));
            }
            "response.reasoning_summary_text.done" if self.thinking_enabled => {
                let key = format!(
                    "thinking_{}_{}",
                    data.get("output_index").and_then(Value::as_i64).unwrap_or(0),
                    data.get("summary_index").and_then(Value::as_i64).unwrap_or(0)
                );
                let idx = self.active_blocks.get(&key).copied().unwrap_or(0);
                out.push_str(&self.sse("content_block_stop", json!({
                    "type": "content_block_stop",
                    "index": idx,
                })));
            }
            "response.completed" => {
                let response = data.get("response").unwrap_or(&Value::Null);
                self.input_tokens = response
                    .get("usage")
                    .and_then(|usage| usage.get("input_tokens"))
                    .and_then(Value::as_i64)
                    .unwrap_or(0);
                self.output_tokens = response
                    .get("usage")
                    .and_then(|usage| usage.get("output_tokens"))
                    .and_then(Value::as_i64)
                    .unwrap_or(self.streamed_output_tokens);
                if let Some(metrics_handle) = self.metrics_handle.take() {
                    metrics_handle.finish(self.input_tokens.max(0) as u64, self.output_tokens.max(0) as u64);
                }
                out.push_str(&self.sse("message_delta", json!({
                    "type": "message_delta",
                    "delta": {"stop_reason": determine_stop_reason(response), "stop_sequence": Value::Null},
                    "usage": {"output_tokens": self.output_tokens},
                })));
                out.push_str(&self.sse("message_stop", json!({"type": "message_stop"})));
            }
            _ => {}
        }

        out
    }

    fn emit_message_start(&mut self) -> String {
        self.message_started = true;
        self.sse(
            "message_start",
            json!({
                "type": "message_start",
                "message": {
                    "id": format!("msg_{}", Uuid::new_v4().simple()),
                    "type": "message",
                    "role": "assistant",
                    "model": self.request_model,
                    "content": [],
                    "stop_reason": Value::Null,
                    "stop_sequence": Value::Null,
                    "usage": {
                        "input_tokens": self.input_tokens,
                        "output_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                    }
                }
            }),
        )
    }

    fn sse(&self, event: &str, data: Value) -> String {
        format!("event: {event}\ndata: {data}\n\n")
    }
}

fn convert_messages(messages: Option<&Vec<Value>>) -> Vec<Value> {
    let mut items = Vec::new();

    for msg in messages.into_iter().flatten() {
        let role = msg.get("role").and_then(Value::as_str).unwrap_or_default();
        let content = msg.get("content").unwrap_or(&Value::Null);

        match role {
            "user" => convert_user_message(content, &mut items),
            "assistant" => convert_assistant_message(content, &mut items),
            _ => {}
        }
    }

    items
}

fn convert_user_message(content: &Value, items: &mut Vec<Value>) {
    match content {
        Value::String(text) => items.push(json!({"role": "user", "content": text, "type": "message"})),
        Value::Array(blocks) => {
            let mut regular_blocks = Vec::new();
            for block in blocks {
                if block.get("type").and_then(Value::as_str) == Some("tool_result") {
                    if !regular_blocks.is_empty() {
                        items.push(json!({
                            "role": "user",
                            "content": convert_content_blocks_to_openai(&regular_blocks),
                            "type": "message",
                        }));
                        regular_blocks.clear();
                    }
                    let result_content = if let Some(arr) = block.get("content").and_then(Value::as_array) {
                        arr.iter()
                            .filter(|entry| entry.get("type").and_then(Value::as_str) == Some("text"))
                            .filter_map(|entry| entry.get("text").and_then(Value::as_str))
                            .collect::<Vec<_>>()
                            .join("\n")
                    } else {
                        block.get("content").and_then(Value::as_str).unwrap_or_default().to_string()
                    };
                    items.push(json!({
                        "type": "function_call_output",
                        "call_id": block.get("tool_use_id").and_then(Value::as_str).unwrap_or_default(),
                        "output": result_content,
                    }));
                } else {
                    regular_blocks.push(block.clone());
                }
            }
            if !regular_blocks.is_empty() {
                items.push(json!({
                    "role": "user",
                    "content": convert_content_blocks_to_openai(&regular_blocks),
                    "type": "message",
                }));
            }
        }
        _ => {}
    }
}

fn convert_assistant_message(content: &Value, items: &mut Vec<Value>) {
    match content {
        Value::String(text) => items.push(json!({"role": "assistant", "content": text, "type": "message"})),
        Value::Array(blocks) => {
            let mut text_parts = Vec::new();
            for block in blocks {
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => text_parts.push(block.get("text").and_then(Value::as_str).unwrap_or_default().to_string()),
                    Some("thinking") => {}
                    Some("tool_use") => {
                        if !text_parts.is_empty() {
                            items.push(json!({
                                "role": "assistant",
                                "content": text_parts.join("\n"),
                                "type": "message",
                            }));
                            text_parts.clear();
                        }
                        items.push(json!({
                            "type": "function_call",
                            "id": format!("fc_{}", block.get("id").and_then(Value::as_str).unwrap_or_default()),
                            "call_id": block.get("id").and_then(Value::as_str).unwrap_or_default(),
                            "name": block.get("name").and_then(Value::as_str).unwrap_or_default(),
                            "arguments": serde_json::to_string(block.get("input").unwrap_or(&json!({}))).unwrap_or_else(|_| "{}".to_string()),
                        }));
                    }
                    _ => {}
                }
            }
            if !text_parts.is_empty() {
                items.push(json!({
                    "role": "assistant",
                    "content": text_parts.join("\n"),
                    "type": "message",
                }));
            }
        }
        _ => {}
    }
}

fn convert_content_blocks_to_openai(blocks: &[Value]) -> Vec<Value> {
    let mut out = Vec::new();
    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => out.push(json!({"type": "input_text", "text": block.get("text").and_then(Value::as_str).unwrap_or_default()})),
            Some("image") => {
                let source = block.get("source").unwrap_or(&Value::Null);
                match source.get("type").and_then(Value::as_str) {
                    Some("base64") => {
                        let media = source.get("media_type").and_then(Value::as_str).unwrap_or("image/png");
                        let data = source.get("data").and_then(Value::as_str).unwrap_or_default();
                        out.push(json!({
                            "type": "input_image",
                            "image_url": format!("data:{media};base64,{data}"),
                        }));
                    }
                    Some("url") => out.push(json!({
                        "type": "input_image",
                        "image_url": source.get("url").and_then(Value::as_str).unwrap_or_default(),
                    })),
                    _ => {}
                }
            }
            _ => {}
        }
    }
    out
}

fn convert_tools(tools: &[Value]) -> Vec<Value> {
    tools
        .iter()
        .filter_map(|tool| match tool.get("type").and_then(Value::as_str) {
            None | Some("custom") => Some(json!({
                "type": "function",
                "name": tool.get("name").and_then(Value::as_str).unwrap_or_default(),
                "description": tool.get("description").and_then(Value::as_str).unwrap_or_default(),
                "parameters": tool.get("input_schema").cloned().unwrap_or_else(|| json!({})),
                "strict": false,
            })),
            _ => None,
        })
        .collect()
}

fn convert_tool_choice(tool_choice: &Value) -> Option<Value> {
    match tool_choice.get("type").and_then(Value::as_str) {
        Some("auto") => Some(Value::String("auto".to_string())),
        Some("any") => Some(Value::String("required".to_string())),
        Some("none") => Some(Value::String("none".to_string())),
        Some("tool") => Some(json!({
            "type": "function",
            "name": tool_choice.get("name").and_then(Value::as_str).unwrap_or_default(),
        })),
        _ => None,
    }
}

fn extract_content_blocks(output: Option<&Vec<Value>>, thinking_enabled: bool) -> Vec<Value> {
    let mut blocks = Vec::new();

    for item in output.into_iter().flatten() {
        match item.get("type").and_then(Value::as_str) {
            Some("message") => {
                if let Some(parts) = item.get("content").and_then(Value::as_array) {
                    for part in parts {
                        match part.get("type").and_then(Value::as_str) {
                            Some("output_text") => blocks.push(json!({"type": "text", "text": part.get("text").and_then(Value::as_str).unwrap_or_default()})),
                            Some("refusal") => blocks.push(json!({"type": "text", "text": format!("[Refused]: {}", part.get("refusal").and_then(Value::as_str).unwrap_or_default())})),
                            _ => {}
                        }
                    }
                }
            }
            Some("function_call") => {
                let input = item
                    .get("arguments")
                    .and_then(Value::as_str)
                    .and_then(|arguments| serde_json::from_str::<Value>(arguments).ok())
                    .unwrap_or_else(|| json!({}));
                blocks.push(json!({
                    "type": "tool_use",
                    "id": item.get("call_id").or_else(|| item.get("id")).and_then(Value::as_str).unwrap_or_default(),
                    "name": item.get("name").and_then(Value::as_str).unwrap_or_default(),
                    "input": input,
                }));
            }
            Some("reasoning") if thinking_enabled => {
                if let Some(summary) = item.get("summary").and_then(Value::as_array) {
                    for entry in summary {
                        if entry.get("type").and_then(Value::as_str) == Some("summary_text") {
                            blocks.push(json!({
                                "type": "thinking",
                                "thinking": entry.get("text").and_then(Value::as_str).unwrap_or_default(),
                                "signature": "",
                            }));
                        }
                    }
                }
            }
            _ => {}
        }
    }

    blocks
}

fn determine_stop_reason(resp: &Value) -> &'static str {
    if resp.get("status").and_then(Value::as_str) == Some("incomplete") {
        return "max_tokens";
    }

    if resp
        .get("output")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .any(|item| item.get("type").and_then(Value::as_str) == Some("function_call"))
    {
        return "tool_use";
    }

    "end_turn"
}

fn copy_key(body: &Value, req: &mut Map<String, Value>, key: &str) {
    if let Some(value) = body.get(key) {
        req.insert(key.to_string(), value.clone());
    }
}
