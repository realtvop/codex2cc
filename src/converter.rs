use std::collections::HashMap;

use serde_json::{json, Map, Value};
use uuid::Uuid;

use crate::metrics::RequestMetricsHandle;

pub fn anthropic_to_openai_request(body: &Value, model_map: &HashMap<String, String>) -> Value {
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .map(|model| {
            model_map
                .get(model)
                .cloned()
                .unwrap_or_else(|| model.to_string())
        })
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
            let budget = thinking
                .get("budget_tokens")
                .and_then(Value::as_i64)
                .unwrap_or(1024);
            let effort = if model.starts_with("gpt") {
                if budget >= 8192 {
                    "xhigh"
                } else if budget >= 2048 {
                    "high"
                } else if budget >= 512 {
                    "medium"
                } else {
                    "low"
                }
            } else if budget >= 8192 {
                "high"
            } else if budget >= 2048 {
                "medium"
            } else {
                "low"
            };
            req.insert(
                "reasoning".to_string(),
                json!({"effort": effort, "summary": "auto"}),
            );
        }
    }

    req.insert("store".to_string(), Value::Bool(false));
    Value::Object(req)
}

pub fn openai_to_anthropic_response(
    resp: &Value,
    request_model: &str,
    thinking_enabled: bool,
) -> Value {
    let usage = resp.get("usage").cloned().unwrap_or_else(|| json!({}));
    let input_tokens = usage
        .get("input_tokens")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let output_tokens = usage
        .get("output_tokens")
        .and_then(Value::as_i64)
        .unwrap_or(0);

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
    pub fn new(
        request_model: String,
        thinking_enabled: bool,
        metrics_handle: Option<RequestMetricsHandle>,
    ) -> Self {
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
                    let item_id = item
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    let call_id = item
                        .get("call_id")
                        .and_then(Value::as_str)
                        .unwrap_or(&item_id)
                        .to_string();
                    let idx = self.block_index;
                    self.active_blocks.insert(item_id.clone(), idx);
                    self.function_call_buffers.insert(
                        item_id,
                        json!({
                            "name": item.get("name").and_then(Value::as_str).unwrap_or_default(),
                            "call_id": call_id,
                            "arguments": "",
                        }),
                    );
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
                        data.get("output_index")
                            .and_then(Value::as_i64)
                            .unwrap_or(0),
                        data.get("content_index")
                            .and_then(Value::as_i64)
                            .unwrap_or(0)
                    );
                    let idx = self.block_index;
                    self.active_blocks.insert(key, idx);
                    out.push_str(&self.sse(
                        "content_block_start",
                        json!({
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {"type": "text", "text": ""},
                        }),
                    ));
                    self.block_index += 1;
                }
            }
            "response.output_text.delta" => {
                let key = format!(
                    "text_{}_{}",
                    data.get("output_index")
                        .and_then(Value::as_i64)
                        .unwrap_or(0),
                    data.get("content_index")
                        .and_then(Value::as_i64)
                        .unwrap_or(0)
                );
                let delta_text = data
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                self.streamed_output_tokens += 1;
                let idx = self.active_blocks.get(&key).copied().unwrap_or(0);
                out.push_str(&self.sse(
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "text_delta", "text": delta_text},
                    }),
                ));
            }
            "response.output_text.done" => {
                let key = format!(
                    "text_{}_{}",
                    data.get("output_index")
                        .and_then(Value::as_i64)
                        .unwrap_or(0),
                    data.get("content_index")
                        .and_then(Value::as_i64)
                        .unwrap_or(0)
                );
                let idx = self.active_blocks.get(&key).copied().unwrap_or(0);
                out.push_str(&self.sse(
                    "content_block_stop",
                    json!({
                        "type": "content_block_stop",
                        "index": idx,
                    }),
                ));
            }
            "response.function_call_arguments.delta" => {
                let item_id = data
                    .get("item_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let delta = data
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if let Some(buffer) = self.function_call_buffers.get_mut(item_id) {
                    let current = buffer
                        .get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    *buffer = json!({
                        "name": buffer.get("name").and_then(Value::as_str).unwrap_or_default(),
                        "call_id": buffer.get("call_id").and_then(Value::as_str).unwrap_or_default(),
                        "arguments": format!("{}{}", current, delta),
                    });
                }
                let idx = self.active_blocks.get(item_id).copied().unwrap_or(0);
                out.push_str(&self.sse(
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "input_json_delta", "partial_json": delta},
                    }),
                ));
            }
            "response.function_call_arguments.done" => {
                let item_id = data
                    .get("item_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let idx = self.active_blocks.get(item_id).copied().unwrap_or(0);
                out.push_str(&self.sse(
                    "content_block_stop",
                    json!({
                        "type": "content_block_stop",
                        "index": idx,
                    }),
                ));
            }
            "response.reasoning_summary_text.delta" if self.thinking_enabled => {
                let key = format!(
                    "thinking_{}_{}",
                    data.get("output_index")
                        .and_then(Value::as_i64)
                        .unwrap_or(0),
                    data.get("summary_index")
                        .and_then(Value::as_i64)
                        .unwrap_or(0)
                );
                let idx = if let Some(idx) = self.active_blocks.get(&key).copied() {
                    idx
                } else {
                    let idx = self.block_index;
                    self.active_blocks.insert(key.clone(), idx);
                    out.push_str(&self.sse(
                        "content_block_start",
                        json!({
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {"type": "thinking", "thinking": ""},
                        }),
                    ));
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
                    data.get("output_index")
                        .and_then(Value::as_i64)
                        .unwrap_or(0),
                    data.get("summary_index")
                        .and_then(Value::as_i64)
                        .unwrap_or(0)
                );
                let idx = self.active_blocks.get(&key).copied().unwrap_or(0);
                out.push_str(&self.sse(
                    "content_block_stop",
                    json!({
                        "type": "content_block_stop",
                        "index": idx,
                    }),
                ));
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
                    metrics_handle.finish(
                        self.input_tokens.max(0) as u64,
                        self.output_tokens.max(0) as u64,
                    );
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
        Value::String(text) => {
            items.push(json!({"role": "user", "content": text, "type": "message"}))
        }
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
                    let result_content =
                        if let Some(arr) = block.get("content").and_then(Value::as_array) {
                            arr.iter()
                                .filter(|entry| {
                                    entry.get("type").and_then(Value::as_str) == Some("text")
                                })
                                .filter_map(|entry| entry.get("text").and_then(Value::as_str))
                                .collect::<Vec<_>>()
                                .join("\n")
                        } else {
                            block
                                .get("content")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string()
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
        Value::String(text) => {
            items.push(json!({"role": "assistant", "content": text, "type": "message"}))
        }
        Value::Array(blocks) => {
            let mut text_parts = Vec::new();
            for block in blocks {
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => text_parts.push(
                        block
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                    ),
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};
    use std::collections::HashMap;

    fn parse_sse(output: &str) -> Vec<(String, Value)> {
        output
            .split("\n\n")
            .filter(|chunk| !chunk.trim().is_empty())
            .map(|chunk| {
                let mut event = None;
                let mut data = None;
                for line in chunk.lines() {
                    if let Some(rest) = line.strip_prefix("event: ") {
                        event = Some(rest.to_string());
                    } else if let Some(rest) = line.strip_prefix("data: ") {
                        data = Some(serde_json::from_str(rest).expect("valid SSE JSON"));
                    }
                }
                (
                    event.expect("missing SSE event name"),
                    data.expect("missing SSE data"),
                )
            })
            .collect()
    }

    #[test]
    fn anthropic_request_maps_core_fields_and_messages() {
        let model_map = HashMap::from([("claude-3-7-sonnet".to_string(), "gpt-5".to_string())]);
        let body = json!({
            "model": "claude-3-7-sonnet",
            "system": [
                {"text": "system line 1"},
                {"text": "system line 2"}
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "image", "source": {"type": "url", "url": "https://example.com/image.png"}},
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": [
                                {"type": "text", "text": "tool line 1"},
                                {"type": "text", "text": "tool line 2"}
                            ]
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "visible reply"},
                        {"type": "thinking", "thinking": "hidden reasoning"},
                        {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"city": "Paris"}}
                    ]
                }
            ],
            "temperature": 0.2,
            "top_p": 0.9,
            "stream": true,
            "max_tokens": 123,
            "tools": [
                {
                    "type": "custom",
                    "name": "lookup",
                    "description": "Look up a city",
                    "input_schema": {"type": "object"}
                },
                {
                    "type": "computer",
                    "name": "ignored"
                }
            ],
            "tool_choice": {"type": "tool", "name": "lookup"}
        });

        let request = anthropic_to_openai_request(&body, &model_map);
        let input = request["input"].as_array().expect("input array");

        assert_eq!(request["model"], "gpt-5");
        assert_eq!(request["instructions"], "system line 1\nsystem line 2");
        assert_eq!(request["temperature"], json!(0.2));
        assert_eq!(request["top_p"], json!(0.9));
        assert_eq!(request["stream"], json!(true));
        assert_eq!(request["max_output_tokens"], json!(123));
        assert_eq!(request["store"], json!(false));
        assert_eq!(
            request["tools"],
            json!([{
                "type": "function",
                "name": "lookup",
                "description": "Look up a city",
                "parameters": {"type": "object"},
                "strict": false
            }])
        );
        assert_eq!(
            request["tool_choice"],
            json!({"type": "function", "name": "lookup"})
        );

        assert_eq!(
            input[0],
            json!({
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "hello"},
                    {"type": "input_image", "image_url": "https://example.com/image.png"}
                ],
                "type": "message"
            })
        );
        assert_eq!(
            input[1],
            json!({
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "tool line 1\ntool line 2"
            })
        );
        assert_eq!(
            input[2],
            json!({
                "role": "assistant",
                "content": "visible reply",
                "type": "message"
            })
        );
        assert_eq!(
            input[3],
            json!({
                "type": "function_call",
                "id": "fc_call_1",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": "{\"city\":\"Paris\"}"
            })
        );
    }

    #[test]
    fn anthropic_request_maps_reasoning_effort_and_tool_choice_shortcuts() {
        let model_map = HashMap::new();

        // Non-GPT model: budget thresholds unchanged
        for (budget, expected_effort) in [(1024, "low"), (2048, "medium"), (8192, "high")] {
            let body = json!({
                "model": "claude",
                "system": "system text",
                "messages": [],
                "thinking": {"type": "enabled", "budget_tokens": budget},
                "tool_choice": {"type": "auto"}
            });

            let request = anthropic_to_openai_request(&body, &model_map);
            assert_eq!(request["instructions"], "system text");
            assert_eq!(
                request["reasoning"],
                json!({"effort": expected_effort, "summary": "auto"})
            );
            assert_eq!(request["tool_choice"], json!("auto"));
        }

        // GPT model: xhigh tier at 8192, shifted thresholds
        let mut gpt_map = HashMap::new();
        gpt_map.insert("claude-3".to_string(), "gpt-5".to_string());
        for (budget, expected_effort) in [
            (256, "low"),
            (512, "medium"),
            (2048, "high"),
            (8192, "xhigh"),
        ] {
            let body = json!({
                "model": "claude-3",
                "messages": [],
                "thinking": {"type": "enabled", "budget_tokens": budget}
            });

            let request = anthropic_to_openai_request(&body, &gpt_map);
            assert_eq!(
                request["reasoning"],
                json!({"effort": expected_effort, "summary": "auto"}),
                "GPT model with budget {budget} should map to effort \"{expected_effort}\""
            );
        }

        let required = anthropic_to_openai_request(
            &json!({"messages": [], "tool_choice": {"type": "any"}}),
            &model_map,
        );
        assert_eq!(required["tool_choice"], json!("required"));

        let none = anthropic_to_openai_request(
            &json!({"messages": [], "tool_choice": {"type": "none"}}),
            &model_map,
        );
        assert_eq!(none["tool_choice"], json!("none"));
    }

    #[test]
    fn openai_response_converts_content_blocks_and_usage() {
        let response = json!({
            "id": "resp_123",
            "status": "completed",
            "usage": {"input_tokens": 11, "output_tokens": 7},
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "hello"},
                        {"type": "refusal", "refusal": "nope"}
                    ]
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": "{\"city\":\"Paris\"}"
                },
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "reasoning summary"}]
                }
            ]
        });

        let anthropic = openai_to_anthropic_response(&response, "claude-3", true);

        assert_eq!(anthropic["id"], "resp_123");
        assert_eq!(anthropic["model"], "claude-3");
        assert_eq!(anthropic["stop_reason"], "tool_use");
        assert_eq!(anthropic["usage"]["input_tokens"], json!(11));
        assert_eq!(anthropic["usage"]["output_tokens"], json!(7));
        assert_eq!(
            anthropic["content"],
            json!([
                {"type": "text", "text": "hello"},
                {"type": "text", "text": "[Refused]: nope"},
                {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"city": "Paris"}},
                {"type": "thinking", "thinking": "reasoning summary", "signature": ""}
            ])
        );
    }

    #[test]
    fn openai_response_omits_thinking_when_disabled_and_detects_incomplete_stop_reason() {
        let response = json!({
            "status": "incomplete",
            "output": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "reasoning summary"}]
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "hello"}]
                }
            ]
        });

        let anthropic = openai_to_anthropic_response(&response, "claude-3", false);

        assert_eq!(anthropic["stop_reason"], "max_tokens");
        assert_eq!(
            anthropic["content"],
            json!([{"type": "text", "text": "hello"}])
        );
        assert_eq!(determine_stop_reason(&json!({})), "end_turn");
    }

    #[test]
    fn build_error_data_maps_status_codes_and_messages() {
        let auth_error = build_error_data(401, &json!({"error": {"message": "bad key"}}));
        assert_eq!(auth_error["error"]["type"], "authentication_error");
        assert_eq!(auth_error["error"]["message"], "bad key");

        let api_error = build_error_data(502, &json!({"message": "upstream exploded"}));
        assert_eq!(api_error["error"]["type"], "api_error");
        assert_eq!(api_error["error"]["message"], "upstream exploded");
    }

    #[test]
    fn stream_converter_emits_message_start_once_and_text_events() {
        let mut converter = StreamConverter::new("claude-3".to_string(), false, None);

        let first = parse_sse(&converter.process_event("response.created", &json!({})));
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].0, "message_start");
        assert_eq!(first[0].1["message"]["model"], "claude-3");

        assert!(converter
            .process_event("response.in_progress", &json!({}))
            .is_empty());

        let start = parse_sse(&converter.process_event(
            "response.content_part.added",
            &json!({
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text"}
            }),
        ));
        assert_eq!(start[0].0, "content_block_start");
        assert_eq!(start[0].1["index"], json!(0));
        assert_eq!(start[0].1["content_block"]["type"], "text");

        let delta = parse_sse(&converter.process_event(
            "response.output_text.delta",
            &json!({
                "output_index": 0,
                "content_index": 0,
                "delta": "Hello"
            }),
        ));
        assert_eq!(delta[0].0, "content_block_delta");
        assert_eq!(delta[0].1["delta"]["type"], "text_delta");
        assert_eq!(delta[0].1["delta"]["text"], "Hello");

        let stop = parse_sse(&converter.process_event(
            "response.output_text.done",
            &json!({
                "output_index": 0,
                "content_index": 0
            }),
        ));
        assert_eq!(stop[0].0, "content_block_stop");
        assert_eq!(stop[0].1["index"], json!(0));
    }

    #[test]
    fn stream_converter_emits_function_call_reasoning_and_completion_events() {
        let mut converter = StreamConverter::new("claude-3".to_string(), true, None);

        let tool_start = parse_sse(&converter.process_event(
            "response.output_item.added",
            &json!({
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "lookup"
                }
            }),
        ));
        assert_eq!(tool_start[0].0, "content_block_start");
        assert_eq!(tool_start[0].1["content_block"]["type"], "tool_use");
        assert_eq!(tool_start[0].1["content_block"]["id"], "call_1");

        let tool_delta = parse_sse(&converter.process_event(
            "response.function_call_arguments.delta",
            &json!({
                "item_id": "fc_1",
                "delta": "{\"city\":\"Paris\"}"
            }),
        ));
        assert_eq!(tool_delta[0].0, "content_block_delta");
        assert_eq!(tool_delta[0].1["delta"]["type"], "input_json_delta");
        assert_eq!(
            tool_delta[0].1["delta"]["partial_json"],
            "{\"city\":\"Paris\"}"
        );

        let tool_stop = parse_sse(&converter.process_event(
            "response.function_call_arguments.done",
            &json!({"item_id": "fc_1"}),
        ));
        assert_eq!(tool_stop[0].0, "content_block_stop");

        let thinking_delta = parse_sse(&converter.process_event(
            "response.reasoning_summary_text.delta",
            &json!({
                "output_index": 0,
                "summary_index": 0,
                "delta": "thinking..."
            }),
        ));
        assert_eq!(thinking_delta.len(), 2);
        assert_eq!(thinking_delta[0].0, "content_block_start");
        assert_eq!(thinking_delta[0].1["content_block"]["type"], "thinking");
        assert_eq!(thinking_delta[1].0, "content_block_delta");
        assert_eq!(thinking_delta[1].1["delta"]["type"], "thinking_delta");
        assert_eq!(thinking_delta[1].1["delta"]["thinking"], "thinking...");

        let thinking_stop = parse_sse(&converter.process_event(
            "response.reasoning_summary_text.done",
            &json!({
                "output_index": 0,
                "summary_index": 0
            }),
        ));
        assert_eq!(thinking_stop[0].0, "content_block_stop");

        let completed = parse_sse(&converter.process_event(
            "response.completed",
            &json!({
                "response": {
                    "usage": {"input_tokens": 5, "output_tokens": 3}
                }
            }),
        ));
        assert_eq!(completed.len(), 2);
        assert_eq!(completed[0].0, "message_delta");
        assert_eq!(completed[0].1["usage"]["output_tokens"], json!(3));
        assert_eq!(completed[0].1["delta"]["stop_reason"], "end_turn");
        assert_eq!(completed[1].0, "message_stop");
    }

    #[test]
    fn stream_converter_suppresses_reasoning_events_when_disabled() {
        let mut converter = StreamConverter::new("claude-3".to_string(), false, None);
        assert!(converter
            .process_event(
                "response.reasoning_summary_text.delta",
                &json!({
                    "output_index": 0,
                    "summary_index": 0,
                    "delta": "hidden"
                }),
            )
            .is_empty());
    }
}
