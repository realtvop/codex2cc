# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**codextocc** is a Rust proxy server that translates Anthropic-style `/v1/messages` requests into OpenAI Responses API calls. Clients send Claude API-compatible requests, and the proxy converts and forwards them to OpenAI endpoints, converting responses back to Anthropic format.

## Build & Test Commands

```bash
cargo build                    # Dev build
cargo build --release --locked # Release build (CI uses this)
cargo test --locked            # Run all tests
cargo test converter           # Run tests for a specific module
cargo test -- --nocapture      # Run tests with stdout visible
cargo run -- serve             # Run the server (auto-generates config.yaml if missing)
cargo run -- pool login <alias> # Add an account to the pool (account_pool mode)
cargo run -- pool list          # List pool accounts
```

Rust edition 2021, MSRV 1.85. CI builds for linux-x86_64, windows-x86_64, macos-x86_64, macos-aarch64.

## Architecture

Six source files in `src/`:

- **main.rs** — Axum HTTP server and CLI entrypoint. Handles `/v1/messages` for both streaming and non-streaming requests. Implements the credential fallback/retry strategy: on 401/403/429 in account_pool mode, retries with the next eligible account. CLI subcommands (`pool login/list/enable/disable/remove/refresh`) are dispatched here.

- **converter.rs** — Bidirectional Anthropic <-> OpenAI protocol translation. `anthropic_to_openai_request()` converts message blocks (text, images, tool_use, tool_result), system instructions, tools, tool_choice, and thinking/reasoning budget. `openai_to_anthropic_response()` converts back. `StreamConverter` is a stateful converter that processes OpenAI SSE events and emits Anthropic SSE format incrementally.

- **config.rs** — Loads `config.yaml` (or `CONFIG_PATH` env var) with env var overrides for all fields. Auto-generates a default config with a random client API key if the file is missing. Three auth modes: `local_codex`, `config_api_key`, `account_pool`.

- **upstream_auth.rs** — Credential resolution for `local_codex` mode. Reads `~/.codex/auth.json`, handles JWT expiry detection (60s skew), OAuth token refresh via OpenID discovery, file-modification-time caching, and fallback to configured API key.

- **account_pool.rs** — Multi-account OAuth management for `account_pool` mode. Device-code login flow, persistent JSON store with 0600 permissions, quota tracking via response headers, and intelligent account selection (sorted by: remaining quota > 0, earliest reset_at, highest remaining, least recently used).

- **metrics.rs** — Simple request tracking with atomic ID allocation and active request set.

## Key Patterns

- **Async everywhere**: Tokio runtime, all I/O is async. Streaming uses `tokio_stream`.
- **Shared state**: `Arc<AppConfig>`, `Arc<MetricsRegistry>`, `Arc<Mutex<T>>` for mutable shared state.
- **Error handling**: `anyhow::Result<T>` throughout, `anyhow::bail!()` for early returns.
- **Config layering**: YAML file values are overridden by environment variables when both are present.
- **model_map**: Config-driven model name translation (e.g., `claude-3` -> `gpt-5`) applied in converter.

## Configuration

See `config.example.yaml` for the full template and `README.md` for all fields and env var overrides. Key auth modes:

- `local_codex`: reads `~/.codex/auth.json`, refreshes tokens, falls back to configured API key
- `config_api_key`: uses only `upstream.api_key`
- `account_pool`: manages multiple OAuth accounts, no local Codex fallback, returns 429 when exhausted
