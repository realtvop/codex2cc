# codextocc

Rust proxy that translates Anthropic-style `/v1/messages` requests into OpenAI Responses API calls.

## Configuration

The server loads configuration from `config.yaml` by default. Set `CONFIG_PATH` to use a different file.

If the configured path does not exist at startup, the server generates a new config file, writes a random 32-character lowercase hex client API key to `api_keys`, enables local Codex credential lookup, and exits immediately. The generated key is only written to the config file, not printed to the terminal.

Environment variables override YAML values when both are present.

### Server fields

- `server.host`
  Default: `0.0.0.0`
- `server.port`
  Default: `8082`

### Client auth fields

- `api_keys`
  Optional list of client API keys accepted by this proxy. If omitted or empty, client auth is disabled. Auto-generated configs include one random key by default.

### Upstream fields

- `upstream.base_url`
  Default: `https://api.openai.com/v1`
  Env override: `OPENAI_BASE_URL`
- `upstream.api_key`
  Default: empty
  Env override: `OPENAI_API_KEY`
- `upstream.prefer_local_codex_credentials`
  Default: `false`
  Env override: `PREFER_LOCAL_CODEX_CREDENTIALS`
  Auto-generated configs set this to `true`
- `upstream.local_codex_auth_path`
  Default: `~/.codex/auth.json`
  Env override: `LOCAL_CODEX_AUTH_PATH`
- `upstream.refresh_local_codex_tokens`
  Default: `true`
  Env override: `REFRESH_LOCAL_CODEX_TOKENS`
- `upstream.local_codex_oauth_client_id`
  Default: derived from `tokens.id_token.aud[0]`
  Env override: `LOCAL_CODEX_OAUTH_CLIENT_ID`
- `upstream.local_codex_oauth_token_endpoint`
  Default: discovered from OpenID configuration
  Env override: `LOCAL_CODEX_OAUTH_TOKEN_ENDPOINT`

## Local Codex Credentials

This proxy can optionally read upstream credentials from the local Codex credential file, which is `~/.codex/auth.json` by default.

Auto-generated configs enable this behavior on the next startup and leave `upstream.api_key` empty as an optional fallback you can add later.

Supported fields from that file:

- `OPENAI_API_KEY`
- `tokens.access_token`
- `tokens.refresh_token`
- `tokens.id_token`
- `tokens.account_id`

When `upstream.prefer_local_codex_credentials` is enabled, upstream auth resolution is:

1. `OPENAI_API_KEY` from the local Codex auth file
2. `tokens.access_token` from the local Codex auth file
3. `refresh_token` refresh flow for a local access token
4. configured `OPENAI_API_KEY` / `upstream.api_key`

If the proxy refreshes a local Codex access token successfully, it updates the token fields in the local auth file and uses an atomic replace to avoid partial writes.

## Failure Behavior

- Missing or malformed local auth file: falls back to configured `upstream.api_key`
- Missing local `OPENAI_API_KEY` and unusable local token: falls back to configured `upstream.api_key`
- Upstream `401` or `403` while using a local access token: invalidates the cached token, reloads the local file, attempts one refresh, then retries once
- Upstream `401` or `403` after local credential failure: retries once with configured `upstream.api_key` if present
- `count_tokens` keeps its existing local estimation fallback if the upstream call still fails

## Security Notes

- Do not commit `config.yaml` with real secrets.
- Do not commit `~/.codex/auth.json`.
- This proxy never logs raw credential values.
- Treat local Codex credential support as optional compatibility behavior. Configured API-key auth remains the stable baseline.
