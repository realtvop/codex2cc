# codextocc

Rust proxy that translates Anthropic-style `/v1/messages` requests into OpenAI Responses API calls.

## Configuration

The server loads configuration from `config.yaml` by default. Set `CONFIG_PATH` to use a different file.

If the configured path does not exist at startup, the server generates a new config file, writes a random 32-character lowercase hex client API key to `api_keys`, and exits immediately.

Environment variables override YAML values when both are present.

### Server fields

- `server.host`
  Default: `0.0.0.0`
- `server.port`
  Default: `8082`

### Client auth fields

- `api_keys`
  Optional list of client API keys accepted by this proxy. If omitted or empty, client auth is disabled.

### Upstream auth mode

- `upstream.auth_mode`
  Values: `local_codex | config_api_key | account_pool`
  Default: derived from legacy `upstream.prefer_local_codex_credentials` (`true => local_codex`, `false => config_api_key`)
  Env override: `AUTH_MODE`

### Common upstream fields

- `upstream.base_url`
  Default: `https://api.openai.com/v1`
  Env override: `OPENAI_BASE_URL`
- `upstream.api_key`
  Default: empty
  Env override: `OPENAI_API_KEY`
- `upstream.codex_base_url`
  Default: `https://chatgpt.com/backend-api/codex`
  Env override: `CODEX_BASE_URL`
  Used for local Codex OAuth and account pool OAuth credentials.

## Legacy local Codex mode (`auth_mode=local_codex`)

Supported local file fields (`~/.codex/auth.json` by default):

- `OPENAI_API_KEY`
- `tokens.access_token`
- `tokens.refresh_token`
- `tokens.id_token`
- `tokens.account_id`

Legacy fields:

- `upstream.prefer_local_codex_credentials`
  Env override: `PREFER_LOCAL_CODEX_CREDENTIALS`
- `upstream.local_codex_auth_path`
  Env override: `LOCAL_CODEX_AUTH_PATH`
- `upstream.refresh_local_codex_tokens`
  Env override: `REFRESH_LOCAL_CODEX_TOKENS`
- `upstream.local_codex_oauth_client_id`
  Env override: `LOCAL_CODEX_OAUTH_CLIENT_ID`
- `upstream.local_codex_oauth_token_endpoint`
  Env override: `LOCAL_CODEX_OAUTH_TOKEN_ENDPOINT`

Resolution order in this mode:

1. `OPENAI_API_KEY` from local file
2. local `tokens.access_token`
3. local `refresh_token` flow
4. configured `upstream.api_key`

## Account pool mode (`auth_mode=account_pool`)

This mode is fully decoupled from local Codex credentials and does not read/write `~/.codex/auth.json`.

### Account pool config fields

- `upstream.account_pool.store_path`
  Default: `~/.codextocc/account-pool.json`
  Env override: `ACCOUNT_POOL_PATH`
- `upstream.account_pool.quota_refresh_interval_secs`
  Default: `30`
  Env override: `ACCOUNT_POOL_REFRESH_INTERVAL_SECS`
- `upstream.account_pool.quota_headers.remaining`
  Default: `x-ratelimit-remaining-requests`
  Env override: `ACCOUNT_POOL_QUOTA_REMAINING_HEADER`
- `upstream.account_pool.quota_headers.reset`
  Default: `x-ratelimit-reset-requests`
  Env override: `ACCOUNT_POOL_QUOTA_RESET_HEADER`
- `upstream.account_pool.oauth.client_id`
  Required for `pool login`
  Env override: `ACCOUNT_POOL_OAUTH_CLIENT_ID`
- `upstream.account_pool.oauth.scopes`
  Default: `["openid","profile","email","offline_access"]`
  Env override: `ACCOUNT_POOL_OAUTH_SCOPES` (comma-separated)
- `upstream.account_pool.openid_config_url`
  Default: `https://auth.openai.com/.well-known/openid-configuration`
  Env override: `ACCOUNT_POOL_OPENID_CONFIG_URL`
- `upstream.account_pool.oauth_token_endpoint`
  Optional override
  Env override: `ACCOUNT_POOL_OAUTH_TOKEN_ENDPOINT`
- `upstream.account_pool.oauth_device_authorization_endpoint`
  Optional override
  Env override: `ACCOUNT_POOL_OAUTH_DEVICE_AUTHORIZATION_ENDPOINT`

### Device-code OAuth management commands

- `codextocc pool login [alias]`
  Starts OAuth device-code login, prints verification URL + code, polls until authorized.
- `codextocc pool list`
- `codextocc pool enable <id-or-alias>`
- `codextocc pool disable <id-or-alias>`
- `codextocc pool remove <id-or-alias>`
- `codextocc pool refresh`
  Manually refreshes quota metadata for enabled accounts.

### Account selection and failover

- Selection prefers:
  1. enabled accounts with usable tokens and `remaining > 0`
  2. earlier `reset_at`
  3. higher `remaining`
  4. least recently used (`last_used_at`)
- Accounts with unknown quota are lower priority but still usable.
- On `401/403/429`, proxy retries once with the next eligible pool account.
- If all accounts are exhausted, proxy returns `429` with `error.reset_at` (ISO8601 when available).

## Failure behavior summary

- `auth_mode=config_api_key`: only uses configured `upstream.api_key`
- `auth_mode=local_codex`: keeps legacy local-file + fallback behavior
- `auth_mode=account_pool`:
  - no local Codex fallback
  - no configured API key fallback
  - exhausted pool returns `429`

## Security notes

- Do not commit `config.yaml` with real secrets.
- Do not commit `~/.codex/auth.json` or account pool store files.
- Account pool tokens are stored in plain JSON with file permission hardening (`0600` on Unix) and atomic replace writes.
- This proxy never logs raw credential values.
