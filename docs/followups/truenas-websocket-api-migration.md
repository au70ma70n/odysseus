# Follow-up: Migrate TrueNAS integration to WebSocket API

**Status:** Planned  
**Priority:** Before TrueNAS 26.04 upgrade  
**Opened:** 2026-06-22  
**Trigger:** TrueNAS alert — deprecated REST API authenticated 138+ times in 24h from `10.0.0.11` (Odysseus host).

## Problem

TrueNAS Scale 25.x deprecates the REST wrapper (`/api/v2.0/*` + Bearer auth). It is **removed in 26.04**. Odysseus still uses REST for almost all TrueNAS `api_call` traffic; only container logs use WebSocket (`core.subscribe` on `/api/current`).

## Goal

Route TrueNAS integration traffic through **JSON-RPC 2.0 over WebSocket** (`wss://<host>/api/current`) while keeping the agent-facing `api_call` surface unchanged (same paths/methods in skills and prompts).

## Scope

### In scope

- `execute_api_call()` for TrueNAS integrations — translate virtual REST paths to WS methods
- Integration health test (`POST /integrations/{id}/test`) — use WS `system.info`
- Extend `src/truenas_ws.py` with a reusable WS client (connect, auth, call, subscribe)
- Method mapping for paths used by `deploy-comfyui-truenas` and agent rules:

| Current REST | WebSocket method |
|--------------|------------------|
| `GET /api/v2.0/system/info` | `system.info` |
| `GET /api/v2.0/app` | `app.query` |
| `GET /api/v2.0/app/id/{name}` | `app.get_instance` |
| `GET /api/v2.0/app/available` | `app.available` |
| `GET /api/v2.0/app/used_ports` | `app.used_ports` |
| `GET /api/v2.0/app/gpu_choices` | `app.gpu_choices` |
| `GET /api/v2.0/pool/dataset` | `pool.dataset.query` (verify name) |
| `GET /api/v2.0/core/get_jobs` | `core.get_jobs` |
| `POST /api/v2.0/app` | `app.create` |
| `POST /api/v2.0/app/start` | `app.start` |
| `POST /api/v2.0/app/stop` | `app.stop` |
| `POST /api/v2.0/app/redeploy` | `app.redeploy` |
| `DELETE /api/v2.0/app/id/{name}` | `app.delete` |
| `GET /api/v2.0/app/container_logs` | Already WS (`core.subscribe` + `app.container_log_follow`) |

- Auth: `auth.login_with_api_key` with API key once per connection (not Bearer per request)
- Tests with mocked WebSocket responses

### Out of scope (for now)

- Removing REST fallback (can keep behind env flag during transition)
- TrueNAS UI / manual API usage outside Odysseus

## Key files

- `src/truenas_ws.py` — WS client + log subscribe (extend)
- `src/integrations.py` — `execute_api_call()`, preset docs, health path
- `routes/auth_routes.py` — integration test route
- `tests/test_truenas_ws.py` — expand coverage
- `data/skills/general/deploy-comfyui-truenas/SKILL.md` — no path changes expected

## Implementation notes

- WebSocket URL: `base_url` with `https` → `wss`, path `/api/current`
- `verify_ssl: false` on integration must apply to WSS (same as current REST)
- Job-returning writes return a job id — polling stays `core.get_jobs`
- Connection pooling: reuse one WS connection per request batch or short-lived per `execute_api_call` (start simple: open → auth → call → close)
- Reference: existing log fetch in `fetch_app_container_logs()` already proves auth + subscribe on 25.10

## Test plan

1. Unit tests: REST path → WS method routing, auth failure, SSL skip
2. Integration (manual): `GET system/info`, `GET app`, deploy ComfyUI skill flow, container logs
3. Confirm TrueNAS alert counter stops incrementing REST auth after migration

## Acceptance criteria

- [ ] All TrueNAS `api_call` paths used by skills/agents work via WebSocket
- [ ] Integration test in Settings succeeds without REST
- [ ] No new dependency beyond `websockets` (already in `requirements.txt`)
- [ ] Documented in TrueNAS integration preset description
