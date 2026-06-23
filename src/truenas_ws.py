"""TrueNAS Scale WebSocket helpers (JSON-RPC 2.0 on /api/current).

Container logs are NOT available over the legacy REST /api/v2.0 surface on
TrueNAS 25.x — they require subscribing to the ``app.container_log_follow``
event via WebSocket.

Follow-up: migrate all TrueNAS integration traffic off deprecated REST
(removed in TrueNAS 26.04). See docs/followups/truenas-websocket-api-migration.md
"""
from __future__ import annotations

import asyncio
import json
import logging
import ssl
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

log = logging.getLogger(__name__)

try:
    import websockets
except ImportError:  # pragma: no cover - optional until requirements install
    websockets = None  # type: ignore


def _ws_url(base_url: str) -> str:
    base = (base_url or "").strip().rstrip("/")
    parsed = urlparse(base)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{scheme}://{parsed.netloc}/api/current"


def container_log_event_name(
    app_name: str,
    container_id: str,
    tail_lines: int = 100,
) -> str:
    """Build the dynamic event name for ``core.subscribe``."""
    payload = {
        "app_name": app_name,
        "container_id": container_id,
        "tail_lines": int(tail_lines),
    }
    return f"app.container_log_follow:{json.dumps(payload, separators=(',', ':'))}"


async def _fetch_container_id(
    base_url: str,
    api_key: str,
    app_name: str,
    *,
    verify_ssl: bool,
    headers: Dict[str, str],
) -> str:
    url = base_url.rstrip("/") + f"/api/v2.0/app/id/{app_name}"
    async with httpx.AsyncClient(timeout=30.0, verify=verify_ssl) as client:
        response = await client.get(url, headers=headers)
    if not response.is_success:
        raise RuntimeError(f"GET /api/v2.0/app/id/{app_name} failed: HTTP {response.status_code}")
    data = response.json()
    details = ((data.get("active_workloads") or {}).get("container_details") or [])
    if not details:
        raise RuntimeError(
            f"App '{app_name}' has no container_details in active_workloads "
            f"(state={data.get('state')!r})"
        )
    cid = details[0].get("id")
    if not cid:
        raise RuntimeError(f"App '{app_name}' container_details missing id")
    return str(cid)


async def fetch_app_container_logs(
    base_url: str,
    api_key: str,
    *,
    app_name: str,
    container_id: Optional[str] = None,
    tail_lines: int = 100,
    verify_ssl: bool = True,
    collect_timeout: float = 20.0,
) -> str:
    """Tail container logs for a TrueNAS app via WebSocket.

    Returns the collected log text (may be empty if the container produced no
    lines in the tail window).
    """
    if websockets is None:
        raise RuntimeError(
            "websockets package is not installed — required for TrueNAS container logs"
        )

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    cid = container_id or await _fetch_container_id(
        base_url, api_key, app_name, verify_ssl=verify_ssl, headers=headers
    )
    event_name = container_log_event_name(app_name, cid, tail_lines=tail_lines)

    ssl_ctx = None
    if _ws_url(base_url).startswith("wss://"):
        ssl_ctx = ssl.create_default_context()
        if not verify_ssl:
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE

    lines: List[str] = []
    async with websockets.connect(
        _ws_url(base_url),
        ssl=ssl_ctx,
        open_timeout=15,
        max_size=8_000_000,
    ) as ws:
        await ws.send(json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "auth.login_with_api_key",
            "params": [api_key],
        }))
        auth_msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        if auth_msg.get("error") or auth_msg.get("result") is not True:
            raise RuntimeError(f"TrueNAS WebSocket auth failed: {auth_msg}")

        await ws.send(json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "core.subscribe",
            "params": [event_name],
        }))
        sub_msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        if sub_msg.get("error"):
            raise RuntimeError(f"TrueNAS log subscribe failed: {sub_msg['error']}")

        deadline = asyncio.get_running_loop().time() + collect_timeout
        while asyncio.get_running_loop().time() < deadline:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 3.0))
            except asyncio.TimeoutError:
                if lines:
                    break
                continue
            msg = json.loads(raw)
            method = msg.get("method")
            if method == "collection_update":
                fields = (msg.get("params") or {}).get("fields") or {}
                data = fields.get("data")
                if data:
                    lines.append(str(data))
            elif method == "notify_unsubscribed":
                break

    if not lines:
        return "(no log lines returned — container may have no recent output)"
    return "".join(lines)


def is_truenas_integration(integration: Dict[str, Any]) -> bool:
    label = f"{integration.get('preset') or ''} {integration.get('name') or ''}".lower()
    return "truenas" in label
