#!/usr/bin/env python3
"""Register the local ferroxide CalDAV account in Odysseus user prefs."""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

DEFAULT_PREFS = Path("/app/data/user_prefs.json")
DEFAULT_KEY = Path("/app/data/.app_key")
ENC_PREFIX = "enc:"


def _load(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _encrypt(plaintext: str, key_path: Path) -> str:
    if not plaintext:
        return plaintext or ""
    if plaintext.startswith(ENC_PREFIX):
        return plaintext
    from cryptography.fernet import Fernet

    key = key_path.read_bytes()
    token = Fernet(key).encrypt(plaintext.encode("utf-8")).decode("ascii")
    return ENC_PREFIX + token


def main() -> int:
    url = (os.environ.get("PROTON_CALDAV_URL") or "http://ferroxide:8081/").strip()
    username = (os.environ.get("FERROXIDE_PROTON_USER") or "").strip()
    password = (os.environ.get("FERROXIDE_BRIDGE_PASSWORD") or "").strip()
    owner = (os.environ.get("ODYSSEUS_ADMIN_USER") or "admin").strip()
    prefs_path = Path(os.environ.get("ODYSSEUS_USER_PREFS") or DEFAULT_PREFS)
    key_path = Path(os.environ.get("ODYSSEUS_APP_KEY") or DEFAULT_KEY)

    if not username or not password:
        print(
            "Set FERROXIDE_PROTON_USER and FERROXIDE_BRIDGE_PASSWORD in .env, "
            "then re-run ./scripts/ferroxide-sync.sh",
            file=sys.stderr,
        )
        return 1
    if not key_path.is_file():
        print(f"App key not found at {key_path}", file=sys.stderr)
        return 1

    all_prefs = _load(prefs_path)
    if "_users" not in all_prefs:
        all_prefs = {"_users": {}}
    user_prefs = dict(all_prefs["_users"].get(owner) or {})

    accounts = list(user_prefs.get("caldav_accounts") or [])
    label = "Proton Calendar (ferroxide)"
    existing = next(
        (a for a in accounts if (a.get("label") or "").strip() == label),
        None,
    )
    acc = dict(existing) if existing else {"id": str(uuid.uuid4()), "label": label}
    acc["url"] = url.rstrip("/")
    acc["username"] = username
    acc["password"] = _encrypt(password, key_path)

    if existing:
        accounts = [acc if a.get("id") == acc["id"] else a for a in accounts]
    else:
        accounts.insert(0, acc)

    user_prefs["caldav_accounts"] = accounts
    user_prefs.pop("caldav", None)
    all_prefs["_users"][owner] = user_prefs
    _save(prefs_path, all_prefs)

    print(f"Wired CalDAV account for {owner}:")
    print(f"  label:    {label}")
    print(f"  url:      {acc['url']}")
    print(f"  username: {username}")
    print("Open Odysseus Calendar or POST /api/calendar/sync to pull events.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
