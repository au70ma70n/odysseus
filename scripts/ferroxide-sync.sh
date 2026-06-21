#!/usr/bin/env bash
# Wire ferroxide into Odysseus prefs (if needed) and trigger a CalDAV pull.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [ -z "${FERROXIDE_PROTON_USER:-}" ] || [ -z "${FERROXIDE_BRIDGE_PASSWORD:-}" ]; then
  echo "Set FERROXIDE_PROTON_USER and FERROXIDE_BRIDGE_PASSWORD in .env first." >&2
  exit 1
fi

# Script may not be in the image yet — copy the repo version in.
docker compose exec -T odysseus mkdir -p /app/scripts
docker cp scripts/ferroxide-wire-odysseus.py "$(docker compose ps -q odysseus)":/app/scripts/ferroxide-wire-odysseus.py

docker compose exec -T \
  -e FERROXIDE_PROTON_USER="${FERROXIDE_PROTON_USER}" \
  -e FERROXIDE_BRIDGE_PASSWORD="${FERROXIDE_BRIDGE_PASSWORD}" \
  -e PROTON_CALDAV_URL="${PROTON_CALDAV_URL:-http://ferroxide:8081/}" \
  -e ODYSSEUS_ADMIN_USER="${ODYSSEUS_ADMIN_USER:-admin}" \
  -e ODYSSEUS_USER_PREFS=/app/data/user_prefs.json \
  -e ODYSSEUS_APP_KEY=/app/data/.app_key \
  odysseus python3 /app/scripts/ferroxide-wire-odysseus.py

ADMIN_USER="${ODYSSEUS_ADMIN_USER:-admin}"
ADMIN_PASS="${ODYSSEUS_ADMIN_PASSWORD:-}"
if [ -z "$ADMIN_PASS" ]; then
  echo "ODYSSEUS_ADMIN_PASSWORD not set — open Calendar in Odysseus to sync, or set the password in .env and re-run."
  exit 0
fi

COOKIE_JAR="$(mktemp)"
trap 'rm -f "$COOKIE_JAR"' EXIT
curl -sf -c "$COOKIE_JAR" -b "$COOKIE_JAR" \
  -H 'Content-Type: application/json' \
  -d "{\"username\":\"${ADMIN_USER}\",\"password\":\"${ADMIN_PASS}\"}" \
  "http://127.0.0.1:${APP_PORT:-7000}/api/auth/login" >/dev/null

curl -sf -c "$COOKIE_JAR" -b "$COOKIE_JAR" \
  -X POST "http://127.0.0.1:${APP_PORT:-7000}/api/calendar/sync" | python3 -m json.tool
