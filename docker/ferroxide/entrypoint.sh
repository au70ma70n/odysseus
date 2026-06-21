#!/bin/sh
set -eu

CONFIG_HOME="${FERROXIDE_CONFIG_HOME:-/config}"
CALDAV_HOST="${FERROXIDE_CALDAV_HOST:-0.0.0.0}"
CALDAV_PORT="${FERROXIDE_CALDAV_PORT:-8081}"

if [ "${1:-caldav}" = "auth" ] || [ "${1:-}" = "status" ] || [ "${1:-}" = "help" ]; then
  exec ferroxide -config-home "$CONFIG_HOME" "$@"
fi

if ! ferroxide -config-home "$CONFIG_HOME" status 2>/dev/null | grep -qE '[1-9][0-9]* logged in user'; then
  echo "ferroxide: no Proton account configured in ${CONFIG_HOME}/ferroxide" >&2
  echo "Run once (interactive — Proton password + 2FA if enabled):" >&2
  echo "  ./scripts/ferroxide-auth.sh" >&2
  echo "  # or: docker compose run --rm -it --entrypoint /entrypoint.sh ferroxide auth \${FERROXIDE_PROTON_USER}" >&2
  echo "Save the printed bridge password to FERROXIDE_BRIDGE_PASSWORD in .env, then restart ferroxide:" >&2
  echo "  docker compose up -d ferroxide" >&2
  echo "ferroxide: idle (CalDAV not listening until auth completes)" >&2
  sleep infinity
fi

echo "ferroxide: starting CalDAV on ${CALDAV_HOST}:${CALDAV_PORT}"
exec ferroxide -config-home "$CONFIG_HOME" \
  -caldav-host "$CALDAV_HOST" \
  -caldav-port "$CALDAV_PORT" \
  caldav
