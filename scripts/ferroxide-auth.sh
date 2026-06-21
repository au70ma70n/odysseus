#!/usr/bin/env bash
# One-time interactive login for the ferroxide sidecar.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

USER="${FERROXIDE_PROTON_USER:-}"
if [ -z "$USER" ]; then
  echo "Set FERROXIDE_PROTON_USER in .env first (e.g. you@proton.me)" >&2
  exit 1
fi

DATA_DIR="${APP_DATA_DIR:-./data}/ferroxide"
mkdir -p "$DATA_DIR"

if [ "${1:-}" = "--reset" ]; then
  echo "Clearing ferroxide auth cache in ${DATA_DIR}..."
  rm -rf "${DATA_DIR}/ferroxide" "${DATA_DIR}/auth.json" 2>/dev/null || true
  shift
fi

echo "Logging in to Proton as ${USER}..."
echo "You will be prompted for your Proton password (and 2FA if enabled)."
echo "Copy the bridge password printed at the end into FERROXIDE_BRIDGE_PASSWORD in .env"
echo "If auth failed before with 401 on keys/salts, rebuild ferroxide first:"
echo "  docker compose build ferroxide"
echo

docker compose run --rm -it --entrypoint /entrypoint.sh ferroxide auth "$USER"
