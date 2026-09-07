#!/usr/bin/env bash
# Run the DR-Sight API locally with the config in ../.env, so a tunnel
# (cloudflared / ngrok) can expose it to the published Base44 app.
#
#   ./scripts/serve-local.sh            # port 8000
#   PORT=9000 ./scripts/serve-local.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  echo "missing .env — copy .env.example and fill it in" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

export PYTHONPATH=.
PORT="${PORT:-8000}"

if [[ ! -x .venv/bin/uvicorn ]]; then
  echo "no .venv — run:  uv venv --python 3.13 .venv && uv pip install -r requirements.txt" >&2
  exit 1
fi

echo "DR-Sight API  ->  http://127.0.0.1:${PORT}   (backbone=${DR_BACKBONE:-b3})"
exec .venv/bin/uvicorn src.api.main:app --host 127.0.0.1 --port "${PORT}"
