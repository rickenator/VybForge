#!/usr/bin/env bash
set -euo pipefail

exec python3 app/configurator.py \
  --backend ollama \
  --model "${VYBAICONF_MODEL:-qwen3:8b}" \
  --schema config/mock-system.schema.json \
  --response-schema config/agent-response.schema.json \
  --config config/mock-system.json \
  "$@"
