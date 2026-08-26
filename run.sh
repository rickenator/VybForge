#!/usr/bin/env bash
set -euo pipefail

backend="${VYBAICONF_BACKEND:-ollama}"
case "$backend" in
  ollama)
    endpoint="${VYBAICONF_ENDPOINT:-http://127.0.0.1:11434}"
    model="${VYBAICONF_MODEL:-qwen3:8b}"
    ;;
  openai-chat)
    endpoint="${VYBAICONF_ENDPOINT:?Set VYBAICONF_ENDPOINT to an OpenAI-compatible base URL ending in /v1}"
    model="${VYBAICONF_MODEL:?Set VYBAICONF_MODEL to the provider model name}"
    ;;
  openai-responses)
    endpoint="${VYBAICONF_ENDPOINT:-https://api.openai.com/v1}"
    model="${VYBAICONF_MODEL:?Set VYBAICONF_MODEL to the OpenAI model name}"
    ;;
  *)
    echo "Unsupported VYBAICONF_BACKEND: $backend" >&2
    exit 2
    ;;
esac

exec python3 app/configurator.py \
  --backend "$backend" \
  --endpoint "$endpoint" \
  --model "$model" \
  --schema config/mock-system.schema.json \
  --response-schema config/agent-response.schema.json \
  --config config/mock-system.json \
  "$@"
