#!/usr/bin/env bash
set -euo pipefail

backend="${VYBFORGE_BACKEND:-ollama}"
case "$backend" in
  ollama)
    endpoint="${VYBFORGE_ENDPOINT:-http://127.0.0.1:11434}"
    model="${VYBFORGE_MODEL:-qwen3:8b}"
    ;;
  openai-chat)
    endpoint="${VYBFORGE_ENDPOINT:?Set VYBFORGE_ENDPOINT to an OpenAI-compatible base URL ending in /v1}"
    model="${VYBFORGE_MODEL:?Set VYBFORGE_MODEL to the provider model name}"
    ;;
  openai-responses)
    endpoint="${VYBFORGE_ENDPOINT:-https://api.openai.com/v1}"
    model="${VYBFORGE_MODEL:?Set VYBFORGE_MODEL to the OpenAI model name}"
    ;;
  *)
    echo "Unsupported VYBFORGE_BACKEND: $backend" >&2
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
