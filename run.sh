#!/usr/bin/env bash
set -euo pipefail

# Resolve VYBFORGE_<NAME> if set, else VYBAICONF_<NAME> (deprecated fallback),
# else "". Keep both alive for one cycle so legacy shells do not die.
getenv() {
  local name="$1"
  local val=""
  eval "val=\"\${VYBFORGE_${name}:-\${VYBAICONF_${name}:-}}\""
  printf '%s' "$val"
}

backend="$(getenv BACKEND)"
[ -z "$backend" ] && backend=ollama

case "$backend" in
  ollama)
    endpoint="$(getenv ENDPOINT)"
    [ -z "$endpoint" ] && endpoint="http://127.0.0.1:11434"
    model="$(getenv MODEL)"
    [ -z "$model" ] && model="qwen3:8b"
    ;;
  openai-chat)
    endpoint="$(getenv ENDPOINT)"
    if [ -z "$endpoint" ]; then
      echo "Set VYBFORGE_ENDPOINT (or deprecated VYBAICONF_ENDPOINT) to an OpenAI-compatible base URL ending in /v1" >&2
      exit 2
    fi
    model="$(getenv MODEL)"
    if [ -z "$model" ]; then
      echo "Set VYBFORGE_MODEL (or deprecated VYBAICONF_MODEL) to the provider model name" >&2
      exit 2
    fi
    ;;
  openai-responses)
    endpoint="$(getenv ENDPOINT)"
    [ -z "$endpoint" ] && endpoint="https://api.openai.com/v1"
    model="$(getenv MODEL)"
    if [ -z "$model" ]; then
      echo "Set VYBFORGE_MODEL (or deprecated VYBAICONF_MODEL) to the OpenAI model name" >&2
      exit 2
    fi
    ;;
  *)
    echo "Unsupported VYBFORGE_BACKEND: $backend (legacy VYBAICONF_BACKEND still honored)" >&2
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
