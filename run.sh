#!/usr/bin/env bash
# run.sh — backend-neutral launcher for the VybForge desired-state interviewer
# (Vyb-native driver: app/configurator.vyb; the Python original is gone).
#
# Extra settings are env vars, not argv (the Vyb JIT has no argv surface):
#   VYBFORGE_STRUCTURED_OUTPUT=json_schema|json_object|prompt
#   VYBFORGE_SCHEMA / VYBFORGE_RESPONSE_SCHEMA / VYBFORGE_CONFIG / VYBFORGE_PROMPT
#   VYBFORGE_API_KEY / VYBFORGE_API_KEY_ENV for hosted providers
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$root/vybenv.sh" || exit 1   # VYBHOME / VYB / VYB_STDLIB (VybForge#15, rickenator/Vyb#424)

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

export VYBFORGE_BACKEND="$backend"
export VYBFORGE_ENDPOINT="$endpoint"
export VYBFORGE_MODEL="$model"
export VYBFORGE_SCHEMA="${VYBFORGE_SCHEMA:-config/mock-system.schema.json}"
export VYBFORGE_RESPONSE_SCHEMA="${VYBFORGE_RESPONSE_SCHEMA:-config/agent-response.schema.json}"
export VYBFORGE_CONFIG="${VYBFORGE_CONFIG:-config/mock-system.json}"
export VYBFORGE_PROMPT="${VYBFORGE_PROMPT:-prompts/system.md}"

cd "$root"
exec env VYB_STDLIB="$VYB_STDLIB" \
  "$VYB" app/configurator.vyb --module-path tools --module-path native/json
