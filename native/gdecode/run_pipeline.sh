#!/usr/bin/env bash
# G-decode end-to-end vertical-slice pipeline (pure Vyb runtime; Python only for
# baseline verification). Composes: tokenizer encode -> on-GPU decode ->
# tokenizer detokenize -> agent-response contract emission + jsonschema check.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$root/vybenv.sh" || exit 1   # VYBHOME / VYB / VYB_STDLIB (VybForge#15, rickenator/Vyb#424)
cd "$root"

echo "== stage A: prompt -> token ids =="
ids="$("$VYB" native/gdecode/pipeline_encode.vyb --module-path native/tokenizer 2>/dev/null | head -1)"
echo "prompt ids: $ids"

echo "== stage B: on-GPU decode (decode_driver) =="
VYBFORGE_DECODE_PROMPT="$ids" "$VYB" native/host/decode_driver.vyb --module-path native/llm

echo "== stage C: detokenize -> agent-response contract =="
"$VYB" native/gdecode/pipeline_emit.vyb --module-path native/gdecode --module-path native/tokenizer

echo "== stage D: validate contract (jsonschema, verification-only) =="
# The oracle needs the repo venv (jsonschema); bare python3 does not have it.
PY="${PY:-}"
if [ -z "$PY" ]; then
  if [ -x "$root/.venv/bin/python" ]; then PY="$root/.venv/bin/python"; else PY=python3; fi
fi
"$PY" native/gdecode/verify_contract.py
