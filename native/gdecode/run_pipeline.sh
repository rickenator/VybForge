#!/usr/bin/env bash
# G-decode end-to-end vertical-slice pipeline (pure Vyb runtime; Python only for
# baseline verification). Composes: tokenizer encode -> on-GPU decode ->
# tokenizer detokenize -> agent-response contract emission + jsonschema check.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VYB="${VYB:-/home/rick/Projects/Vyb/build/vyb}"
cd "$root"

echo "== stage A: prompt -> token ids =="
ids="$("$VYB" native/gdecode/pipeline_encode.vyb --module-path native/tokenizer 2>/dev/null | head -1)"
echo "prompt ids: $ids"

echo "== stage B: on-GPU decode (decode_driver) =="
VYBFORGE_DECODE_PROMPT="$ids" "$VYB" native/host/decode_driver.vyb

echo "== stage C: detokenize -> agent-response contract =="
"$VYB" native/gdecode/pipeline_emit.vyb --module-path native/gdecode --module-path native/tokenizer

echo "== stage D: validate contract (jsonschema, verification-only) =="
python3 native/gdecode/verify_contract.py
