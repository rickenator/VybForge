#!/usr/bin/env bash
# run_chat_render_gate.sh — Phase 2 gate for native/train/render_chat.vyb (P2.2a part 1).
#
# The Vyb program must render the configurator corpus through the Qwen3 chat template
# carried in the Qwen3-4B GGUF metadata, exactly as
# training/train_lora.py's tokenizer.apply_chat_template(..., tokenize=False,
# add_generation_prompt=False) does. Checks:
#
#   1. BASELINE  — the 720-record render (text + byte-offset index) hashes to the pinned
#                  values; /tmp-independent, works with no Python at all.
#   2. BOUNDARY  — native/legit/fixtures/chat_render_cases.jsonl renders to its pinned hash.
#                  Those cases cover what the corpus never exercises: a `</think>` split,
#                  a multi-turn record (only the last assistant gets the think block), a
#                  system message that is not first, reasoning that is not last, unicode,
#                  and leading/trailing newlines.
#   3. REFUSAL   — the two refuse fixtures must be refused: each record reports a reason,
#                  nothing is written, and main reports 3. (The Vyb JIT prints main's return
#                  value and exits 0, so the check is the printed value, not `$?`.)
#   4. ORACLE    — when the repo's `.venv` (transformers/numpy/safetensors + llama_cpp) and
#                  the Qwen3-4B GGUF are both present, re-render the corpus and the boundary
#                  cases with native/train/render_chat_ref.py — llama.cpp's own Jinja
#                  implementation of the same template — and require byte-identity, plus
#                  require the GGUF's template to still hash to the pinned value. Skipped
#                  with a notice when either is missing (the pinned hashes above still hold).
#
# Usage: ./native/legit/run_chat_render_gate.sh
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$root/vybenv.sh" || exit 1

WORK="${TMPDIR:-/tmp}/chat-render-gate.$$"
mkdir -p "$WORK"
fail=0
step() { printf '%-52s %s\n' "$1" "$2"; }

cd "$root"
fx="native/legit/fixtures"
base="$fx/chat_render_baseline.sha256"
GGUF="${VYBFORGE_QWEN3_GGUF:-/home/rick/Models/qwen3/Qwen3-4B-Q4_K_M.gguf}"
if [ ! -s "$base" ]; then step "baseline fixture" "MISSING ($base)"; echo; echo "CHAT RENDER GATE: FAIL"; exit 1; fi
want_txt_sha="$(awk '$3=="native/out/chat_render.txt"{print $1}' "$base")"
want_txt_sz="$(awk '$3=="native/out/chat_render.txt"{print $2}' "$base")"
want_idx_sha="$(awk '$3=="native/out/chat_render.idx"{print $1}' "$base")"
want_tpl_sha="$(awk '$3 ~ /qwen3_chat_template.jinja/{print $1}' "$base")"
want_cases_sha="$(awk '$3=="cases.txt"{print $1}' "$base")"

# 1. corpus render vs pinned baseline
VYBFORGE_CHAT_OUT="$WORK/render.txt" VYBFORGE_CHAT_IDX="$WORK/render.idx" \
  "$VYB" --module-path native/json native/train/render_chat.vyb >"$WORK/log" 2>&1
rc=$?
if [ $rc -ne 0 ] || ! grep -q "^rendered: 720 records" "$WORK/log"; then
  step "corpus render (720 records)" "FAIL (exit $rc)"; tail -4 "$WORK/log" | sed 's/^/      /'; fail=1
else
  got_sha="$(sha256sum "$WORK/render.txt" | cut -d' ' -f1)"
  got_sz="$(wc -c < "$WORK/render.txt")"
  got_idx="$(sha256sum "$WORK/render.idx" | cut -d' ' -f1)"
  if [ "$got_sha" = "$want_txt_sha" ] && [ "$got_sz" = "$want_txt_sz" ] && [ "$got_idx" = "$want_idx_sha" ]; then
    step "corpus render (720 records)" "byte-identical ($got_sz B, text+idx, ${got_sha:0:8}…)"
  else
    step "corpus render (720 records)" "FAIL (text $got_sz B ${got_sha:0:8}…, want $want_txt_sz B ${want_txt_sha:0:8}…)"
    fail=1
  fi
fi

# 2. boundary cases vs pinned hash
VYBFORGE_CHAT_SRC="$fx/chat_render_cases.jsonl" VYBFORGE_CHAT_OUT="$WORK/cases.txt" \
  VYBFORGE_CHAT_IDX="$WORK/cases.idx" \
  "$VYB" --module-path native/json native/train/render_chat.vyb >"$WORK/cases.log" 2>&1
case_sha="$(sha256sum "$WORK/cases.txt" | cut -d' ' -f1)"
if grep -q "^rendered: 8 records" "$WORK/cases.log" && [ "$case_sha" = "$want_cases_sha" ]; then
  step "boundary cases (8, think/multi-turn/…)" "byte-identical (${case_sha:0:8}…)"
else
  step "boundary cases (8, think/multi-turn/…)" "FAIL (${case_sha:0:8}… want ${want_cases_sha:0:8}…)"
  tail -3 "$WORK/cases.log" | sed 's/^/      /'; fail=1
fi

# 3. refusals: reason reported, nothing written, main reports 3
ref_ok=1
for f in chat_render_refuse chat_render_refuse2; do
  rm -f "$WORK/ref.txt"
  out="$(VYBFORGE_CHAT_SRC="$fx/$f.jsonl" VYBFORGE_CHAT_OUT="$WORK/ref.txt" \
        VYBFORGE_CHAT_IDX="$WORK/ref.idx" \
        "$VYB" --module-path native/json native/train/render_chat.vyb 2>&1)"
  # last non-empty line is main's printed return value
  got_rc="$(printf '%s\n' "$out" | grep -E '^[0-9]+$' | tail -1)"
  if [ "$got_rc" != "3" ] || [ -e "$WORK/ref.txt" ]; then
    step "refusal: $f" "FAIL (reported '${got_rc:-none}', output $([ -e "$WORK/ref.txt" ] && echo written || echo absent))"
    printf '%s\n' "$out" | tail -3 | sed 's/^/      /'; ref_ok=0; fail=1
  fi
done
[ $ref_ok -eq 1 ] && step "refusals (tool role/content/tool_calls)" "ok (3 cases refused, nothing written)"

# 4. optional live oracle cross-check
if [ -x .venv/bin/python ] && [ -f "$GGUF" ]; then
  if .venv/bin/python native/train/render_chat_ref.py --src data/vybos-configurator-all.jsonl \
       --out "$WORK/py.txt" --idx "$WORK/py.idx" --gguf "$GGUF" --template-sha "$want_tpl_sha" \
       >"$WORK/py.log" 2>&1; then
    r1=0; cmp -s "$WORK/py.txt" "$WORK/render.txt" || r1=1
    cmp -s "$WORK/py.idx" "$WORK/render.idx" || r1=1
    .venv/bin/python native/train/render_chat_ref.py --src "$fx/chat_render_cases.jsonl" \
      --out "$WORK/py2.txt" --idx "$WORK/py2.idx" --gguf "$GGUF" >"$WORK/py2.log" 2>&1
    cmp -s "$WORK/py2.txt" "$WORK/cases.txt" || r1=1
    if [ $r1 -eq 0 ]; then
      step "oracle cross-check (llama.cpp Jinja)" "ok (corpus + boundary cases byte-identical)"
    else
      step "oracle cross-check (llama.cpp Jinja)" "FAIL (Vyb and oracle disagree)"; fail=1
    fi
  else
    step "oracle cross-check (llama.cpp Jinja)" "FAIL (oracle run)"
    tail -4 "$WORK/py.log" | sed 's/^/      /'; fail=1
  fi
else
  miss=""
  [ -x .venv/bin/python ] || miss=".venv"
  [ -f "$GGUF" ] || miss="$miss GGUF"
  step "oracle cross-check (llama.cpp Jinja)" "skipped (missing:$miss; pinned hashes still enforced)"
fi

rm -rf "$WORK"
echo
if [ $fail -eq 0 ]; then echo "CHAT RENDER GATE: PASS"; else echo "CHAT RENDER GATE: FAIL"; fi
exit $fail
