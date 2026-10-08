#!/usr/bin/env bash
# run_decode_oracle_gate.sh — the independent decode oracle for VybForge#22.
#
# Every other inference gate compares the GPU against the numpy reference, and both implement
# the same conventions: GPU == numpy proves self-consistency, not correctness. That is how #11
# (token salad) hid for weeks. This gate compares instead against llama.cpp on the same GGUF —
# an implementation that shares no code with ours — per token.
#
# Fixtures live in native/legit/fixtures/llama_decode/ and are captured by
#   env -u PYTHONPATH .venv/bin/python native/tools/llama_decode_capture.py
# Each pins: the prompt ids, llama.cpp's greedy continuation, and the provenance (llama.cpp
# version + GGUF id). Greedy streams are only stable where the decision is decisive, so
# fixtures declare a mode:
#   exact       every generated token must equal llama.cpp's
#   membership  the token must be one of the fixture's allowed ids (a documented near-tie)
# A provenance change means the oracle moved: the gate FAILS and says to recapture, rather than
# letting a stale fixture pass quietly.
#
# Usage: ./native/legit/run_decode_oracle_gate.sh
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$root/vybenv.sh" || exit 1

WORK="${TMPDIR:-/tmp}/decode-oracle.$$"
mkdir -p "$WORK"
fail=0
step() { printf '%-52s %s\n' "$1" "$2"; }

cd "$root"
fx_dir="native/legit/fixtures/llama_decode"
GGUF="${VYBFORGE_QWEN3_GGUF:-}"

# ---- prerequisites: absent oracle skips with a notice, it does not fail -------------------
[ -d "$fx_dir" ] || { step "fixtures" "MISSING ($fx_dir)"; echo; echo "DECODE ORACLE GATE: FAIL"; exit 1; }
if [ ! -x "$root/.venv/bin/python" ] || ! "$root/.venv/bin/python" -c "import llama_cpp" >/dev/null 2>&1; then
  step "oracle (llama_cpp)" "SKIP — no llama_cpp in .venv; the pinned fixtures still hold"
  echo; echo "DECODE ORACLE GATE: SKIP"; exit 0
fi
if [ -z "$GGUF" ] || [ ! -s "$GGUF" ]; then
  step "oracle model" "SKIP — no GGUF at \$VYBFORGE_QWEN3_GGUF"
  echo; echo "DECODE ORACLE GATE: SKIP"; exit 0
fi

live_ver="$("$root/.venv/bin/python" -c 'import llama_cpp;print(llama_cpp.__version__)' 2>/dev/null)"
live_gguf="$(head -c 1048576 "$GGUF" | sha256sum | cut -c1-12)"
step "oracle provenance" "llama_cpp $live_ver, gguf $live_gguf"

field() { awk -v k="$2" '$1==k{$1="";sub(/^ /,"");print;exit}' "$1"; }

for fx in "$fx_dir"/*.fix; do
  name="$(basename "$fx" .fix)"
  mode="$(field "$fx" mode)"
  want_ver="$(field "$fx" llama_version)"
  want_gguf="$(field "$fx" gguf_id)"
  ids="$(field "$fx" prompt_ids)"

  if [ "$want_ver" != "$live_ver" ] || [ "$want_gguf" != "$live_gguf" ]; then
    step "$name" "FAIL (oracle moved: llama_cpp $want_ver/$live_ver, gguf $want_gguf/$live_gguf)"
    echo "      recapture: env -u PYTHONPATH .venv/bin/python native/tools/llama_decode_capture.py"
    fail=1
    continue
  fi

  # ids file for the driver (newline-separated, as decode_driver expects)
  printf '%s\n' $ids > "$WORK/$name.ids"

  if [ "$mode" = "exact" ]; then
    want="$(field "$fx" expect_ids)"
    gen="$(printf '%s\n' $want | wc -l)"
  else
    want="$(field "$fx" allowed_ids)"
    cs="$(field "$fx" check_step)"; cs="${cs:-0}"
    gen=$((cs + 1))          # must run far enough to reach the step under test
  fi

  VYB_PROMPT_IDS="$WORK/$name.ids" VYB_GEN="$gen" VYB_STDLIB="$VYB_STDLIB" \
    "$VYB" native/host/decode_driver.vyb --module-path native/llm >"$WORK/$name.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    step "$name" "FAIL (driver exit $rc)"; tail -4 "$WORK/$name.log" | sed 's/^/      /'; fail=1; continue
  fi

  got="$(grep -E '^GEN step [0-9]+ token=' "$WORK/$name.log" | sed 's/.*token=//' | tr '\n' ' ' | sed 's/ *$//')"
  got_n="$(printf '%s\n' $got | wc -l)"

  if [ "$mode" = "exact" ]; then
    if [ "$got" = "$want" ]; then
      step "$name" "PASS (exact, $got_n tokens: $(field "$fx" expect_text))"
    else
      step "$name" "FAIL (exact mismatch)"
      echo "      llama : $want"
      echo "      vyb   : $got"
      fail=1
    fi
  else
    got_at="$(printf '%s\n' $got | sed -n "$((cs + 1))p")"
    if printf '%s\n' $want | grep -qx "$got_at"; then
      step "$name" "PASS (membership at step $cs: token $got_at is one of [$want])"
    else
      step "$name" "FAIL (step $cs token $got_at not in [$want])"
      echo "      $(field "$fx" note)"
      fail=1
    fi
  fi
done

echo
if [ $fail -eq 0 ]; then echo "DECODE ORACLE GATE: PASS"; else echo "DECODE ORACLE GATE: FAIL"; fi
exit $fail
