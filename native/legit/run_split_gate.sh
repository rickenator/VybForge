#!/usr/bin/env bash
# run_split_gate.sh — Phase 2 gate for training/split_dataset.vyb (P2.5a).
#
# The Python it replaces was the inline `python3 -` heredoc in
# training/generate-data.sh:
#
#     rows = [json.loads(l) for l in all.splitlines() if l]
#     for path, split in ((train,'train'), (eval,'eval')):
#         path.write_text(''.join(json.dumps(r, separators=(",",":")) + "\n"
#                                 for r in rows if r['metadata']['split'] == split))
#
# Checks, none of them in a runtime path:
#   1. PARITY — splitting the committed data/vybos-configurator-all.jsonl with the Vyb
#      splitter must reproduce the committed train.jsonl and eval.jsonl byte-for-byte.
#      All three files are tracked, so the committed corpus is the oracle (verified
#      first: the Python original also reproduces them byte-for-byte).
#   2. NO PYTHON — training/generate-data.sh must contain no python3 and must call
#      the Vyb splitter.
#   3. REGENERATION — the generator is rebuilt with the Vyb toolchain, its stdout is
#      captured to a scratch file, and that scratch file must (a) match the committed
#      all.jsonl and (b) split byte-identically. Nothing tracked is overwritten.
#
# Usage: ./native/legit/run_split_gate.sh
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$root/vybenv.sh" || exit 1

WORK="${TMPDIR:-/tmp}/split-gate.$$"
mkdir -p "$WORK"
fail=0
step() { printf '%-52s %s\n' "$1" "$2"; }

cd "$root"

all="data/vybos-configurator-all.jsonl"
trn="data/vybos-configurator-train.jsonl"
evl="data/vybos-configurator-eval.jsonl"
for f in "$all" "$trn" "$evl"; do
  [ -s "$f" ] || { step "input $f" "MISSING"; fail=1; }
done
[ $fail -eq 0 ] || { echo "SPLIT GATE: FAIL"; exit 1; }

# 1. port parity on the committed corpus
VYBFORGE_SPLIT_SRC="$all" \
VYBFORGE_SPLIT_TRAIN="$WORK/train.jsonl" \
VYBFORGE_SPLIT_EVAL="$WORK/eval.jsonl" \
  "$VYB" training/split_dataset.vyb --module-path native/json >"$WORK/split.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  step "split_dataset.vyb" "FAIL (exit $rc)"; tail -4 "$WORK/split.log" | sed 's/^/      /'; fail=1
else
  ok=1
  cmp -s "$WORK/train.jsonl" "$trn" || { step "train.jsonl byte parity" "FAIL"; ok=0; fail=1; }
  cmp -s "$WORK/eval.jsonl"  "$evl" || { step "eval.jsonl byte parity"  "FAIL"; ok=0; fail=1; }
  if [ $ok -eq 1 ]; then
    step "port parity (committed corpus)" \
         "byte-identical ($(grep -c . "$trn") train / $(grep -c . "$evl") eval records)"
  fi
fi

# 2. the shipped script is Python-free and uses the port
if grep -q "python3" training/generate-data.sh; then
  step "generate-data.sh python-free" "FAIL (python3 still present)"; fail=1
elif grep -q "split_dataset.vyb" training/generate-data.sh; then
  step "generate-data.sh python-free" "ok (calls split_dataset.vyb)"
else
  step "generate-data.sh python-free" "FAIL (splitter not wired in)"; fail=1
fi

# 3. regeneration: generator -> scratch all.jsonl -> split, nothing tracked touched
"$VYB" training/generate_dataset.vyb --build "$WORK/vybos-training-data" -O2 >"$WORK/gen.log" 2>&1
if [ $? -ne 0 ]; then
  step "generate_dataset.vyb build" "FAIL"; tail -4 "$WORK/gen.log" | sed 's/^/      /'; fail=1
else
  "$WORK/vybos-training-data" | sed -n '/^{/p' >"$WORK/all.jsonl"
  if cmp -s "$WORK/all.jsonl" "$all"; then
    VYBFORGE_SPLIT_SRC="$WORK/all.jsonl" \
    VYBFORGE_SPLIT_TRAIN="$WORK/rtrain.jsonl" \
    VYBFORGE_SPLIT_EVAL="$WORK/reval.jsonl" \
      "$VYB" training/split_dataset.vyb --module-path native/json >/dev/null 2>&1
    if cmp -s "$WORK/rtrain.jsonl" "$trn" && cmp -s "$WORK/reval.jsonl" "$evl"; then
      step "regenerated corpus splits identically" "ok ($(wc -c < "$WORK/all.jsonl") B, byte-identical)"
    else
      step "regenerated corpus splits identically" "FAIL"; fail=1
    fi
  else
    step "regenerated corpus == committed all.jsonl" \
         "DIFFERS (generator/artifact drift; $(wc -c < "$WORK/all.jsonl") B vs $(wc -c < "$all") B)"
  fi
fi

rm -rf "$WORK"
echo
if [ $fail -eq 0 ]; then echo "SPLIT GATE: PASS"; else echo "SPLIT GATE: FAIL"; fi
exit $fail
