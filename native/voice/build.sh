#!/bin/sh
# build.sh — the L3 dispatcher gate. Run from anywhere:
#   ./native/voice/build.sh
set -e
here=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$here/../.." && pwd)
VYB=${VYB:-$HOME/Projects/Vyb/build/vyb}
export VYB_STDLIB=${VYB_STDLIB:-$HOME/Projects/Vyb/stdlib}
PY=${PY:-$HOME/miniconda/bin/python}
TZFILE=${VOICE_TZFILE:-/usr/share/zoneinfo/America/Los_Angeles}

cd "$here"
echo "== L3 dispatcher (tz.vyb + dispatch.vyb)"
VOICE_TZFILE="$TZFILE" VOICE_UTTERANCES="$ROOT/fixtures/voice/utterances.txt" \
"$VYB" host/dispatch_probe.vyb \
    --module-path "$ROOT/native/voice" \
    --module-path "$ROOT/native/tensor" \
    --module-path "$VYB_STDLIB" \
    > out/dispatch.txt 2>&1 || true
tail -1 out/dispatch.txt
OFF=$(grep -A1 "^OFFSET$" out/dispatch.txt | tail -1 | tr -d ' ')
"$PY" check_dispatch.py out/dispatch.txt "$TZFILE" "$OFF"
