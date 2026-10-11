#!/bin/sh
# build.sh — L4's synthesis-core gate. Run from anywhere:
#   ./native/tts/build.sh
set -e
here=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$here/../.." && pwd)
VYB=${VYB:-$HOME/Projects/Vyb/build/vyb}
export VYB_STDLIB=${VYB_STDLIB:-$HOME/Projects/Vyb/stdlib}
PY=${PY:-$HOME/miniconda/bin/python}

cd "$here"
echo "== L4 synthesis core (Klatt cascade formants, WAV out)"
TTS_OUT="$here/out" "$VYB" host/synth_probe.vyb \
    --module-path "$here" \
    --module-path "$VYB_STDLIB" \
    > out/synth.txt 2>&1 || true
tail -1 out/synth.txt
"$PY" check_vowels.py out/synth.txt
