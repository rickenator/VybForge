#!/bin/sh
# round_trip.sh — the loop's closed-loop measurement: text -> audio -> text.
#
#   ./native/tts/round_trip.sh hello
#
# Synthesises the argument with the Vyb synthesiser, then transcribes the result
# with the Vyb Whisper and prints both. This is NOT wired into build.sh as a gate:
# it does not pass yet, and a gate that is permanently red for work not yet done
# hides the failures that matter. It is here as a measurement, so the gap between
# "makes a sound with the right spectrum" and "says a word" can be watched closing.
set -e
here=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$here/../.." && pwd)
VYB=${VYB:-$HOME/Projects/Vyb/build/vyb}
export VYB_STDLIB=${VYB_STDLIB:-$HOME/Projects/Vyb/stdlib}
WORD=${1:-hello}

cd "$here"
VOICE_TEXT="$WORD" TTS_OUT="$here/out" "$VYB" host/say_probe.vyb \
    --module-path "$here" --module-path "$VYB_STDLIB" > out/say.txt 2>&1
grep "^WORD" out/say.txt || true

cd "$ROOT"
WHISPER_WAV=native/tts/out/say.wav WHISPER_STAGES=text "$VYB" \
    native/whisper/host/forward_driver.vyb \
    --module-path native/mel --module-path native/torchload \
    --module-path native/json --module-path native/tensor \
    --module-path "$HOME/Projects/Vyb/bindings/cuda" \
    > native/tts/out/roundtrip.txt 2>&1 || true

echo "sent:     $WORD"
echo -n "heard:    "
grep -A1 "^TEXT$" native/tts/out/roundtrip.txt | tail -1
