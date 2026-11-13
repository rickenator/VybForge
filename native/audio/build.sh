#!/usr/bin/env bash
# build.sh — build and run the L1 audio gates (voice loop, native/audio).
#
#   ./build.sh              # capture gate: 4 s from the default input -> WAV + stats
#   ./build.sh playback     # playback gate: 2 s 440 Hz tone to the default output
#   ./build.sh all          # both
#
# Everything runs natively (vyb --build), because the shim library is not on the
# compiler's RTLD_GLOBAL preload list: native linking resolves its symbols
# directly, while a JIT run would need `LD_PRELOAD=$PWD/libvybaudio.so`.
#
# No root and no -dev packages: the shim dlopens libasound.so.2 at runtime.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"

VYBHOME="${VYBHOME:-$HOME/Projects/Vyb}"
VYB="${VYB:-$VYBHOME/build/vyb}"
export VYB_STDLIB="${VYB_STDLIB:-$VYBHOME/stdlib}"

echo "== shim: libvybaudio.so"
cc -O2 -fPIC -shared -o libvybaudio.so vyb_audio.c -ldl

build() {   # build() <probe-name>
    echo "== vyb --build $1"
    "$VYB" "$1.vyb" --module-path . --link "./libvybaudio.so" --build "$1"
}

run_capture() {
    build capture_probe
    echo "== capture gate"
    ./capture_probe
}

run_playback() {
    build playback_probe
    echo "== playback gate (listen for a 2 s 440 Hz tone)"
    ./playback_probe
}

case "${1:-capture}" in
    capture)  run_capture ;;
    playback) run_playback ;;
    all)      run_capture; run_playback ;;
    *)        echo "usage: $0 [capture|playback|all]" >&2; exit 2 ;;
esac
