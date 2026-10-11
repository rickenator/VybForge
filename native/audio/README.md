# native/audio — L1, the voice loop's native audio module

The bottom box of the conversational loop: get mono float frames **into and out
of Vyb** on the host, with no Python and no service in the path. Everything else
in the loop (STT, command processing, TTS) sits on top of this module.

Full architecture: `../../doc/VOICE-LOOP.md`.

## Files

| File | Role |
|---|---|
| `vyb_audio.c` / `vyb_audio.h` | the shim: dlopens `libasound.so.2` at runtime and exposes 7 functions (init / lasterr / error / open / read / write / close) |
| `audio/vyb_audio_binding.vyb` | FFI binding, hand-written in `vyb bindgen` output format (7 declarations) |
| `audio/mod.vyb` | the Vyb-facing module: `start`, `open_capture`, `open_playback`, `read_block`, `write_block`, `stop`, `last_error`, `error_text` |
| `capture_probe.vyb` | capture gate: 4 s from the default input → `capture_vyb.wav` + level stats |
| `playback_probe.vyb` | playback gate: 2 s 440 Hz tone (windowed, amp 0.15) to the default output |
| `build.sh` | compiles the shim, builds and runs the gates |

## Run

```sh
./build.sh              # capture gate
./build.sh playback     # playback gate (listen)
./build.sh all
```

No root, no `-dev` packages: the shim resolves ALSA through `dlopen`, so it
needs nothing but the runtime `libasound.so.2` that PipeWire already pulls in.
`default` follows the system default PCM, i.e. whatever PipeWire's default
source/sink is — the USB audio interface today, the BT earbuds once paired.

Runs are **native** (`vyb --build ... --link ./libvybaudio.so`). The library is
not on the compiler's `RTLD_GLOBAL` preload list (`src/main.cpp`), so a JIT run
would need `LD_PRELOAD=$PWD/libvybaudio.so` to make the symbols visible to ORC's
process-symbol generator. Native linking is the clean path.

## Gates and their evidence

Capture (2026-10-11, godzilla, input = `alsa_input.usb-Generic_USB_Audio-00`):

* Vyb read **64000 frames** (4 s @ 16 kHz mono) in 16 blocks, and wrote a
  128044-byte WAV (`ch 1, rate 16000, width 2, frames 64000`).
* Vyb's own stats: `rms 0.002580  peak 0.019274`.
* Same WAV re-measured independently in Python: `rms 0.002576  peak 0.019257`.
* Same microphone via a standalone C harness: `rms 0.002574  peak 0.019212`.
  Three independent measurements agree, so the frames are real room audio, not
  silence, a stuck buffer, or a mis-scaled copy.

Playback:

* Vyb synthesised the tone (`sin`, fades) and wrote **32000 frames** (2 s).
* Captured at the sink's monitor during that run: `peak 0.149994` (the
  programmed `AMP = 0.15`), dominant frequency **440.43 Hz**, **99.9 %** of the
  spectral energy inside 420-460 Hz. The tone really reached the audio server.

## Vyb/FFI pitfalls found while building this (each cost a debugging round)

These are compiler behaviour, not shim bugs; they are recorded here so the next
FFI module does not rediscover them. All were observed with `build/vyb`
`c14d4768`.

1. **A module function named `read`, `write`, or `open` breaks the module.**
   Exporting `read` produced `Undefined identifier: va_open` for an unrelated
   imported symbol; exporting `open` produced that plus a segfault at runtime.
   Renaming to `read_block` / `write_block` / `open_capture` fixed it with no
   other change. Avoid prelude names for module exports.
2. **A scalar out-param (`xx<loc<CInt>>`) inside a `freedom` block in a module
   miscompiles into a bad pointer** — `snd_pcm_open` segfaulted on a garbage
   device pointer, while byte-identical code at file scope worked. The shim
   therefore has no `int *` out-params at all; failures travel through
   `va_lasterr()`.
3. **A `Bool` parameter in a single `open(dev, rate, channels, capture)` entry
   point yielded a handle whose reads always returned 0 frames**, while the same
   call shape with literal arguments read real audio. Hence two literal-argument
   entry points (`open_capture` / `open_playback`) — which is the honest API
   anyway, since the loop is mono 16 kHz both ways.
4. **`their<T>` parameters must be called with an explicit `borrow(...)`**;
   passing the `Vec` directly fails with `Argument type mismatch: Expected ptr
   but got { ptr, i64, i64 }`. Same idiom as `stdlib/png/mod.vyb`.
5. **No `CFloat` → `Float` cast exists.** Vyb's `Float` is `CDouble`, so a
   `float *` shim buffer cannot be read element-wise. The shim takes `double *`
   and widens/narrows internally (`VA_CHUNK` staging buffer).
6. **Strings returned as `const char *` must bind as `loc<CChar>`** and be
   copied out with `strlen` + `String::from_bytes` (a `CString` return value
   cannot be assigned to a `String`). Same trick as `bindings/sqlite`.
7. `loc(...)` of a local array works for pointer params (`loc(buf)` on
   `[Float; 1024]`), and `String` variables pass fine to `CString` params.

## Next steps for this module

* **Bluetooth**: pair the earbuds, then confirm the profile. The loop wants the
  headset profile (mSBC, 16 kHz) held permanently, because an A2DP/headset flip
  costs seconds and kills a conversation. Nothing in this module needs changing
  for BT — `default` follows the route — but the *policy* (pin the profile, keep
  16 kHz) belongs to the loop, not the shim.
* **Upstream posting**: once the loop depends on it, this should become
  `Vyb/bindings/alsa/` (shim + binding + module) with `libasound.so.2` added to
  the compiler's `RTLD_GLOBAL` dlopen list in `src/main.cpp` — a compiler-repo
  change, so it goes through the implementation agent / a PR on Rick's say-so.
* **Devices beyond `default`**: `va_open` already takes an ALSA PCM name, so
  `plughw:X,Y` works without code changes.
