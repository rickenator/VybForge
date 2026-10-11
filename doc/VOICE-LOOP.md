# VOICE-LOOP — the fully self-contained Vyb conversational loop

Status: architecture mandate (Rick), 2026-10-11. Owner: native/VybForge + Vyb bindings.
Companion: `native/ROADMAP.md` (substrate phases), `doc/SUBSTRATE-SCOPE.md` (generality rule).

## Where it stands

**P0 (host audio path) is done and gated** — see `native/audio/README.md`. A Vyb
program read 64000 frames (4 s @ 16 kHz mono) from the host microphone and wrote a
valid WAV whose levels agree with two independent measurements; a second Vyb
program synthesised a 440 Hz tone that was captured at the sink's monitor at the
programmed amplitude, 99.9 % of its energy inside 40 Hz around 440 Hz.

**P1 (mel front-end) is done and gated** — see `native/mel/README.md`. Vyb-native
FFT (Bluestein over a radix-2 kernel, because 400 = 2^4*5^2 is not a power of two),
Slaney filterbank, reflect-padded framing and the log/floor/scale normalisation.
Against the reference: filterbank exact to 1e-12, the full 80x3000 log-mel within
1.3e-5 (float32-vs-float64 rounding) of `transformers.WhisperFeatureExtractor` on
the same PCM.

**P3 (dispatch) is done.** `native/voice/dispatch.vyb` routes a normalised
utterance to a reply, with the clock as a PARAMETER (text, epoch, utc offset) so
every case is reproducible rather than "whatever time it happened to be". It is
deliberately not a model: a fixed phrase set routed through an LLM adds a second
thing that can be wrong and makes the layer untestable. What it needed first was a
local time, and Vyb has none — the stdlib gives epoch seconds and nothing else — so
`native/voice/tz.vyb` parses the TZif (zoneinfo) file directly rather than shelling
out to `date` or adding a C shim, both of which would put a non-Vyb dependency in a
path the mandate says is self-contained. The gate recomputes the expected time and
date in Python from the same epoch and offset, so the calendar conversion is
checked rather than echoed.

**P2 (STT) — the whole Whisper encoder runs Vyb-native on the GPU and is gated
stage by stage.** `native/whisper` computes mel -> conv1 -> conv2 -> transpose ->
+positional -> four transformer blocks -> final layer norm, with all 14 stages
compared against the real model's own intermediates in one run (relative
deviations 1e-06..3e-05; the single 1.4e-02 absolute figure on layer 3 is the
float32 reference's accumulation noise, proved by an independent float64
implementation disagreeing on exactly the same entries). Convolutions are im2col +
the verified f64 GEMM rather than a new convolution kernel; the checkpoint's f32
weights are reordered and widened on the GPU in one pass; device math comes from
`native/kernels/vmath.vyb` because kernel mode has no libm (no `exp`, no `erf`).
What remains for P2 is the DECODER: self- and cross-attention over the same
`attn` kernel, the prompt (`[50257, 50362]`), greedy decode with a chunked argmax
over 51864 logits, and the tokenizer's id->text path. The reference tensors for all
of it are already captured under `fixtures/whisper/gold_stt/`.

Nothing else exists: no command layer, no TTS.

## The loop, as mandated

    mic (BT earbud) -> [L1 native audio in] -> [L2 Vyb-native STT] -> text
      -> [L3 VybForge command processing] -> action
      -> [L4 native TTS] -> BT earbud speaker -> back to the human

Every box is Vyb. No Python at runtime, no external ASR/TTS service, no cloud.
Host: godzilla (ASUS-Z590-E), RTX 3090 24 GB, PipeWire 1.0.5 + BlueZ 5.72.
Transport in/out: Bluetooth earbuds (HSP/HFP mic + A2DP/HFP playback).

## What already exists (verified on this host, this checkout)

| Need | State |
|---|---|
| GPU substrate for neural nets | VybForge `native/`: GEMM/RMSNorm/RoPE/SiLU-MLP/GQA-attn/F32+BF16/dequant/GGUF/safetensors/tokenizer/sampler/stack/decode — all Vyb-native, all oracle-gated |
| Vyb FFI pattern for native libs | proven: `bindings/sdl3/mod.vyb` (compiler-side) and VybWin `src/sdl3/sdl3_win_shim.h` + auto-generated `sdl3_win_binding.vyb` (`vyb bindgen`) |
| SDL3 on the host | built at `~/sdl3local` (3.4.16) — **`-DSDL_AUDIO=OFF`**, video only |
| Concurrency | Vyb stdlib `threads`, `tasks`, `channels` — capture can own a thread |
| Audio server / BT stack | PipeWire 1.0.5 + WirePlumber + `libspa-0.2-bluetooth` + BlueZ 5.72 active |
| FFI dev headers for audio | **absent** (`libpipewire-0.3-dev`, `libasound2-dev`, `libpulse-dev` not installed) |
| FFT / mel | **none anywhere in Vyb** — no `fft` in stdlib, no mel, no windowing |
| Any STT/TTS code or model | **none in any repo**. Cached but unusable as-is: `Systran/faster-whisper-base{,.en}` (CTranslate2 `model.bin`, not safetensors) |
| Oracle tooling | VybForge `.venv` has numpy 2.5.3 + transformers + tokenizers (no torch; other venvs have torch) |

## Layer contracts

### L1 — native audio input/output module (`native/audio`)  [done, gated]
- Public surface, all in Vyb: `start()`, `open_capture(dev, rate)`, `read_block(h, frames) -> Vec<Float>?`,
  `open_playback(dev, rate)`, `write_block(h, samples)`, `stop(h, drain)`, `last_error()`. Raw FFI stays
  inside `freedom { }` in the module.
- Backend: **ALSA through PipeWire**, reached by a C shim we compile ourselves (`vyb_audio.c`) that
  `dlopen`s `libasound.so.2` and declares the ABI facts it needs by hand. Chosen over SDL3 audio because
  (a) SDL3 on this host is built `-DSDL_AUDIO=OFF` and the audio dev headers are absent, (b) no root is
  available in an agent session to install them, and (c) `pipewire-alsa` already routes `default` to the
  same devices SDL3 would enumerate. The cost is one small C shim; the win is zero new dependencies.
- Contract: mono float, opened at **16 kHz** in both directions, 1024-frame blocks (64 ms). The shim
  widens/narrows float32 <-> float64 because Vyb has no `CFloat` -> `Float` cast.
- WAV dump exists from day one, and the playback gate is verified by capturing the sink's monitor
  rather than by trusting a frame count.
- Compiler pitfalls found and worked around (module function names that collide with the prelude,
  `loc<CInt>` out-params in modules, a `Bool` parameter in the open entry point, `borrow(...)` for
  `their<T>` params, no `CFloat` cast) are listed in `native/audio/README.md` — read that before
  adding to the module.

### L2 — Vyb-native STT
- Model: **Whisper tiny.en** (39 M params, d=384, 4+4 layers, 6 heads, 1500 source
  positions, vocab 51864), weights from the HF safetensors checkpoint already
  downloaded to `/usr/export/LLM/hf/hub/models--openai--whisper-tiny.en`. Then
  `base.en` if the loop needs the accuracy.
- **Placement is decided by measurement, not taste**: Vyb CPU arithmetic is
  ~107 MFLOP/s (128³ GEMM, `-O2`), and tiny's encoder is ~28 GFLOP — about four
  minutes per utterance on the host CPU against a 1.5 s latency budget. The
  encoder and decoder therefore run on the **GPU kernels** under `native/`
  (verified GEMM + attention + the `tensor::` wrapper + the params-buffer launch
  pattern). The mel front-end stays on the CPU for now; it is ~3 s and has an
  obvious 7x cut available (frames that no longer touch real audio are exactly the
  floor constant).
- New kernels required, in build order (none of these exist today):
  1. **conv1d** — kernel 3, pad 1, stride 1 (conv1) and stride 2 (conv2). The
     existing `tools/conv_authority.c` is ggml's *depthwise causal* SSM_CONV, i.e.
     a different op; this is new work.
  2. **gelu** (the exact erf form, not a tanh approximation).
  3. **layernorm** (the substrate has RMSNorm only).
  4. **non-causal self-attention** with no mask, and **cross-attention**
     (queries from the decoder, keys/values from the encoder) — the existing
     attention kernel is causal self-attention.
  5. **embedding gather** for the token and positional tables (the `embed` kernel
     exists from the Qwen work and is reusable).
  6. **argmax over 51864 logits** with a running best, chunked if needed.
- Weights path: safetensors (f32 in this checkpoint) via
  `native/torchload/safetensors.vyb`, uploaded once and kept resident — the
  serving-latency lesson from the Qwen slice applies (model load dwarfs generation).
- Gate: staged against `fixtures/whisper/gold_stt/`, which the oracle captured from
  the real model. The order that localises a bug instead of guessing at it:
  mel -> conv1(+gelu) -> conv2(+gelu) -> +positional -> encoder layer 0 sub-stages
  (ln1, attn, residual, ln2, mlp) -> each layer -> final LN -> decoder embed ->
  decoder layer 0 (self-attn, cross-attn, mlp) -> first-step logits -> greedy ids.
  Acceptance is **token-for-token on the first 24 ids** of "And so my fellow
  Americans ask not what your country can do for you, ...", with `843` (" And") as
  the first generated id.
- Tokenizer: only *decoding* is needed for the loop (ids -> text); the prompt is
  the fixed pair `[50257, 50362]`. `native/tokenizer` already has the reverse-vocab
  and glyph->byte machinery; Whisper's byte-level BPE differs from Qwen's
  pre-tokenizer, so the encoder side would be a new profile if a later feature
  (e.g. timestamp prompting) needs it.
- Segmentation: energy/zero-crossing VAD in Vyb first (Silero VAD is a later
  upgrade), 8-10 s windows with overlap, so the loop is not "wait for silence".

### L3 — command processing in VybForge
- Reuse `native/host/decode_driver.vyb` + `native/sampler/sampler.vyb` + the verified GGUF path.
- Utterance text -> constrained output: a small action grammar (verb + slot, emitted as one JSON object)
  rather than free chat, so a mis-decode fails safe. Dispatch table in Vyb: each verb maps to a Vyb
  action (VybOS module, home-service HTTP via stdlib `http`, or local playback/volume).
- Evidence: a fixture file of N utterance -> expected-action pairs, gate = exact match count, no
  "mostly works" claim. `tools/reply_parse.vyb` / `validate_proposal.vyb` show the parse-then-validate
  pattern to copy.

### L4 — native TTS
Neural TTS is the highest-risk box (see risks). Order:
1. **Loop-closing synth**: pure-Vyb source-filter (Klatt-lite) or diphone concatenation, 16 kHz, no
   neural net. Robotic but real: it proves L1 playback + the turn structure end to end.
2. **Neural target**: Piper/VITS-class model (encoder + duration/pitch predictor + HiFi-GAN vocoder).
   New kernel family needed: **transposed conv / upsampling** (the existing conv work is neither),
   plus ISTFT (reuses the P1 FFT) and a text->phoneme step. Phonemization is a real dependency decision:
   either link espeak-ng (non-Vyb, breaks "entirely self-contained") or write a small English G2P in Vyb
   (rule set + exception lexicon — tractable, honest, limited to English).
3. Gate: reference-audio comparison offline (mel-cepstral distortion / spectral distance), plus an
   ASR round-trip: synth text -> L2 STT transcribes it back -> string match. The round-trip gate is the
   one that actually proves the loop.

## Hard constraints to design around (not solvable by writing more code)

1. **BT headset-profile bandwidth.** The earbud mic exists only while the headset profile (HFP,
   mSBC 16 kHz / CVSD 8 kHz) is active; A2DP playback and a live mic are mutually exclusive on most
   earbuds. Consequence: **hold the headset in HFP permanently** and design the whole chain at 16 kHz
   mono. Never let the codec flip per turn — the switch costs 1-2 s and kills the conversation.
2. **24 GB GPU is one card.** A 27B at W4A16 eats ~23.8 GB and leaves no room for STT + TTS. The voice
   loop must be sized together: Qwen3-4B class LLM + Whisper tiny/base + a Piper-size vocoder is ~4-5 GB
   total and leaves headroom. Do not design against the 27B.
3. **Latency budget.** Target < 1.5 s from end-of-speech to first audio out, measured on godzilla with
   timestamps, not felt. Rough allocation: 150 ms segment+mel, 200-400 ms STT, 300-500 ms LLM (short
   command, few tokens), 200-400 ms TTS. Barge-in (stop TTS the moment he speaks) is a requirement, not
   a nicety.

## Phase plan with gates

| Phase | Deliverable | Gate (evidence) |
|---|---|---|
| P0 | `native/audio`: shim + binding + module; 16 kHz capture and playback | **DONE** — 4 s WAV from the host mic with levels matching two independent measurements; 440 Hz tone confirmed at the sink monitor (440.43 Hz, 99.9 % in band, amplitude exact). Earbud-specific replay still pending the earbuds being paired |
| P1 | Vyb-native FFT + Hann + mel + log (`native/mel/`) | **DONE** — rfft vs numpy max|diff| 1.5e-6 (rel 4e-8); filterbank exact to 1e-12; full 80x3000 log-mel vs WhisperFeatureExtractor max|diff| 1.3e-5, 0 values over 2e-5 |
| P3 | Vyb-native text->action dispatch (no model, but a real clock) | **done** — `./native/voice/build.sh`, 12 utterances, time/date recomputed independently |
| P2 | Vyb-native Whisper tiny.en on the GPU kernels | **encoder done** — 14 stages gated vs the real model, relative error 1e-06..3e-05; the decoder and greedy decode remain; final acceptance is token-for-token on the first 24 ids of the jfk fixture |
| P3 | VybForge command layer: action grammar + dispatch + fixture battery | N/N utterance -> action exact match, reported as a count |
| P4a | Loop-closing Vyb synth | synth -> L2 STT round-trip string match |
| P4b | Neural VITS/Piper-class TTS (transposed conv, ISTFT, G2P) | MCD vs reference audio + round-trip ASR match |
| P5 | Full duplex: VAD, barge-in, latency instrumentation | measured end-of-speech -> first-audio-out on godzilla; loop survives a 10-turn conversation |

## Open decisions (need Rick)

1. **Target host**: run the loop as a Vyb application on the godzilla host first (everything above
   exists today), or build it inside VybOS from the start (which needs new `modules/bluetooth.vyb` +
   `modules/audio.vyb` in the VybOS framework — PipeWire/WirePlumber/BlueZ services — before any audio
   can exist at all).
2. **TTS first cut**: interim pure-Vyb synth (source-filter / diphone) to close the loop early, then
   neural; or straight to the neural vocoder path.
3. **Earbuds**: which profile to pin. The recommendation is to hold the headset profile (mSBC, 16 kHz
   mono) permanently rather than flipping to A2DP for playback — a flip costs seconds per turn. Worth
   confirming against the actual earbuds, since a few models behave badly when pinned to mSBC.
