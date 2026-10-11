# native/whisper — the Whisper slice of the voice loop (P2, in progress)

Whisper tiny.en as the loop's speech recogniser: 80 x 3000 log-mel in (from
`native/mel`), English text out. Architecture and phase plan:
`../../doc/VOICE-LOOP.md`.

## State

| Piece | State | Gate |
|---|---|---|
| Checkpoint inventory | **done** | `./build.sh weights` — 167 tensors, names/dtypes/shapes/bytes exact |
| Encoder front half (conv1, conv2, +positional) on the GPU | **done** | `./build.sh stages` — all four stages within 1.3e-5 of the real model |
| Encoder transformer blocks (layernorm, attention, MLP) | not started | — |
| Decoder + greedy decode | not started | — |

### Stage evidence (RTX 3090)

```
conv1: n=1152000 max|diff|=4.710e-06 rel=6.469e-07 over 1e-4: 0  [ok]
conv2: n= 576000 max|diff|=1.288e-05 rel=3.857e-06 over 1e-4: 0  [ok]
e_pos: n= 576000 max|diff|=9.995e-07 rel=9.995e-07 over 1e-4: 0  [ok]
e_in:  n= 576000 max|diff|=1.274e-05 rel=3.015e-06 over 1e-4: 0  [ok]
```

### How the front half works

* **Convolutions are im2col + the verified GEMM**, not a new convolution kernel:
  `out[C_out,T] = W[C_out, k*C_in] @ X_unf[k*C_in, T]`, zero padding 1, stride 1
  then 2. The weight reorder (f32 -> f64, and the GEMM's `(K, C_in)` tap order)
  happens on the GPU in one pass.
* **Device math comes from `native/kernels/vmath.vyb`**: kernel mode has no libm
  (`exp`/`erf` are compile errors — "not available in NVPTX device code"), so
  `exp` is `vexp` and `erf` is the Numerical Recipes rational approximation
  (|error| <= 1.2e-7, two orders below the gate tolerance). The tanh gelu
  approximation would be ~1e-3 off and is unusable here.
* **Layouts**: the conv stack is (channels, time); every later stage is
  (time, width). `native/mel` emits band-major, and the encoder transposes once.

### Traps hit here (each looked like a kernel bug, none was)

1. **A checker that swallows the element-count line** shifts every comparison by
   one and reads exactly like a one-frame kernel shift. The dump format is
   `SECTION <name> n=` + count + values, and the count is not a value.
2. **`add_bias_row` is not an elementwise add.** Using it for the positional
   table (a full `(T, D)` matrix) gives the right answer for row 0 and the wrong
   one for the other 1499. Elementwise is `add2`.
3. **An undersized f32 staging buffer makes the bulk DMA fail**, surfacing as
   "stage failed" with no size in the message. Size it for the largest tensor
   uploaded that way (the positional table here is bigger than conv2's weight).
4. `their<[Float; N]>` parameters cannot be written element-wise, and a struct
   cannot hold a readable fixed array — kernels take/return packed by-value
   arrays (skill `vyb-native-audio-io`).

## What is done here

**Checkpoint inventory — gated.** `weights_probe.vyb` reads
`openai/whisper-tiny.en/model.safetensors` through the existing
`native/torchload/safetensors.vyb` reader and emits every tensor's name, dtype and
shape. `check_weights.py` compares that against an independent parse of the same
file (`oracle_weights.py`, which parses the JSON header itself) and against
transformers' own `state_dict`:

    tensors: vyb=167 ref=167  missing=0 extra=0 mismatched=0
    TOTAL_BYTES: vyb=151041024 ref=151041024   TENSORS: vyb=167 ref=167
    WEIGHTS_GATE: OK

Two structural facts that came out of it, both load-bearing:

* **The output projection is tied**: transformers' `state_dict` has 168 entries,
  the file has 167 — the missing one is `proj_out.weight (51864, 384)`, i.e. the
  decoder reuses `embed_tokens.weight`. Do not allocate a second 51864x384 table.
* The checkpoint is **f32** (not f16/bf16), so the f32-only substrate reads it
  without a dtype conversion pass.

**Staged reference tensors — captured.** `../../fixtures/whisper/oracle_stt.py`
runs the real model and dumps every intermediate the implementation must reproduce,
so a wrong forward pass is localised at a stage instead of guessed at:
conv1(+gelu), conv2(+gelu), positional embedding, encoder layer 0 sub-stages
(ln1/attn/residual/ln2/mlp), all four layer outputs, the encoder final norm, the
decoder prompt (`[50257, 50362]`), the decoder embedding, decoder layer 0
(self-attn/cross-attn/residuals/mlp), the first-step logits and the greedy ids.

Reference answer for the fixture (`jfk.wav`):

    text: " And so my fellow Americans ask not what your country can do for you, ask what you can do for your country."
    ids : 843 523 616 5891 3399 1265 407 644 534 1499 460 466 329 345 11 1265 644 345 460 466 329 534 1499 13
    cfg : d_model 384, 4+4 layers, 6 heads, 1500 source positions, vocab 51864

## Run

```sh
./build.sh stages     # ~2 min: compile kernels, run the driver, gate every stage
./build.sh weights    # checkpoint inventory gate (SKIPs cleanly if the file is absent)
./build.sh all
WHISPER_STAGES=conv2 ./build.sh stages     # one stage while iterating
./build.sh gold       # regenerate every reference artifact
```

## What is left

The encoder's four transformer blocks and then the decoder. New pieces needed:
`layernorm` (the substrate has RMSNorm only), non-causal self-attention and
cross-attention (the existing attention kernel is causal self-attention), an
embedding gather (exists in the Qwen path — reuse it), and a chunked argmax over
51864 logits. Why this cannot be CPU work: Vyb CPU measures ~107 MFLOP/s and the
encoder is ~28 GFLOP, against a 1.5 s end-to-end latency budget.

Gate order: mel -> conv1 -> conv2 -> +positional -> encoder layer 0 sub-stages ->
each layer -> final norm -> decoder embed -> decoder layer 0 -> first-step logits ->
greedy ids; acceptance is token-for-token on the 24 ids above.
