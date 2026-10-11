# native/whisper — the Whisper slice of the voice loop (P2, in progress)

Whisper tiny.en as the loop's speech recogniser: 80 x 3000 log-mel in (from
`native/mel`), English text out. Architecture and phase plan:
`../../doc/VOICE-LOOP.md`.

## State

| Piece | State | Gate |
|---|---|---|
| Checkpoint inventory | **done** | `./build.sh weights` — 167 tensors, names/dtypes/shapes/bytes exact |
| Decoder: prompt embedding, four layers, cross-attention, MLP | **done** | `./build.sh decoder` — 1.0e-06 … 2.6e-06 per stage |
| Encoder front half (conv1, conv2, +positional) on the GPU | **done** | `./build.sh stages` — within 1.3e-5 of the real model |
| Encoder transformer blocks (attention, MLP, residuals) + final norm | **done** | all 14 stages green in one run |
| Decoder: four layers, final norm, tied head, argmax | **done** | `./build.sh decoder` — logits 5.2e-05 over 51864 values; **first token 843** |
| Greedy decode loop (24 tokens) + id-to-text | not started | acceptance: the first 24 ids token-for-token |

### Stage evidence (RTX 3090)

```
conv1     1152000  max|diff| 4.710e-06  rel 6.5e-07
conv2      576000  max|diff| 1.288e-05  rel 3.9e-06
e_pos      576000  max|diff| 9.995e-07  rel 1.0e-06
e_in       576000  max|diff| 1.274e-05  rel 3.0e-06
e_l0_ln1   576000  max|diff| 1.650e-05  rel 1.1e-06     layer 0 sub-stages:
e_l0_attn  576000  max|diff| 6.656e-06  rel 1.7e-06       ln1, attn, residual,
e_l0_res1  576000  max|diff| 1.257e-05  rel 4.6e-06       ln2, mlp
e_l0_ln2   576000  max|diff| 6.618e-05  rel 4.6e-06
e_l0_mlp   576000  max|diff| 2.022e-05  rel 3.8e-06
e_layer0   576000  max|diff| 2.269e-05  rel 3.3e-06
e_layer1   576000  max|diff| 2.384e-05  rel 2.9e-06
e_layer2   576000  max|diff| 2.214e-05  rel 2.2e-06
e_layer3   576000  max|diff| 1.439e-02  rel 2.3e-05     activations to 631
e_final    576000  max|diff| 3.975e-05  rel 3.3e-06
STAGES_GATE: OK
```

The last block's absolute figure is the float32 REFERENCE's own accumulation
noise, not this implementation's: an independent float64 numpy implementation of
that block, fed this implementation's own `e_layer2`, differs from the float32
reference by 1.46e-02 on exactly the same three entries. The gate's bar is set
from that measurement (abs 1e-3 / rel 3e-4) — an order above the measured
reference noise, three orders below a real bug's signature.

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

1. **A transpose mode validated only on square matrices.** The linear-weight
   transpose passed every attention projection (all 384x384) and then folded the
   FFN's non-square weights, because the routine was handed the row/column words
   in the wrong order — invisible at 384x384. Test a layout mode at a
   non-degenerate geometry, or it will pass exactly where you are looking.
2. **A scratch buffer sized for the model width, not for the widest consumer.**
   A 384-value bias staging buffer was reused for a 1536-value FFN bias, so the
   widen pass ran off the end into the weight buffer next to it. The result was a
   *partially* wrong MLP (correlation ~0.5 with the reference) — not an error,
   not obvious, and it survived one round of "fixed it" because the second
   problem was hiding behind the first.

3. **A checker that swallows the element-count line** shifts every comparison by
   one and reads exactly like a one-frame kernel shift. The dump format is
   `SECTION <name> n=` + count + values, and the count is not a value.
4. **`add_bias_row` is not an elementwise add.** Using it for the positional
   table (a full `(T, D)` matrix) gives the right answer for row 0 and the wrong
   one for the other 1499. Elementwise is `add2`.
5. **An undersized f32 staging buffer makes the bulk DMA fail**, surfacing as
   "stage failed" with no size in the message. Size it for the largest tensor
   uploaded that way (the positional table here is bigger than conv2's weight).
6. `their<[Float; N]>` parameters cannot be written element-wise, and a struct
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

## Decoder (in progress)

`host/decoder_driver.vyb` + the same `attn`/`embed`/`layerwise` kernels. Two
differences the existing kernels already cover: the decoder's self-attention is
CAUSAL (attn's `causal` param word — 0 for the encoder and cross-attention, 1 for
decoder self-attention) and its cross-attention is the same entry point with K/V
pointing at the encoder's output.

The tied head matters here: `proj_out.weight` exists in transformers' state_dict
but NOT in the file, so the 51864x384 token table is loaded once (in ~2M-element
pieces — a single 79.7 MB Vyb String is a lot to hold) and used twice, as a gather
for the input and as the transposed projection.

Rather than lift the encoder into a module two drivers share, the driver is now
`host/forward_driver.vyb` and runs the whole forward pass — the decoder needs the
encoder's final output and a Vyb tensor cannot travel through a file, so one
process is the honest shape of the thing. `./build.sh stages` gates the encoder
stages, `./build.sh decoder` the decoder stages, same binary.

The decoder is complete as a single step: `dec_layer` (causal self-attention,
cross-attention over the encoder output, MLP, residuals) runs four times, the final
norm feeds the tied head, and a single-thread argmax picks the token.

```
d_embed        768  max|diff| 1.0e-06      d_l0_crossattn  768  1.2e-06
d_l0_ln1       768  max|diff| 1.1e-06      d_l0_mlp        768  2.6e-06
d_l0_selfattn  768  max|diff| 1.0e-06      d_step0_logits 51864 2.5e-05
d_l0_res1      768  max|diff| 1.0e-06
first-token gate: OK (843 = " And")
```

**Next step**: the greedy loop — feed the sampled token back in, re-run, stop at
50256 — and the id-to-text path. Note the decode loop must re-run the decoder per
token (no KV cache yet): CL grows by one each step, which is correct but wasteful,
and the first thing to optimise once the loop is honest.

## A gold tensor whose NAME implied more than it measured

`d_final_ln` was the oracle's final decoder norm applied to **layer 0's output** —
a one-layer probe — while the name said "the decoder's final norm". The four-layer
driver matched the logits to 1.2e-06 (rel) and the first token exactly, and failed
that gate across all 768 values. The implementation was right and the CHECK was
wrong, which is the more expensive way round to find out.

The rule: when a gate fails and a downstream gate on the same data passes, question
what the gold tensor IS before touching the code. It is renamed
`d_layernorm_on_l0_out` in the gold set and in the oracle that generates it, and the
real four-layer path is gated by `d_step0_logits` — the same norm, feeding the head.

## Facts about this checkpoint that the file itself settles

* `k_proj.bias` occurs ZERO times: every k_proj in the model is biasless — the
  encoder's and the decoder's self-attention AND the decoder's cross-attention.
  Three projections in the decoder look identical and are not.
* `proj_out.weight` is in transformers' state_dict and absent from the file: the
  head is tied to `embed_tokens`. The table is loaded once, TRANSPOSED
  (`stage_table`), which is the layout the head needs as a GEMM operand — and the
  gather then reads it with one index change instead of needing a second copy.

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
