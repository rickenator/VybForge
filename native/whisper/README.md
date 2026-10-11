# native/whisper — the Whisper slice of the voice loop (P2, in progress)

Whisper tiny.en as the loop's speech recogniser: 80 x 3000 log-mel in (from
`native/mel`), English text out. Architecture and phase plan:
`../../doc/VOICE-LOOP.md`.

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
./build.sh weights    # checkpoint inventory gate (SKIPs cleanly if the file is absent)
./build.sh gold       # regenerate the reference inventory
```

## What is left

The forward pass itself, on the GPU kernels under `native/` — see the P2 section of
`doc/VOICE-LOOP.md` for why it cannot be CPU work (Vyb CPU measures ~107 MFLOP/s,
the encoder is ~28 GFLOP) and for the kernel work list: conv1d (stride 1 and 2),
exact-erf gelu, layernorm, non-causal self-attention, cross-attention, embedding
gather (exists), and a chunked argmax over 51864 logits.

Gate order: mel -> conv1 -> conv2 -> +positional -> encoder layer 0 sub-stages ->
each layer -> final norm -> decoder embed -> decoder layer 0 -> first-step logits ->
greedy ids; acceptance is token-for-token on the 24 ids above.
