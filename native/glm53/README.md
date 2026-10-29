# native/glm53 — model-level references for GLM-5.3-Flash (P2+)

Companion to `native/exl3` (P1: EXL3 expert dequant, gated). This directory holds
the per-block references that the Vyb kernels are gated against, for
`doc/ROADMAP-GLM53-ON-VYB.md`.

## P2 — MoE block

`moe_ref.py` pins the block semantics from exllamav3 (the engine that serves this
checkpoint) and `verify_moe.py` gates them:

- `MOE_ROUTE_VERIFY: OK` — numpy router vs the SHIPPED reference
  `exllamav3.modules.block_sparse_mlp_routing._routing_ds3_torch` (the composition
  the CUDA router kernel is tested against) on the real layer-3 router weights:
  8/8 tokens with identical expert sets, max |Δweight| 1.2e-4 (the oracle returns
  fp16 weights).
- `MOE_BLOCK_SANITY: OK` — numpy vs torch on identical weights: rel diff 5.7e-16.
- **`MOE_VYB_MATCH: OK`** — the Vyb kernels (`native/kernels/moe.vyb`:
  `moe_route`, `moe_act`, `moe_down`, driven by `native/host/moe_driver.vyb`,
  `make moe`) on the prepared fixture: 4/4 tokens identical expert sets, max
  |Δweight| 3e-8, block output rel 8e-8 (scale 0.745).

P2 is complete. Fixture caveat: only expert 0's trellis is in the P1 fixture, so
every lane uses expert 0's weights (the routing is still fully exercised — the
selection, weights and lane dispatch all agree with the oracle). Distinct
per-expert weights arrive with the loader (P4).

Semantics (scoring_func=sigmoid, topk_method=noaux_tc, n_group=topk_group=1):

    logits = y @ gate.weight^T           # [S, 288] fp32
    scores = sigmoid(logits)
    choice = scores + e_score_correction_bias    # bias is SELECTION-only
    idx    = top8(choice)
    w      = scores[idx] / (sum + 1e-20) * routed_scaling_factor (2.5)
    FFN:   g = min(gate_proj(x), 10.0); u = clamp(up_proj(x), ±10.0)
           e = (silu(g) * u) @ down_proj        # swiglu_limit = 10.0
    out = Σ_i w_i e_i + shared_expert(x)        # shared expert, same clamp

Layout note: the checkpoint stores `[out, in]` (HuggingFace convention), while
the P1 dequantised experts come out `[in, out]` — `load_shared()` transposes.

Fixture (gitignored): `native/glm53/fixture/`, layer 3's router + shared-expert
weights pulled from `lefty:/srv/models/hf` (see `native/exl3/README.md` for the
extractor pattern). Only expert 0's trellis is in the P1 fixture, so the block
sanity gate maps expert 0 to every lane; the 8-distinct-expert case arrives with
the loader (P4).

## Remaining for P2

P2 is done (`MOE_VYB_MATCH: OK`). Carried into P4 with the loader: distinct
per-expert weights, a per-expert pointer table instead of the single expert-0
set, and the grouped/batched GEMM shape that beats the simple per-lane loop.

## Hyper-connections (mHC) — P3, gated

The body carries `hc_mult = 4` parallel fp32 residual streams per token. Every
sublayer site has its own mixer: `hc_attn` and `hc_ffn` per layer, tensors
`model.language_model.layers.N.hc_{attn,ffn}_{fn,base,scale}`:
`fn` bf16 [24, 16384] (rows = `(2+H)*H`: pre 4, post 4, comb 16), `base` f32 [24],
`scale` f32 **[3 scalars]** (pre/post/comb scales, not a matrix).

    flat = rms_norm_unweighted(streams.reshape(R, 4*4096))   # fp32 streams
    mix  = flat @ fnᵀ                    # pre(4) | post(4) | comb(4x4)
    pre  = sigmoid(pre_w*pre_s + pre_b) + hc_eps        # hc_eps = 1e-6
    post = 2 * sigmoid(post_w*post_s + post_b)
    comb = softmax(comb_w*comb_s + comb_b, -1) + hc_eps
    comb = comb / (colsum + hc_eps); then 5 x {rowsum, colsum} normalise (sinkhorn)
    collapsed = Σ_h pre_h * stream_h                    # what the sublayer consumes
    x ← post ⊗ y + combᵀ x                              # apply_ residual update

`native/glm53/hc_ref.py` (numpy) + `verify_hc.py`: **HC_REF_VERIFY: OK** against
exllamav3's shipped path (`HyperConnection._mix_unfused` / `apply_`, which dispatches
to the fused `ext.hc_mix` CUDA kernel on a GPU) on the real layer-3 weights:
post |Δ| 7e-8, comb |Δ| 1e-7, residual update 1e-7, collapsed bit-exact at fp16
(the fused path stores `collapsed` as half; 1 of 12288 elements differs, a rounding
boundary). Observed on real weights: `pre` spans 0.0002-0.54 — some streams are
nearly switched off — and comb columns sum to exactly 1.0 after sinkhorn.

## Remaining for P3

gated delta-net (34 layers) and MLA + DSA indexer (11 layers) still to pin against
their torch paths; then the per-layer composition (P4) can be assembled.

## Vyb host/kernel gotchas found here

- **A 4-byte DtoH into a `Float` gives denormals.** The buffers are f32; copying
  4 bytes into an 8-byte Float yields values like 4.8e-315. Read the word into an
  `Int` and decode the 4 bytes in python instead.
- **A bf16→f32 conversion applied twice silently zeroes the data** (`astype(uint32)`
  on an f32 array truncates it to 0/1). `load_router()` returns f32 now; nothing
  downstream converts again.
- **Fixture scale matters for routing tests.** With `y ~ N(0, 0.02)` the selection
  bias (≈7.4-7.8) dominated the gate logits and every token routed identically;
  `y ~ N(0, 0.3)` makes the logits decide.
- **The harness reaps long background jobs (~6 min).** Split slow steps and cache
  their outputs (the f64 dequant of three 2048x4096 experts is minutes of numpy;
  `prepare_moe_fixture.py` caches each array in `prep/`).
