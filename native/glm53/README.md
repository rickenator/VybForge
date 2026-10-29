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

- Vyb kernels: grouped GEMM over the selected experts + dispatch/gather, and the
  router (sigmoid/top-8/renorm) on device; then `MOE_VYB_MATCH` against
  `moe_ref.moe_block` on the same fixture.
