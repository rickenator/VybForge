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

## Layer 3 MoE (Gate B) — gated on the oracle's own reproducibility

`verify_layer3_moe.py` runs the REAL layer 3 (DSA + 288-expert MoE) twice against
HF: pass 1 with zero experts captures HF's own h2 and top-k (the router bias is an
`nn.Buffer`, not a parameter — it must be assigned explicitly or HF routes without
it); the 8 selected experts are dequantized from EXL3 and loaded in place; pass 2
compares the MoE block on identical (h2, top-k, weights). **LAYER3_MOE_VERIFY: OK**
— rel(f64 w) 4.8e-3, rel(bf16-rounded w) 5.1e-3 on |out| 0.93.

The controlling finding: **expert selection is chaotically sensitive** — the MoE
output changes by **335% for one bf16 ulp on h2** (a near-tied score flips the
top-8 set). Consequences, both recorded here:
- Any implementation whose h2 differs by even 1 ulp (different accumulation order,
  fp32 vs bf16 routing arithmetic) can select a different expert set — outputs are
  then NOT comparable element-wise, only distributionally. The Vyb router must
  reproduce the reference precision path (bf16 routing arithmetic) for token-level
  parity, or parity gates must use logprob/distribution criteria (which P8 already
  specifies).
- Gate criteria for MoE layers must be measured against the oracle's own
  reproducibility (this gate measures HF's h2-ulp sensitivity and gates against it)
  rather than against fixed fp64 tolerances.

## Tokenizer (P5) — contract pinned

`tokenizer.json` (20 MB) extracted from the checkpoint into `fixture/`. Facts:
HF `TokenizersBackend`, `vocab_size` 154,820 (`len(t)` 154,856), **eos = pad =
154820, no BOS**, byte-level pretokenization (BPE over bytes, so any unicode
string round-trips exactly). `native/glm53/verify_tokenizer.py` →
**TOKENIZER_REF: OK** — encodes a 10-string corpus (code, unicode, chat
template, whitespace runs, 300-char repeat) with all round-trips exact and
writes `native/out/tokenizer_ref_ids.txt`, the golden ids a Vyb-side
implementation must reproduce byte for byte. 1 M-context cache math is
straightforward at these dims: 11 DSA layers x 1 M tokens x 512 latent x 2 B
= ~11 GB KV + 45 x 1 M x 128 x 2 B indexer keys = ~11.5 GB.

## MLA + DSA indexer — P3, gated (selection)

Config: `q_lora_rank 1536`, `kv_lora_rank 512`, 64 q-heads, `qk_nope_head_dim 256`,
**`qk_rope_head_dim 0`** (no rope anywhere in the main attention), `v_head_dim 256`;
indexer `index_n_heads 32`, `index_head_dim 128`, `index_topk 2048`,
`index_kpool 4` (compress ratio), `kpool_compress` on, `always_select_tail` on, and
all 45 layers are `"full"` indexers.

    q   = q_b(q_a_layernorm(q_a(x)))            # [.., 64, 256]
    ckv = kv_a_layernorm(kv_a_proj(x)[..., :512])
    k[h] = ckv @ W_UK[h] ; v[h] = ckv @ W_UV[h]        # from kv_b_proj
    o[h] = softmax(q[h]·k[h]/sqrt(256) + causal) @ v[h]
    out  = o_proj(o.reshape(.., 64*256))

    indexer: qk = index_qk_proj(x) -> q[32*128] + raw_k[128]
             q = q_layernorm(q) then rope ; pooled = mean over cr=4 blocks, k_layernorm, rope@block start
             score = relu(q·pooled).sum(heads) * 1/sqrt(128)
             blocks visible iff block < (pos+1)//cr ; top-k blocks -> cr tokens each ; plus the
             tail block's tokens ; drop tokens past pos ; pad -1 to a multiple of 32

`native/glm53/mla_ref.py` + `verify_mla.py`:
- **DSA_SELECT_VERIFY: OK** — numpy selection vs the shipped
  `QSAIndexer.select_indices_ref` at real indexer shape (32x128, kpool 4,
  block_topk 512, 2 seqs x 24 queries over 64 tokens of history): **0 differing
  entries out of 99,840**, i.e. identical token sets, order and -1 padding.
- **MLA_ABSORB_VERIFY: OK** — the absorbed-query form (what decode runs) vs the
  explicit `ckv·W_UK` / `ckv·W_UV` form: rel 1.4e-15.

Not yet gated (needs real weights, P4): the indexer's projection/norm/rope path and
the MLA q_a/q_b/kv_a latent projections — the gates above cover the selection
geometry and the attention algebra, not the checkpoint weights feeding them.

## Linear attention (KDA) — P3, gated

Config: `linear_attn_config = {num_heads 64, head_dim 128, short_conv_kernel_size 4,
gate_lower_bound -5.0}`, and `kda_layers` covers **all 34** linear layers, so every
linear layer is the KDA (per-channel forget gate) variant. Layer schedule is
`GGGAGGGAGGGAG...` (34 linear / 11 `deepseek_sparse_attention`).

    qkv   = qkv_proj(x)                     # q|k|v, 3 x 64 x 128
    conv  = silu(causal depthwise conv k=4) over q|k|v   (weight [C,4], w[...,3] = current token)
    z     = g_b_proj(g_a_proj(x))           # low-rank SIGMOID output gate
    beta  = sigmoid(b_proj(x)) * beta_scale
    f     = f_b_proj(f_a_proj(x))           # low-rank forget gate
    g     = -5 * sigmoid((f + dt_bias) * exp(a_log))     # per-k-channel log decay
    q̂ = l2norm(q)·dk^-0.5 ; k̂ = l2norm(k)              # eps 1e-6
    per step: state *= exp(g_i) ; kv = (state*k̂_i).sum(-2)
              state += k̂_i ⊗ ((v_i - kv)*beta_i) ; o_i = (state*q̂_i).sum(-2)
    out = gated_rmsnorm(o, z)               # KDA: sigmoid gate AFTER the weighted norm
    → o_proj → hyper-connection apply

`native/glm53/gdn_ref.py` + `verify_gdn.py` → **GDN_RULE_VERIFY: OK**:
- delta rule vs the module's own labelled reference (`torch_recurrent_kda`,
  "mirroring the GLM5.3/Kimi reference") at 64 heads x 128, 12 steps: **99.991% of
  98,304 outputs round to the identical bf16 value** (the reference returns bf16;
  the rest are last-bit ties).
- short conv + silu vs `causal_conv1d_update_function_torch`: rel 9.2e-8.
- gated RMSNorm vs `GatedRMSNorm.forward` (sigmoid form): rel 1.3e-7.

Still open in P3: MLA + DSA indexer (11 layers), then the GDN layer assembled from
real weights (projections are EXL3 trellis → the P1 dequant path).

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
