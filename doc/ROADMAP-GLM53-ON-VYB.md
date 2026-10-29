# Roadmap — GLM-5.3-Flash on Vyb (replacing TensorFold as the daily driver)

Status: written 2026-10-28 after the rig postmortem (dsv41-exl3-nfs restart lost the inner
mount; glm53-flash-tf Exited (1) for 27 h; Hermes silently fell back to deepseek-flash).
Rick's direction: *"ultimately it needs to be vyb."* This doc is the honest gap list and the
phased path. It deliberately does NOT promise dates.

## The target, stated plainly

Serve **GLM-5.3-Flash** (`Glm5NextForConditionalGeneration`, `model_type glm5_next`) from the
Vyb-native substrate on the two DGX Sparks behind an OpenAI-compatible endpoint on
`http://192.168.1.249:8000/v1`, at parity or better vs the current TensorFold rig
(mia-tf v0.6.0: 80-90 tok/s math/code decode, 4 streams, 1,048,576 ctx, DFlash2 drafts,
vision on). Until phase P8 passes its A/B gate, **TensorFold stays the daily driver** and
nothing in `~/.hermes/config.yaml` changes.

## Ground truth (read off the real weights, lefty:/srv/models/hf snapshot 078455ff)

- 45 layers, `hidden_size 4096`, vocab 154,880, `rms_norm_eps 1e-5`, max ctx 1,048,576,
  untied embeddings. Weights: 83 bf16 safetensors shards, 166 GB on `/srv/models`.
- **Attention is a hybrid, not one thing** (`layer_types`): 34× `linear_attention`
  (gated delta-net: 64 heads × head_dim 128, `short_conv_kernel_size 4`,
  `gate_lower_bound -5.0`) + 11× `deepseek_sparse_attention` (MLA:
  `kv_lora_rank 512`, `q_lora_rank 1536`, `qk_nope_head_dim 256`, `qk_rope_head_dim 0`,
  `v_head_dim 256`, 64 heads, `mla_use_nope true`) with a lightning-style **indexer**
  (`index_topk 2048`, `index_head_dim 128`, `index_n_heads 32`, `index_kpool 4` +
  compress, `indexer_rope_interleave`, 45× `full`).
- **Hyper-connections everywhere**: `mhc true`, `hc_mult 4`, `hc_sinkhorn_iters 20`,
  `hc_eps 1e-6` — residual/hidden state is 4-wide, not 1-wide.
- **MoE**: 42 sparse layers (first 3 dense, `intermediate_size 12288`),
  `n_routed_experts 288`, top-8 per token (`num_experts_per_tok 8`), 1 shared expert,
  `moe_intermediate_size 2048`, sigmoid scoring, `topk_method noaux_tc` (n_group 1 /
  topk_group 1), `norm_topk_prob true`, `routed_scaling_factor 2.5`,
  `moe_router_dtype float32`, `swiglu_limit 10.0` (clamped SwiGLU).
- **Quant is narrow**: `quantization_config.scope == "glm53_routed_experts_only"` —
  ONLY the routed experts are EXL3 4bpw (`codebook mcg`, `head_bits 16`). Attention,
  shared expert, embeddings, norms, dense MLPs are bf16 in the shards. This is the single
  best piece of luck in the whole spec: the exotic dequant surface is the expert stack,
  nothing else.
- MTP: `num_nextn_predict_layers 1` (TensorFold serves it as DFlash2 drafts).
- Vision: separate ViT (24 deep, 1024 wide, 448 px, patch 14 → out 4096). Optional;
  text-only serving is acceptable for Hermes use.

## Gap table (what native/ has vs what this model needs)

| Capability | Standing today | Needed for GLM-5.3 |
|---|---|---|
| Dense decoder forward (GQA, RoPE, RMSNorm, SiLU MLP) | DONE + verified (Qwen3-4B, LAYER0/PREFILL/DECODE/KV all OK) | template only — GLM attention differs |
| bf16 safetensors reader | primitive (`native/torchload/safetensors.vyb` + probes) | scale it: 83-shard index, stream bf16 → device, no f64 |
| Grouped/batched GEMM (per-expert) | none | MoE top-8 dispatch over 288 experts |
| EXL3 4bpw dequant | none (only GGUF q4_0/q4_K/q6_K) | routed experts (mcg codebook, trellis dequant) |
| MoE router (sigmoid, noaux_tc, renorm, ×2.5) | none | every sparse layer |
| Gated delta-net linear attention | none (no recurrent/scan kernels) | 34 of 45 layers |
| MLA (q/kv LoRA, nope+rope split) | none | 11 DSA layers |
| DSA indexer (topk-2048 token selection, kpool compress) | none | same 11 layers |
| Hyper-connections (mhc ×4 + sinkhorn) | none | cross-layer residual plumbing |
| GLM tokenizer (154,880 BPE) | none (Qwen3 BPE port is the template) | encode/decode parity |
| 1M-ctx KV pool (paged, 2.1M-token scale) | per-layer slabs at 4B scale | real context window |
| TP=2 over RoCE (all-gathers, split weights) | none | the model cannot fit one Spark (~181 GB) |
| MTP/drafter, vision tower | none | optional, last |

## Phases (each ends in a committed verify gate; no phase starts without the previous gate green)

- **P0 — proving harness (no new kernels).** Dense GGUF model (Qwen3-4B family) served by
  Vyb on one machine behind a minimal OpenAI-compatible endpoint; greedy + sampled
  generations match the numpy reference; tok/s measured and recorded. Gate:
  `SERVE_E2E_OK` + benchmark table in this doc. Purpose: exercise the serving path
  (load → KV loop → detokenize → HTTP) without touching Sparks or TensorFold.
- **P1 — EXL3 4bpw dequant (routed experts).** Spec extracted from exllamav3 source +
  `mcg` codebook; numpy reference dequant (`*_ref.py`, verification-only Python like
  `layer0_ref.py`); golden fixture = one real expert tensor's packed bytes read off a
  lefty shard; Vyb kernel `deq_exl3` in `native/kernels/`. Gates: `EXL3_REF_VERIFY` and
  `EXL3_VYB_MATCH` (maxerr bound vs oracle, on real model bytes).
  **PROGRESS 2026-10-28**: `EXL3_REF_VERIFY: OK` — `native/exl3/exl3_ref.py` matches
  the exllamav3 oracle (`ext.reconstruct_had_slice`) on real expert bytes from
  `lefty:/srv/models/hf` (max abs diff 1.2e-4, fp16-butterfly noise only; rotated
  basis bit-exact). Key discovery: the window->slot placement is NOT fully
  hand-derivable — the 128 s-windows (j%8 in 4..7) break the sliding model; the
  verified per-slot composition is committed as
  `native/exl3/tile_slot_composition.json` (see `native/exl3/README.md` for the
  format, the oracle setup and its gotchas). Remaining: the Vyb kernel port
  (`deq_exl3`) gated as `EXL3_VYB_MATCH`.
  **DONE 2026-10-28**: `EXL3_VYB_MATCH: OK` — `native/kernels/exl3.vyb` +
  `native/host/exl3_driver.vyb` (`make exl3`) on the real fixture, ENTIRELY on
  GPU (`exl3deq` rotated basis + `exl3hgen`/`exl3had1`/`exl3had2` for the
  per-128-block Hadamard and suh/svh scales; no numpy in the pipeline): rotated
  basis max abs 9.8e-4 / rel 4.9e-4 (the reference's own fp16 intermediate
  rounding), full W max abs 4.4e-5, 0/8.4M over 3e-3. P1 complete.
- **P2 — MoE block.** Reference (numpy): router fp32 sigmoid → noaux_tc top-8 → renorm →
  ×2.5, shared expert + clamped SwiGLU experts, weighted sum. Vyb: grouped GEMM +
  dispatch kernels. Fixture: real bf16 router/shared weights + dequant'd experts from P1
  fixtures. Gates: `MOE_REF_VERIFY`, `MOE_VYB_MATCH`.
- **P3 — attention stack.** 3a gated delta-net (short conv + recurrent state update),
  3b MLA dense-equivalent (nope+rope, kv LoRA), 3c DSA indexer top-k selection +
  kpool compress. Oracles: the HF transformers implementation for glm5_next (or
  Qwen3-Next for the delta-net pieces). Gates: `LA_VYB_MATCH`, `MLA_VYB_MATCH`,
  `DSA_VYB_MATCH`. This is the highest-risk phase; budget it as such.
- **P4 — hyper-connections + one real GLM layer end-to-end.** bf16 body weights
  streamed from lefty, experts via P1, one full `glm5_next` layer forward vs the
  transformers reference on the godzilla 3090 (single layer fits; the full model never
  will). Gate: `LAYER_GLM53_MATCH`.
- **P5 — tokenizer + context.** GLM BPE port (reuse the Qwen3 BPE machinery; new vocab +
  merges + pre-tokenizer rules), then the paged 1M-token pool. Gates:
  `GLM_TOKENIZER_MATCH`, `POOL_1M_OK`.
- **P6 — TP=2 over RoCE in Vyb.** Split weights across lefty+righty, all-gather over
  both CX7 rails (reuse the cabling/`f0` facts in the glm53 skill), correctness first,
  then bandwidth. Gate: `TP2_VYB_MATCH` vs single-layer P4 output.
- **P7 — optional**: MTP drafter (needs P6 to be worth it), vision tower.
- **P8 — cutover.** Vyb server binds `:8000` behind the same contract as TensorFold
  (model id `GLM-5.3-Flash`, thinking flag, vision off acceptable), A/B: fixed-prompt
  logprob parity + tok/s vs the recorded TensorFold numbers, both nodes' containers up.
  Only then `~/.hermes/config.yaml` moves (via `hermes config set`, never hand-edited).

## Non-goals / guards

- Do not "improve" the TensorFold rig while this runs — it is the baseline and the
  fallback, and its numbers live in `references/mia-tf-settings-and-benchmarks.md`.
- Dev/verification runs on godzilla's 3090 with small fixtures (real tensor slices, not
  whole models). The Sparks serve production; a dev start on them must follow the glm53
  skill's GPU-freeing procedure and stop cleanly after.
- Python appears only as reference/verification harnesses (`*_ref.py`), per the
  substrate rule in `doc/SUBSTRATE-SCOPE.md`.
- Never commit model weights; central store stays `~/Models/` (GGUF) and
  `lefty:/srv/models/hf` (GLM shards).
