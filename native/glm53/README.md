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

## DSA indexer — the WEIGHTS path, corrected (dsa_ref.py + verify_dsa.py)

`mla_ref.indexer_select` models **exllamav3's `QSAIndexer`** (rope'd q/k, *mean*
pooling of complete cr-blocks). The checkpoint does not contain those tensors. Layers
3-44 carry, verbatim:

    self_attn.indexer.wq_b.weight                [4096, 1536]   (from the q_a latent)
    self_attn.indexer.wk.weight                  [128, 4096]    (from the layer input)
    self_attn.indexer.k_norm.{weight,bias}       [128]          LayerNorm, eps 1e-6
    self_attn.indexer.weights_proj.weight        [32, 4096]
    self_attn.indexer.index_kpool_compress_ape   [4, 128]       per-slot pooling bias
    self_attn.indexer.index_kpool_compress_gate  [128, 4096]    learned pooling gate

i.e. **learned gate-weighted pooling with a per-slot bias, no rope** — exactly HF's
`Glm5NextTextIndexer`. `DSA_SELECT_VERIFY` could not catch this: it compares
*selection geometry* against exllamav3 with synthetic q/pooled arrays (no checkpoint
weights), and at S=1 (the only length the earlier gates ran) every token is visible,
so any pooling rule selects the same set. The weights path was simply ungated.

`dsa_ref.py` is the corrected reference (HF semantics, batch 1, no cache):
`indexer_topk()` + `masked_attention()` + `layer_attention()`. Two layout facts that
cost time and are now pinned by the gate:
- HF **drops pools that are not fully valid** (`keep = pool_valid.any(0)`) before
  `select_k = min(index_topk // kpool, n_pools)`; dropping a pool narrows the pool
  block and moves the appended tail's column offset. Pool count therefore drives the
  topk *layout*, not just the selection.
- the appended incomplete tail lands at column `select_k*kpool`, padded with -1 to
  `index_topk + kpool - 1` (= 2051) columns.

`verify_dsa.py` (real layer-3 tensors, S=22, layers 0-2 chained in numpy first):
`DSA_INDEXER_VERIFY: OK` — per-query selected token sets identical on all 22 queries,
and 45102/45122 topk entries **bit-identical**; the 20 that differ are the *order of
the selected pool blocks* (same tokens, two near-tied pool scores swapping under
bf16), the documented lane-order boundary class. `DSA_ATTN_VERIFY: OK` — masked MLA
attention out rel 8.3e-3 on |out| 1.9 (bf16 side's own rounding); projections on the
way: input_layernorm 6.0e-3, q_a latent 7.9e-3, q_b 6.0e-3, kv_a_layernorm 7.4e-3.

Not exercised (and cannot be at these lengths): actual top-k *pruning* needs
> index_topk tokens (8,192+ for pools to outnumber `index_topk/kpool`); every query
here sees all tokens causally, so selection == causal mask. The pool-score ordering,
pooling math and tail layout are gated; pruning is gated by construction once a long
context run exists.

## Vyb checkpoint index (P4) — GX_INDEX_MATCH: OK

`native/glm53/gx.vyb` resolves a tensor name to (shard, byte offset, length, dtype)
in pure Vyb: a substring scan of `model.safetensors.index.json` gives the shard
(the weight_map has ~157k entries and the JSON module has no key iteration, but
`index_of()` over the raw text is cheap), then the gated safetensors header reader
gives the byte range, then `gx_read` returns the bytes for upload.
`native/host/gx_check.vyb` + `native/glm53/gx_ref.py` (independent safetensors-library
oracle) agree **byte for byte** on 10 real tensors — layer-0 attention/mlp/hc,
layer-3 router/shared/hc, three EXL3 expert parts and expert 287's trellis — with the
deliberate bogus name reported MISSING on both sides. Resolution is ~40 s for a
one-off batch (5 MB index scan + 3 header parses); shard opens must be cached per
layer for the full 45-layer load.

## Vyb-side loader (P4) — bytes to the device, gated

Python `loader.py`/`kv_cache.py` stay the oracle; the runtime path is Vyb.
**`native/host/gx_load_driver.vyb` + `native/glm53/gx_load_ref.py` → `GX_LOAD_MATCH: OK`**
(`make -f native/Makefile gxload`): reads the byte range `gx` resolved, uploads the raw
payload to the 3090 (`cuMemcpyHtoD_v2` straight from the Vyb String, no per-word loop),
then reads three 8-byte words back from the device (first / 8-aligned middle / last) and
compares them to the host bytes in-driver, so a lossy transfer fails here rather than in
a kernel. 10 real tensors: 64 MB and 96 MB dense bodies, a 12-byte F32 buffer, the router
table, a shared expert, a DSA `kv_b_proj`, EXL3 `trellis`/`suh` parts. The oracle parses
the shards independently (header `data_offsets` + a plain file handle) and the listings
must match byte for byte.

Vyb gotchas hit here: no module-level constants (a string literal must live inside the
function), there is no `Int.to_char` (slice `"0123456789abcdef"`), `cuMemFree_v2` takes
the device pointer value, not `loc(ptr)`, and the ctx/alloc loop belongs inside one
`freedom { }` block as the other host drivers do.

## EXL3 dequant as a callable Vyb path — `make glm3x` → EXL3_LOAD_VYB: OK

The MoE's routed experts are EXL3 4 bpw. The P1 gate proved that pipeline on one fixture
tensor; this makes it a **repeatable call at real expert shape**: layer 3, expert 0,
`gate_proj` [4096, 2048] → 8,388,608 f64 values in [in, out] order, gemm-ready.

`native/host/glm_exl3_check.vyb` runs trellis → `exl3deq` → `exl3hgen`/`exl3had1`/
`exl3had2` with the 256-slot tile map re-headered at load time (cw = out features,
ntiles = (in/16)·(out/16)), i.e. no Python in the path and no per-shape Python tooling.

Against the P1 oracle: sq rel 8.4e-7, max rel 7.8e-5 (max abs ≈ 1.2e-5), abs rel 5.5e-7
— inside the band the P1 gate established (max abs 4.4e-5, 0/8.4M over 3e-3).

Two things worth knowing:
- **`extern "C"` FFI declarations do not cross module boundaries.** `native/glm53/exl3_load.vyb`
  holds the cleaned-up version of this function, but importing it fails: the module's own
  `share(all)` functions cannot see its `extern "C"` cu\* declarations ("Undefined
  identifier: cuMemFree_v2"). The sanctioned route is `import cuda_binding` (Vyb's
  `bindings/cuda`), which exposes alloc/free/module-load/launch but **no bulk host→device
  copy**, so a 4 MB trellis currently uploads in 8-byte words (that is the 37 s). A bulk
  copy in `cuda_binding` would make this fast and make the module importable — a Vyb-side
  gap, not a VybForge one.
- Reading a dequantized expert back to the host is 67 MB; the MoE should keep it on the
  device and gemm against it (the check driver downloads only to make a digest).

## Layer 3's MoE router + shared expert in Vyb — `make glm3moe` → both gates OK

`native/host/glm_l3_moe_fwd.vyb` runs the ffn site's front half on real weights: the router
(sigmoid scores, selection-only `e_score_correction_bias`, top-8, renorm ×2.5) and the
clamped shared expert (`min(gate,10)`, `clamp(up,±10)`, silu·u @ down).

- `L3_MOE_ROUTE_VYB: OK` — **22/22 tokens select the identical top-8 expert set**, weights
  agree to 2.1e-10. This matters more than it looks: the routing gate in `verify_layer3_moe`
  measured a 335 % output change for ONE bf16 ulp on h2, i.e. selection is chaotic — the Vyb
  router reproduces numpy's selection exactly on all 22 tokens, not merely "close".
- `L3_MOE_SHARED_VYB: OK` — shared-expert output 1.5e-11 relative.

Still missing at this site: the 109 routed experts these 22 tokens touch (EXL3-packed,
4 bpw). That needs the P1 `exl3` dequant pipeline lifted into a reusable Vyb function
(today it lives inside `native/exl3/exl3_driver.vyb` for a single fixture tensor), after
which the expert FFN can reuse the same gemm calls as the shared expert.

## Layer 3's DSA attention in Vyb — `make glm3` → L3_ATTN_VYB_MATCH: OK

The first Vyb code for the sparse-attention path (11 of the 45 layers):
`native/host/glm_l3_attn_fwd.vyb` runs q_a→q_b, the kv_a latent, the per-head absorbed
keys/values out of `kv_b`, the **gpool indexer** (learned pooling over 4-token pools with
the per-slot bias, per-head score weights, top-k pools + the visible tail), the **masked
MLA attention** and `o_proj` — all on real layer-3 weights.

Input `h_a` is the layer-3 input chained through layers 0-2 in numpy (dumped as scaled
integers; Vyb has no f64 reinterpret), so this gate measures the DSA implementation
itself. Results against the numpy gold:

| stage | rel (sum / sq / max) |
|---|---|
| ha (input round-trip) | 1.5e-11 |
| qres (q_a projection) | 2.4e-11 |
| qb (q_a norm + q_b) | 4.1e-12 |
| ckv (kv_a latent) | 1.6e-12 |
| ao (attention, pre-o_proj) | 1.2e-11 |
| **attn (DSA out)** | **2.7e-12 / 3.4e-12 / 1.9e-12** |
| indexer selections | **22/22 queries identical token sets** |

Two bugs found here: the q_a/kv_a latents each have their OWN RMSNorm weight (passing an
empty weight vector silently computes an unweighted norm — qb and ckv were 40-200% off
while qres matched), and `var` is a reserved word in Vyb.

Cost: 143 s wall for the 22-token batch — dominated by the host per-head attention loops
(2 billion-element naive form would never finish; materialising the probabilities per
(query, head) brings it to ~100 M ops). This path wants the kernel next.

Remaining for a full layer: the 288-expert MoE (EXL3 + `moe` kernels) at layer 3's ffn
site, then the 45-layer pass, TP=2 and the endpoint.

## Layer 0 COMPLETE in Vyb — `make glm0` → LAYER0_VYB_MATCH: OK (all stages)

`native/host/glm_layer0_fwd.vyb` runs the entire GLM-5.3 layer 0 inside Vyb on the real
checkpoint weights: 4-wide hyper-connection mix → input_layernorm → **KDA attention**
(conv+silu on q/k/v, sigmoid beta, log-decay gate, delta-rule recurrence, gated RMSNorm,
o_proj) → hc apply → ffn hc mix → post-attn layernorm → dense SwiGLU MLP → hc apply.
22 real tokens, fp64, 27 s wall on the 3090 (weight dequant + 14 GEMMs + the host
delta-rule sweep).

Per-stage agreement with the numpy gold (`native/glm53/gold_layer0.py`):

| stage | rel (sum / sq / max) |
|---|---|
| x (embeddings) | 0 / 0 / 0 (bf16→f64 exact) |
| hc mix (mv, collapsed, post) | 7e-15 / 2.6e-12 / 3.3e-12 |
| h (input_layernorm) | 1.8e-12 |
| q/k/v projections (qpre) | 1.2e-10 |
| conv+silu (qc) | 5.4e-12 |
| l2-norm + scale (qn/kn) | 7e-13 |
| beta, fraw, zraw | 2e-12, 8e-13, 2e-12 |
| log-decay gate (gdec) | 8.2e-12 |
| **attn (KDA out)** | **3.3e-11 / 9.6e-12 / 1.3e-11** |
| h2 (post-attn norm) | 3.2e-11 |
| **mlp (SwiGLU out)** | **5.9e-11 / 2.3e-12 / 2.3e-11** |
| **streams2 (layer out)** | **5.2e-11 / 1.4e-11 / 0** |

Three real bugs this surfaced, all now fixed and commented in place:
- **conv1d weights are bf16, dt_bias/A_log are f32** in this checkpoint; decoding dt_bias
  as bf16 produced NaN gates. (`_keep_in_fp32_modules_strict` is an HF *load-time* list,
  not the stored dtype — check the shard header, not the config.)
- **dt_bias has no token axis**: indexing it by the flattened `(token, head, channel)`
  position read past the array for every token but the first.
- **A push-built vector follows push order**: iterating channels outside tokens in
  `conv_silu` silently produced a transposed result — and because the gate compared only
  sum/abs/sq/max, the digests still matched. Hence the extra per-value dumps and the
  permuted-vs-wrong lesson: digests alone cannot catch a layout error.

## Loader performance: three substrate hot spots fixed (measured, not guessed)

The first version of this path was ~300x too slow to load a 163.6 GiB checkpoint. Each
fix below was found by timing the real work, not by reading the code:

| what | before | after | why |
|---|---|---|---|
| `json_parse` of the 15.5 MB weight_map | **134,489 ms** | **439 ms** | `parse_string_raw` appended the body **one character at a time** (`out = out + src.substring(i, i+1)`): an allocation plus a growing copy per char. Fast path now scans for the closing quote with `char_at` and takes ONE `substring`; the escape path still runs per-char, but JSON strings rarely take it. Synthetic scaling after: 44 MB/s (32k entries in 66 ms). |
| `st_open` (one shard header) | **5,534 ms** (1185 tensors) | **22 ms** | it called `st_arr_ints()`/`jgetstr()` three times per tensor, and those take the `JsonDoc` **by value** — Vyb structs are value types, so each call deep-copied all ~9k nodes (0.94 ms). The reads are inlined through the doc's own methods. |
| `gx_index` (name -> shard for 150k names) | **159,000 ms** | **555 ms** | `JsonDoc.get(id, key)` is a *linear* scan over an object's children, so one keyed lookup per name is O(n^2). Children are now walked by index (`elem` + `ckey_at`), which is O(n). |

Whole-checkpoint plan (`make gxplan`): **2.3 s** for 150,226 tensors / 163.6 GiB
(34.8 s before the two Vyb runtime fixes below), digest-identical to the independent
oracle (`GX_PLAN_MATCH: OK`). The same fix sped the
older gates up too: `make gx gxload` went from ~90 s to 3.2 s.

## Two Vyb runtime defects this path hit — FIXED in Vyb (`3347863c`)

Both were found by this loader, fixed in the Vyb repo, and are now covered by Vyb tests.

1. **Loop-body temporaries leaked stack (SIGSEGV).** Codegen emitted `alloca` at the point
   of use, so a temporary inside a loop body took a fresh stack slot every iteration and
   LLVM only reclaims stack at function exit. A 130k-iteration loop copying a 5-field
   struct out of a `Vec` overflowed the default 8 MB stack and died in
   `__vyb_string_release` (crash point varied run to run, backtrace useless). Fixed by
   hoisting every constant-size alloca into the entry block (`hoistStaticAllocas`);
   reproducer `repro/loop_stack_leak.vyb`, test `test/memory/loop_stack_no_leak.vyb`.
2. **String registry live-count drift (progressive slowdown, then SIGSEGV).** Reusing a
   tombstone slot decremented tombstones but not the live count, while `release()` always
   decrements it: under churn the live count drifted down, the table stopped growing and
   linear-probe clusters grew without bound. Measured here: parsing all 83 shard headers
   took **1.4 s with an empty heap and 127 s with 150k live Strings**; a rehash now
   re-derives the counter from the table. Same workload after the fix: **1.09 s**
   (`native/glm53/perf/st_open_all_shards.vyb` reproduces the measurement).

Effect on this repo's numbers: `make gxplan` (150,226 tensors, 163.6 GiB) went from
**34.8 s to 2.3 s**, and the older `make gx gxload` pair stays at ~3 s. The consequence for
the engine also stands: still resolve per shard / per layer and never hold a global
name→tensor map, because a large live heap is expensive even when it is correct.

Also worth knowing: Vyb's `^` is XOR, not power (`2 ^ 8` is 10) — a header-length decode
using it silently produced garbage. `JsonDoc` has no iterator; `keys()` materialises a
Vec of every key (150k Strings) — prefer `nchildren`/`ckey_at`/`elem` in loops.

## Layer 3 complete (Gate C) — the full layer matches HF

`verify_layer3_attn.py` composes the ENTIRE layer 3 in numpy on the chained input
from layers 0-2 — hc mix → input_layernorm → DSA attention (absorbed MLA: q_b from
the q_a latent, `kv_b [32768,512]` split per head into W_UK [256,512] / W_UV
[256,512], softmax weight exactly 1 at S=1) → hc apply → hc mix → post-norm → routed
MoE (selected experts dequantized from EXL3) → hc apply — and compares every stage
against HF on identical weights: attention out, h2, MoE out each 3-7e-3 (the bf16
side's own rounding), routing **sets** identical (lane order varies with accumulation
order — the Gate B boundary class), full-layer output streams 9.9e-3.
**LAYER3_FULL_VERIFY: OK** (commit `7ad3172`).

Loader lesson (cost a day, keep it): the hc tensors (`attn_hc.fn/base/scale`) are
silently skipped by a parameters-only loading loop — HF then runs the hc mix on
random init, which looks like a 19-46% "attention divergence" and varies run to run.
**Any loader must iterate `named_buffers()` as well as `named_parameters()`.**

With Gates A/B/C the composition is proven end to end: every architecture piece is
gated, chained dense layers match, and one complete DSA+MoE layer matches HF. The
45-layer forward follows by induction (same per-layer code, per-layer weights).
Remaining for a running model: the Vyb loader orchestration + KV caches (port of
loader.py/kv_cache.py per the Vyb-only rule), the DSA/MoE Vyb kernels, TP=2, server.

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
