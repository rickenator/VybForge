# HANDOFF — Porting GLM-5.3-Flash to a Vyb-native engine

**Written 2026-10-30.** Read this first; it is the entry point for a fresh session.

The goal: GLM-5.3-Flash (the EXL3 4 bpw checkpoint the TensorFold rig currently serves on
the dual DGX Sparks) running as a **Vyb-native inference engine** in
`VybForge/native/glm53` — no Python in the runtime path. Every stage is gated against a
numpy/HF oracle derived from the real checkpoint before it is allowed to stand.

Priority per Rick: **speed and fidelity to the model.** The Sparks are the deployment
target; the 3090 (godzilla) is the reference machine while the Sparks cannot read the
weights (see §8).

---

## 1. Where things stand

| thing | state |
|---|---|
| loader (83 shards, 164 GB, 150,226 tensors) | gated, 2.3 s for the whole plan |
| layer 0 (KDA attention + dense MLP) | gated, every stage ≤6e-11 |
| layer 3 (DSA attention + router + MoE + shared + hc) | **L3_FULL_VYB_MATCH: OK**, all six stages |
| full 45-layer prefill | **runs to completion**, `PREFILL_OK layers=45`, **914 s (15:14), peak RSS 7.1 GB** — was 5697 s / 53.6 GB; see §6 |
| KV cache / decode | **step decode in Vyb: DEC_STEP_VYB OK** — `glm_decode_step.vyb` runs layer 0 one token at a time with the KDA recurrence and conv rings carried, matching the oracle's two teacher-forced steps at worst 1.9e-12 (`make -f native/Makefile decstep`). **Layer 3 too: DEC_L3_STEP_VYB OK** (`glm_decode_l3.vyb`, `make decl3step`) — ckv [T,512] / indexer k+gsc [T,128] caches carried and appended per step, decode-layout pool+topk, masked MLA over cached ckv, router + top-8 EXL3 experts; attn elementwise 2.6e-11, mlp/streams2 digest ≤6.7e-6 vs the 2e-4 L3_FULL convention (the elementwise residue is EXL3 dequant noise). Next: all 45 layers chained with the caches, then the MTP head |
| TP=2 across the Sparks | not started |
| endpoint | not started; `verify_full_forward.py` is still a stub |

Repos: Vyb `fd0bf4ee`, VybForge `081d6a5` (both pushed, worktrees clean).
Everything below is reproducible from those two commits.

---

## 2. Architecture — the facts that cost the most time to establish

45 decoder layers (**0..44**) plus **layer 45 = the MTP (multi-token-prediction) head**, which
is *not* a decoder layer: it carries `eh_proj` / `enorm` / `hnorm` and its own 288 experts.
Driving layer 45 through the decoder path fails with `PF_FAIL attention layer 45` — that is
correct behaviour, not a bug. The MTP head is for speculative decoding only.

    attention: KDA (gated delta-rule)  on n % 4 != 3   → 34 layers
               DSA (sparse/deepseek attn) on n % 4 == 3 → 11 layers (3,7,...,43)
    ffn:       dense SwiGLU MLP on layers 0-2 only
               288-expert top-8 MoE on layers 3-44
    every layer has two hyper-connection (hc) sites: `hc_attn_*` and `hc_ffn_*`

Model constants: H=4096, HEADS=64, HDIM=128, NOPE=256, VDIM=256, KV_LORA=512, Q_LORA=1536,
IDIM=128, IHEADS=32, KPOOL=4, indexer top-k=2048, INTER=12288, EINTER=2048 (MoE expert
hidden), 288 experts, top-8, routed_scaling_factor=2.5, EPS=1e-5, hc eps=1e-6,
sinkhorn_iters=20, 45 layers.

Checkpoint (godzilla): `/mnt/apex/rick/vybforge/glm53-checkpoint`
(`model.safetensors.index.json`, `model-000NN-of-00083.safetensors`, `.COMPLETE`).
Only the routed experts are EXL3-quantized (`quantization_config.scope =
glm53_routed_experts_only`, bits=4, codebook `mcg`); everything else is bf16/f32.

**The checkpoint is the authority, not exllamav3.** The indexer is HF's
`Glm5NextTextIndexer` (learned gpool, per-slot bias, no rope) — exllamav3's `QSAIndexer`
does not match the shards. See `native/glm53/dsa_ref.py`.

Vyb is a 3 KDA : 1 DSA pattern, so **the same driver code must dispatch per layer** — this is
where a naive "layers 0-2 KDA, 3-44 DSA" assumption burns a run.

---

## 3. Layout

    native/glm53/          python oracles (numpy fp64) + verifiers, one per stage
      loader.py kv_cache.py mla_ref.py dsa_ref.py layer0_ref.py gdn_ref.py moe_ref.py hc_ref.py
      gold_layer0.py gold_l3_attn.py gold_l3_moe.py gold_l3_hc.py gold_exl3_deq.py
      verify_layer0_vyb.py verify_l3_vyb.py verify_l3_moe_vyb.py verify_l3_tail_vyb.py
      verify_l3_full.py verify_l3_hc.py verify_exl3_load.py verify_l3_moe_all.py
      exp_env.py gx_ref.py gx_plan_ref.py README.md
    native/host/           the drivers (Vyb programs)
      glm_layer0_fwd.vyb   layer 0 end to end (KDA + dense MLP)
      glm_l3_attn_fwd.vyb  layer 3 attention site alone
      glm_l3_moe_fwd.vyb   router + shared expert
      glm_l3_moe_all.vyb   router + every routed expert + shared, all 22 tokens
      glm_l3_expert_fwd.vyb one token's routed experts
      glm_exl3_check.vyb   EXL3 dequant at real expert shape
      glm_l3_hc_check.vyb  the hc mix/apply in isolation (fast localiser, seconds)
      glm_l3_full.vyb      layer 3 start to finish, one process
      glm_l3_tail_fwd.vyb  hc_ffn mix -> norm -> MoE -> hc_apply
      glm_prefill.vyb      THE 45-layer forward (embeddings -> 45 layers -> streams)
    native/exl3/           EXL3 reference + map.bin (256-slot tile map) + gen_map_bin.py
    native/kernels/        Vyb kernels: exl3.vyb (deq/hgen/had1/had2), gemm.vyb, bf16.vyb, ...
    native/torchload/      safetensors.vyb (st_open, st_arr_ints)
    native/json/           json_parse.vyb (fast paths; do not regress them)

Makefile targets (from the repo root): `gx`, `gxload`, `gxplan`, `glm3`, `glm3moe`,
`glm3exp`, `glm3hc`, `glm3tail`, `glm3all`, `glm3x`, `glm3full`, `prefill`.

Invocation (every driver):

    VYB_STDLIB=~/Projects/Vyb/stdlib ~/Projects/Vyb/build/vyb \
      --module-path native/json --module-path native/torchload --module-path native/glm53 \
      --module-path native/kernels --module-path native/tensor \
      --module-path ~/Projects/Vyb/bindings/cuda --module-path ~/Projects/Vyb/stdlib/collections \
      native/host/<driver>.vyb

Python oracle env: `env -u PYTHONPATH ~/venvs/exl3/bin/python`. The conda
`conda-anaconda-tos` warning on every shell is harmless.

---

## 4. Gates — what passes, with the tolerances that were actually measured

| gate | target | result |
|---|---|---|
| `GX_INDEX_MATCH` | 150,226 names vs the python loader | OK |
| `GX_LOAD_MATCH` | tensor bytes, byte-exact | OK |
| `GX_PLAN_MATCH` | whole-checkpoint sha256 digest, 2.3 s | OK |
| `LAYER0_VYB_MATCH` | KDA + dense MLP, every stage | ≤6e-11 |
| `EXL3_LOAD_VYB` | expert gate_proj [4096,2048] vs P1 oracle | sq 8.4e-7, max 7.8e-5 |
| `L3_ATTN_VYB_MATCH` | DSA attention, 22 tokens | attn 2.7e-12, selections 22/22 |
| `L3_MOE_ROUTE_VYB` | top-8 sets | 22/22 identical |
| `L3_MOE_SHARED_VYB` | shared expert | 1.5e-11 |
| `L3_EXPERT_VYB` | one token's routed contribution | abs 2.3e-5, max 8.1e-5 |
| `L3_MOE_ALL_VYB` | full mlp [22,4096], 109 experts | abs 5.3e-6, max 1.1e-5 |
| `L3_HC_VYB` | hc mix/apply in isolation | comb 2.2e-8, streams1 2.1e-9 |
| `L3_TAIL_VYB` | hc_ffn -> MoE -> hc_apply | h2 7e-11, mlp 8.5e-6, streams2 1e-5 |
| `L3_FULL_VYB_MATCH` | layer 3 end to end | h_a 5.8e-11, attn 2.8e-11, streams1 2.1e-9, h2 3.4e-9, mlp 5.3e-6, streams2 3.3e-6 |

Notes on the tolerances: the EXL3 path carries fp16 butterfly noise, so `mlp`/`streams2` sit
at 1e-5..1e-6 and a 2e-4 gate is the honest bar. When comparing a *sum* of signed values that
nearly cancels, compare against the `abs` sum, not the sum's own magnitude — a 1e-4 gate on a
1/700-cancelling sum measures noise.

Prefill beyond layer 3 has **no oracle**. Layers 0-3 are gold-matched end to end, every site
is gated individually, and the 45-layer run's per-layer digests are finite and monotone in
magnitude — but "plausible" is the strongest claim for layers 4-44 today:

    max |stream| per layer, 45-layer run:
    0: 0.059  3: 2.65  10: 9.4  18: 24.1  29: 27.1  43: 47.8  44: 290

The 6× jump at layer 44 is the first thing to check once a reference exists past layer 3.

---

## 5. Runtime bugs found and fixed on the way (all in Vyb, all pushed)

These were blockers, not polish. Each has a repro or a test.

1. **`alloca` in loop bodies** (SIGSEGV in long loops). Codegen emitted temporaries at point
   of use; a 130k-iteration loop overflowed the 8 MB stack. Fixed by `hoistStaticAllocas`
   (`src/vre/llvm/cgen_util.cpp`), filtering on `isa<ConstantInt>(getArraySize())`.
2. **String registry live-count drift** (progressive slowdown). Reusing a tombstone slot did
   not increment the live count while `release()` always decremented → the table stopped
   growing, `keyed get`/parse went O(n²). 83 shard headers: 1.4 s empty vs 127 s with 150k
   live Strings. Fixed in `runtime/vyb_runtime.c` (rehash re-derives the live count).
3. **`read_all`/`read_at` over-registered their buffer** (the OOM killer). This is the big
   one. The buffer is handed back as a String *value* that the destination retains (+1) and
   releases; registering it with a hard-coded refcount of 1 on top left every destination at
   refs 2 with one release, so **each call leaked the whole buffer**: 200 reads of a 4 MB
   file grew RSS 74 MB → 891 MB. The prefill reads ~327 tensors per MoE layer (~1.3 GB) and
   died at layer 25. Fixed by `vyb_string_register_refs(p, refs)` with
   `__vyb_string_register` (refs 1, transfer producers) and
   `__vyb_string_register_borrowed` (refs 0, value hand-offs); the two file-read paths use
   the borrowed form. Fixtures: `repro/string_read_leak.vyb`, `repro/string_assign_leak.vyb`.
   **Do not** apply the borrowed form to the other 33 String-returning runtime helpers
   (net/tls/regex/exec/base64/ed25519/env): that is use-after-free on paths the codegen
   classifies as transfers — five tests catch it (chain, chain_signed, websocket, ed25519,
   nested-Vec write-back). They need per-site classification.
4. **`@noinline` was stripped at -O2** by the attribute sweep that exists to undo the
   Phase-6.4 trace default. Codegen now tags an explicit `@noinline` with
   `vyb-explicit-noinline`; the sweep skips it. `test/kernel/test_noinline_helper.vyb`.
5. **Vyb has no FFI across module boundaries.** An `extern "C"` block inside an imported
   module is invisible to that module's own functions (`Undefined identifier: cuMemFree_v2`).
   Modules must go through `bindings/cuda` (`cuda_mem_alloc`, `cuda_memcpy_h2d`,
   `cuda_launch`, `cuda_sync`, `cuda_mem_free`); `cuda_launch` takes ONE argument, so a
   kernel with several parameters needs a params buffer (as the exl3 kernels do) or
   `cuda_launch4i`, which only works from a driver.
6. **Vyb language traps** that cost real time: `fail` and `var` are reserved words; `^` is
   XOR, not power; structs are value types (by-value parameters deep-copy — this silently
   turned 9k-node JSON docs into 5.5 s per shard header); loop counters must be declared in
   the enclosing scope; `return <int>` inside a `Vec<Float>` function is a cast error (return
   a typed empty vector); `.to_string()` on an Int is fine but there is no bitcast from Int
   to String.

Driver bugs fixed (in VybForge): the hc comb softmax used a **global** max instead of a
per-row max, so rows far below the block peak underflowed to exp(0)=0 and the small entries
were wrong by 6× (streams1 off 1.8e-3); and `TOPK` collided — the attention site's indexer
top-k is 2048 while the router wants top-8, so defining `TOPK` once made the router select
288 experts per token (it looked exactly like "routing is chaotic under a 1e-10 h2 change",
ran the expert path hundreds of extra times and OOM-killed the driver). The router constant
is `ROUTE_TOPK`.

Two more driver-level root causes were found and fixed later, both measured: the per-element
`htod_vec`/`dtoh_vec` copy path in `native/tensor/tensor.vyb` (now `bulk_htod`/`bulk_dtoh`),
and `gx_resolve` calling `st_open` once per wanted name instead of once per shard. See §6 —
those two were 4.3x of wall time and 7.8x of peak RSS on the 45-layer prefill.

---

## 6. Performance — measured, and what it means

45-layer prefill, 22 tokens, 3090: **914 s (15:14), peak RSS 7.1 GB** after the fixes below.
The progression, each step verified value-identical at all 45 layers: 5697 s (as written) ->
1315 s (bulk DMA + shard-header cache) -> 908-914 s (deq scale/map bulk copies, `st_find` borrow,
index hoist). 6.2x total; the last three micro-fixes moved the layer-3 driver from 16.4 s to
13.5 s but are within noise at 45 layers, because the drift described at the end of this section
dominates at depth. Per-layer digests are identical to the pre-fix run at every layer
(max|stream| 0.059 / 2.653 / 9.405 / 24.151 / 27.072 / 47.781 / 290.367 on layers 0/3/10/18/29/43/44),
so both fixes are value-exact.

**The old profile blamed the fp64 gemm. It was the copy path.** Instrumented layer 3 (the
`T_*` prints in `native/host/glm_l3_full.vyb`), milliseconds:

| phase | before | after |
|---|---|---|
| indexer deq (4 bf16 tensors) | 12 | 6 |
| `mm_host` qidx / kraw / gsc / wproj | 522 / 328 / 330 / 322 | 2 / 1 / 1 / 1 |
| uk/uv build + K/V gemms ("pool_sel 62 s" in the old note) | 64 540 | 171 |
| poolk / index scores / selection | 1 / 0 / 0 | 1 / 0 / 0 |
| masked MLA | 18 | 18 |
| o_proj | 1596 | 16 |

Two root causes, both now fixed:

1. **`htod_vec`/`dtoh_vec` in `native/tensor/tensor.vyb` moved one 8-byte element per
   `cuMemcpy`.** ~4 us of driver overhead per element: a 90k-element activation upload cost
   330 ms and the 16.7M-element `kv_b` download cost ~60 s. `bulk_htod` / `bulk_dtoh` use
   `addr(v)` — a `Vec<Float>` is `{data ptr, size, cap}`, so the first word of the header is
   the element buffer — and move the whole buffer in one call. `bulk_htod` takes
   `their<Vec<Float> const>` and is called as `view(x)`: a by-value Vec parameter would
   deep-copy 134 MB before the copy even started. All 26 call sites in `glm_l3_full.vyb` and
   `glm_prefill.vyb` now use the bulk form.
2. **`gx_resolve` called `st_open` once per wanted NAME** (`native/glm53/gx.vyb`): 9 header
   parses (8 ms each) per expert resolve, and the `StIndex` structs accumulated — that was the
   "residual ~1.2 GB/layer". Each expert is resolved once per layer, 109 experts per MoE layer,
   so it cost ~8 s/layer and the growth that made late layers 2.5x slower than early ones.
   Now the last opened header is kept and reopened only when the shard changes: one 9-name
   resolve went 98 ms -> 18 ms and 100 resolves 9.85 s/1.18 GB -> 1.84 s/0.22 GB.

Result: the prefill's per-layer time is flat (~23 s/layer) instead of rising, and RSS peaks at
6.9 GB — small enough that a KV cache now fits on the 3090 box.

Per-expert profile inside a MoE layer, before these fixes (still useful for what remains):

    resolve (gx_resolve)      89 ms   -> ~10 ms (fixed: header reuse)
    deq (3 matrices)          93 ms      bulk DMA, fine
    gemms + host math        212 ms      mostly gone: mm_host was per-element DMA
    ────────────────────────────────
    ~380 ms/expert × 109 = ~42 s/layer of the ~210 s

Two more, same class, found by instrumenting the MoE layer the same way (layer 3, 109 routed
experts, milliseconds per layer):

| phase | before | after |
|---|---|---|
| resolve (`gx_resolve` per expert) | 2346 | 1418 at ONE layer / 2355-4700 per layer at depth — see 4. |
| deq pipeline (9 tensors) | 9013 | 6843 |
| the three gemms | 228 | 232 |
| gather / SwiGLU / scatter host math | 3 | 7 |

3. **The EXL3 map body and both fp16 scale vectors were uploaded 8 bytes at a time** inside
   `exl3_deq` — 1536 `cuMemcpyHtoD` calls per matrix, ~3.3 s per MoE layer. Both are Strings, so
   each is now ONE copy (`cuMemcpyHtoD_v2(sud, suh, nsuh)`); the lengths are hoisted out of the
   `freedom` block because `.len()` on an outer String is not readable inside one.
4. **Tried: `gx_resolve` once per layer for all 288 experts' 2592 names** (indexed with
   `i0 = e * 9`). It is 40% cheaper at a single layer (1418 ms vs 2346 ms) and gives identical
   digests, **but it does not hold at depth**: in the 12-layer prefill the resolve term grew
   ~1 s per layer (1.7 -> 9.5 s by layer 11, measured), and the full 45-layer run went from
   21:55 to 25:53. Per-expert resolution also drifts but only ~0.3 s/layer. String churn is the
   suspected mechanism (a 2592-entry `Vec<String>` per layer vs 109 short-lived 9-name ones);
   it is a compiler/runtime question, not a driver one. **Reverted in the prefill**
   (`glm_prefill.vyb`), kept in the single-layer gate driver (`glm_l3_full.vyb`) where it is a
   straight win. This is the one change here that was measured, looked good, and was undone.

5. **Two more, after the reverted experiment, both kept:**
   - `st_find` took `StIndex` BY VALUE, so every name lookup deep-copied the shard header
     (~1185 names + six Vecs). Resolving one expert's 9 tensors was 9 header copies. It now
     takes `their<StIndex const>` and is called as `view(ix)`: layer 3's resolve went
     1450 -> 801 ms. Call sites updated: `native/glm53/gx.vyb`, `native/whisper/host/forward_driver.vyb`.
   - The 15.5 MB index was re-read and re-parsed for every expert. `gx_resolve` now delegates to
     `gx_resolve_idx(index, ...)`, and the prefill's MoE loop reads the index once per layer:
     ~200-500 ms/layer, small but free.

   Neither fixes the **drift**: resolve still rises ~250 ms per layer (1866 ms at layer 3 ->
   4160 ms at layer 11 in the 12-layer test), while deq/mm stay flat and the digests stay
   identical. At this point the evidence is that String-heavy code slows down as the process
   ages (fewer strings created does not help, borrowed params do not help, one big resolve does
   not help) — i.e. the String registry / allocator, not the driver. That needs a Vyb-side
   experiment, and Vyb is the implementation agent's territory.

Finally, the deq itself: instrumented per stage, one matrix costs deq 1.0 ms, hgen 0.0 ms,
**had1 8.0 ms, had2 8.2 ms** — so the two Hadamard passes are 5.3 s of a ~8.7 s MoE layer, and
they are fp64. **That, not the gemm, is the next speed lever**: the gemm pair is 232 ms/layer
(0.7 ms per expert), so wiring `gemm32_p` into `mm_host` would buy ~2%, while an f32
deq/hadamard path is where the remaining time actually is. f32 also halves the 8-byte-per-element
writes on a bandwidth-starved Spark.

Layer-3 driver after all of this: 16.4 s -> 13.5 s, `L3_FULL_VYB_MATCH` unchanged at
streams2 abs 3.32e-06.

**The biggest remaining item is the growth, not the arithmetic.** Across the 45-layer run the
per-layer `gx_resolve` term drifts from 2.3 s (layer 3) to 12.2 s (layer 44) while deq/mm stay
flat, so at layer 44 resolve is ~half the layer. It is the same String-churn shape that made the
resolve-all variant lose; anything that touches how names/index/headers are handled per layer
should be measured over 12+ layers, never on one. A Vyb runtime question (String registry live
count / Vec<String> release) is the plausible root, and it is worth one focused experiment
before more driver-level tuning.

---

## 7. Memory budget traps (environment, not code)

- **The Hermes gateway cgroup caps `MemoryMax=4 GiB`.** Every long job launched from a
  session is OOM-killed at ~4 GB RSS regardless of the host's 125 GB. Symptom:
  `rc=137`, `Memory cgroup out of memory: Killed process … anon-rss:4159220kB`. Launch heavy
  runs with
  `systemd-run --user --scope -q -p MemoryMax=80G --unit=<name> bash -c '<cmd>'`.
- `malloc_trim(0)` once per layer is worth keeping (EXL3 trellis churn retains ~0.5-0.8 GB
  per layer in allocator arenas) but it is not a substitute for the leak fix.
- Residual growth after the leak fix is **gone**: the ~1.2 GB/layer was `gx_resolve` retaining a
  parsed `StIndex` per wanted name (§6). The 45-layer peak is now 6.9 GB, so a KV cache fits.

---

## 8. Where the weights live (and why the Sparks cannot run yet)

    godzilla : /mnt/apex/rick/vybforge/glm53-checkpoint   83 shards, 164 GB, readable
               /etc/exports is EMPTY — not NFS-exported
    lefty    : /root/.cache/huggingface/hub/models--neko-legends--GLM-5.3-Flash-Uncensored-EXL3/
               snapshots/07135ec0…/   ← what the TensorFold rig actually serves (prod.env)
               root-owned; rick cannot read it; rick's sudo is scoped to drop_caches
               the rig is live on :8000 and answers /v1/models with GLM-5.3-Flash
    righty   : /mnt/apex mounted but EMPTY, root fs 100% full (942 MB free)
    lefty    : /mnt/apex mounted but EMPTY, 23 GB free, 116 GB RAM held by the TF rig
    both     : CUDA 13 + nvcc, aarch64, GB10, `~/Projects/Vyb/build/vyb` exists

So: the model *is* on a Spark (root's HF cache) and the Spark is the deployment target, but
the engine cannot read those weights today. Unblock with either
`chown -R rick` on that snapshot (or a bind-mount), or export `/mnt/apex/rick` from godzilla
and mount it on righty (righty has RAM free; lefty does not). Once either is done, run the
same 4-layer gate on the Spark (`VYBFORGE_NLAYERS=4`) for a real number instead of the
extrapolation below.

Spark expectations (from spec, not measurement): GB10 LPDDR5X ≈ 273 GB/s vs the 3090's
936 GB/s, fp64 also weak, 121 GB usable. Bandwidth-bound work (dequant writing f64) should be
~3× slower per byte; capacity-bound work is far better. This is exactly why the f32
conversion and kernelized `pool_sel` matter more on the Sparks than on the 3090.

---

## 9. Next steps, in order

**Done since this was written:** the layer-3 profile (the 76 s / 62 s attribution was wrong — it
was the per-element copy path), the bulk-copy fix, the `gx_resolve` header-reuse fix, and the
45-layer re-run at 1315 s / 6.9 GB. Details in §6.

1. **Unblock the Spark weights** (§8), then run `VYBFORGE_NLAYERS=4` there and record the real
   per-phase numbers; re-plan speed work against those. This is now the cheapest real work
   available and it is blocked on access, not code.
2. **The per-layer `gx_resolve` growth** (§6): 2.3 s at layer 3 -> 12.2 s at layer 44, ~40% of a
   late layer, while deq and the gemms stay flat. Same String-churn family as the resolve-all
   regression, so it is one experiment in the Vyb runtime (String registry live count /
   `Vec<String>` release) rather than more driver tuning. Measure over 12+ layers.
3. **f32 in the EXL3 deq + Hadamard path** — after the growth this is the next lever (had1+had2
   = 5.3 s of a ~8.7 s MoE layer, all fp64; the gemms are 232 ms). The 3090's fp64 rate is 1/64
   of fp32 and GB10's is weak too, and f32 halves the writes. Gate: `L3_FULL_VYB_MATCH` must
   stay OK (tolerance for mlp/streams is 2e-4, so f32's ~1e-6 is not the risk — the risk is the
   `deq -> had1 -> had2 -> gemm` dtype plumbing, so add an f32 gemm at the same time).
4. **KV cache + one-token decode.** Prefill needs no cache (at S=22 each attention sees the
   whole prompt); generation does. `native/glm53/kv_cache.py` is the reference sketch and the
   `rdelta*` kernels already carry the KDA recurrence — what is missing is carrying the
   per-layer KDA state and the DSA latent/indexer keys across steps.

   **The step-decode oracle now exists** (`native/glm53/gold_decode_step.py`, layer 0 =
   KDA + dense MLP + both hc sites). It threads the two pieces of state the batched gold throws
   away: the KDA recurrence (64x128x128 per layer) and the k=4 depthwise conv, which needs the
   last 3 PRE-CONV vectors per channel for q, k and v (a ring buffer, not a scalar carry).
   Self-gates, both printed by the run:
     conv_step_vs_batch                        max_abs 0.0            (the ring order is the check)
     stepwise walk vs gated gold_l0_streams2    max_abs 1.388e-16      (fp64 noise)
   so the per-token path IS the gated layer-0 math. It also emits 2 teacher-forced decode steps
   (token ids 13, 10788 = the prompt's last two) with `gold_dec_step{0,1}_{streams2,attn,mlp}.npy`
   and `gold_decode.json`. Next unit: a Vyb driver (`native/host/glm_decode_step.vyb`) that
   reproduces those two arrays, then the same shape for layer 3 (DSA latent + indexer key cache),
   then all 45 layers.

   The oracle also dumps the exact state a decode step STARTS from, so that driver can run one
   step in isolation instead of re-deriving the 22-token prefill. Both `.npy` and raw
   little-endian f64 `.bin` (the `.bin` is what a Vyb driver can actually read — it has no .npy
   parser), in `native/out/`:
     gold_dec_state_kda_start.bin  [64,128,128]  8,388,608 B   carried recurrence
     gold_dec_ring_{q,k,v}.bin     [3,8192]        196,608 B   pre-conv history for the k=4 conv
     gold_dec_streams_prev.bin     [4,4096]        131,072 B   the last prefill token's streams2,
                                                               i.e. decode step 0's input
     gold_dec_step{0,1}_{streams2,attn,mlp}.npy                the two steps' expected outputs

   So the gate for the driver is: read those five files, run one step, match
   `gold_dec_step0_attn.npy` and `gold_dec_step0_streams2.npy`. The new logic it needs is exactly
   two state-threading changes to the already-gated layer-0 code: `delta_rule` must take and
   return `st` instead of zeroing it per call, and `conv_silu` must take the 3-row ring instead of
   rebuilding the conv from the prompt. Everything else (hc mix/apply, rmsnorm, the gemms, the
   gated norm, o_proj, the dense MLP) is unchanged and already gated at 1.87e-11.
5. **Extend the oracle past layer 3**: a gold for layer 4 (KDA + MoE — the one combination no
   gold covers) using the same numpy path as `gold_l3_moe.py`, then check the layer-44
   magnitude jump (still 6.08x, reproduced at 47.781 -> 290.367).
6. **TP=2 across lefty/righty**, then the OpenAI-compatible endpoint, then
   `verify_full_forward.py` stops being a stub.

---

## 10. Working agreements that keep this moving

- Gate every change: `L3_FULL_VYB_MATCH` (3 min), `L3_HC_VYB` (seconds) and `L3_TAIL_VYB`
  (4 min) catch most regressions; the full Vyb suite is 1357 tests, ~2.5 min, and
  `test/suite_count_check.vyb` fails if a documented suite count drifts.
- One oracle per claim. If a gate passes on synthetic data, it has not been shown to pass.
- Vyb's `main` ruleset blocks direct pushes: disable ruleset `22855500`
  (`gh api -X PUT repos/rickenator/Vyb/rulesets/22855500 --input off.json`), push, re-enable.
  `git add -A` has swept build directories into commits twice — `.gitignore` now covers
  `build-qt/` and `native/`; check `git show --stat` before pushing.
- Text-slice assembly of a big Vyb driver from another driver's functions is a trap: slice to
  a function's closing brace, not to the next function you remember, and expect a copied
  prelude to re-zero parameters and to arrive untyped.
- Reports want HOST + FULL path per artifact; apex NFS (`/mnt/apex/rick`) is the landing zone.
