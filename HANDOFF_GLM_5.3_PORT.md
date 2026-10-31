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
| full 45-layer prefill | **runs to completion**, `PREFILL_OK layers=45`, 5697 s, peak RSS 53.6 GB |
| KV cache / decode | not started |
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

---

## 6. Performance — measured, and what it means

45-layer prefill, 22 tokens, 3090: **5697 s (95 min), peak RSS 53.6 GB**. ~259 s/token.
DSA layers dominate (~210 s each); KDA layers are much cheaper.

Per-expert profile inside a MoE layer (layer 3):

    resolve (gx_resolve)      89 ms
    deq (3 matrices)          93 ms      ← bulk DMA, fine
    gemms + host math        212 ms
    ────────────────────────────────
    ~380 ms/expert × 109 = ~42 s/layer of the ~210 s

Attention site inside the same layer:

    proj_qk (q_a/q_b/kv_a)     0.7 s
    indexer_proj              76 s
    pool_sel (gpool + top-k)  62 s
    masked MLA **0.018 s**    ← the actual attention is free

So the GPU is idle and the time is in (a) the fp64 gemm kernel's memory access pattern and
(b) Vyb host loops. Two concrete observations to start from:

- Inside `indexer_proj` the individually timed calls (`deq_bf16` ×4 ≈ 8 ms, `mm_qidx`
  0.5 s, `mm_wproj` 0.3 s, `ln_rows` ~0) sum to ~1 s, yet the block reports **76 s**. The
  time is therefore in the untimed `mm_host(kraw)` / `mm_host(gsc)` and the `dtoh_vec`s, or
  in deferred GPU work being flushed at the first sync — i.e. **the fp64 gemm kernel is the
  suspect**. It is one-thread-per-output with strided f64 reads: for M=22, K=1536, N=4096 it
  moves ~1.1 GB uncoalesced. Do not trust intuition here — instrument those two calls first.
- `pool_sel` is 62 s of Vyb host loops over 2048 pool columns per query; that is
  pointer-chasing Vyb code, the worst shape for a 273 GB/s unified-memory part.

Levers, in the order the evidence supports:

1. **f32 instead of f64** through dequant + hadamard + gemm. Halves write traffic, and the
   3090's fp64 rate is 1/64 of fp32 (GB10's is weak too), so a tiled f32 gemm is the single
   biggest win on both machines. `native/kernels/gemm32_probe.vyb` exists as a starting point.
2. **Kernelize `pool_sel`** (pooling + top-k) — removes host loops entirely.
3. Cache dequantized experts across decode steps (only worth it once decode exists; 288
   experts × 67 MB in f64 does not fit a 3090, 9.6 GB in f32 does not fit either — so this
   is a TP=2 / Spark-memory decision, not a 3090 one).

---

## 7. Memory budget traps (environment, not code)

- **The Hermes gateway cgroup caps `MemoryMax=4 GiB`.** Every long job launched from a
  session is OOM-killed at ~4 GB RSS regardless of the host's 125 GB. Symptom:
  `rc=137`, `Memory cgroup out of memory: Killed process … anon-rss:4159220kB`. Launch heavy
  runs with
  `systemd-run --user --scope -q -p MemoryMax=80G --unit=<name> bash -c '<cmd>'`.
- `malloc_trim(0)` once per layer is worth keeping (EXL3 trellis churn retains ~0.5-0.8 GB
  per layer in allocator arenas) but it is not a substitute for the leak fix.
- Residual growth after the leak fix is ~0.6-1.2 GB/layer; 45 layers peak at 53.6 GB. Suspects:
  `gx_resolve`'s per-expert `st_open` and the `mm_host` download Vecs. It fits 80 GB but will
  not fit 80 GB plus a KV cache, so it is worth another pass before decode.

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

1. **Instrument `mm_host(kraw)` / `mm_host(gsc)` and the `dtoh_vec`s in `indexer_proj`** to
   pin the 76 s (suspicion: the fp64 gemm kernel). Then replace the gemm with a tiled f32
   kernel and re-run `make glm3full` — `L3_FULL_VYB_MATCH` must stay OK.
2. **Kernelize `pool_sel`** (62 s/layer): the gpool pooling and top-k over 2048 columns.
   Same gate must stay green.
3. **Unblock the Spark weights** (§8), then run `VYBFORGE_NLAYERS=4` there and record the
   real per-phase numbers; re-plan speed work against those.
4. **KV cache + one-token decode.** Prefill needs no cache (at S=22 each attention sees the
   whole prompt); generation does. `native/glm53/kv_cache.py` is the reference sketch.
5. **Extend the oracle past layer 3**: a gold for layer 4 (KDA + MoE — the one combination no
   gold covers) using the same numpy path as `gold_l3_moe.py`, then check the layer-44
   magnitude jump.
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
