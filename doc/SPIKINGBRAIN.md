# SpikingBrain in Vyb — mapped plan (spiking linear-attention LLM + fly-brain integration)

> Status: PLANNED (2026-10-01) — not started. Primary deliverable: SpikingBrain-7B
> inference in pure Vyb on the RTX 3090, plus a shared spike core adopted into VybFly's
> fly-brain simulation. Step order is checkpointed: every step has a gate, and the tree
> is green at each gate before the next starts.
>
> **Resequenced 2026-10-02 (Rick): substrate first.** A new **S0** phase (first subsection of
> section 4) builds the pieces two architectures both need — dtype, tensor/config contract,
> loaders, attention family — in *general* form, so SpikingBrain is the second instance that
> proves the substrate instead of a second copy of the Qwen3-shaped code. Scope, evidence, the
> port-vs-consume rule and the park-list: `doc/SUBSTRATE-SCOPE.md`.

Source: "SpikingBrain: Spiking Brain-inspired Large Models", arXiv 2509.05276
(accepted TMLR 2026), BICLab (CAS Institute of Automation) + HK PolyU / Beihang /
Zhongguancun Academy / MetaX / et al. Code: github.com/BICLab/SpikingBrain-7B.

---

## 1. What we are implementing (verified against paper + repo code)

### 1.1 SpikingBrain-7B architecture (from `configuration_gla_swa.py` + `gla_attention.py`)

- Base: Qwen2.5-7B. hidden 3584, 28 layers, 28 Q heads / 4 KV heads (GQA 7:1),
  head_dim 128, RoPE theta 1e6, SwiGLU FFN (intermediate 18944), vocab 152064,
  max_position_embeddings 131072 (the 4M-token claim extrapolates via the 1M RoPE
  base + long-context conversion training), bf16, untied embeddings (lm_head
  separate), model norm eps 1e-6. All values from the model's config.json.
- **1:1 inter-layer hybrid attention**: `attn_layers = range(1, 28, 2)` →
  GLA on odd layers (1,3,...,27), sliding-window softmax attention on even layers
  (0,2,...,26), window 4096. No full-softmax layers.
- **GLA (Gated Linear Attention)** per layer:
  - q/k/v/gate projections from the residual stream; k,v are 4x128 (KV heads only),
    repeated 7x for Q heads.
  - q and k pass through **ReLU** (keeps the attention map non-negative, per the
    paper's conversion requirement).
  - gate `gk`: low-rank (3584 -> 16 -> 512), then `g = logsigmoid(gk) / 16` per step.
  - recurrence: `S_t = diag(g_t) . S_{t-1} + k_t v_t^T`, `o_t = q_t S_t`,
    state `S` is [B, 28, 128, 128] (~1.8 MB per GLA layer; ~26 MB total).
  - output: per-head-dim RMSNorm (eps 1e-5) then o_proj (no bias).
  - prefill: chunked parallel scan (`chunk_gla`); decode: fused recurrent.
  - no short convolution in this checkpoint (default off).
- **SWA**: standard causal softmax within a 4096-token window, GQA, RoPE — a
  windowed version of the existing VybForge GQA layer kernel.
- Inference memory is (partially) constant in sequence length: GLA state +
  14 SWA layers x 4K window KV (~1.6 GB). This is what enables the long-context claims.

### 1.2 Spiking scheme (paper sec 3.3, repo `W8ASpike/Int2Spike/`)

- **Adaptive-threshold spiking** (single-step, straight-through gradients in training):
  `V_th(x) = mean(|x|) / k`, `s_INT = round(x / V_th(x))` — integer spike counts
  per channel, applied to the input activations of all linear projection layers.
  `k` is the firing-rate hyperparameter (their Appendix A.8 sweeps it).
- **Spike coding** (inference): expand `s_INT` into sparse spike trains and replace
  the dense matmul with event-driven accumulation:
  - binary `{0,1}`;
  - ternary `{−1,0,1}` (symmetric quantization; halves timesteps and firing rate);
  - bitwise (pure / bidirectional / two's complement): bit `t` of the count is one
    timestep, `y = V_th · Σ_t 2^t · W s_t`. 8-bit counts → up to 8 timesteps.
- **Measured stats (SpikingBrain-7B, bitwise-ternary)**: 94.1% of |s_INT| in [0,7],
  ~1% exceed 16; average 1.13 spikes fired per channel; 18.4% of channels fire nothing.
  Filling inactive bits within a 3-step window → **69.15% sparsity**.
- **W8ASpike**: symmetric INT8 weights + spiking activations, calibrated on 128 text
  samples (~1.5 h, single GPU, 15 GB). Benchmark drop ~1–3% (MMLU/CMMLU/commonsense).
- **Energy (paper's estimate, idealized async hardware — NOT a measured number)**:
  FP16 MAC 1.5 pJ, INT8 MAC 0.23 pJ, event-driven INT8 add 0.03 pJ →
  1.13 spikes x 0.03 pJ = 0.034 pJ/MAC = 97.7% below FP16 (43.48x), 85.2% below INT8.

### 1.3 Headline results and honest reading of the claims

- ">100x TTFT at 4M tokens" is **relative to quadratic-attention Transformer
  baselines** at 4M; absolute prefill on one card is still minutes. Our benchmark
  reports absolute numbers plus the ratio.
- "Matches open-source Transformer baselines with ~150B tokens of conversion CPT"
  (2% of a from-scratch budget) — training is out of scope here; we load released
  weights.
- The 97% energy figure is the paper's model of event-driven hardware, with
  explicit caveats (paper sec C.2). Our gate measures joules on the 3090 (two-track,
  VybFly sec-17 discipline) and reports the pJ/MAC estimate separately as a model.
- SpikingBrain-76B (hybrid LA+SWA+full-softmax, 128 sink tokens, 16-expert top-1 MoE
  + 1 shared, 7 dense layers at [1,2,3,5,7,9,11], gate tied to key `g = 1 − k`) and
  SpikingBrain 2.0 (2026-04) are deferred — see step 8.

---

## 2. Scope and decisions

1. **Inference only, 7B first.** No training, no MetaX. The conversion pipeline
   (QKV reuse, MoE upcycling alpha=0.98, CPT/SFT stages) is recorded above for
   reference only.
2. **No GGUF.** We do not want to own the GGUF format/toolchain. Weights are read
   **natively in Vyb** from their published formats:
   - **safetensors** (W8ASpike: 7 shards, ~29.8 GB — dtype map to be confirmed on
     download; safetensors = 8-byte header length + JSON tensor map + raw
     little-endian bytes, trivially ownable), and
   - **torch .bin** (V1-7B-base: 15 bf16 shards, ~14.1 GB; torch.save zip + a
     finite pickle opcode subset).
   Python is an **oracle only** (tensor dump for verification, benchmark reference
   model). Nothing Python runs in the production path.
   Fallback if the pickle subset proves gnarly: one-off Python exporter to a
   header-less flat binary (VybFly canonical-binary convention), Vyb reads raw
   arrays. Last resort, not the plan.
3. **VybFly (step 7) is first-class, not an afterthought** — the spike core is
   shared, and the fly-brain side reuses VybFly's DES engine, energy instrumentation,
   and ingest conventions (table in step 7).
4. **76B / SpikingBrain 2.0 = deferred note (step 8)**, after the 7B line is done.
5. Host: RTX 3090 (24 GB). bf16 7B (~14 GB) fits (native bf16 on sm_86);
   W8ASpike fits with room for the long-context benchmark.
6. **Substrate first (2026-10-02).** The pieces both architectures need are built in general form
   in S0 and consumed by the later steps; nothing gets built "just for this model" when the general
   form is the same work. Bounded deliberately — an S0 item must be **forced by a model in the
   queue**, and each lands with its oracle + gate plus proof that the existing Qwen3 gates stay
   green (generality demonstrated by not regressing instance #1). Park-list and reasoning:
   `doc/SUBSTRATE-SCOPE.md`.

---

## 3. Reuse of existing Vyb assets

| Asset (verified) | From | Used for |
|---|---|---|
| GEMM / RMSNorm / vmath kernels + `tensor::` wrapper | VybForge `native/kernels`, `native/tensor` | all projections, GLA state updates, SWA |
| GQA causal-softmax layer kernel (LAYER_VERIFY ~5e-5) | VybForge `native/kernels/layer.vyb` | generalised in **S0.4** into an attention family (the window becomes a parameter); its Qwen3 gate must stay green through that change |
| RoPE kernel | VybForge `native/kernels/qwen3.vyb` | both attention types (theta 1e6) |
| Multi-layer stack + greedy decode + stochastic sampler | VybForge `native/llm`, `native/sampler` | the 7B model driver |
| Qwen BPE tokenizer + detokenizer (stdlib/vllm) | VybForge / stdlib | identical vocab path (Qwen2.5 tokenizer ships with the model) |
| JSON value parser (G-json) | VybForge `native/json` | safetensors header parsing |
| CUDA DES engine (calendar queue, SoA state, spike-for-spike validated, 6.6–7M events/s) | VybFly `src/vyb/des.vyb` | step-7 fly-brain event execution; design template for step-6 spike-train kernel |
| CUDA kernel lessons (short descriptors, micro-units, zero'd buffers, 8-byte readback, st widths) | VybFly `docs/VYB-PORT.md` | all new NVPTX kernels |
| Energy instrumentation (RAPL pattern, J/bio-s, uJ/spike; M4) | VybFly `results/phase3`, M4 | step-6 two-track energy gate |
| Python-oracle / Vyb-production / probe-gate convention | VybFly `docs/VYB-PORT.md`, VybForge docs | every step's verification |

**Oracle policy (2026-10-02, Rick).** The Python oracles are **kept, not converted**.
They are the independent second implementation that catches our defects, and they
may themselves still be wrong — which is exactly why they stay independent of the
Vyb code they check. Hold period: until the model sweep (this doc's steps plus the
fleet rigs) is done. When a step later replaces an oracle with a Vyb implementation,
**leave a comment in the replacement naming the oracle it replaced and what it
asserted**, so the retired oracle's expectations stay traceable. Full statement:
`doc/PYTHON-CLEANUP.md` → "Oracle policy".

---

## 4. Steps (each gate must pass before the next starts)

### S0 — substrate (added 2026-10-02; precedes S1, blocks S2 and S4)

Why: every requirement SpikingBrain adds lands on a *general* gap rather than a model-specific one, so
the same work written generally serves this model, the previous one, and the next. Evidence, the
port-vs-consume rule and the park-list: `doc/SUBSTRATE-SCOPE.md`. This is the work order.

**S0.1 — dtypes: bf16 + f16 storage, fp32 accumulation. LANDED 2026-10-02.**
`native/dtype/dtype.vyb` is now the single place that names a dtype (container spelling → id,
width, and the storage-vs-accumulate policy): a half is *stored* f16/bf16, read through the
device intrinsic `ld_f16`/`ld_bf16` (exact widening), accumulated in Vyb `Float`, and written
back through `st_f16`/`st_bf16` only at an explicit conversion point. The device intrinsics
already existed (Vyb#203, `doc/CUDA.md`); what S0.1 adds is the policy, the single naming
point, and the proof.
**Gate:** `native/legit/run_dtype_gate.sh`. `native/kernels/dtcvt.vyb` converts on the GPU and
compares every element **on the device** against expectations from numpy/torch
(`native/dtype/dtcvt_ref.py`), so the corpora can be millions of values: all 65,536 f16 and
bf16 encodings widened, 4,194,304 real values widened from two real checkpoints, and
2,621,440 structured + 2,000,000 random f32 narrowed to f16 (every rounding branch of f16's
10-bit mantissa, both signs, all 256 exponents). Result: widening byte-exact everywhere; f16
narrowing has **0 value differences** against torch. NaN payload differences are reported
separately rather than failed — IEEE 754 leaves payload propagation on a conversion
implementation-defined — but a NaN that comes back as a *number* is a failure, which is what
catches the bug below. The dtype module's table is also checked against the container
readers' own tables so the two cannot drift.
**Found by this gate:** `st_bf16` truncated instead of rounding — ~50% of values were one ulp
off torch and a NaN could silently become Inf (`0x7f807fff` → `0x7f80`). Filed as **Vyb#441**
and fixed in **rickenator/Vyb#443** (`cgen_expr_kernel.cpp`): round-to-nearest-even
(`+ 0x7FFF + lsb` on the f32 pattern) plus an Inf/NaN guard, since the rounding constant
carries out of the exponent — Inf passes through, a NaN gets the canonical quiet bit. All
eight conversions gate now (0 value differences, was 6 of 8). The kernel Makefile rule depends
on `$(VYB)` for the same reason the gate rebuilds from the current toolchain: a kernel is an
artifact of the *compiler*, and a stale `.ptx` would have let the gate certify the previous
toolchain. Importing `native/tensor` also made a file's own `extern "C"` declarations
unresolvable (**Vyb#442**) — **fixed 2026-10-02** in Vyb `2f1d5477` (`a module resolves its own
extern "C" declarations after importing a binding`, + fixture conventions `e05cc7f1`), so the
workaround is no longer needed. Verified from this tree, not taken on report: a file that does
`import tensor` and declares its own `extern "C"` now JIT-runs (`TENSOR_REPRO_OK`, with
`--module-path native/tensor --module-path ~/Projects/Vyb/bindings/cuda` — the binding lives in the
compiler tree at `bindings/cuda/`, not under `native/tensor/`). The `native/tensor` module can
therefore be imported by a driver alongside its own externs; the drivers still hand-wire for now,
and switching them over is a simplification available, not a requirement.

**S0.2 — tensor core + model config contract (audit G1). PLAN 2026-10-02.**
Host-side tensor layer (shape/strides/dtype/alloc, broadcast) plus a **config contract**: model dims
(layers, hidden, heads, kv heads, head_dim, ffn, vocab, rope theta, eps, tied/untied) read from the
model's own config instead of literals. Need is measured, not assumed: Qwen3-4B's values are
hardcoded across the tree — hidden 2560 in **25 files**, 36 layers in **34**, ffn 9728 in **24**,
vocab 151936 in **13** — so a second architecture otherwise edits those files or forks them.
**Gate:** re-express ONE existing Qwen3 driver through the config contract + tensor layer and keep its
gate green. That re-expression is the generality proof and the guard against a substrate designed for
imagined needs; no Qwen3 gate may regress.

Mapped steps (each one gated, in order):

- **S0.2a — config contract. LANDED 2026-10-02** (`native/config/model_config.vyb`, `mc_probe.vyb`;
  gate `native/legit/run_config_gate.sh`). One `ModelConfig` (arch, layers, hidden, heads, kv heads,
  head_dim, ffn, vocab, ctx, rope theta, rms eps, tied + derived `nq`/`nkv`) read from **either** a
  **GGUF**'s own metadata (the `<arch>.` namespace comes from `general.architecture`; `vocab` falls
  back to `token_embd.weight` when `<arch>.vocab_size` is absent; `tied` is decided by the tensor
  table — no `output.weight` means tied) or an **HF `config.json`** (via `native/json`, with the
  aliases in use: `rms_norm_eps` | `norm_eps`, `num_hidden_layers` | `n_layer`, no `head_dim` at
  all). Every field records its provenance — `defaulted` (the source did not state it), `derived`
  (computed from another fact in the same file), `notes` (the file's two sources disagree).
  **Gate green** against: llama.cpp's own `gguf-dump` (**18 fields**, metadata *and* tensor table),
  the python `gguf` package, `transformers.AutoConfig` on a **second real model**
  (SpikingBrain-7B, 13 fields), and the rope_theta → invfreq table the kernels actually consume
  (`max|diff| = 0`). Discrimination cases all fire: a mutated `hidden_size` is reported as mutated,
  a removed `num_key_value_heads` triggers the labelled fallback, and a truncated GGUF is refused
  (`check=64`, `notes=truncated in metadata`) instead of looking valid.
  Two things this already caught: `head_dim` is **not** `hidden/heads` (Qwen3-4B is 2560/32 with
  head_dim 128, so the naive derivation gives 80 and poisons every downstream size — it comes from
  `attention.key_length` / the attention tensor instead), and GGUF stores each 2-D weight in the
  layout its kernel consumes (`attn_output` is `[nq, hidden]` but `attn_k` is `[hidden, nkv]`).
- **S0.2b — tensor core. LANDED 2026-10-02** (`native/tensor/core.vyb` + `dt_width` in
  `native/dtype`; gate `native/legit/run_tensor_gate.sh`). Shape, element strides, numel/byte size,
  contiguity, strided offsets and numpy's broadcast rules (right-aligned, stride-0 expansion,
  incompatible shapes **refused**). Deliberately **pure — no CUDA and no allocation**, for two
  reasons: importing `native/tensor/tensor.vyb` dragged `cuda_binding` into the importer and made the
  importer's own `extern "C"` decls unresolvable (Vyb#442 — **now fixed** in Vyb `2f1d5477`, so this
  reason has expired), and because a shape library must be numpy-checkable on the CPU (this reason
  stands).
  **Gate green: 586 cases vs numpy** (strides from real arrays, offsets from `np.ravel_multi_index`,
  contiguity from `.flags["C_CONTIGUOUS"]` on real views, broadcast from `np.broadcast_shapes` and
  `np.broadcast_to(...).strides`), including 24 refusal and 29 non-contiguous cases. The table is
  *required* to contain refusals and non-contiguous cases, and the checker is fed a perturbed
  expectation to prove it reports a mismatch. It earned that: two real bugs in the core were found
  and fixed (a wrong `rank<=1 ⇒ contiguous` shortcut for rank-1 views, and a broadcast stride that
  must be 0 on any source axis of length 1 even when the target axis matches).
- **S0.2c — re-express `native/host/model_driver.vyb` through both. LANDED 2026-10-02.** The dim
  literals (`D=2560; H=32; KVH=8; HD=128; FF=9728` + `VOCAB=151936` + `MAXL=36` + `EPS=0.000001`)
  are gone: the driver loads `mc_load(MODEL)` and refuses to run on a config that does not check
  (`CFG_FAIL`), prints `CFG_SRC`/`CFG_DERIVED`/`CFG_DEFAULTED` and its `DIMS` line, and takes every
  buffer size from the tensor core — `tt_nbytes(SD|SQ|SK|SF|SV|D1|S1, dt_f64())` for activations and
  `dt_bytes_for(numel, dt_f64())` for the weight/embedding buffers, replacing 39 hand-written
  `S * D * 8`-style expressions. Gate `make prefill` (numpy `prefill_ref.py` vs the driver,
  `verify_prefill.py`, TOL 2e-3): **`PREFILL_HIDDEN_MATCH: OK`, bad=0, maxrel 3.4e-4, top1 MATCH.**
  `EPS` now comes from the model (9.99999997e-07, the f32 image of 1e-6) rather than a literal.
  **This step uncovered a real pre-existing bug, which it also fixes.** `make prefill` had been RED
  since **2026-09-13**: the 2D-weight orientation fix (`7adc585`) updated the numpy ``read_weight``
  and `decode_driver.vyb` but **never touched `model_driver.vyb`**, whose `load_quant` hardcoded the
  dequant kernel's in-dim argument to `0` (no transpose) — so for three weeks this gate compared a
  corrected reference against a driver with a hidden whole-matrix transpose. The driver's own
  embedding always agreed with the reference (5e-8), which is what localized the divergence to the
  weight path. Mirroring the fix (`load_quant`/`stage_one` take `inz`; Wq/Wk/Wv→D, Wo→NQ, Wg/Wu→D,
  Wd→FF, norms→0) takes the gate from `maxrel 5.4e3 / top1 MISMATCH` to `maxrel 3.4e-4 / MATCH`.
  Two consequences to be aware of: (i) the pre-fix driver's frozen gold `[31784, 31784]` in
  `native/tools/verify_chat_real.py` was captured from that wrong forward — its token does not appear
  in llama.cpp's top 10 at either position — so it was REPLACED: `verify_chat_real.py` now compares the
  driver's per-position top1 against a numpy reference regenerated on every run
  (`native/gguf/prompt_ref.py` + `native/tools/emit_prompt_ids.py`), on the principle the Makefile
  states at that target — "a committed gold file must never be the thing that is trusted, because a
  stale gold and a buggy forward can agree and leave the gate green". The consequence to keep in mind
  is that `make chat-real` is now a GPU-vs-numpy comparison and so proves self-consistency only; the
  INDEPENDENT check for the same prompt is S0.4 below. (ii) at the arbitrary ids this gate uses
  (`[0,1]`) the next-token distribution is nearly flat (p≈0.04, four tokens within 0.04 logprob), so
  a top1-argmax comparison is a coin flip: the *hidden* comparison is the meaningful one, and the
  real-prompt gate (`chat-prompt`, "The capital of France is" → ` Paris`) is the semantic one.
- **S0.2d — battery step + the no-regression sweep. LANDED 2026-10-02.** `run_config_gate.sh`,
  `run_tensor_gate.sh` and the S0.2c `make prefill` step are in `run_phase2_battery.sh`; the battery
  is green.

  **The five-driver orientation sweep — DONE 2026-10-02.** Every remaining site that handed the
  dequant kernel a literal `0` in-dim now passes the tensor's own in-dim (`Wq/Wk/Wv/Wg/Wu → D`,
  `Wo → NQ`, `Wd → FF`, norms → 0), each with its own gate re-run:

  | driver | helper | fix | gate result |
  |---|---|---|---|
  | `native/host/model_driver.vyb` | `load_quant` | per-weight in-dim | `make prefill` green (maxrel 3.4e-04) |
  | `native/host/layer0_driver.vyb` | `load_quant` | 7 call sites | `make layer0` green — measured red→green: **maxrel 1.03e+03 → 4.86e-06**, bad 5115/5120 → 0 |
  | `native/host/kv_driver.vyb` | `dequant_gpu` | 14 call sites | `make decode-kv` green (`DECODE_REAL_MATCH: OK`) |
  | `native/host/wcheck_driver.vyb` | `dequant_gpu` | 6 call sites | **had no gate at all** — added `verify_wcheck.py` + `make wcheck`; green, all 5 tensors within 2e-6 of numpy |
  | `native/train/kvctx.vyb` | `deq_w` | 7 call sites | **OK** — see (iii): the gate was blind to the rope and both KV drivers embedded a stale id file |
  | `native/train/kvrespfwd.vyb` | `deq_w` | 14 call sites | **OK** — corr **1.000000**, `|g| = |r| = 6.92e+03`, `max|g-r| = 5.55e-04` (see (ii)) |

  Two more real bugs surfaced while gating these, both pre-existing and neither a regression from the
  in-dim change:
  (i) `wcheck_driver` dequantized **layer 0's `attn_v` with the Q4 kernel while the tensor is type 14
  (Q6_K)** — its own label printed `ty14` next to a `q4fn` launch — yielding values like `1.3e+06`.
  It also compared its numbers to the reference **by eye**: there was no verifier, and `wcheck_ref.py`
  read its numpy loader from a hardcoded `/home/rick/Projects/VybAIConf`, a tree this repo no longer
  lives in.
  (ii) `kvrespfwd` died at the first response-forward dequant with **CUDA 700 (illegal address)**,
  and the cause was not in that call at all (VybForge#17): the driver reads
  `native/out/kvresp_ids.bin` and `kvresp_labels.bin`, and **nothing in the tree writes them any
  more** — the producer was Python, and the Phase-1 cleanup removed it while the Vyb driver kept
  reading its outputs. `read_bin` on a missing path returns a short String, `download` copies the
  requested byte count out of it anyway and returns 0 (the sibling ctx-ids read *is* length-checked;
  this one was not), so `RIDS` held garbage, the response-embed kernel indexed the embedding table
  with a garbage token id, and the resulting illegal address surfaced — asynchronously — at the next
  *checked* API call. Ruled out one run at a time: a probe of the identical call placed earlier in
  the process returns 0; all 26 response allocations check out; all nine unchecked memcpys were made
  to report and return 0; both unchecked launches between the loops now report and return 0; skipping
  the emb dump changes nothing; nothing is ever freed and the card was empty (131 MiB). The reported
  call's arguments were sane throughout (`nl` = `attn_q`'s [4096,2560], `inz` = D, valid src/dst), and
  behavior is byte-identical with and without the in-dim fix. `kvresp_ref.py` now writes both inputs
  and the driver length-guards both reads, so a missing input reports `RESP_IDS_MISSING` rather than a
  CUDA 700 fifty lines from its cause. With the inputs restored, the driver runs to completion
  (`KVRESP masked CE loss = 8.28567`) and the gate returns its first real verdict: `FAIL`, with the
  in-dim fix worth corr **-0.003884 → 0.775046** (|g| = 7.88e+03 vs |r| = 6.92e+03). This driver's
  layout is therefore proven right as well, and its residual — corr 0.775 rather than 0.995+, with a
  14% norm gap and a diffuse maxrel — is the same class of non-layout cause as (iii). Note that an
  earlier A/B of this driver compared the fixed file against itself (the in-dim change was already
  committed, so stashing the working tree reverted only the new guards) and printed bit-identical
  numbers; the pre-change revision has to be checked out by SHA, which is what produced the figures
  above.
  (iii) `kvctx` is red for a reason that is **not** orientation: the fix moved it from
  **corr −0.67…0.20 to 0.995…0.999** (a wrong layout cannot correlate at 0.999), but a diffuse
  10–30% error remains across all 64 dumped values — not a rope-pair or transpose signature. The
  driver's 9 context ids are byte-identical to the reference's, so the inputs agree; the divergence is
  inside the build (LoRA fusing or rope application) and belongs to the interviewer-runtime workstream
  rather than this sweep.

  (A missing `inz` parameter is not by itself proof of a bug — a helper can take the in-dim from a
  per-weight lookup instead — so each row needed its own gate run before being called broken. Every
  row above was run; the reference `.bin` files for the two training drivers were absent from
  `native/out/`, so `make kvctx` and `make kvrespfwd` now regenerate gold and driver output inside the
  target — the same stale-gold trap that had let a frozen constant stay green for weeks.)

- **S0.2e — a capability / layer descriptor, and a NAMED refusal. LANDED 2026-10-08 (VybForge#10
  phase 2).** `model_config.vyb` answers what a model's DIMENSIONS are; `native/config/model_caps.vyb`
  answers what it REQUIRES of the engine — layer kinds and their state, the quant types present,
  and the optional vision tower / MTP head / tokenizer — and refuses to run a model this build
  cannot express. It reads the capability-shaped metadata (`<arch>.ssm.*`,
  `<arch>.nextn_predict_layers`, `<arch>.full_attention_interval`, `clip.*`, `tokenizer.*`, all
  namespaced by `general.architecture`, so a new architecture is unsupported DATA rather than a
  missing code path) and counts layer kinds from the TENSOR TABLE — one `blk.N.ssm_alpha.weight`
  per recurrent layer, one `blk.N.attn_q.weight` per attention layer — so a layer that merely
  looks recurrent by name cannot fool it. The engine's own capability table (`eng_type`, `eng_gdn`,
  `eng_mtp`, `eng_vision`) is what "this build implements" means, and the printed report and the
  refusal both read it, so they cannot drift apart.

  Measured, gate `native/legit/run_caps_gate.sh` / `make caps` (S0.2e in the Phase-2 battery):

  * Qwen3-4B → **SUPPORTED**: dense, 36/36 attention layers, no GDN/MTP/vision, types
    `F32:145,Q4_K:216,Q6_K:37`. The gate CROSS-CHECKS those counts against the independent Python
    parser (`native/gguf/ridge_inventory.py`), so the two implementations must agree — a Vyb-only
    check could be self-consistently wrong.
  * Qwen3.8-27B Ridge → **UNSUPPORTED** with exactly five named reasons: IQ2_S(160), Q5_K(51),
    IQ3_S(32), the Gated-DeltaNet layer kind (48 layers) and the MTP head
    (`nextn_predict_layers=1`). Q8_0 (96 tensors) left this list on 2026-10-08 when the kernel
    landed — see S0.5. It independently reproduces the hybrid layout the issue describes:
    64 text blocks = 16 attention + 48 recurrent, every 4th block an attention layer,
    `ssm.state_size 128`, `conv_kernel 4`, `group_count 16`, `time_step_rank 48`, `inner_size 6144`.
  * mmproj → **UNSUPPORTED as a vision-encoder** (BF16 110 tensors + vision), not as a broken text
    model — the layout field is what keeps those two failures distinguishable.
  * a file truncated inside the metadata → **UNSUPPORTED with `UNSUPPORTED_READ …`**, never a
    profile full of zeros. A prefix cut inside the DATA region still describes its model correctly;
    that is deliberate, since the profile is a property of the metadata and tensor table, and the
    gate's comment says so rather than implying the descriptor checks the weights.

  Also settled here, because it decides how every gate reads a probe: **the JIT prints main()'s
  return value and always exits 0** (`doc/PYTHON-CLEANUP.md`), so gates parse the `*_DONE` /
  verdict lines and never trust `$?`. `mc_probe.vyb`'s header claimed its exit code was
  `mc_check()`, which was never true — corrected.

- **S0.5 — Q8_0 dequant on real Ridge tensors. LANDED 2026-10-08 (VybForge#10 phase 3, first
  kernel).** `native/kernels/q8_0.vyb` (`q8_0deq`), the simplest of the quant family: 34-byte
  blocks of one f16 delta plus 32 SIGNED int8 with no scale table and no min. It is the type the
  Gated-DeltaNet state path needs (`ssm_alpha`/`ssm_beta`, 96 tensors in the file).

  Gate `native/legit/run_q8_0_gate.sh` / `make q8_0`, on real tensors from the 12 GiB GGUF:
  * the numpy reference `native/tools/q8_0_ref.py` is **bit-identical to the independent python
    `gguf` package's dequantizer on 4/4 tensors**; without that second implementation the
    comparison would only prove our code agrees with our code (the #22 lesson);
  * the GPU kernel matches that reference to **maxrel 4.2e-6** (worst of four tensors), which is
    the f32-multiply rounding floor — device code multiplies in f32 while the references are f64.
    The gate's tolerance is 1e-5 for that reason, and a layout bug would show as O(1), not 1e-6.

  Two producer bugs surfaced while wiring this up, both of the "looks fine, decodes to garbage"
  kind, and both are now fixed in `native/gguf/ridge_inventory.py`:
  1. offsets were emitted GGUF-relative while the repo's convention (and `layer0_ref.py`'s
     `fh.seek`) is ABSOLUTE file offsets;
  2. the data region starts at the table end rounded up to the file's alignment (32 by default,
     `general.alignment` otherwise) — using the raw table end put every offset 12 bytes early.
  After the fix all 866 offsets match the `gguf` package exactly, which is how the second bug was
  caught: the first symptom was NaN deltas, not an obviously broken inventory.

- **S0.6 — IQ2_S dequant on real Ridge tensors. LANDED 2026-10-08 (VybForge#10 phase 3, the
  largest type).** `native/kernels/iq2s.vyb` (`iq2sdeq`), 160 tensors / 4.25 GiB — the mid-stack
  FFN, and the type that decides whether the 27B is runnable. Layout (82 B): `d` f16, `qs[64]`
  (2-bit indices in the first 32 bytes, SIGN BITS in the rest), `qh[8]`, `scales[8]`; per 32-value
  group a 10-bit index selects one of 1024 grid entries.

  **The authority here is llama.cpp's own C, compiled.** The `gguf` package implements Q8_0 but
  NOT IQ2_S, so `native/tools/iq2_s_c_authority.py` extracts `dequantize_row_iq2_s`, the
  `block_iq2_s` struct and the 1024-entry `iq2s_grid` + `kmask_iq2xs` tables VERBATIM from the
  local checkout (commit `4df29be4`), adds only macro shims, compiles it, and the reference must
  agree with it on whole 4096-element slices of real tensors — measured: **3/3 bit-identical**.
  The grid table is never retyped: `native/tools/gen_iq2s_tables.py` extracts it into
  `native/gguf/iq2s_tables.py` (generated, with the upstream commit recorded).

  **The grid rides in the packed buffer.** A 1024×8-byte table cannot be derived and the shared
  launcher (`cuda_launch4i`) passes exactly four integers, so `q` points at an 8192-byte grid
  prefix and the blocks start at `q + 8192`; `iq2_s_ref.py` writes the grid image so driver and
  reference cannot disagree about it. Measured: GPU vs reference **maxrel 1.7e-6** (tolerance
  1e-5, the f32 floor).

  Gate `native/legit/run_iq2_s_gate.sh` / `make iq2_s`, S0.6 in the Phase-2 battery. With Q8_0 and
  IQ2_S implemented, the descriptor's Ridge refusal is down to **four** reasons (IQ3_S, Q5_K, the
  GDN layer kind, MTP), and the caps gate asserts both new types are absent from it.

- **S0.4 — an INDEPENDENT decode oracle: llama.cpp, per token. LANDED 2026-10-07 (VybForge#22).**
  Every other inference gate here compares the GPU against the numpy reference, and both implement
  the same conventions, so agreement proves self-consistency and not correctness — which is how #11
  (token salad) stayed open while every hidden-state gate passed at 5e-6. This gate compares against
  llama.cpp on the same GGUF instead, sharing no code with us. Fixtures
  `native/legit/fixtures/llama_decode/*.fix` are captured by
  `native/tools/llama_decode_capture.py` (`env -u PYTHONPATH .venv/bin/python …`; `llama-cli` cores
  on `--help` in this environment, so the binding in the repo `.venv` is the route — token *ids* from
  the low-level greedy loop, *probabilities* from `create_completion(logprobs=…)`, because
  `llm.scores` comes back unfilled here). Gate `native/legit/run_decode_oracle_gate.sh`, wired into
  the battery as S0.4 (SKIP-with-notice when `llama_cpp` or the GGUF is absent; a provenance change —
  llama.cpp version or GGUF id — FAILS with a recapture instruction rather than letting a stale
  fixture pass). Each fixture declares a mode: `exact`, or `membership` for a documented near-tie.
  Measured 2026-10-07: `decode_weather` exact 8/8, `decode_capital_teacherforced` exact,
  `decode_capital_neartie` membership — and at that tied step **llama.cpp's own two routes
  disagree** (the low-level loop takes `.`, `create_completion` ranks `,` first by 0.25 logprob),
  which is why the case asserts membership. That is the honest boundary: the gate proves agreement
  whatever llama's decision is decisive, not bit-exactness of the distribution. Related: the stale
  gold `[31784, 31784]` that S0.2c left in `native/tools/verify_chat_real.py` has since been replaced
  by a regenerated numpy comparison, so `make chat-real` is green but self-consistent — this gate is
  the independent check for that same prompt (`The capital of France is`, `decode_capital_*`).

### Prompts: what the driver is given, and why the `[0,1]` run looks Chinese and gibberish

`model_driver.vyb` now takes its prompt three ways, and the choice is recorded in the log as
`PROMPT_MODE`:

  - **default** — a real English sentence (`The capital of France is`), ids from the model's own
    tokenizer. A bare run and the chat gates read as an actual conversation.
  - **`VYB_PROMPT=<text>`** — any other real prompt.
  - **`VYB_PROMPT_RAW=1`** — the synthetic 2-token prefix `[0,1]`. No tokenizer, no chat template, no
    text at all. **This is what `make prefill` uses**, because its numpy reference (`prefill_ref.py`)
    is built for exactly that prefix, and the numeric gate should not depend on a tokenizer.

What the synthetic prefix *is*: ids 0 and 1 detokenize to the two characters `!"`. So its
`LLM_CHAT_RESPONSE` is not an answer to anything and no language is being requested of the model. The
argmax there is the top of a nearly flat distribution — p = 0.015 at position 0, p = 0.046 with four
tokens inside 0.005 logprob at position 1 — so the winner is decided by fp noise rather than by
language: llama.cpp independently picks a Chinese-leaning token at position 0 too (a partial UTF-8
byte fragment that can begin 开), and at position 1 the entire top-5 is English words (`" The"`,
`" I"`, `" ("`, `" he"`, `" said"`). Read `PREFILL_HIDDEN_MATCH` / `PREFILL_TOP1_MATCH` for that run,
never the decoded text.

The real prompt is the opposite, and is the semantic check: for `The capital of France is`, llama.cpp
(`-ngl 0`, greedy, same GGUF) returns **12095 = `" Paris"`** at logprob −0.435 against −2.748 for the
runner-up — a decisive, correct English continuation, not a tie. Both the numpy reference and the GPU
driver must reproduce that token-for-token (`make chat-prompt` / `make chat-real`). llama.cpp's
`/tokenize` and the driver's own tokenizer also agree exactly on the prompt ids
(`785, 6722, 315, 9625, 374` — `"The" " capital" " of" " France" " is"`), which is the independent
check on the encoder path.

Three dead gates were found and fixed while wiring this up — all the same failure mode, a check that
cannot fail when it is wrong:

  1. `verify_chat_real.py` hardcoded the gold `[31784, 31784]`, which was the *un-mirrored driver's
     own output* (31784 = `"Cog"`, which is why the broken run printed `<CogCog>`). A gold captured
     from the buggy forward cannot detect the bug. It now requires `PROMPT_MODE=text`, compares the
     per-position top1 against the regenerated numpy gold, compares the final hidden against
     `prompt_hidden_ref.txt`, and requires the decoded response to contain ASCII letters.
  2. `prompt_top1_ref.txt` was dated **2026-09-11**, i.e. before the 09-13 orientation fix in the
     numpy reference, and its ids were not even for `CHAT_PROMPT` (they belonged to another prompt,
     "Add a package to a QEMU x86_64 boot test."). Combined with item 3 below, `chat-prompt` had
     never compared anything at all: it could not run, and its gold was stale and unrelated. The gold
     is now regenerated inside the `chat-prompt` / `chat-real` targets on every run, the way
     `make prefill` always did. The gate that *was* green while wrong is item 1: `chat-real`, measured
     against the frozen `[31784, 31784]` that the buggy forward itself had produced.
  3. `make chat-prompt` could not run at all: `CHAT_PROMPT ?= "The capital of France is"` combined
     with `VYB_PROMPT="$(CHAT_PROMPT)"` produced doubled quotes, and under `/bin/sh` the recipe ran
     `capital` as a command (`/bin/sh: 1: capital: not found`). The variable is now unquoted. Related:
     `native/out/prompt_ids.txt` held newline-separated ids for an unrelated prompt
     ("Add a package to a QEMU x86_64 boot test."), so the committed gold never corresponded to
     `CHAT_PROMPT`. `emit_prompt_ids.py` now requires the log's `PROMPT=<...>` line, refuses a log
     whose prompt does not match the expected one, and rewrites the file in one canonical
     comma-separated line.

**S0.3 — loaders (this is S3's content, scheduled here). LANDED 2026-10-02.**
`native/torchload/` holds two container readers, both host-mode and model-agnostic:
- **safetensors** — `safetensors.vyb` (8-byte LE header length + JSON map via `native/json` + raw LE
  data region), driven by `st_probe.vyb`. Indexing reads only the prefix and header, so a 4 GB shard
  costs one header read; tensor bytes are read on demand.
- **torch `.bin`** — `torchzip.vyb` (zip central directory, data offsets resolved from local headers,
  ZIP64 handled: torch writes a ZIP64 EOCD record + locator *before* the classic EOCD) plus
  `torchpickle.vyb` (a protocol-2 pickle VM over a structure-of-arrays arena; one storage per
  `data/<k>` record, so `persistent_load` needs no offset splitting), driven by `tb_zip_probe.vyb`
  and `tb_pkl_probe.vyb`.

Each yields name → {dtype, shape, stride, storage key, byte range} plus a bounded-window sha256, so a
caller can stream tensor bytes without loading a shard whole.
**Gates (both green):** `native/legit/run_safetensors_gate.sh` — byte-exact listing parity against an
independent stdlib parse plus a `safetensors`-library cross-check by name; verified on a 66 MB adapter
and 4.6 GB / 8.1 GB shards from three model families (incl. AWQ packed int32 + fp16 scales).
`native/legit/run_torchbin_gate.sh` — container layer against Python's `zipfile` (whose SIZE comes from
the OS, and which reads every hashed record twice), object graph against CPython's pickle VM, plus
`torch.load(map_location='meta')` as the authoritative third side compared by name; verified on all 15
SpikingBrain shards: 395 tensors (== the index's count), 0 mismatches, pickle STOP at the last byte.
`.npy` is not needed by this model and is deferred — `.bin` + safetensors cover the queue.

**S0.4 — attention family + KV/state abstraction.**
One attention module with the causal mask as a *parameter*: full causal (Qwen3 today), sliding window
(SWA), and a state path for recurrent attention (GLA, in S2) — plus a KV/state abstraction covering
dense KV, rolling window and per-layer recurrent state, so S2/S4 do not grow a third copy of the
attention plumbing.
**Gate:** SWA ≡ full-softmax-on-windowed-mask; a window of ∞ ≡ the existing full causal path; the
existing Qwen3 layer gate stays green (LAYER_VERIFY ~5e-5 unchanged).

**Park-list — explicitly NOT in S0:** autograd generality, generic optimizers, ONNX/OpenCV/PaddleOCR,
BLAS/FFT/reduce-scan surfaces, Unicode property classes (risk 7). Entry requirement remains "forced by
a model in the queue".

**Ordering:** S0.1 and S0.2 are the front (S2 and S4 both wait on them). **S1 is host-only and
independent of all of S0** — it can go first or in parallel. S0.3 blocks S4 only.

### S1 — Spike core in Vyb (`spike.vyb`)

Pure host Vyb module, no GPU:
- adaptive-threshold spiking: `V_th(x) = mean(|x|)/k`, `s = round(x / V_th)`;
- three codings: binary, ternary (symmetric), bitwise (pure / bidirectional /
  two's complement) with spike-train expansion (`s_t` per timestep, `2^t` weights);
- stats: per-tensor silent-channel fraction, mean spikes/channel, firing rate,
  sparsity at a given step window;
- `k` exposed as a parameter.

**Gate:** bit-for-bit match vs the repo's published `Int2Spike/demo.py` +
`test.py` on fixed vectors (binary/ternary/bitwise), then reproduce the paper's
measured stats (1.13 avg / 18.4% silent / 69.15% sparsity) on activations sampled
from a real 7B forward (any layer set; stats are per-linear-layer).

### S2 — GLA + SWA kernels (NVPTX, `native/kernels/`)

**Consumes:** S0.1 (bf16 storage / fp32 accumulate) and S0.4 (the SWA window is a *parameter* of the
attention family, not a fork of `layer.vyb`). GLA is genuinely new work either way.

- `gla.vyb`:
  1. fused-recurrent decode first (S = diag(g).S + k v^T; o = q S; per-head
     RMSNorm; o_proj) — correctness gate;
  2. chunked parallel prefill (intra-chunk causal linear attention with per-token
     gate decay + inter-chunk state scan) — TTFT path.
- `swa.vyb`: windowed causal softmax (4K), GQA 7:1, RoPE, sliding KV cache of one
  window (no full KV growth) — S0.4's windowed path, instantiated for this model's 4K window.
- GQA, ReLU-on-q/k, logsigmoid/16 gate, per-head RMSNorm all per the repo code.
- Weights bf16 (3090 has native bf16); accumulation fp32, noted per gate.

**Gate:** per-kernel numpy golden gates (the `GEMM_OK` / `LAYER_VERIFY` pattern):
random tensors, chunked vs recurrent equivalence at random split points, SWA vs
full-softmax-on-windowed-mask equivalence. Tolerance 1e-4 relative (bf16/fp32 mix
noted per gate).

### S3 — Native weight loaders

**Timing:** scheduled inside S0 as **S0.3** (2026-10-02). The content below is unchanged — it moved
earlier because it is generic and blocks S4.

- `safetensors.vyb`: header length + JSON map (via `native/json`) + raw LE tensor
  reads via `io`. Generic, reusable for any safetensors model.
- `torchbin.vyb`: torch.save zip container (EOCD + central directory) + minimal
  pickle subset (tensor entries: shape/dtype/storage offsets; storage objects;
  the ~20 opcodes needed), reading raw tensor bytes from zip entries.
- Both produce a name -> {dtype, shape, byte-buffer} index the model driver maps
  to Qwen2.5/SpikingBrain tensor names.

**Gate:** on each downloaded shard, every tensor's name/shape/dtype matches the
torch/safetensors reference listing, and sampled tensor bytes hash-match (a few
tensors per shard, full-tensor check on one small tensor per dtype). First
download: inspect the W8ASpike dtype map (repo is ~29.8 GB — confirm int8
weights + scales layout before finalizing the model wiring; if it carries fp32
tensors too, note them).

### S4 — SpikingBrain-7B forward in Vyb

**Consumes:** S0.2 (dims from the config contract, not literals), S0.3 (loaders), S0.4 (attention
family + state abstraction).

- Model driver (`native/llm/spikingbrain.vyb`, main-less module + thin runner):
  embed -> 28 layers (even: SWA, odd: GLA) -> final RMSNorm -> lm_head -> logits;
  prefill chunked (carries GLA state + rolling SWA window across chunks), decode
  recurrent.
- Tokenizer: reuse stdlib/vllm Qwen BPE (tokenizer.json/vocab.json ship with the
  model — verify byte-BPE equivalence vs the HF tokenizer on a prompt corpus).

**Gate:** fixed prompt + fixed seed vs the HF reference (repo `hf_7B_model` code
in a venv, same weights): **exact greedy token match** for N>=256 tokens on >=5
prompts; logit cosine > 0.999 on a few held-out random inputs; perplexity within
1% on the standard eval slice the paper uses (or a fixed 10k-token reference
corpus, whichever is cheaper to run).

### S5 — Long-context + TTFT benchmark

- TTFT table at 32K / 128K / 1M / 4M tokens (chunked prefill), plus tokens/sec
  decode at the same contexts; memory curve (expect ~26 MB GLA state + ~1.6 GB
  window KV, near-constant growth).
- Ratio vs a quadratic baseline: run Qwen2.5-7B (or the repo's baseline numbers,
  clearly labeled) at the same lengths where feasible on one card; otherwise
  report our absolute numbers against the paper's published baseline.

**Gate:** measured table + memory curve as artifacts; the "100x at 4M" claim
reported as relative-to-quadratic with our own absolute numbers beside it.

### S6 — W8ASpike event-driven path + measured energy

- INT8 weight GEMM kernel (weights int8 + scales; activations via spike counts).
- Spike-train execution of the linear projections: expand s_INT per the chosen
  coding (default ternary-bitwise, 3-step window), execute event-driven
  accumulation (skip silent channels), fold by `V_th`. Design follows the VybFly
  calendar-queue / SoA pattern; a dedicated NVPTX kernel, not the general DES.
- Calibration: 128-sample calibration set for scales (port the repo's approach);
  verify the ~1–3% benchmark-drop claim on our eval slice.
- Energy, two-track (VybFly sec-17 discipline):
  - **measured**: RAPL wall power, J/token and J/spike, dense-bf16 vs
    spiking-int8 on identical prompts (3090; the paper's 45nm pJ numbers are NOT
    ours);
  - **modeled**: pJ/MAC via the paper's formula with OUR measured spike
    statistics (expect ~1.13 spikes/ch, ~18% silent on our activations).

**Gate:** spiking path output within stated tolerance of the dense-bf16 path on
the same prompts; sparsity stats match the paper's within 5% absolute; energy
table with both tracks labeled.

### S7 — VybFly integration (fly brain) — first-class

Adopt the S1 spike core in the fly-brain simulation; this is where the two
projects share machinery in both directions.

- **Adaptive-threshold neuron model** in the VybFly DES: SpikingBrain's
  IF-with-adaptive-threshold neuron is exactly PROJECT-VYBFLY.md Phase 1's
  "later candidate: adaptive LIF". `k` becomes a per-experiment tunable (their
  Appendix A.8 documents the firing-rate tradeoff; the fly side sweeps k for its
  own target sparsity).
- **Ternary/bitwise event encoding** for the fly DES: replace plain binary LIF
  events with signed/weighted events where the plasticity rule supports them —
  fewer, sparser events = less queue pressure and atomic contention, which is
  the stated bottleneck for the 100x / 14M-neuron M15 attempt (PROJECT-VYBFLY.md
  sec 23).
- **Reuse, concretely:**
  - `des.vyb` calendar queue + SoA entity state: unchanged substrate, new neuron
    model + event payload (sign, bit weight);
  - CUDA DES (M3): extended with signed atomic accumulation (M3's fixture
    re-validation, spike-for-spike, as the gate pattern);
  - energy instrumentation (M4): add the spiking variant to the 1x/2x/5x/10x
    curves → J/bio-s and J/spike for LIF vs adaptive-threshold spiking → feeds
    M13/M14 capability/watt honestly (two tracks, per sec 17);
  - ingest convention: flat binary + meta.json for any new per-neuron parameter
    arrays (threshold stats etc.);
  - probe/gate discipline: spike-for-spike vs the Python `flyscale` oracle on a
    fixture before any scale run.
- **Promotion:** once S1 + S7 both consume it, promote the spike core to
  `stdlib/spike` in the Vyb repo (the vllm/chain promotion pattern: module-defined
  `share(all)` structs, env-var artifact paths if it ever owns files, refman
  regen + section-4 digest + suite-count sync). Both repos then import one
  implementation.

**Gate:** (a) LIF vs adaptive-threshold spike-for-spike on the tiny-net fixture
against the extended Python oracle; (b) CUDA equivalence re-run at 512 neurons;
(c) at least two scales (1x and 2x) with full energy curves for both neuron
models; (d) `stdlib/spike` green in the Vyb suite.

### S8 — Deferred (note, after the 7B line is done)

- **SpikingBrain-76B**: intra-layer parallel LA+SWA hybrid (1:1) with full-softmax
  attention layers interleaved at 1:6, 128 learnable sink tokens (unmasked among
  themselves,
  prepended embeddings, custom attention mask), MoE FFN (16 routed top-1 + 1
  shared, upcycled alpha = 0.98, 7 dense layers at [1,2,3,5,7,9,11]), GLA gate
  tied to key (`g = 1 − k_t`, sigmoid QK). New kernels needed: router + expert
  dispatch, sink-token masking. Weights on ModelScope.
- **SpikingBrain 2.0** (BICLab/SpikingBrain2.0, 2026-04): architecture and
  training-token-efficiency upgrades — diff against this plan and re-scope.
- **VLM** (SpikingBrain-7B-VL) if the vision path is wanted.
- **Training**: out of scope by decision 2; only the 76B MoE upcycling details
  matter if/when training enters scope.

---

## 5. Verification and honesty rules

- Every gate is a **real run with an artifact path** (the dogfooding rule):
  probe program + output file + one-line verdict, in the style of the existing
  `*_verify` gates and VybFly's `results/` tree.
- Oracle discipline: Python (torch/HF reference, `flyscale` package) is the
  golden reference; Vyb matches it — never tune a tolerance to make a check pass,
  mismatches stay in the output.
- Energy claims are two-track and labeled: **measured on the 3090** vs
  **modeled** (paper's pJ formula or biological-equivalent anchor). The 97.7%
  figure is never quoted as a measurement without that split.
- The 100x TTFT claim is quoted as relative-to-quadratic-baseline with absolute
  numbers beside it.
- Toolchain drift: if a Vyb program that ran fine starts crashing with no source
  change, check `build/vyb` mtime + `git status` first (impl-agent hot-rebuilds)
  before debugging; probe with a minimal standalone file.

---

## 6. Known Vyb constraints to build against (verified, see VybFly VYB-PORT.md + skills)

- Structure-of-arrays, never `Vec<Vec<T>>` (crash/garbage family).
- Caller state mutates only via `their<T>`; `borrow()` once at the owner,
  propagate after (re-borrow segfaults).
- Kernel entry takes exactly ONE descriptor pointer; keep it short (long
  descriptors lose their tail); zero every device buffer after alloc; no zero
  grid dims; device intrinsics return CInt (`as Int` at assignment); descriptor
  floats travel as micro-units; 8-byte slots for scalar readback.
- Ranges are INCLUSIVE (`0..n-1`); explicit return types on every function;
  `String.to_int` absent (ship a parser); `io` byte buffers do not cross module
  boundaries; `for (x in structVec)` fails (bind to a local first);
  method-chain on a function's Vec return dies in codegen (bind to a local).
- `return 1` from main exits 0 under JIT — gates must `exit(1)`.
- New `.vyb` tests / stdlib modules: sync the documented suite counts and run
  `tools/docstatus.vyb` / refman regen before pushing to the Vyb repo.

---

## 7. Open questions / risks

1. **W8ASpike layout**: repo is ~29.8 GB across 7 safetensors — larger than int8
   7B implies. Confirm dtype map on first download (int8 + scales, or fp32
   components too) before finalizing S4/S6 wiring. S3's loader gate covers this.
2. **torchbin pickle subset**: finite but unproven in Vyb. If the opcode subset
   balloons, fall back to the flat-binary exporter (VybFly convention) — decision
   2 keeps this door open without committing.
3. **ModelScope download**: weights are CN-hosted (modelscope.cn); plan for the
   modelscope CLI or direct file URLs. HF mirror `maujim/spikingbrain-v1-7b-base`
   (same 15 .bin shards) is the fallback source.
4. **4M-token TTFT on one 3090**: compute-bound (minutes, O(n) prefill). The
   benchmark is a table; if 4M wall-time is impractical on the tuned kernel,
   report up to 1M + extrapolation, labeled.
5. **Chunked GLA numerical drift** at very long contexts vs the reference
   Triton implementation: S2's chunked-vs-recurrent gate at random split points
   bounds it; S4's logit gate catches what remains.
6. **S7 plasticity compatibility**: the MB learning rule (Phase 8) is weight-based
   on LIF; signed/bitwise events must keep the rule's inputs equivalent — the
   fixture spike-for-spike gate is designed to catch any behavior change.
7. **Tokenizer non-ASCII fidelity (added 2026-10-02).** The reused Qwen BPE pre-tokenizer is exact on
   ASCII and *approximate* on non-ASCII, because Vyb has no Unicode-property-class regex — the ported
   splitter classifies remapped byte-glyphs by glyph identity (see `doc/SUBSTRATE-SCOPE.md` and
   `native/tokenizer/` for the divergence trail). S4's byte-BPE equivalence gate against HF will hit
   this. Either accept and document the divergence for the classes that actually occur, or add Unicode
   property classes to `stdlib/regex` — a Vyb-repo task, to be filed upstream rather than worked
   around here.

---

## 8. File layout (where things land)

- VybForge `native/tensor/` — dtype + tensor core + config contract (S0.1/S0.2);
  `native/torchload/{safetensors,torchbin}.vyb` — loaders (S0.3);
  `native/kernels/attn.vyb` — the attention family (S0.4).
- VybForge `native/spike/spike.vyb` (+ probes) — S1 core (promoted to
  `stdlib/spike` in S7d).
- VybForge `native/kernels/gla.vyb`, `native/kernels/swa.vyb`,
  `native/kernels/spike_gemm.vyb` (+ `.ll` artifacts are build junk — delete
  before committing) — S2/S6.
- VybForge `native/torchload/safetensors.vyb`, `native/torchload/torchbin.vyb`
  (or `native/tensor/` if the tensor module is a better home) — S3.
- VybForge `native/llm/spikingbrain.vyb` (main-less) + `native/llm/sb7b_run.vyb`
  (runner) — S4; benchmark drivers in `native/tools/` — S5.
- VybFly `src/vyb/adaptive_neuron.vyb` (or extension of `des.vyb`) + extended
  `scripts/phase*` drivers + `results/` artifacts — S7.
- Vyb repo: `stdlib/spike/` after promotion (refman page + section-4 digest +
  suite-count sync required).

Cross-references: VybFly `PROJECT-VYBFLY.md` (Phase 1 later-candidates, sec 17
energy model, sec 23 GPU scaling), VybFly `docs/VYB-PORT.md` (Vyb constraints +
CUDA lessons), VybForge `doc/LLM-FORWARD.md` (facade pattern this extends).

  **Resolution: all five drivers are green.** The two reds above turned out to share one root cause, and
  finding it needed one gate improvement:

  - **`kvctx_ctx_ids.bin` was stale.** Both KV drivers embed the context-token ids read from that
    binary file; `kvctx_ref.py` wrote only the text form and tokenizes live. The file on disk held a
    *different prompt* (`[28497, 419, …]` vs the reference's `[40, 1366, …]` — only the final id 13
    agreed), so the context build forwarded different tokens than the reference and, because both sides
    ran the same valid architecture, the outputs still correlated at 0.995–0.999 with a huge maxrel.
    `kvresp` inherited it end to end, since it builds the cache its response forward attends to from
    the same file. `kvctx_ref.py` now writes the `.bin` as well, so a `make kvctx` run regenerates it.
  - **The gate could not see the rope.** Every comparison used the first 64 values of each `[S,NKV]`
    dump — row 0, i.e. token 0 — whose rope angle is `(POS + 0) * freq = 0`, so the rope is the identity
    there by construction. `kvctx`'s rope had never been loaded (`DF` uninitialized — see #19) and no
    gate could tell. The gate now compares row 3 as well, which is genuinely rotated.
  - With the ids current and the rope load in place, all 20 kvctx comparisons pass (`corr 1.00000`,
    `maxrel` 2.5e-06–2.4e-04) and `kvrespfwd` passes at `corr 1.000000`, `|g| = |r| = 6.92e+03`,
    `max|g-r| = 5.55e-04` — down from `corr 0.775046`, `|g| = 7.88e+03`, `max|g-r| = 4.21e+02`.

  The lesson worth carrying: both reds *looked* like numeric drift and were input faults. A gate whose
  sample point makes a whole stage a no-op (row 0 under a rotary) cannot fail that stage, and a
  comparison that passes at 0.995 on wrong inputs is more dangerous than one that fails outright.
