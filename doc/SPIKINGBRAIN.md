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
toolchain. Importing `native/tensor` also makes a file's own `extern "C"` declarations
unresolvable (**Vyb#442**), which is why the driver hand-wires its CUDA calls for now.

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
  incompatible shapes **refused**). Deliberately **pure — no CUDA and no allocation** (importing
  `native/tensor/tensor.vyb` drags `cuda_binding` into the importer, Vyb#442, which a shape library
  must not do to its consumers), which is also what makes it numpy-checkable on the CPU.
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
  in llama.cpp's top 10 at either position — so `make chat-real` is now red against a stale constant
  and needs re-blessing or a set/tolerance comparison; (ii) at the arbitrary ids this gate uses
  (`[0,1]`) the next-token distribution is nearly flat (p≈0.04, four tokens within 0.04 logprob), so
  a top1-argmax comparison is a coin flip: the *hidden* comparison is the meaningful one, and the
  real-prompt gate (`chat-prompt`, "The capital of France is" → ` Paris`) is the semantic one.
- **S0.2d — battery step + the no-regression sweep. LANDED 2026-10-02.** `run_config_gate.sh`,
  `run_tensor_gate.sh` and the S0.2c `make prefill` step are in `run_phase2_battery.sh`; the battery
  is green. **Still open from this step:** five more drivers hand the dequant kernel a literal `0`
  in-dim — `native/host/kv_driver.vyb`, `layer0_driver.vyb`, `wcheck_driver.vyb`,
  `native/train/kvctx.vyb`, `kvrespfwd.vyb` — each needs the same per-weight in-dim treatment and its
  own gate re-run; the sweep table for that is in `doc/SPIKINGBRAIN.md`'s S0.2d notes below.

  Sweep for the remaining literal-`0` in-dim sites (grep the dequant helper *definition* for an
  `inz` parameter, then the launch for a literal `0` in the 4th slot):

  | driver | helper | in-dim today | gate |
  |---|---|---|---|
  | `native/host/model_driver.vyb` | `load_quant` | **fixed** (per-weight) | `make prefill` green |
  | `native/host/layer0_driver.vyb` | `load_quant` | `0` | `make layer0` |
  | `native/host/kv_driver.vyb` | `dequant_gpu` | `0` | — |
  | `native/host/wcheck_driver.vyb` | `dequant_gpu` | `0` | — |
  | `native/train/kvctx.vyb` | `deq_w` | `0` | `make kvctx-*` |
  | `native/train/kvrespfwd.vyb` | `deq_w` | `0` | `make kvrespfwd-*` |
  | `native/host/decode_driver.vyb`, `resident_driver.vyb`, `loradec_driver.vyb`, all of `native/train/*` | — | fixed | green |

  (A missing `inz` parameter is not by itself proof of a bug — a helper can take the in-dim from a
  per-weight lookup instead — so each row needs its gate run before it is called broken. The two rows
  with a literal `0` in the launch and no lookup are the ones to treat as suspects.)

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
