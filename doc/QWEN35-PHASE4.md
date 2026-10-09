# Phase 4 — what stands between here and a first token from the 27B

Reconnaissance note, written after phase 3 closed (every quant type in the file has a kernel, each
bit-exact against an implementation that shares no code with ours). Phase 4 is the layer wiring, and
the descriptor names exactly two blockers, both of them hardcoded:

    native/config/model_caps.vyb:140   eng_gdn()<Int> -> { return 0 }   // Gated DeltaNet layers
    native/config/model_caps.vyb:141   eng_mtp()<Int> -> { return 0 }   // native draft head

    :387  if (eng_gdn() == 0) { ... "UNSUPPORTED_LAYER_KIND gated-deltanet layers=48" }
    :390  if (eng_mtp() == 0) { ... "UNSUPPORTED_CAPABILITY mtp nextn_predict_layers=1" }

## What the model actually asks for (from the GGUF, not from the name)

    general.architecture          qwen35          (NOT qwen3next — worth knowing before searching)
    block_count                   65              (64 text + 1 draft)
    embedding_length              5120
    feed_forward_length           17408
    full_attention_interval       4               -> 16 full-attention layers, 48 recurrent
    attention.head_count/_kv      24 / 4
    attention.key_length          256
    attention.value_length        256
    rope.dimension_count          64
    rope.dimension_sections       [11, 11, 10, 0]  (IMROPE — not the usual pairs)
    rope.freq_base                1e7
    ssm.conv_kernel               4
    ssm.group_count               16
    ssm.inner_size                6144
    ssm.state_size                128
    ssm.time_step_rank            48
    nextn_predict_layers          1

## The reference implementation is already on this box

`~/Projects/llama.cpp` at `4df29be4` — the same checkout the dequant authority was extracted from,
so the provenance convention carries over unchanged:

* `src/models/qwen35.cpp` — the model: `load_arch_hparams`, `load_arch_tensors` (it reads a `d2t`
  tensor for the draft head), and `build_arch_graph`, which per layer picks
  `build_layer_attn_linear(...)` (recurrent) or `build_layer_attn(...)` (full attention) on the
  interval.
* `llm_build_delta_net_base` — the Gated DeltaNet graph, shared with the other delta-net models. The
  file is `src/models/delta-net-base.cpp`: `build_delta_net_chunk`, `build_delta_net_auto`,
  `build_delta_net_fuse`, `build_delta_net` and `build_conv_state` — a chunked, an auto-selecting
  and a fused variant, so a reference must implement the one a plain CPU run actually uses.
* `src/models/qwen35moe.cpp` — the MoE sibling; useful when reading the shared base.

## Two build facts that decide how the reference is made

* **A built libggml exists here** (`~/Projects/llama.cpp/build/bin/libggml.so`, `libggml-cpu.so`,
  `libggml-base.so`), so an authority harness can LINK the same library llama.cpp runs rather than
  recompiling ggml sources the way `ggml_dequant_authority.py` had to for the quant kernels. That is
  the stronger option: the authority would execute the identical op implementations, not a copy.
* **`llama-cli` dumps core on this box** (also known from earlier work — it cores on `--help`). The
  working entry point to llama.cpp's own inference is `llama-server` + a completion request. Any
  end-to-end oracle for phase 4 must go through the server, not the CLI.

## What the repo has today

Nothing of the recurrence. The only files that mention `ssm_*` are the descriptor (which reads the
metadata as evidence and refuses on it), the inventory, and the Q8_0 dequant fixtures — whose real
tensors happen to BE the `ssm_alpha`/`ssm_beta` pair, which is why the type list looked familiar.

So phase 4 is genuinely new code, in this order:

1. **The recurrent layer math** (`conv1d` with `conv_kernel=4`, the gated delta rule with
   `state_size=128`, `group_count=16`, `time_step_rank=48`, and the state carried between tokens),
   ported from `llm_build_delta_net_base` and checked against it the way every dequant kernel was
   checked — an authoritative reference first, then the GPU.
2. **The hybrid schedule** (`full_attention_interval=4`: 16 attention + 48 recurrent interleaved),
   plus IMROPE (`[11,11,10,0]`) — verified absent today: no file under `native/` mentions
   `dimension_sections`, and the prefill reference calls plain `rope(q, k, pos)`.
3. **The MTP head** (the 65th block + `d2t`), which is the smaller piece — one layer, and the
   descriptor's `eng_mtp()` is a capability flag rather than a big kernel.
4. Only then `eng_gdn()`/`eng_mtp()` flip to 1 and the descriptor stops refusing the model.

## Phase 4, item 1 — first unit: the recurrence, checked against ggml's own op

`native/tools/gdn_authority.c` + `native/tools/gdn_ref.py`. The harness LINKS the libggml this
llama.cpp was built with (`~/Projects/llama.cpp/build/bin`) and calls `ggml_gated_delta_net`
directly, so it runs the identical op implementation llama.cpp uses, not a copy.

### Established

* The op's buffer split is confirmed by running it: with `K=1` the result is the attention output
  `[S_v, H_v, T, B]` immediately followed by the new state `[S_v, S_v, H_v, B]`
  (`ggml/src/ggml-cpu/ops.cpp:10797-10803`), and `kda` is decided by the gate's `ne0`
  (`ops.cpp:10783`) — our gate is `ne0 == 1`, so the scalar-decay branch.
* **The numpy port reproduces ggml's op EXACTLY at the smallest geometry** (S=4, one head, one
  token): output and new state match to 6e-8 relative, which is the authority's own f32 floor.
  That validates the formula and the state orientation — including that the state is stored
  transposed (`s_out[j*S_v + i] = S[i][j]`, `ops.cpp:10849-10850`) and that the read-out happens
  AFTER the rank-1 update, scaled by `1/sqrt(S)` (`ops.cpp:10852-10882`).
* **The op has TWO kernels, and the multi-token path is a different one.** With multiple tokens the
  op's token-0 output CHANGES when more tokens follow — impossible for a causal recurrence
  (demonstrated by feeding identical tokens at T=1,2,3). The kernel read at `ops.cpp:10740-10896`
  writes scores but not the state when `K=1`; a second entry point,
  `ggml_compute_forward_gated_delta_net_f32` (`ops.cpp:10899`), carries chunking logic. Multi-token
  comparison therefore needs that path's arrangement established first, and the reference
  deliberately refuses to claim it (`--tokens 1`).

### NOT established — the open problem, with its evidence

At model geometry the comparison fails, and not by rounding. Three experiments narrowed it down.

**1. The delta rule itself is right, and so is the v-head mapping.** A setup where the update is the
only effect (`state = 0`, `beta = 1`, `k = e_0`, `q = e_0`, `v[:,h] = V_h`) returns `out = V_h / sqrt(S)`
per head — exactly what the port predicts, including the outer-product orientation and the scale.

**2. Two heads fed identical memory give DIFFERENT results.** With `H_v = 2` and the state, v, gate
and beta repeated along the head axis, the op returns two different head outputs, and a head-0 result
that differs from the same head-0 computation at `H_v = 1`. Identical input slices cannot do that
under the layout the model asserts.

**3. A delta-function scan maps the addressing outright.** Put a single `1.0` at flat state slot `k`
(decay 1, `beta = 0`, `q = 1`) and read which `(out index, head)` picks it up. Result: **24 of 32
slots land where the documented layout says they should not** — the op behaves as if the state's axes
were `(i, head, j)` while its own asserts (`delta-net-base.cpp:399`) require
`ne[0]=ne[1]=S_v, ne[2]=H_v`.

### Where this stands after following the data path

`build_rs` (`llama-graph.cpp:3361-3382`) reshapes the cache to `(state_size, s->ne[1])` and gathers
rows via `get_state_rows`. With one sequence that gather is trivial, so the op receives the per-layer
cache row in its NATURAL order — the same order the harness builds. Meanwhile the reshape in
`qwen35.cpp:401-402` therefore does not reorder anything either.

So the two readings now contradict each other, and one of them is wrong:

* the harness's INPUT is natural (just argued from the source), yet
* the delta-function scan says the op's input addressing is not natural.

The cheapest experiment that separates them, and the one to run first next: **test the OUTPUT
layout, not the input.** Feed `q = e_j` for one `j` at a time with `state = identity` per head, and
read both the score output and the state output. If the score output lights up at the slot the
kernel's formula (`attn_out_base + (iv3*n_tokens*H + iv1)*S_v`, `ops.cpp:10839`) predicts, the
input side is at fault; if it lights up transposed, the fault is in how the result buffer is read —
which is the cheap possibility, since every conclusion drawn so far depends on that read.

### And yet the source says both my input and my output read are correct

Read one step further and the contradiction gets worse, which is worth stating plainly rather than
leaving as a tidy story:

* the kernel derives `S_v = src_v->ne[0]`, `H = src_v->ne[1]` (`ops.cpp:10758-10761`) — my tensors
  give 4 and 2, as assumed;
* the state is asserted contiguous and addressed `(iv3*H + iv1)*S_v*S_v` with the row walk
  `j*S_v + i` (`ops.cpp:10835`, `10849-10850`) — i.e. flat layout `(i, j, head)`, which is exactly the
  `(S_v, S_v, H_v)` tensor the harness builds;
* the score output is addressed `attn_out_base + (iv3*n_tokens*H + iv1)*S_v` (`ops.cpp:10839`) — i.e.
  head `h` at offset `h*S_v`, which is exactly how the harness reads it.

So input layout, output layout and the recurrence all check out against the source, and the
delta-function scan still says 24 of 32 slots land elsewhere. One of those two things is wrong and I
have not found which; more probes on my own harness would only re-confirm my own assumptions.

**The next step is therefore to stop probing the harness and have llama.cpp answer**: instrument the
real path — a debug print in the op (or ggml's own debug facilities) under a real run of the model,
dumping the actual `src_state` / `src_g` / `src_beta` pointers and strides as the graph hands them
over. That establishes the ground truth directly instead of inferring it from arithmetic that both
sides believe is right.

### The instrumented run — and what it eliminated

Done as the previous step proposed. A guarded dump was added to the op's entry point
(`GGML_GDN_DEBUG=1`), ggml-cpu rebuilt, and the harness run against it. The dump is saved as
`native/tools/ggml_gdn_debug.patch` (llama.cpp's source was then restored and rebuilt clean).

It confirms — from inside the op, not by inference — that the tensors are exactly what the harness
feeds:

    GDN_DBG  state type=0 ne=(128,128,48,1) nb=(4,512,65536,3145728) contig=1
    GDN_DBG  dst   type=0 ne=(6144,129,1,1)  nb=(4,24576,3170304,3170304) contig=1
    GDN_DBG  v ne=(128,48,1,1)  g ne=(1,48,1,1)  beta ne=(1,48,1,1)  q,k ne=(128,16,1,1)

So the state is contiguous in the (i, j, head) order the harness builds, `dst` is the scores
followed by the state (6144 then 128×6144), and every input is shaped as assumed. **The tensor-layout
hypothesis is dead** — the delta-function scan's oddities were being read through an output layout
that is in fact correct, so that scan's inference (mine, not the op's) was the flawed part.

Two corrections to the record, both mine:

* the first instrumented runs printed nothing because `gdn_ref.py` captures the harness's stderr and
  discards it — not because the op took a different path. The op's path (`one_chunk`) was right all
  along.
* the "24 of 32 slots land elsewhere" conclusion came from that same misread and should not be
  trusted as evidence of anything on the op's side.

### The intermediate diff, run self-contained

`native/tools/gdn_diff.py` does the whole loop in one process — generate, write, run the harness with
the op's dump, run the port, diff — because an earlier ad-hoc comparison of mine read a file that a
later run had regenerated, and reported nonsense. That ad-hoc comparison, *including* its "the op's
values sit at flat indices 0..3" search result, is void and should not be cited.

What a clean run says, for head 0, first token:

    decay  op=0.695724607  port=0.695724589     (agree, f32 vs f64)
    beta   op=0.476222813  port=0.476222813     (identical)
    i=0    k/q/v MATCH exactly; delta op=0.0349552184 port=0.0255914107; attn op=0.00426558638 port=0.00110170828
    i=1    k op=-0.125186399 port=-0.0155698974 ; q op=-0.134002402 port=-0.0744439587
    i=2    k op=-0.172529504 port=-0.0382126644 ; q op=0.198019758 port=-0.183470666
    i=3    k op=0.00990023743 port=-0.0622620061 ; q op=-0.011945161 port=0.0357212573
    whole-tensor maxrel out=1.356 state=5.364e-01

So element 0 of the inputs agrees and elements 1+ do not — on the same buffer, with the op's own dump
reporting `nb0 = 4` for k, q and v (contiguous). That is a stride/offset disagreement between the two
readings, not a formula disagreement, and it is the first signal in this whole investigation that
points at a specific place rather than at "the op".

Next step: inside the same op call, print `src_k->data`, `src_k->nb[0..3]` and the first four floats
at the pointer the op will use, *next to* the first four floats the harness wrote. That distinguishes
"the harness handed over a different buffer than it wrote" from "one side's pointer arithmetic is not
what the source reads as" — and it is a single run, not a probe series.

### The op is exonerated — the fault is on my side of the comparison

The decisive diagnostic: the harness prints the k block it *intended* to load next to what is actually
in the tensor it handed the op. Both read

    0.00886083394 -0.125186399 -0.172529504 0.00990023743

which is EXACTLY what the op printed for `k_d[0..3]`. So the op reads its inputs correctly, and the
numbers my port printed for those same indices (k[1] = -0.0155698974, …) came from somewhere other
than the array the script itself wrote out. The disagreement is therefore in my reference or in how
the diff script feeds and prints it — not in ggml, and not in any stride or layout of the op.

### ROOT CAUSE FOUND: numpy's fastest axis is the last one, ggml's is `ne0`

    kd   = k[:,0,0,0]     = [ 0.00886083 -0.0155699  -0.03821266 -0.06226201]
    flat = k.ravel()[:4]  = [ 0.00886083 -0.1251864  -0.1725295   0.00990024]

`k` is (128,16,1,1), so `k[:,0,0,0]` steps 16 elements per sample, while the buffer's real element
order is `ne0`-fastest. Every array the port reads this way has been transposed on its trailing axes;
the op was right all along, and the delta-function scan's "24 of 32 slots land elsewhere" was my
transposed view talking, not the op.

This explains the whole shape of the investigation: the minimal geometry (S=4, H=1, T=1, B=1) has all
strides equal to 1, so a transposed read coincides with a correct one — which is exactly why that case
matched to 6e-8 while nothing at model geometry did. It also explains why the harness's dump agreed
with every expectation: the tensors really were right; only the reference's *reading* of them was not.

### Fixed and verified: the reference now matches the op at model geometry

`native/tools/gdn_verify.py` (self-contained, corrected convention) reports

    GDN_VERIFY geometry S=128 H_k=16 H_v=48 T=1 B=1
    GDN_VERIFY broadcast=mod  maxrel out=1.555e-07 state=8.651e-08  MATCH
    GDN_VERIFY_DONE the port reproduces ggml's op within maxrel 1e-05

1.6e-7 relative is the authority's own f32 floor, so the recurrence, its state orientation, the
read-out after the rank-1 update, the `1/sqrt(S)` scale and the head geometry all now agree with
ggml's implementation at the real geometry. **The q/k→v broadcast is `mod`** (`h % H_k`) — what the
kernel source said, now established by data instead of assumed.

Phase-4 item 1's first unit is therefore done: a numpy reference for one Gated DeltaNet step, checked
against ggml's own op rather than against itself. The harness (`gdn_authority.c`, linking the same
libggml llama.cpp runs) plus `gdn_verify.py` is the reusable piece; `gdn_ref.py` still carries the
old transposed indexing and should be deleted or rewritten onto `gdn_verify.py`'s convention.

What this does NOT cover, and what follows: the multi-token/prefill path (the op's other kernel), and
the rest of the layer around the recurrence — the short convolution, the l2 normalisation, the gated
RMS epilogue and the projections — before a Vyb kernel and its gate can be written.

### Second unit: the short convolution, also verified against ggml's own op

`native/tools/conv_authority.c` runs `GGML_OP_SSM_CONV` from the same libggml; `native/tools/conv_verify.py`
is the port and the check. At the model's real width (`d_conv=4`, `d_inner=10240`, `n_t=5`, i.e. 51200
values):

    CONV_VERIFY n=51200 maxabs=5.971e-08 maxrel=7.606e-08 (authority scale 7.850e-01)
    CONV_VERIFY_DONE within maxrel 1e-06 of ggml's ssm_conv

So the causal depthwise convolution — `out[t,ch] = Σ_{j<4} s[t+j,ch] * w[j,ch]`, the reason the layer
carries the last three frames as state — matches ggml's implementation. Note the numpy shapes are
`(d_inner, ncs)` and `(d_inner, d_conv)` for exactly the ne0-fastest reason recorded above: reading
them the other way is what cost the earlier investigation.

Both novel ops of a recurrent layer are now checked against the real implementation. Next: the
l2 normalisation and the gated RMS epilogue (ordinary ops, and closer to the existing prefill path),
then the layer wiring, then the Vyb kernel and its gate.

### Third unit: the l2 norm, the RMS norm and the gated epilogue, verified against ggml's own ops

Read first-hand rather than taken from a summary, because the two variants differ by an ulp and a
summary is not evidence. `ggml/src/ggml-cpu/ops.cpp:4198-4206` (l2) and
`ggml_compute_forward_rms_norm_f32` (rms):

    l2 :  ggml_float sum = 0.0;  sum += (ggml_float)(xi * xi);   scale = 1.0f/fmaxf(sqrtf(sum), eps)
    rms:  ggml_float sum = 0.0;  sum += (ggml_float)(x[i00]*x[i00]);
          mean = sum/ne00;                                       scale = 1.0f/sqrtf(mean + eps)

So the two conventions are opposite, and both are easy to get wrong: for the l2 norm **eps is a FLOOR
applied AFTER the sqrt**; for the RMS norm **eps is INSIDE the sqrt**. In both, the products `x*x` are
f32, the accumulation is double, the narrowing to f32 happens once and the reciprocal is f32. A port
that puts eps inside the l2 sqrt, or accumulates in f32, disagrees exactly where the norm is small.

`native/tools/norm_authority.c` is one harness with three modes — `l2`, `rms`, and `epilogue` (the
epilogue builds `RMSNorm(output, ssm_norm) * SiLU(z)` out of the same ops the model's graph uses,
matching `qwen35.cpp build_norm_gated`); `native/tools/norm_verify.py` is the port and the check. At
the model geometry (`n_col = 128`, l2 over 16 k-heads × 4 tokens, epilogue over 48 v-heads × 4
tokens, `eps = f_norm_rms_eps = 1e-6`), and with a deliberately tiny-norm leading row so the two eps
conventions are distinguishable:

    NORM_VERIFY l2       n=8192  maxrel=0.000e+00   opposite convention 7.145e-02
    NORM_VERIFY rms      n=24576 maxrel=0.000e+00   opposite convention 6.603e-03
    NORM_VERIFY epilogue n=24576 maxrel=8.775e-08

The l2 and RMS ports reproduce the authority **bit-exactly** (the f32/double split above is the whole
of it), the epilogue is at the f32 floor (SiLU's `expf` against numpy's), and the opposite convention
is off by 7e-2 and 7e-3 — so the check can fail, which is what makes the zeros mean something. The
l2 norm is per head over 128 (`qwen35.cpp:440-443`) and the epilogue's weight (`ssm_norm`) is
per-channel `{head_v_dim}`; both ops normalise along `ne0` only, so every trailing dim of the model's
tensor is just another row in the harness.

**All three units of item 1's reference work are now pinned to the real implementation**, and they no
longer live only in the transcript: `native/legit/run_gdn_ops_gate.sh` runs all three verifiers and
`native/legit/run_phase2_battery.sh` carries it as step **P4.1** (SKIP without the llama.cpp
checkout, so a machine without one cannot quietly look green).

### Fourth unit: the five projections and the beta/alpha gates, verified against ggml's ops

`native/tools/mm_authority.c` (+ `mm_verify.py`) covers the rest of the layer's input path, at the 27B
Ridge geometry read from the GGUF's own metadata (`n_embd = 5120`, `head_k_dim = head_v_dim = 128`,
`n_k_heads = 16`, `n_v_heads = 48`, so `key_dim = 2048`, `value_dim = 6144`, `conv_dim = 10240`):

    wqkv      5120 -> 10240
    wqkv_gate 5120 ->  6144
    ssm_beta  5120 ->    48      beta  = sigmoid(mm)
    ssm_alpha 5120 ->    48      alpha = softplus(mm + ssm_dt) * ssm_a
    ssm_out   6144 ->  5120

Measured, against `ggml_mul_mat` / `ggml_sigmoid` / `ggml_softplus` called directly:

    MM_VERIFY mm wqkv      maxrel=4.714e-07   transposed-read 1.355e+00
    MM_VERIFY mm wqkv_gate maxrel=5.065e-07   transposed-read 1.344e+00
    MM_VERIFY mm ssm_beta  maxrel=4.599e-07
    MM_VERIFY mm ssm_alpha maxrel=4.304e-07
    MM_VERIFY mm ssm_out   maxrel=5.331e-07   transposed-read 1.084e+00
    MM_VERIFY beta         maxrel=2.742e-06
    MM_VERIFY alpha exact  maxrel=2.797e-07   (scale 2.591e+02, real ssm_dt/ssm_a)
    MM_VERIFY alpha log1p / no_threshold       inf

5e-07 is the f32 accumulation floor of a K = 5120 dot product, not slack. Three conventions were
pinned by this unit, and each has a rejection measurement to show the check bites:

* **`ggml_mul_mat(w, x)` is `w^T x`**, so a weight of ne (K, N) has the memory of a numpy (N, K)
  array. The harness was written the other way first and every projection came out at maxrel ~1.3 —
  the same axis lesson as the recurrence port, now caught by the same "mis-read buffer" check the
  verifier prints (transposed-read ~1.2, must be large).
* **`softplus` is `(x > 20.0f) ? x : logf(1.0f + expf(x))`** (ggml-cpu/unary-ops.cpp `op_softplus`),
  with the threshold and without `log1p`: the `log1p` port and the threshold-less port both diverge to
  inf on this input. The threshold is not a corner case for this model either — the real `ssm_dt`
  bias reaches 19.25, and 23 of 192 values in the check land in the threshold branch.
* **`ssm_a` is already the exponential** (llama.cpp names it `SSM_A_NOSCAN` and multiplies by it;
  qwen35.cpp comments it `-A_log.exp()`), so the port must not re-apply the exp. The verifier reads
  `blk.0.ssm_a` from the GGUF and asserts it is all negative; measured `[-3.376e-01, -3.839e-03]`.

So the layer's whole non-recurrent input path is now referenced: projections, both gates, and (from
units 2-3) the convolution, the two norms and the epilogue. What is left before a Vyb kernel can be
written is the layer wiring itself and the multi-token/prefill kernel noted below.

### Fifth unit: one whole recurrent block, wired and checked stage by stage

`native/tools/layer_authority.c` builds the block the way `build_layer_attn_linear` + the block loop
do — `attn_norm` → `wqkv`/`wqkv_gate` → `ssm_beta`/`ssm_alpha` gates → conv window + `ssm_conv` + SiLU →
the q|k|v views → l2 norms → `ggml_gated_delta_net` → gated epilogue → `ssm_out` → the attn residual —
out of the real ops, in one graph, and dumps 14 named stages. `native/tools/layer_verify.py` is the
reference: the four earlier units' ports, assembled in that order, compared **per stage**.

At the 27B Ridge geometry, T=1 (the sequential-kernel geometry), zero conv window and zero delta-net
state (a fresh sequence, i.e. first-token decode), with the model's real `ssm_dt`/`ssm_a`:

    xn         maxrel=0.000e+00        conv_silu  maxrel=2.915e-07
    qkv        maxrel=2.332e-07        q_norm     maxrel=2.582e-07
    z          maxrel=2.463e-07        k_norm     maxrel=3.696e-07
    beta       maxrel=1.951e-07        gdn_out    maxrel=3.985e-07
    gate       maxrel=2.208e-08        epi        maxrel=2.867e-07
    conv_in    maxrel=2.332e-07        y          maxrel=5.581e-07
                                       layer_out  maxrel=1.212e-07

Every stage is at the f32 floor, and the check has teeth: the same reference with the epilogue's
`SiLU(z)` factor dropped is off by 2.95e-01, and with the residual taken on the normed input instead of
the block input by 8.79e-02. Two classes of wiring mistake — a missing factor and a misplaced
residual — are therefore visibly rejected rather than silently tolerated.

What this does NOT cover, and what is left before a Vyb kernel: the FFN half of the block (unchanged
from the dense path already gated in phase 2), the recurrent path with a NON-empty state (this unit is
one step from a fresh sequence; carrying state across tokens needs the multi-token kernel below), and
the real-weight variant (the weights here are synthetic, deliberately — the op semantics are what is
under test).

### Sixth unit: the layer's kernels, on the GPU

`native/kernels/gdn.vyb` is the first Vyb code for the recurrent layer — the three pieces of a qwen35
linear-attention block that are not gemms:

    l2norm      q, k normalised along ne0 (128 per head), eps as a floor after the sqrt
    delta_step  the recurrence: one thread per head, no barrier and no shared memory
    norm_gated  the gated RMS epilogue: RMSNorm(out, ssm_norm) * SiLU(z)

`native/host/gdn_driver.vyb` loads `native/build/gdn.ptx` and runs all three on one token at the 27B
geometry; `native/tools/gdn_kernel_verify.py` writes the fixture, drives it, and compares **every**
result element, read back as the raw 64-bit pattern of each fp64 value (so nothing hides in the six
significant digits `Float.to_string()` prints — the lesson of the quant gates). Measured, 802816
elements over five sections:

    q_norm     2048    maxrel=3.494e-16        gdn_out    6144    maxrel=6.304e-11
    k_norm     2048    maxrel=5.235e-16        state_out 786432  maxrel=5.421e-11
    epi        6144    maxrel=1.609e-11

The reference here is **fp64, matching the kernels**, because the subject is the kernels rather than a
precision floor. (The pure-op gates do the opposite — they mirror ggml's f32 bodies — because there
the authority IS f32.)

The read-out axis is where this kernel was wrong first, and it is now the printed negative: the op's
read-out is `out = m @ q`, contracting q with the state's SECOND axis, so

    kq = sum_i k[i]*q[i];   out[j] = scale * (gexp * sum_i state[j,i]*q[i] + delta[j]*kq)

Written with the contraction on the first axis instead (the natural reading of "the state is [j,i], so
sum over j"), every output was off by maxrel 1.45e0 while `state_out` stayed correct to 5e-11 — the
per-stage comparison is what localised it in one run. The gate reports that alternative (and a
no-decay variant, 1.98e-01) on every run, so the check is visibly able to fail.

Gate: `native/legit/run_gdn_kernel_gate.sh`, step **P4.2** of the phase-2 battery; `make -f
native/Makefile gdn` builds the PTX and runs the verifier. SKIPs without a CUDA device or a Vyb
toolchain.

What unit 6 does NOT cover: one token from a state read off disk (no carry-over across calls, so the
state is exercised as maths but not as a running sequence), synthetic fp64 weights (the engine will
use quantized ones), the projections and the conv (existing gemm/quant kernels), and speed — one
thread per head is the simplest correct mapping, not a fast one.






With layout eliminated, the remaining untested piece was the q/k→v head broadcast — the one thing
ggml's header says it cannot yet select between (`ggml.h:2564`). Both candidates were implemented
and both fail against the op: interleaved `h % H_k` gives maxrel 1.36, tiled `h // (H_v/H_k)` 1.23.
So the broadcast is not the explanation either, and the port still disagrees at `H_v = 48`.

The next step is not another black-box probe: it is to **instrument the kernel's own intermediates**
(decay, the pre-update prediction, delta, and the first state row/column for head 0) for one call,
and compare those against the port's same quantities. That localises the divergence to a specific
line of arithmetic instead of to "the whole op", which is where five probes have now left it.

Also noted, not touched: `src/models/qwen35.cpp` in that checkout carries 27 uncommitted lines that
are not mine (the sibling `build-fastmtp` tree suggests MTP experimentation). They do not affect the
harness, which links only libggml, but anyone reproducing this should know they are there.

### Seventh unit: the whole block wired on the GPU, stage by stage against ggml

`native/host/gdn_layer_driver.vyb` builds one qwen35 linear-attention block out of the Vyb kernels —
`rmsnorm` (from the existing module), `mm_nt`, `sigmoid_k`, `alpha_gate`, `interleave_qkv`, `conv1d_k`,
`silu_k`, `l2norm`, `delta_step`, `norm_gated`, `add_k` — in the order `build_layer_attn_linear` uses,
and dumps 12 stages. `native/tools/gdn_layer_kernel_verify.py` feeds the same fixture to it and to
unit 5's authority (the same block built from ggml's own ops) and compares the two **stage by stage**.

Small geometry on purpose (n_embd=512, S=32, H_k=4, H_v=8, d_conv=4 — quick, and every stride > 1,
which is where the axis traps live), one token, state zero, the model's real `ssm_a`/`ssm_dt`:

    xn        1.109e-07      conv_silu 1.897e-07      gdn_out 2.095e-07
    qkv       1.818e-07      q_norm    2.978e-07      epi     2.966e-07
    z         1.245e-07      k_norm    1.076e-07      y       2.906e-07
    beta      8.905e-08      conv_silu/window         layer_out 8.149e-08
    gate      8.844e-08

2e-7 is the authority's own f32 floor (the kernels are f64), so this is agreement, and the run prints
its negatives too: the residual taken on the normed input is off by 7.85e-02 and with no residual at
all 9.45e-01, against a 1e-4 bar.

Three bugs the per-stage split named in one run each, all worth keeping in mind for the next layer:

* **the conv's axis order is the frame, not the channel.** `ggml_ssm_conv` takes ne `(ncs, qkv_dim)`
  with `ncs = d_conv-1+T`, so its memory is `frame + ncs*ch`, and the weight's ne `(d_conv, qkv_dim)`
  is `j + d_conv*ch` (numpy `(qkv_dim, ncs)` and `(qkv_dim, d_conv)`). Written channel-fastest, the
  conv was the first stage to diverge.
* **the projection cannot write the conv buffer directly.** `build_conv_state` gets its token frames
  by `ggml_concat` along ne0, which INTERLEAVES them per channel, so a plain qkv block is the wrong
  layout — hence `interleave_qkv`, and a conv buffer that starts as zeros.
* **a NaN must fail the gate.** The comparison used `r > MAXREL`, and every comparison with a NaN is
  False, so a NaN-filled stage printed DIFFERS and still let the gate report DONE. It is `not (r <=
  MAXREL)` now. (The NaN itself was a file-layout mismatch: the fixture's window section was
  `d_conv-1` frames where the driver read `ncs` frames, so the state upload landed past EOF.)

Gate: `native/legit/run_gdn_layer_gate.sh`, step **P4.3**; `make -f native/Makefile gdn-layer` runs the
verifier. SKIPs without a toolchain, libggml or CUDA.

### Eighth unit: two chained steps — the state carry-over a sequence needs

The seventh unit's driver now runs **two** decode steps and the gate compares both: step 2's conv
window is the last `d_conv-1` frames of step 1's qkv (the `conv_shift` kernel slides the buffer, the
interleave writes the new token) and step 2's delta-net state is step 1's new state (the two state
buffers swap roles each step, as a decode loop would). The authority runs the same sequence as two
chained single-token steps, so this is the carry-over checked against ggml's own ops, not against our
own loop. All 26 stages across both steps: `s1_*` and `s2_*` at 2.4e-08..2.4e-07, including
`state_out` (8192 values per step).

The trap here is not numerical: an 8192-byte upload into a 4096-byte device allocation
(`cuMemAlloc(loc(DX), NE*8)` left behind when the fixture grew to two tokens) silently clobbered the
neighbouring buffers and made the FIRST step read zeros — every stage then differed while the driver
reported no error at all. Device allocation sizes must move with the fixture's sizes; the symptom of
getting it wrong is wrong numbers, not a fault.

### Ninth unit: the layer check now runs at the model's own geometry, on real quantized weights

The layer check runs at 27B Ridge geometry (`n_embd=5120`, `S=128`, `H_k=16`, `H_v=48`, `d_conv=4`) and
`ssm_alpha` / `ssm_beta` are the model's OWN `blk.0.*.weight` Q8_0 tensors: read RAW from the GGUF at
the inventory's offsets, uploaded, and dequantised on the GPU by the existing `q8_0deq` kernel, with
the authority fed the same values dequantised in numpy. The Q8_0 math is separately gated (S0.5,
bit-exact against the `gguf` package), so what this adds is the pipeline: real GGUF bytes → GPU dequant
→ the layer's projections → the block's output.

    26 stages over two chained steps, worst maxrel 7.866e-07, gate wall clock 23 s

That is the authority's f32 floor over K = 6144 dot products, and the negatives still read
9.5e-01 / 1.1e-01 against the 1e-4 bar.

**The cost estimate that kept this dormant was wrong, and measuring it is what unblocked it.**
`native/host/mmnt_bench.vyb` times one full-size `5120 -> 10240` projection at 0.53 s including the
419 MB upload: a single-token step is a matrix-VECTOR product (M=1, ~52 M MACs), so the check is
I/O-bound and cheap. The ~270 GFLOP figure describes a PREFILL, where M is the prompt length — that is
the case that needs a tiled/gemm projection (and, if it grows, the shared DGX Sparks cluster rather
than this box's single 3090).

The model's F32 tensors came along for free: `attn_norm` and `ssm_conv1d` are read RAW and used
directly (no dequant). So four of the layer's weight sets are the model's own — `attn_norm`,
`ssm_conv1d`, `ssm_alpha`, `ssm_beta` — and the check still lands at 2.5e-07..4.8e-08 on those stages.

Still synthetic, and BLOCKED for a reason worth recording: the three remaining projections
(`blk.0.attn_qkv`, `blk.0.attn_gate`, `blk.0.ssm_out`) are all **Q4_K**, and Q4_K is the one quant type
this repo has no numpy reference for (`native/tools/` has `q5k_ref.py`, `q8_0_ref.py`, `iq2_s_ref.py`,
`iq3_s_ref.py`, `bf16_ref.py` — no `q4k_ref.py`) and no S0.x gate in the phase-2 battery. Using it here
without that would be the self-consistency the project forbids: the authority has to be fed the same
weights the GPU dequantised, and "our numpy agrees with our kernel" proves nothing. The `gguf` package
does implement Q4_K, so the S0.5 pattern applies directly — write `q4k_ref.py`, cross-check it against
the package, add the gate, and then the layer check can take the three remaining weights as data.

### `eng_gdn()` stays 0 — and why

The descriptor's `eng_gdn()` (native/config/model_caps.vyb:140) is what makes the caps gate report
`UNSUPPORTED_LAYER_KIND gated-deltanet` for a model with 48 recurrent layers. It is tempting to flip it
now that the layer runs on the GPU, and it would be wrong: the engine path (`model_driver`, the chat
server, the state cache) still has no recurrent layer, no multi-token/prefill path, and no weight
loading for a quantized model. Flipping it would make the descriptor claim a capability the engine
cannot deliver — the green-that-means-nothing this project's gates exist to prevent. What is left is
listed below; the flip is the last step of that list, not the first.

## The sequencing decision

The MTP head is smaller and would leave the Gated DeltaNet as the single refusal — a tidy
milestone, but on its own it buys no tokens, because a draft head without the trunk is useless. The
recurrent layer math is the piece that actually unblocks inference, and it is the larger one: a new
recurrence, a new state, and IMROPE touching the existing attention path.

Recommended: start with the recurrent layer math (item 1), because every other item is either
sizing (2), or means nothing without it (3). The risk to watch is that the reference is a graph
builder over ggml ops, not a standalone function — so the honest first step is to extract the
per-layer math from `llm_build_delta_net_base` into something a numpy reference can be checked
against, exactly as `ggml_dequant_authority.py` did for the quants.

## Unit 9c step 1 — the per-layer dispatch, from the tensor table (DONE, commit 3f34465)

Unit 9c is the engine integration, and its first requirement is the least glamorous one: the
per-layer loop has to know which branch a block takes. Getting that from the architecture name, or
from a list of block indices copied off this model, is the kind of assumption that runs the wrong
branch on the wrong block and still produces plausible numbers — 48 of the Ridge model's 64 text
blocks are recurrent, so most of the output would be wrong and nothing would say so.

The rule now lives in one place. `model_caps::mc_layer_kind(name)` returns **2** for a recurrent
block, **1** for an attention block, **0** otherwise, from the block's own weight names, and the
capability descriptor counts with it (`caps_tensors`), so the printed report and the engine's
dispatch cannot drift apart. `mc_layer_kinds(path)` returns the per-block plan.

The test is a SUFFIX match, and the two traps make that necessary rather than stylistic:

* `blk.N.attn_qkv.weight` and `blk.N.attn_gate.weight` carry the `attn_` prefix but belong to the
  RECURRENT block — a prefix test, or a `.contains("attn")`, classifies them as attention;
* `blk.N.attn_q_norm.weight` contains `attn_q` without being the attention marker, and
  `blk.N.ssm_norm.weight` is recurrent-only without being the recurrent marker.

So the two markers are the exact names `blk.N.attn_q.weight` and `blk.N.ssm_alpha.weight`. The probe
(`native/config/layerkind_probe.vyb`) carries a table of these names with the kind the dispatch must
report for each, and runs it before it opens a file. Measured teeth: replacing the suffix test with
`.contains(".attn_q")` fails two rows, and a rule that passes the table while double-counting real
tensors (a second `.ssm_out.weight` marker) is caught by the plan counts instead — 113 blocks and 96
recurrent where 65 and 48 are correct.

The plan is the real interleave, and it agrees with an INDEPENDENT parser (the Python inventory's
tensor names), per block, on both models: Ridge is 64 text blocks = **16 attention at `N % 4 == 3`
+ 48 recurrent**, Qwen3-4B is 36 attention + 0.

One fact worth carrying into the engine work, because it is exactly the sort of thing a summary
hides: **the tensor table classifies 65 blocks and only 64 are text.** Block 64 is the MTP draft head
and it carries `attn_q.weight` too. A kind-only count therefore reports "17 attention layers" for a
16-attention model. The descriptor's 16 is `n_layers - mtp_layers`; the engine's loop must run blocks
below `text_layers` and must not feed the draft head into the text attention branch.

`mc_layer_kinds` is deliberately THREE-VALUED, and collapsing the last two was the first draft's
bug: `""` is an unreadable table (a truncated download), `"-"` is a readable table with no
identifiable block (the mmproj tower), anything else is the plan. The tower read as a broken model
until the two were separated, and the gate now asserts they stay distinguishable.

Gate: `native/legit/run_layerkind_gate.sh`, step **P4.4** of the phase-2 battery. 9 cases, 0 skipped
on godzilla (the Ridge and Qwen3 GGUFs are present there; a checkout without them SKIPs those cases,
and a gate that proved nothing is a FAIL). The S0.2e capability gate is unchanged by the refactor and
still PASSes.

## Unit 9c step 2 — the block on the ENGINE's own weight path (DONE, commit cd78628)

P4.3 ran the block out of Vyb kernels, but on a fixture, and it multiplied with `mm_nt`, whose B
operand is [out,in]. The engine finds each tensor by NAME in the GGUF, dequantises the packed types
with the kernels it uses for every layer, and multiplies with `gemm` (layer.ptx), whose B operand is
[in,out] — the layout the dequant kernels' transposing write produces when their `z` argument is the
tensor's IN dim (VybForge#11). Those are different staging paths and a different matmul, so P4.3's
green says nothing about the engine's. This unit runs the engine's path for one recurrent block and
compares it against the same authority on the same real weights.

`native/host/gdn_engine_driver.vyb` stages all ten of blk.0's tensors from the live GGUF —
`attn_qkv`/`attn_gate`/`ssm_out` Q4_K and `ssm_alpha`/`ssm_beta` Q8_0 through `q4kdeq`/`q8_0deq`,
`attn_norm`/`ssm_conv1d`/`ssm_norm`/`ssm_a`/`ssm_dt` F32 through `f32expand` — runs the chain with
the engine's `gemm` for the six projections and the engine's `rmsnorm`/`resid`, carries the conv
window and the delta-net state across two decode steps, and dumps the same 13 stages. Its geometry
and strides are DERIVED from blk.0's own tensor numels and cross-checked against each other, so
nothing is spelled out that the file can answer.

`native/tools/gdn_engine_verify.py` reuses P4.3's fixture and authority runner verbatim: the
reference is unchanged, the driver under test is what changed. Measured on the real weights, two
chained steps from a zero state: **26/26 stages within maxrel 1e-4, worst 3.903e-06**, including the
786432-element delta-net state at 4.1e-07.

The engine's tensor index for Ridge is generated rather than hand-written:
`native/tools/inventory_to_tsv.py` projects `ridge_inventory.py`'s TSV into the engine's
`name / dims / type / off=<absolute> / size=0` form and asserts every tensor's byte count against its
type before writing, so a change in the alignment rule cannot quietly shift the weights.

### The bug this check caught — and why stage comparison alone could not

`delta_step` and the `ssm_out` projection gemm SHARE the `P2` param block. `delta_step` needs nine
words there and writes a POINTER into `P2+48` (the `stout` state buffer) — exactly where `gemm` reads
`alpha`. The projection therefore ran with alpha = the bits of a device pointer reinterpreted as f64,
and the whole `y` vector came back as denormals (~1e-308).

What made it instructive is how it READ: not as a wrong projection but as an almost-perfect one.
`epi`, which feeds it, agreed to 1.5e-06; the state, the norms and the conv agreed to 1e-7; and the
reported failure was "one index differs, maxrel 1.000e+00". A maxrel of exactly 1.0 is the signature
of "the output is all zeros": the difference vector is the reference itself, so the argmax of |diff|,
and the index it names, are arbitrary. The tell was counting, not reading — 5120 of 5120 slots were
denormal, not 1.

The lesson, in the form the next session needs it: **a param block is not an argument list.** The
kernel reads whatever words are at the offsets it reads; nothing tells you when a second kernel has
reused them. Two defences are now in place here: the alpha/beta words are re-stated immediately
before that gemm (with the comment saying why), and the verifier compares the STAGED operand slot by
slot at the addresses `gemm` reads — because a block that is right about everything except what it
was fed is invisible to a stage-by-stage comparison.

## Unit 9c step 3 (started) — the block moves INTO the engine's own driver (gate P4.5)

The step-2 driver was a side harness: it re-parsed the tensor index, had its own copy of the staging
helpers, and existed only to be verified. That is the wrong home for code the loop has to run, and a
second copy of a chain is a second thing to keep true. So the block moved into
`native/host/model_driver.vyb` — the file the layer loop lives in — behind a probe selector:

    VYB_GDN_PROBE=<layer>   run ONE recurrent block and return (verification)
    unset                   run the model (unchanged)

In probe mode the ENGINE's driver stages blk.0's ten tensors from the live GGUF through the engine's
own path (found by name in the tensor index; `q4kdeq`/`q8_0deq`/`f32expand`; transposing writes so
`gemm` sees B as [in,out]), runs the chain with its own `gemm`/`rmsnorm`/`resid` plus the gdn.ptx
ops, carries the conv window and the delta-net state across two chained decode steps, and dumps the
same 13 stages. `native/tools/gdn_engine_verify.py` now drives THAT, and the standalone driver is
deleted. Measured: **26/26 stages, worst 3.903e-06** — identical to step 2, which is the point: the
same block, in the file that will run it.

Landed with it: `gdn.ptx` and `q8_0.ptx` are loaded in the engine's CUDA setup with their ten
function handles; `put_i`/`put_f`/`dump_section`/`dump_slot` are helpers in the engine file; and
`VYB_MODEL`/`VYB_TSV` override the model and its tensor index, so the same driver can be pointed at
another GGUF without editing it (the probe needs the Ridge index; the dense path keeps its default).

What is NOT done, and is the rest of step 3: the loop still stages `attn_q/k/v/output` and calls
`run_layer` for every block. The dispatch (`mc_layer_kind` per block, which `model_caps` already
answers, P4.4), the per-layer state cache (48 x (40960 conv + 786432 delta-net) x 8 B), and the
`run_layer` split into `run_layer_attn` + `run_ffn` are the remaining three. Then the whole-model
logits against llama.cpp — which waits on two things outside this unit: IMROPE for Ridge's attention
layers (`rope.dimension_sections [11,11,10,0]`; the engine's rope is pairs-only) and the multi-token
prefill path for recurrent layers (only the sequential T=1 kernel is characterised so far).

## Unit 9c step 3 — the layer loop dispatches, and the block is inside it (DONE for the loop)

The engine's layer loop no longer assumes every block is attention. It asks `mc_layer_kind` per block —
the same rule the capability descriptor counts with (P4.4) — and for a recurrent block runs the Gated
DeltaNet mixer: the ten mixer weights staged from the GGUF by name through the engine's own path plus
the block's FFN weights, the mixer run ONE TOKEN AT A TIME (the sequential rule, which is what a decode
step is), the conv window carried in place by `conv_shift` and the delta-net state carried in a
per-layer cache slot, and the block's residual written into the SAME `X1` the attention branch writes.
Both branches then feed one shared FFN: `run_layer` became `run_layer_attn` + `run_ffn`.

The state cache is allocated once for every text block — conv windows, and two delta-net slots per
block used alternately so the in/out swap is a pointer rather than a 6 MB copy — and cleared with one
`cuMemsetD8_v2` per buffer. `VYB_GDN_PROBE=<layer>` restricts the loop to a single block with its input
read from the fixture, so the gate now verifies THE LOOP's own dispatch, staging, cache and chain
rather than a copy of them: 26/26 stages, worst 3.903e-06 over two chained tokens, unchanged.

### Three bugs a hybrid model found that a dense one could not

* **Buffer sizing from block 0.** The attention buffers were sized from `blk.0.attn_q/k/v/output` — but
  in a hybrid model block 0 is RECURRENT and carries none of them, so every attention buffer took a
  missing tensor's size and the first allocation failed (main returned 11 before any work). Sizing now
  uses the first ATTENTION block as the template, with sane defaults for tensors an architecture names
  differently.
* **`post_attention_norm`.** qwen35 names the FFN's pre-norm `post_attention_norm`; qwen3 calls it
  `ffn_norm`. A name-based staging call fails outright on the one that is absent, so the recurrent
  branch resolves either name and stages by index.
* **A shadowed variable that zeroed every size.** An activation buffer named `GQKV` shadowed the
  geometry variable of the same name; every size computed from it became 0 and the allocation of a
  419 MB weight buffer was the first casualty. Separately, the 410 MB state cache was being zeroed by
  the host-side `download` helper, which moves 8 bytes per device call — 51 M calls, minutes of wall
  clock — instead of one `cuMemsetD8_v2`.

The lesson worth keeping: **a hybrid model exercises the parts of a loader that a dense one leaves
dormant**, and those parts are exactly where the assumptions live (`blk.0` is an attention layer; the
FFN's norm is called `ffn_norm`). Both assumptions had been true for 36 dense layers.

### What this does NOT yet do

The sequential kernel is the decode rule, so a recurrent block is correct at T=1 per token and slow for
a long prompt; llama.cpp's chunked prefill kernel is still uncharacterised. The state cache is correct
within one forward pass and does not yet persist its slot parity across passes (there is no session
state in the engine — a chat server would need one before a second token can follow a first). And the
gate verifies one block (blk.0), because the authority side of the fixture is built from blk.0's
weights; a second block needs that parameterised. `eng_gdn()` stays 0 until the whole-model logits are
compared against llama.cpp — which also needs IMROPE for Ridge's attention layers and its `attn_q`
output gate, both attention-path work.

### The fourth bug, and the one worth remembering: a classifier is not an existence test

The dispatch started as `mc_layer_kind("blk." + L + ".ssm_alpha.weight") == 2` — which is the shared
rule, applied to a constructed name. That is not a dispatch: `mc_layer_kind` answers "what kind is this
NAME", and every constructed name ends with the marker, so it answered `recurrent` for every block of a
DENSE model. The engine stopped on Qwen3-4B with `GDN_CFG_FAIL the first recurrent block blk.0. is
missing a tensor`, before it had done any work.

The rule that came out of it, now written at the three sites: **presence comes from the tensor table,
the kind comes from the rule.** `find_nm(nmv, name) >= 0` says IF; `mc_layer_kind(name)` says WHAT. The
loop also refuses loudly when a block carries neither marker, rather than quietly treating it as
attention.

What makes it worth a section is how it survived two green gates. P4.5 passed twice while the dense
path was broken, because the probe only exercises the recurrent branch — a check of the NEW path says
nothing about the OLD one. The check that caught it was the regression on the model that does NOT have
the feature, and the reason to run it on every change to a shared dispatch. Note also the failure mode
of the previous regression: with the driver exiting early, `verify_prefill.py` compared the PREVIOUS
run's outputs and reported OK — so a regression gate must assert the driver's own completion marker
(`MODEL_PREFILL_DONE`), not just the final comparison.

## Unit 10 reconnaissance — Ridge's ATTENTION layer, read from the reference

The recurrent side is verified standalone; the attention side of a hybrid block is what stands between
here and a whole-Ridge run, and the phase-4 note has been calling it "IMROPE" since phase 3. It is more
than that, and less. This is the graph, read out of `~/Projects/llama.cpp` at `4df29be`
(`src/models/qwen35.cpp`, `build_layer_attn`, and the rope op it calls), plus the semantics of the rope
from `ggml/src/ggml-cpu/ops.cpp`.

### The graph (order matters)

    Qcur_full = wq @ cur                 // ONE joint projection: (2 * head_dim) * n_head wide
    Qcur      = view of the FIRST head_dim of each head   (stride 2*head_dim per head)
    Qcur      = RMS norm over head_dim, per head          (attn_q_norm: 256 values)
    Kcur, Vcur = wk @ cur, wv @ cur
    Kcur      = reshape (head_dim, n_head_kv, tokens) then RMS norm per head
    gate      = view of the SECOND head_dim of each head, made contiguous  (head_dim * n_head)
    Qcur, Kcur = rope_multi(Qcur), rope_multi(Kcur)       // MRoPE, see below
    cur       = attention(Qcur, Kcur, Vcur)               // kq_scale = 1/sqrt(head_dim)
    cur       = cur * sigmoid(gate)                       // the OUTPUT GATE
    cur       = wo @ cur

So `attn_q.weight` is `5120x12288` because it is Q **and** the gate in one matrix, **interleaved per
head** — head h contributes dims [2h*256, 2h*256+256) to Q and [2h*256+256, 2h*256+512) to the gate — not
"all Q then all gate". Ridge's numbers: n_head 24, n_head_kv 4, head_dim 256, so n_rot dims are rotated
only in the first 64 of each head.

### The rope: "IMROPE" collapses for text, and the rotation may already exist in this repo

`rope.dimension_sections = [11, 11, 10, 0]` is MRoPE (the Qwen2-VL/Qwen3-VL scheme) and `is_imrope`
selects the interleaved assignment: for pair index `sector = (i0/2) % 32`, `sector % 3 == 0` uses the
temporal position, `1` the height, `2` the width, otherwise the extra position. **For a text-only model
all four position ids are the same token position**, so every branch yields the same theta and the
sectioning degenerates: the cache becomes the ordinary geometric progression over the head dims. The
section code matters only for the vision tower.

What does NOT degenerate, and is the real work:

* only the first `n_dims = 64` dims are rotated; dims 64..255 are copied through unchanged
  (`ggml/src/ggml-cpu/ops.cpp`, the `!is_vision` pass-through loop);
* the rotation is the NEOX pairing (`rotate_pairs(n_dims, n_dims/2, ...)`) — pairs (i, i + 32) — which is
  the convention the engine's `qwen3rope` already implements, over a parameterised dim count;
* freq_base 1e7, and `kq_scale = 1/sqrt(256)` unless the GGUF sets `f_attention_scale`.

So the likely finding, to be MEASURED rather than assumed: Ridge's attention rope is the engine's
existing rope kernel with `n_rot = 64` and `head_dim = 256`, not a new kernel. `rotate_pairs`'s exact
index arithmetic is not something to reconstruct by reading (its `ic = i0/scale` comment and the
`n_offset` argument interact in a way that invites an off-by-one); the authority harness settles it, as
it settled every other op in this phase.

### The plan that follows (unit 10, in the repo's usual order)

1. **Rope first, smallest unit.** A harness that links the built libggml and calls `ggml_rope_multi` with
   Ridge's own parameters (n_dims 64, sections [11,11,10,0], mode IMROPE, freq_base 1e7) on known input,
   plus a numpy port compared against it — the `mm_authority`/`norm_authority` pattern. If the engine's
   rope kernel already reproduces it at `n_rot = 64`, this unit ends with a gate, not a kernel.
2. **The attention block authority.** A C harness building the graph above out of ggml's own ops
   (mul_mat, rms_norm, rope_multi, attention, mul/sigmoid, mul_mat) and dumping stages — `layer_authority`
   for a hybrid attention layer, including the q/gate interleave and the output gate.
3. **The engine-side branch.** `run_layer_attn` gains the joint-QG split, the per-head interleave, the
   parameterised rope and the output gate, verified against (2) stage by stage.
4. **Then the whole-model logits against llama.cpp.** `eng_mtp()`/the 65th block still has to be dealt
   with (or the descriptor needs a documented profile without the draft head), and the state-cache
   session question from unit 9c remains open.

`eng_gdn()` stays 0 until step 4 lands: the recurrent block is verified standalone, but the engine has
never produced a token from this model.

## Unit 10 step 1 — the rope spec, MEASURED (gate P4.6), and a correction to the reconnaissance

The reconnaissance above reasoned that the MRoPE sectioning "collapses for a text model". The
measurement says otherwise, in a way that matters, so it is recorded as a correction rather than a
refinement.

`native/tools/rope_authority.c` links the libggml llama.cpp runs and calls `ggml_rope_multi` with
Ridge's own parameters; `native/tools/rope_verify.py` then asks which (mode, rotation layout) pair
reproduces it. Measured on the text case (all four MRoPE positions the same):

    mode=neox    neox_first_nd      8.993e-08  MATCH
    mode=neox    swizzle_2k_in_nd   1.332e+00  differs
    mode=neox    adjacent_in_nd     1.287e+00  differs
    mode=neox    neox_all_dims      1.478e+00  differs
    mode=mrope   (all four layouts) 1.01e+00 .. 1.689e+00  differ
    mode=imrope  (all four layouts) 1.01e+00 .. 1.478e+00  differ

Three things follow, and two of them correct what this document said earlier:

1. **The modes are not interchangeable even for text.** `ggml.h` states it plainly: the sections do
   not change the theta per dim ("idx used for theta: [0..n_dims/2], not reset for each section") —
   they change WHICH DIMS ARE PAIRED (`MROPE: [ttttyyxxttttyyxx00]` vs `IMROPE: [ttyxttyxttyxttyx00]`).
   Equal positions therefore do not make IMROPE and MROPE agree, and "the sections collapse" was
   wrong.
2. **For this architecture the sections are inert anyway.** `llama_model_rope_type` gives qwen35 the
   NEOX mode (it sits with qwen3next), ggml's `mrope_used = mode & MROPE` is then false, and the
   `[11,11,10,0]` metadata never reaches the kernel. So Ridge's attention rope is ordinary rope.
3. **It is still a real piece of work, not a parameterisation.** The spec: rotate ONLY the first
   `n_dims = 64` dims of each 256-dim head with NEOX pairing inside them — pairs (k, k + 32) — and
   pass dims 64..255 through. The engine's `qwen3rope` rotates every head dim with pairs
   (i, i + HD/2); that is the `neox_all_dims` candidate, measured at 1.478, i.e. wrong for Ridge. The
   kernel needs a variant (a `n_rot` bound and a pair offset of `n_rot/2`), and the gate has teeth:
   eleven alternatives are rejected in every run.

What the earlier reconnaissance got right and is worth keeping: the joint Q+gate projection
(`attn_q.weight` is `5120x12288`, interleaved per head) and the output gate `attn * sigmoid(gate)`
before `wo` — that remains the substantial attention-path work, and it is where the next unit starts.
The lesson about method is the one this document keeps relearning: a harness that runs the real op
settles in one run what reading the source gets wrong, and the wrong reading here was mine.

## Unit 10 step 2 — the rope variant kernel: plumbing proven, rotation still WRONG (gate P4.7 RED, not wired)

`native/kernels/rope.vyb`'s `rope_nrot` (NEOX inside the first `n_rot` dims, the rest passed through)
runs on the GPU through `native/host/rope_driver.vyb`, and `native/tools/rope_kernel_verify.py` compares
it against the P4.6 authority. **It does not match yet** — this is recorded as a red gate rather than
wired into the battery, so nobody reads a green battery as covering it.

What IS established:

* The kernel compiles and runs, and the comparison machinery is real: the verifier runs the kernel at
  `n_rot = 64` (must match) AND at `n_rot = HD` (must not), so a pass cannot come from a check that
  ignores the parameter.
* **The fixture plumbing is sound, and that is measured, not assumed.** Running the driver at
  `n_rot = 0` — a pure pass-through — dumps values whose i64 bit patterns decode exactly to the
  fixture's own inputs (4602070306595536896 = 0.4662207755…, the fixture's q[0]). Upload → f32→f64
  expand → kernel → download → dump is therefore correct, and the defect is inside the rotation.
* One real defect found by reading the kernel against ggml's `rotate_pairs` and fixed: a thread sitting
  on the SECOND element of a pair must compute `x[i-half]*sin + x[i]*cos`; the first draft had the two
  multiplied the other way round (a transposed rotation).
* That fix did not clear it: the output is still ~1e18, and identical for `n_rot = 64` and
  `n_rot = 256`, which looks like the q buffer never being written rather than a wrong angle. Two
  leads for whoever picks this up: (1) the `n_rot = 256` run reads `FR + (i % 128)*8` = up to 5 KB out
  of a 1 KB `DFR` allocation — that run is definitively out of bounds and probably corrupts the
  context, and its garbage may be a false lead for the `n_rot = 64` case; (2) rule the crash out by
  launching the rotation on a single head and reading back, i.e. narrow with a fiducial the way the
  `n_rot = 0` run narrowed the plumbing.

The kernel stays SEPARATE from `qwen3rope` until it is green: that one is on the verified dense path,
and merging them while this one is wrong would put the dense regression in the position of testing the
change instead of the behaviour.

### Correction to the paragraph above: the kernel is RIGHT, my PARSER was the bug

The red run recorded above diagnosed the kernel's rotation as still wrong. That was wrong, and the way
it was wrong is worth keeping.

`rope_kernel_verify.py`'s dump parser built its section arrays with `np.array(buf, dtype="<f8")`. That
CONVERTS the integers to floats — so a dumped `4602070306595536896` became the float
4.602070306595537e+18, and the `frombuffer(..., dtype="<f8")` meant to reinterpret the raw i64 bit
patterns then read back those already-converted bytes. The reinterpreting step never reinterpreted
anything. Every "garbage" reading was arithmetic on integer values dressed as doubles, which is why the
error was ~1e18 and identical at both `n_rot` values: the comparison was measuring my parser, not the
kernel. The decisive fiducial was the cheap one — run the driver at `n_rot = 0` (a pure pass-through)
THROUGH the verifier's own parse path; it returned 4.602070306595537e+18 where the fixture holds
0.46622076630592346. The plumbing had been cleared earlier with a manual decode of the printed values,
which is exactly why the bug hid: the printed values were right and the parsed ones were not.

With the parser keeping the raw int64 patterns (`dtype="<i8"`, then one deliberate reinterpretation):

    ROPE_KERNEL_VERIFY n_rot= 64 q maxrel=8.903e-08 k maxrel=6.969e-08 MATCH
    ROPE_KERNEL_VERIFY n_rot=256 q maxrel=1.703e+00 k maxrel=1.237e+00 differs

So the transposed-rotation fix WAS necessary (a transposed pair is an O(1) error, and it was hidden
under a 1e18 parser artifact), and the kernel now reproduces ggml's rope at the f32 floor. Gate P4.7
(`native/legit/run_rope_kernel_gate.sh`, wired into the phase-2 battery): PASS.

The lesson, which is the same one twice over: when a measurement is absurd, suspect the instrument
before the subject — and use the cheapest fiducial that separates them (here, a pass-through launch
through the same parse path).

Lessons for the skill: a dump parser must keep the dumped int64 BIT PATTERNS intact until the single
reinterpretation (an intermediate dtype="<f8" silently converts); and a pass-through/no-op case is the
right fiducial for any load→compute→store→dump chain.
