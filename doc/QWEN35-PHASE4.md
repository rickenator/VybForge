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

Next, outward: **the layer wiring** — assembling conv → l2 norm → recurrence → epilogue → projections
in the order `build_layer_attn_linear` uses, with the residual and the two norms
(`attn_norm` / `attn_post_norm`) around it, checked end to end against a captured layer. Then the Vyb
kernel + driver + gate, and last the `eng_gdn()` flip.






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
