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

### What is still unexplained, and the next step

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
