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

At model geometry the comparison fails, and not by rounding. The isolating probe: feed `H_v = 2`
with **identical** memory for both heads (state, v, gate, beta all repeated along the head axis).
Two consequences, both reproducible:

1. the op's two head outputs are NOT equal to each other, and
2. the op's head 0 does NOT match the port's head-0 result, whereas the same head-0 computation at
   `H_v = 1` matches exactly.

Identical input slices cannot produce different results under the layout that
`build_delta_net_fused` asserts (`s->ne[0]==S_v && s->ne[1]==S_v && s->ne[2]==H_v && s->ne[3]==n_seqs`,
`delta-net-base.cpp:399`) and that the kernel's addressing implies (`(iv3*H + iv1)*S_v*S_v`). So the
op's per-head addressing differs from the documented layout in some way not yet identified — the
candidates are the state's head axis, the v-head/k-head broadcast, or a head count the op derives
from somewhere other than `V->ne[1]`. This must be settled before the reference can be trusted at
model geometry; guessing would be worse than the blank.

Next probe when this resumes: hold everything fixed and vary ONE head-dependent input (state vs v
vs gate) with a per-head multiplier, and read which op output scales with it — that identifies the
op's actual head slice. Only then does the port's geometry mapping get a defensible answer.

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
