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
