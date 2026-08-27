# Native GPU inference substrate (VybForge)

The handoff slice (HANDOFF-VYB-NATIVE-INFERENCE.md) — Vyb-native decode of the
Qwen3-4B configurator on the RTX 3090, with **zero Python in the production
pipeline**. Python is used only for reference *verification*, never at runtime;
every GPU kernel and the reference checks live in Vyb.

This directory is the P0 substrate that was blocked by the two language gaps
(#202 numeric casts, #203 device intrinsics). Both landed at `rickenator/Vyb`
`55593fc`; `build/vyb` from that commit is the toolchain.

## What is verified on-GPU (all in-Vyb, host reference vs GPU readback)

| Piece | Kernel | Result |
|---|---|---|
| Reusable GEMM `C=αAB+βC` (f64, params-buffer launch, multi-block) | `kernels/gemm.vyb` | `GEMM_OK` bad=0 maxerr=0 |
| RMSNorm (native Newton sqrt) | `kernels/rmsnorm.vyb` | `RMSNORM_OK` bad=0 |
| Native `exp` / `sin` / `cos` (pure arithmetic) | `kernels/vmath.vyb` | `VMATH_OK` exp<2.6e-10 rel, sin/cos<6e-12 abs |
| **ONE transformer layer forward** (RMSNorm → GQA/RoPE → causal softmax attn → o_proj → residual → RMSNorm → SiLU MLP → residual) | `kernels/layer.vyb` | `LAYER_VERIFY: OK` — max diff ~5e-5 vs numpy ref at every stage (attn 5.3e-6) |
| **GGUF v3 reader** (header, metadata KV, tensor index; little-endian binary parse) | `gguf/parse_gguf.vyb` | `GGUF_PARSE_VERIFY: OK` on synthetic fixture |
| **Q4_0 dequant on-GPU** (GGUF tensor data → device mem → `deq_q4_0`) | `gguf/dequant_gguf.vyb` | `DEQUANT_OK` bad=0 (32/32) |
| **Generic JSON value parser** (object/array/string/num/bool/null tree) | `json/json_parse.vyb` | verified: parses mock-system.json + vocab slice |
| **Qwen3 BPE tokenizer** (real 151k vocab + merges, Qwen2 pre-tokenizer) | `tokenizer/tokenizer.vyb` | EXACT match 13/13 vs transformers (words, contractions, apostrophes, punctuation, whitespace) |
| **Multi-layer stack** (L× weight-tied decoder, double-buffered) | `host/stack_driver.vyb` | `STACK_VERIFY: OK` at L=4 (~8e-7 rel vs numpy) |
| **Greedy autoregressive decode** (embed→stack→lm_head→argmax, token loop) | `host/decode_driver.vyb` | `DECODE_VERIFY: OK` (generated ids == numpy) |
| **Stochastic sampler** (temp + top_k + top_p + seeded LCG) | `sampler/sampler.vyb` | `SAMPLER_VERIFY: OK` (kept set/probs/12 draws exact vs numpy) |
| **`tensor::` wrapper module** (CUDA ctx / PTX module-load / dev buffer / H2D–D2H / gemm + rmsnorm via single-arg `cuLaunchKernel`) | `tensor/tensor.vyb` | `TENSOR_GEMM_AND_RMSNORM_OK` — both bad=0, exact vs in-Vyb ref |
| **G-decode end-to-end slice** (prompt → encode → on-GPU decode → detokenize → agent-response contract) | `gdecode/run_pipeline.sh` | `CONTRACT_VERIFY: ALL_OK` + round-trip OK (`make gdecode-pipeline`) |

The handoff **P0 go/no-go gate** (single layer) plus the **full vertical slice**
(substrate → GGUF/JSON/tokenizer loaders → multi-layer stack → autoregressive
decode → **stochastic sampling**) all run Vyb-native on the RTX 3090 and are
reference-verified. Remaining: G-decode (config-contract emission) and the rest
of the `tensor::` wrapper (rope/attn/silu/resid ops moved behind it). See
`ROADMAP.md`.

## Build & run

```sh
# from repo root
make -f native/Makefile verify      # compile all kernels + run all drivers
make -f native/Makefile gemm        # GEMM only
```

Kernels must be compiled to PTX with `vyb --kernel` (sm_86) before the host
driver runs; the Makefile handles that. The host driver is plain Vyb that FFI's
to libcuda inside a `freedom` block.

## Launch pattern (the 4-arg limit)

`cuda_launch4i` passes exactly four `Int` args. Multi-tensor kernels therefore
dereference a **packed params buffer** in device memory: host packs
addr/dim/scalar words at fixed byte offsets, copies it to a device buffer, and
passes only that base pointer (plus a few inline scalars). `gemm.vyb` shows the
layout (`A,B,C,M,N,K,alpha,beta` at offsets 0..56).

## Hard-won kernel-mode gotchas (all empirically hit)

- **libm builtins don't reliably lower to device code.** `sqrt` sometimes folds
  to `sqrt.rn.f64` but can also be emitted as an unresolved device extern that
  ptxas rejects; `exp`/`sin`/`cos`/`tan` are *always* unresolved externs. Do not
  use them in kernels — implement in pure `+,-,*,/` (Newton sqrt, polynomial
  exp/sin/cos). Every one here is a plain-arithmetic device function.
- **`X * (Y)` is parsed as a pointer type** (`*` followed by a parenthesized
  group ⇒ `Y` is read as a type to point to: `ptr<X>`), which **silently nulls
  the expression** — the kernel compiles but writes garbage/zero. Happens in
  host `freedom` blocks *and* inside kernels. Always expand to a plain multiply
  chain: `s * (H * HD)` ⇒ `s * H * HD`; compute `rtot`-style sums without
  parenthesized products. This one cost a full layer of debugging.
- **Deeply-nested Horner expressions evaluate to `0`/null** in the runtime.
  Write polynomials as explicit power-sum terms (`p = 1.0 + r*c1 + r2*c2 + ...`),
  not `r * (c1 + r * (c2 + ...))`. This silently zeroed `vexp`.
- **Vyb float literals don't accept scientific notation** (`1e-6` ⇒ parse
  error) — write `0.000001`.
- Ranges are **inclusive**: `0..n-1` for n iterations.
- Fixed-array element **write** is unsupported (#201) — build buffers with
  `Vec` + `push`, or via the device `ld/st` kernels.
- First `cuModuleLoadData` of a non-trivial kernel in a fresh process returns
  `NO_BINARY_FOR_GPU (218)` once; reload from a *fresh* String (both test
  drivers show the pattern).
- `share(all)` functions in a module; import flat modules with the braced form
  `import mod::{a, b}` and `--module-path native/kernels`.

## Vyb vs python reference note

End-state goal (per Rick): the production pipeline must contain **no
Python/torch/Ollama** — Vyb runs the model on-GPU. Python is acceptable only as
a reference-verification harness, which is why the drivers compare against
host libm (`exp`/`sin`/`cos`) where convenient. The GPU kernels themselves never
call into Python.
