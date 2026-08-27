# Native GPU inference substrate (VybAIConf)

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

The single-layer forward is the handoff's **P0 go/no-go gate**: a real decoder
layer (embed→RMSNorm→QKV→RoPE→GQA causal-softmax→o_proj→residual→MLP SiLU→
residual) runs fully on the RTX 3090 in Vyb and matches a reference. Next: the
G4 loaders (GGUF/Q4 base reader, Qwen3 BPE tokenizer, JSON value parser) then
stacking to 36 layers + the config-contract decode.

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
