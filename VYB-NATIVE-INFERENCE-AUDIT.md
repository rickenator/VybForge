# Vyb-native inference of the Qwen3-4B VybOS configurator — capability audit & requirements

**Date**: 2026-08-26 · **Audit by**: Hermes (Vyb observer/reviewer) · **Status**: FOUNDATION PRESENT; language blockers G2 (cast) + G3 (device intrinsics) **RESOLVED** (`55593fc`); remaining G1 (tensor/cuBLAS) + G4 (GGUF/tokenizer) + G5 (fp16) are **implementation work → see `HANDOFF-VYB-NATIVE-INFERENCE.md`**

## 0. Goal

Replace the Python/torch stack in `VybAIConf` with a **Vyb-native** pipeline that
loads the already-trained **Qwen3-4B + LoRA** configuration model (the
`artifacts/vybos-configurator-lora*/adapter_model.safetensors`, rank 16 / alpha 32 on
all of `q,k,v,o,gate,up,down` over base `Qwen/Qwen3-4B`) and runs it on the GPU, via
the CUDA kernel foundation from `#198`. Developers run it locally to produce the
schema-driven VybOS `SystemSpec` config-JSON contract (`{path, op, value, reason}`),
the same contract the deterministic `tools/apply.vyb` backbone already applies.

This is the "framework written in vybey" end-state: model I/O, tokenizer, transformer
layers, GEMM, KV cache, sampling, and the config contract all live in Vyb; no torch,
no Ollama, no hosted model required.

## 1. What the target pipeline needs (decode-only; training is a later milestone)

```
GGUF/Q4 (Qwen3-4B base, or safetensors-fp16) + LoRA adapter (r16) + tokenizer (BPE)
   → embed
   → x 36 layers:
        RMSNorm → Q,K,V (LoRA on q,k,v,o) → RoPE → GQA KV-cache →
        Q·K^T (softmax/scale/causal) → ·V → o_proj(+LoRA o) → residual
        RMSNorm → MLP gate·up→SiLU→down (+LoRA gate/up/down) → residual (+LoRA)
   → final norm → lm_head logits
   → sampling (top_k/top_p/temperature) → next token
```

Per-token decode; KV cache grows with sequence length (config interviews are short,
so ~1–4K tokens is plenty).

## 2. What Vyb already has (grounded, verified)

| Need | Vyb surface | Staus |
|---|---|---|
| Launch arbitrary C (+CUDA) from Vyb | `freedom` blocks + `extern "C"` FFI; optional-C-lib dlopen hook; `bindings/` (sqlite, libgit2, cuda) | ✅ (#174/#197/#198) |
| Compile Vyb → launchable device kernel | kernel mode (`--kernel`): `.visible .entry` for void fns, `tid/blk/dim` sregs, `ld/st` global mem | ✅ (#198 P4) |
| GEMM / matmul in a kernel | Vyb-authored tile matmul fixture (register-tiled, verified vs CPU) | ✅ (primitive proof; not a library) |
| Narrow ints / Float32 / `as` casts | `Int8/16/32`, `UInt8..64`, `Float32`, `as` widening/narrowing, bit ops, byte-packing | ✅ |
| Raw byte buffers | `Bytes` (fat-pointer block); `io`, `archive` (gzip+tar); `http/https` (verified TLS) | ✅ |
| Generic collections | `Vec<T>`, `HashMap`, `BTreeMap`, `String` (immutable, length-known) | ✅ |
| Conv / concurrency | async fibers, threads, channels, agents | ✅ |
| String→num parse | `s.to_int()` / `s.to_float()` | ✅ |
| JSON | `__vyb_*_to_json` (serialize) + `__vyb_complex_from_json` (deserialize **into typed structs via metadata**) | ⚠️ typed-DOM only; no generic JSON value/DOM parser at the Vyb surface |

## 3. Decisive gaps to close (the "enough support from Vyb lang" part)

### P0 — must have to run the trained 4B model natively

**G1. `tensor` / matmul substrate on the GPU (the actual "replace torch" core).**
No vector/tensor type, no general GEMM library, no cuBLAS binding today. Needed:
- a `bindings/cublas/` freedom-FFI binding (cuBLAS `Sgemm`/`Strsm` for fp32; later
  bf16) — hours of work, huge speedup, Vyb orchestrates launches; AND/OR
- a `tensor::` stdlib module: dense N-D device buffers (via `cuMemAlloc_v2` + the
  `ld/st` kernels) with GEMM + elementwise/RMSNorm/softmax/reduction kernels.
  This becomes the NumPy-eating substrate everything else sits on.
- `cuLaunchKernel` multi-arg / batched-launch ergonomics (the `#198` `cuda_launch4`
  helper is fixed-arity; a general launch + kernel-scheduling helper is needed for
  the dozens of tiny elementwise kernels a transformer issues per token).

**G2. Numeric `Int ↔ Float` cast** — **RESOLVED (`55593fc`)**. `as` now crosses
int↔Float/Float32 (SIToFP/UIToFP, FPToSI/FPToUI) + Float32↔Float width
(FPExt/FPTrunc). Replaces the note below (previously: `Float(i)` was a semantic
error; `as` crossed integers only). Test: `test/units/test_numeric_cast.vyb`.

**G3. Kernel device intrinsics expansion** — **RESOLVED (`55593fc`)** for a token
engine. Now available (verified ptxas + on-GPU): `ld/st` for `i8/u8/i16/u16` and
**fp16/bf16** (`ld_f16/ld_bf16` → `Float`), **int4 GGUF q4_0 dequant** via
`deq_q4_0(blockAddr, i) <Float>` (the 4-bit weight path), **shared-memory f64**
staging (`ld/st_shared_f64`, 32 KB addrspace(3)), **global atomics**
(`atomic_add_f64/i32`), **`kernel_barrier`**, and geometry (`tid/blk/dim_z`,
`grid_x/y`, `lane_id`, `warp_size`). q4_K dequant remains a follow-on if needed.
What a token engine still adds at the *implementation* level:

**G4. Model + tokenizer loader at the Vyb surface.**
- a **GGUF reader** (binary header, tensor index, quant blocks) — either Vyb-native
  (`Bytes` + `io`) or freedom-FFI to ggml; safetensors (no header, big-endian tensor
  layout) for the fp16/lora side. The LoRA adapter is `safetensors` → LoRA tensors are
  small (rank 16) — loadable directly.
- **BPE tokenizer** for Qwen3 (151k vocab): regex pre-tokenize (Vyb `regex` ✅) +
  `vocab.json`/`merges.txt` (→ needs G5) + byte-pair merge + chat-template (`{role}`).
- a **generic JSON value/DOM parser** today there's only the typed-metadata
  deserializer; vocab/merges/contract parsing wants a real `json::parse` (or a
  freedom-FFI to a JSON lib). Alternatively use the typed deserializer with declared
  struct shapes since vocab/merges shapes are fixed.

### P1 — efficiency / experience

- **fp16/bf16 scalar types** + kernels (Qwen3 is bf16; halves activation memory and
  unlocks tensor-core GEMM via cuBLAS bf16). f32 compute is correct-but-2x activations
  for the initial port.
- **Fused attention / flash-attention** kernels (paper flash, then own) instead of
  materializing QK^T.
- **Multi-arg launch + launch-batching** in binding/main.cpp so the host side doesn't
  JIT-per-op (JIT call overhead will otherwise dominate at 36 layers × dozens of ops).
- `Bytes` ⇄ device-buffer DMA helpers (host→device copy for weights, mmap/stream GGUF).
- Fix `#201` (fixed-array element write → invalid GEP IR crash) — host-side dense
  buffer building frequently wants mutable arrays; today only `Vec` element writes work.

### P2 — stretch (the "replace all python tools like torch" end-state)

- **Training in Vyb** (autograd, Adam/optimizers, QLoRA/full FT) — a large project on
  its own; recommend inference-first now, training-support as a later milestone.
- Composable Vyb-authored kernels as the default GEMM/attention path (no cuBLAS), with
  shape/broadcast DSL.
- Multi-GPU / sharded KV cache.

## 4. Recommended build order (implements P0)

1. **Close G2** (int↔float cast) + **G5** (JSON value parser) in the compiler/stdlib —
   small, unblocks everything. (RFE to impl agent.)
2. Generalize the `#198` matmul + add **G3** int4/fp16/smem/atomic device intrinsics.
3. **G1** `tensor::` module + a first fused decode of **one** layer (embed → attn →
   mlp → logits) on a tiny random-weight model — proves the substrate with numbers.
4. **G4**: GGUF/Q4 base loader + BPE tokenizer + (later) the committed LoRA adapter.
5. Fill in 36 layers → full decode loop → sampling → feed the `apply.vyb` contract.
6. Deploy (per the SDK/smuggle channel) as a posted `bindings/llm/` + Vyb execution tool.

## 5. Bottom line

**Vyb has the foundation and is implementable**, but today it is **not yet
"enough" on its own**: it lacks a tensor/GEMM substrate (G1), the `Int↔Float` numeric
cast (G2), the wider/narrow+int4 device-intrinsic set + shared mem/atomics (G3), and
a GGUF/tokenizer/JSON-parse loader stack (G4/G5). None of these are architectural
blockers — all are additive compiler/stdlib/binding work on the existing kernel-mode
and freedom-FFI rails. The near-term, high-value slice (run the trained 4B config
model natively, decode-only) is **feasible**; a full training-side torch replacement is
the long tail and should be sequenced behind it.
