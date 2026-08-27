# Handoff → vyb-config agent: implement Vyb-native inference of the Qwen3-4B configurator

**READ THIS FIRST.** You are unblocked: the two Vyb-language gaps the earlier audit
flagged (#202 numeric `Int↔Float` cast, #203 kernel device-intrinsic expansion) are
**landed, verified, and closed** in `rickenator/Vyb` commit `55593fc` (on `main`,
pushed; full suite 1112/0). Everything below is what you need to pick up and build.

This is the implementation slice to take: a **decode-only, Vyb-native** runtime that
loads the trained Qwen3-4B + LoRA configurator and produces the VybOS `SystemSpec`
JSON contract — no torch, no Ollama, no hosted model. Training in Vyb is a later,
separate milestone.

---

## 1. What you now have (verified working, use these)

**Vyb→GPU kernels (#198 + #203).** Kernel mode (`vyb --kernel file.vyb --ptx out.ptx`)
lowers a `main`-less Vyb module to launchable PTX:
- A **VOID top-level function becomes a PTX `.entry` kernel** (launchable); value
  functions stay `.visible .func` device helpers (no marker needed — the void
  convention is the rule).
- `cuda_launch4` / `cuda_launch4i` (compiler process-symbols in `src/main.cpp`,
  dlsym'd against the dlopen'd libcuda) own the `void**` kernelParams packing. Use
  `cuda_launch4(hf, gx,gy,gz, bx,by,bz, a0,a1,a2, a3)` (4th arg Float → XMM) or
  `cuda_launch4i(..., a3<Int>)` (4th arg Int → RDI).

**Kernel-mode device intrinsics (all verified — compile to ptxas-valid PTX, and the
dequant/fp16 math is runtime-verified on the RTX 3090):**
| Intrinsic | Meaning |
|---|---|
| `tid_x/y/z`, `blk_x/y/z`, `dim_x/y/z`, `grid_x/y`, `lane_id`, `warp_size` → `<Int>` | NVPTX special-register reads |
| `ld/st_f64,f32,i64,i32,i8,u8,i16,u16` | device-global load/store (addrspace 1) |
| `ld_f16(addr)` / `st_f16(addr,v)` → `<Float>` | IEEE half ↔ Vyb Float |
| `ld_bf16(addr)` / `st_bf16(addr,v)` → `<Float>` | bfloat16 (top-16-bits-of-f32) |
| `deq_q4_0(blockAddr, i) <Float>` | **GGUF q4_0 dequant** = `(nibble − 8) * d`; block = `[f16 d][32×4-bit]`, 18 B |
| `ld/st_shared_f64(off)` | 32 KB addrspace(3) shared staging (`__vyb_kernel_shared`) |
| `atomic_add_f64(addr,v)` / `atomic_add_i32(addr,v)` | monotonic global atomics |
| `kernel_barrier()` | `bar.sync 0` (only valid where all threads in the block reach it) |

**Numeric casts (#202):** `x as Float`, `f as Int` (truncates), `x as Float32`,
`f as Float32`, and back — full int↔float↔Float32 cross-family. (RoPE positions,
attention scale, softmax, and int-dequant all rely on this now.)

**Host-side FFI you can lean on (#174/#197/#198):** `freedom` blocks + `extern "C"`
inline declarations reach any dlopen'd C lib (sqlite, libgit2, **libcuda**, and you can
add cuBLAS/cuDNN/ggml to the best-effort dlopen list in `run_vyb_code`), plus the
posted `bindings/cuda/` module.

**Environment gotchas (verified):**
- `a..b` ranges are **inclusive** — write `0..n-1`.
- Fixed-array element **write is now supported** (`a[i]=v`, constant or loop index —
  fixed #201) — build host-side dense buffers directly in fixed arrays, or via
  `Vec` / device-memory + the `ld/st` kernels.
- `cuModuleLoadData` of a non-trivial kernel in a fresh process returns
  NO_BINARY_FOR_GPU (218) once and caches per-String — re-read the PTX into a FRESH
  String and reload (the #198 tests show the pattern).

## 2. Your implementation plan (prioritized)

### P0 — make a real single-layer decode run with numbers (the "prove the substrate" slice)
1. **`bindings/cublas/`** freedom-FFI binding → `cublasSgemm` (fp32 `C = A·B`), and/or
   generalize the existing `#198` register-tiled matmul kernel (`fixtures/kernel/
   matmul.vyb`) into a reusable GEMM on device buffers.
2. **`tensor::` stdlib module**: dense N-D buffers backed by `cuMemAlloc_v2`
   (handles as `Int`), a GEMM, and elementwise/RMSNorm/softmax/reduction kernels
   on the intrinsics above. Implement in **fp32 first** (correct; 2× activation mem
   vs fp16; swap in bf16/fp16 later).
3. Suffix decode **one transformer layer** (embed → RMSNorm → GQA/RoPE → softmax QKᵀ →
   ×V → o_proj → MLP SiLU) with **random small weights** on the GPU; verify against a
   torch/numpy reference once. This is your go/no-go gate.

### P0 — model + tokenizer loader (G4)
4. **GGUF reader** (binary header, tensor index, q4_0/q4_K blocks) — Vyb-native on
   `Bytes`+`io`, or freedom-FFI to ggml. Load the 4B base as **Q4** (~2.2 GB) using
   `deq_q4_0`; the LoRA adapter is `safetensors` (rank 16 — small, load directly).
5. **Qwen3 BPE tokenizer**: `vocab.json` + `merges.txt` + regex pre-tokenize
   (Vyb `regex` ✅) + byte-pair merge + chat-template (`artifacts/*/chat_template.jinja`).
6. **JSON value parser** if needed for vocab/contracts (Vyb has
   `__vyb_*_from_json` typed-metadata deserialization; a generic `json::parse` is a
   small stdlib add or an FFI to a JSON lib).

### P1 — make it fast / complete
7. fp16/bf16 GEMM path (halve activation memory; cuBLAS bf16 if bound).
8. Fused attention / own flash-attention kernels (avoid materializing QKᵀ).
9. General/batched launcher + launch-batching (the current `cuda_launch4/4i` are
   fixed-arity; a 36-layer decode issues dozens of small kernel launches per token —
   batch them so JIT/host-call overhead doesn't dominate).
10. LoRA inference merge (`y = xW + α·(x·Wup)·Wdown`, rank 16 — a couple of small
    GEMMs per projection).

### Acceptance line
- A Vyb program loads the Qwen3-4B-Q4 base + `artifacts/vybos-configurator-lora*/`,
  runs the schema-driven config interview decode **fully on-GPU in Vyb** (no CUDA
  FFI except cuMem alloc/copy/launch + cuBLAS), and emits the `{path, op, value,
  reason}` SystemSpec JSON that `tools/apply.vyb` consumes — verified byte-compatible
  with the current Python/Ollama path on a few interviews.
- Full Vyb suite stays green; kernels ptxas-valid + a GPU value check committed.

## 3. Ground truth / references
- `rickenator/Vyb` `main` at `55593fc` (Suite: **1112/0** green). Kernel-mode refs:
  `doc/.../PROGRAMMERS_GUIDE.md` (types/cast), the #198/#203 skill notes
  (`references/nvptx-kernel-mode-198.md`), tests `test/kernel/test_device_intrinsics.vyb`,
  `test/ffi/test_cuda_{axpy,matmul,p203}.vyb`, `test/units/test_numeric_cast.vyb`.
- VybForge: `artifacts/vybos-configurator-lora*/` (base `Qwen/Qwen3-4B`, LoRA r16/
  α32 on q,k,v,o,gate,up,down), `training/train_lora.py` (NF4), `config/*.schema.json`
  (the JSON contract), `README.md`/`HANDOFF.md`.
- This repo's earlier audit: `VYB-NATIVE-INFERENCE-AUDIT.md` (G1/G4/G5 remain open;
  G2/G3 closed per above).
