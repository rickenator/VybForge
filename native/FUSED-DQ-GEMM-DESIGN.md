# Fused-Dequant GEMM: fp32 Compute Path with f64 Oracle

**Status:** design only, no code written. Target: `native/kernels/`, driver in `native/host/`, verification via existing dump/compare tooling.

## Problem

The current inference/training path dequantizes every quantized weight tensor to an
expanded f64 buffer (packed 63.9MB → 807MB f64) and runs GEMMs in f64
(`kernels/gemm.vyb`). On the RTX 3090 (GA102) this is doubly punitive:

- GA102 f64 rate is 1/64 of its fp32 rate.
- The GEMM is bandwidth-bound; reading 8 B/weight of expanded f64 vs ~2 B/weight
  of packed q4_K (~0.5 B/weight effective for q4_K_M) wastes most of the memory
  budget.

Production engines (llama.cpp CUDA, vLLM, ExLlamaV2/3, TensorRT-LLM) all use the
same pattern: weights stay packed, dequant happens to fp16/fp32 **in registers**
inside the matmul, accumulation in fp32, f64 nowhere. This doc specifies that
pattern adapted to Vyb kernel constraints.

## Hard constraints (verified in code)

1. **No shared memory.** Vyb's device surface exposes only global-memory
   `ld_/st_` ops (`ld_f64/f32/f16/i64/i32/i8/u8...`); no `__shared__`. Classic
   tiled GEMM is out of scope. (This also rules out Option B's tile staging
   unless a later compiler change adds sync primitives.)
2. **`cuda_launch4i` caps kernel args at 4 Ints.** All tensor pointers and dims
   ride a 64 B device params buffer (existing pattern in `gemm.vyb`).
3. **Vyb `Float` is f64.** fp32 arithmetic requires explicit `f32`/`Float32`
   types (`cgen_types.cpp:590-599`). Mixed f16/f32 ops need explicit casts; the
   kernel must declare accumulators as `f32` deliberately.
4. **No libm device builtins** — dequant is pure integer/float arithmetic, so
   this constraint doesn't bite here (it did for `exp`/`sin` in `vmath.vyb`).

## Design

### Kernel: `gemm_dq_f32.vyb`

`C[M,N] (f32) = alpha * A_act[M,K] (f32) @ W_packed + beta * C`, where
`W_packed` is the raw GGUF byte slab for one weight tensor (q4_K, q6_K, or
pre-expanded f32), dispatched by a `wty` field in the params buffer.

Per output element `(m,n)`:

```
acc_f32 = 0 (or beta*C[m,n] folded at the end)
for k in 0..K-1:
    a   = ld_f32(A + (m*K+k)*4)              # activation, fp32
    w   = dequant_one(W_packed, k, n, wty)   # in registers, -> f32
    acc += a * w
st_f32(C + (m*N+n)*4, alpha*acc + beta*c)
```

`dequant_one` reuses the proven bit-exact block math from `q4k.vyb` / `q6k.vyb`
unchanged (scale/min extraction from `scales[12]`, nibble selection). Per
output element the dequant recomputes scale/min per 32-element group —
redundant loads (~32 threads re-reading the same 144 B block) are acceptable in
the one-thread-per-output model; L2 absorbs them. Do NOT pre-expand.

Params buffer layout (extends gemm.vyb's):
```
+0  A ptr    +8  W ptr (packed)   +16 C ptr
+24 M        +32 N                +40 K
+48 alpha    +56 beta             +64 wty (0=q4_K, 1=q6_K, 2=f32-slab)
```

### Memory layout change

Drop the expanded-f64 weight residency. Layer loop keeps **packed slabs
resident for all 36 layers** (the PKD direction already in
`train_full_loop_pkd.vyb`), dequant-in-GEMM per step. Norms/activations move to
f32 (`f32expand` becomes f32, not f64).

### Verification: f64 as oracle

The f64 path is retained as the reference oracle — that is its ongoing utility:

1. Run existing f64 layer forward on a fixed input (existing
   `layer0_input.bin` machinery); dump output (existing f64 dump tools).
2. Run the new fused fp32 kernel on the same input; dump f32.
3. Compare: expect max abs diff ~1e-3 or below for a full layer (quantization
   is identical, so residual is pure fp32-vs-f64 rounding over K-length
   accumulation). `TOL = 2e-3` from `verify_layer0.py` remains the gate.
4. Bit-exactness vs llama.cpp dequant is inherited automatically: the dequant
   math is byte-identical to `q4k.vyb`/`q6k.vyb`, which are already 32/32
   verified.

Gate: the fp32 path replaces the f64 path in the training loop only after the
oracle diff is inside tolerance across blk.0 (11 tensors) and a full-layer
forward.

### Expected performance (3090, order-of-magnitude)

- Compute: fp32 35.6 TFLOPS vs f64 0.56 TFLOPS → ~64× on the arithmetic.
- Memory: ~0.56 B/weight (q4_K_M effective) vs 8 B/weight → ~14× on traffic,
  which dominates for these shapes.
- Realistic end state: tens-of-× faster layer forward, model resident footprint
  ~64MB packed + small f32 activations vs 807MB. Do not treat these as
  measured; the oracle run reports actuals.

### Explicitly not built (this iteration)

- Shared-memory tiling / Option B staging — blocked on no-`__shared__`.
- Tensor-core mma (fp16/bf16) — needs inline-PTX/extern surface in Vyb; compiler
  work, file under the Vyb tracker if pursued.
- fp16 activations/KV cache — f32 accumulate everywhere first; f16 is a later
  optimization gated on oracle tolerance.
- Batched/multi-token GEMM shapes — one design, shapes handled by M/N/K params.

## File plan

| Piece | Location |
|---|---|
| `gemm_dq_f32.vyb` kernel (q4_K/q6_K/f32 dispatch) | `native/kernels/` |
| Driver: load packed slabs, launch, dump f32 out | `native/host/` |
| Oracle compare script (f64 ref vs f32, TOL gate) | `native/tools/` |
| Makefile targets mirroring phase-2/3 targets | `native/Makefile` |
