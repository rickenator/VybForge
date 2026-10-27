# Changelog

## [Unreleased] — 2026-10-26

### Added
- **27B Ridge (qwen35) all-Vyb decode** (`native/host/ridge_decode.vyb`, `native/kernels/r*.vyb`),
  with `./ridge-chat`: streaming chat, the ChatML template, thinking on or off, and a gold-exact
  `VYB_EXACT=1` path. See `doc/RIDGE-27B.md`.
- RTX 3090 speed: tg128 37.6 tok/s, pp512 791 tok/s (llama.cpp: 47.35 / 1258.5). Recent steps:
  - Q6_K decode GEMV `rq6kr2` (2 rows per warp, `native/tools/gen_rq6k.py 2 1`), default on sm_8x;
    GB10 keeps `rq6k8`, which measured faster there. Bit-identical
  - `rms_blkg` writes the q8_1 copy of the attention-norm output itself (`VYB_RMSQ8`), dropping the
    separate `xq8` launch per layer; GDN decode small kernels fused into `sg_ag` (sigmoid + alpha
    gate) and `conv_step` (shift + interleave + conv1d + silu) (`VYB_GDNF`). Both bit-identical
    (`VYB_LOGDUMP` cmp); with `rq6kr2`, tg128 35.8 → 37.6 tok/s
  - FFN decode fusions: `silu32q` (SwiGLU + q8_1 quant for `ffn_down`, `VYB_SWQ8`), the `ffn_norm`
    q8_1 copy from `rms_blkg` (now part of `VYB_RMSQ8`), and the token embedding inside the decode
    graph (`q6kdeq_ix`, `VYB_EMBG`). About 128 fewer launches per step, bit-identical; about +0.3%
    tg128 measured interleaved, within the 37.6 headline
  - decode step as one CUDA graph (`VYB_CG`), launched by a direct `cuLaunchKernel` Vyb extern; 34.7 → 35.8 tok/s, bit-identical
  - q8_1 + dp4a GEMVs for IQ2_S / IQ3_S / Q4_K / Q5_K / Q8_0
  - batched prefill
  - multi-block exact RMSNorm (`rms_blkg`, 32.4 → 32.7 tok/s)
  - seeded 6-step Newton sqrt in `l2norm_b` / `norm_gated_b`. It is bit-identical, checked by
    `native/tools/check_vsqrt_seed.py`, and took decode from 32.7 to 34.2 tok/s.
- Tiled f64 GDN state step `delta_step_x` (`native/tools/gen_rdelta2x.py`), now the default
  (`VYB_DSX=0` restores `delta_step_u`). Bit-identical f64 logits; 56 → 47.7 µs per launch,
  tg128 33.9 → 34.3 tok/s.
- `make -C native ridge-kernels` (`native/tools/build_ridge_kernels.sh`): builds every PTX the
  decoder loads, each at its own `--kernel-shared` size.

### Fixed
- `make -C native resident-bench`: the gate read a gitignored `native/out/prompt_ids.txt` that could
  be left over from another prompt and then failed on correct code. It now re-encodes the gold
  prompt every run, and `verify_resident_bench.py` names a stale seed when the ids do not match.
- `ridge_decode.vyb` refuses a missing or empty tensor index with a named `ERR` (it used to fail
  later with "alloc region"). `make -C native ridge-tensors` builds the index;
  `ridge-tensors-check` tests the error.
- `make -C native verify`: the decode step ran `verify_decode.py`, which still expects the old toy
  `wte.txt` fixture. `decode_driver.vyb` has been the real Qwen3-4B decode for a while. The step
  now checks its token stream with `verify_decode_real.py` against `decode_ids_ref.txt`,
  generating that file with `decode_ref.py` if it is missing. `qwen3` was also added to
  `KERNELS` so a fresh checkout builds the module that `decode_driver` loads.

### Accuracy
- Default lossy path vs exact: mean KLD 0.00407, PPL 3.2568, top-1 98.6 %. `VYB_EXACT=1` stays
  bit-identical to gold, and the Qwen3-4B resident gate passes.
