# Changelog

## [Unreleased] — 2026-10-26

### Added
- **27B Ridge (qwen35) all-Vyb decode** (`native/host/ridge_decode.vyb`, `native/kernels/r*.vyb`),
  with `./ridge-chat`: streaming chat, the ChatML template, thinking on or off, and a gold-exact
  `VYB_EXACT=1` path. See `doc/RIDGE-27B.md`.
- RTX 3090 speed: tg128 40.4 tok/s, pp512 791 tok/s (llama.cpp: 47.35 / 1258.5). Recent steps:
  - f32 GDN recurrent state with a register-resident state step `delta_step_g32`
    (`native/tools/gen_rdeltag.py`, structure after llama.cpp `gated_delta_net.cu`, MIT): one DRAM
    read and one write of the state per token instead of the tiled f64 `delta_step_x` (48.6 µs per
    launch, the largest per-kernel gap in an nsys profile against llama.cpp's 6.5 µs). Default unless
    `VYB_EXACT=1` (`VYB_GDN32` / `VYB_DSG`); lossy, KLD-checked: mean KLD 0.00394, top-1 98.04 %,
    PPL 3.2448 vs 0.00407 / 3.2568 for the f64 default. tg128 37.9 → 39.7 tok/s
  - Q4_K / Q5_K q8_1 GEMVs with the super-block loop unrolled ×4 (`gemv_q4k_qu`, `gemv_q5k_qu`;
    `VYB_Q4KU`, `VYB_Q5KU`), bit-identical (`VYB_LOGDUMP` cmp), +0.8 % / +0.4 % tg128 interleaved
  - Q6_K decode GEMV `rq6kr2` (2 rows per warp, `native/tools/gen_rq6k.py 2 1`), default on sm_8x;
    GB10 keeps `rq6k8`, which measured faster there. Bit-identical
  - Q4_K GDN GEMVs (qkv+gate, ssm_out) in llama.cpp's mmvq layout: `gemv_q4k_m` in `rq4kq.vyb`
    (128 threads per row, vdr=2, 8 super-blocks in flight, structure after `mmvq.cu` /
    `vecdotq.cuh`, MIT; see THIRD-PARTY.md), with rq4kq built with a 256 B static shared buffer.
    Default wherever `VYB_Q4Q8` is on (`VYB_Q4KM=0` reverts). KLD-checked: mean 0.00402, top-1
    98.63 %, PPL 3.2491. 3090 tg128 39.69 → 40.16 tok/s; GB10 with `VYB_Q4Q8=1` 14.54 → 14.68
  - Q5_K attention q/k/v GEMVs in the same mmvq layout: `gemv_q5k_m` in `rq5kq.vyb` (high bits from
    `qh` per 4-byte group, rq5kq built with a 256 B static shared buffer; llama.cpp MIT, see
    THIRD-PARTY.md). Default on sm_8x (`VYB_Q5KM=0` reverts); opt-in on GB10, where it measured
    no gain. KLD mean 0.00384 (lowest so far), top-1 98.63 %, PPL 3.2336. 3090 tg128 40.25 → 40.45
  - GB10 (DGX Spark): `VYB_Q8SK` (Q8_0 split-K GEMV) now defaults on for CC >= 10 as well (off under
    `VYB_EXACT=1`); with DSG, tg128 14.55 → 14.76 tok/s; token stream unchanged vs `VYB_Q8SK=0`.
    GB10 `VYB_EXACT=1` checked gold-exact
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
