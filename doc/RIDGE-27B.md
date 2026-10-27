# Ridge 27B (qwen35) — all-Vyb decode

`native/host/ridge_decode.vyb` runs Qwen3.8-27B "Ridge" (the 3.7 bpw GGUF: IQ2_S / IQ3_S / Q4_K /
Q5_K / Q6_K / Q8_0 mix, 64 layers of Gated-DeltaNet plus full attention) on one GPU. Everything
in the inference path is Vyb: the host program, the tokenizer and chat template, sampling,
streaming, and every PTX kernel (`native/kernels/r*.vyb`, built with `vyb --kernel`). Python is
used only for reference checks and code generators. `ridge-chat` is a thin bash wrapper.

## Quick start

```bash
export VYBHOME=~/Projects/Vyb            # a Vyb build that has the kernel intrinsics (Vyb main)
make -C native ridge-kernels             # builds all PTX the decoder loads into native/build/
./ridge-chat                             # interactive chat (/reset, /quit or Ctrl-D)
./ridge-chat -1 "Why is the sky blue?"   # one-shot answer
```

`ridge-chat` env vars:

| Var | Default | Meaning |
|---|---|---|
| `VYB_BIN` | `~/Projects/Vyb/build/vyb` | vyb binary |
| `VYB_STDLIB` | `<vyb>/../../stdlib` | stdlib directory |
| `VYB_MODEL` | `~/Models/qwen38-27b-ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf` | GGUF file |
| `RIDGE_GEN` | 1024 | max new tokens per reply |
| `RIDGE_SYS` | – | system prompt |
| `VYB_THINK` | 0 | 1 turns thinking on |
| `VYB_EXACT` | 0 | 1 = bit-exact gold path (slower, see below) |
| `RIDGE_STATS` | 1 | 0 hides the tok/s line |
| `VYB_MAXS` | 2048 | context length (KV slots) |

The decoder can also be run directly, which is what the benchmarks do:

```bash
VYB_STDLIB=$VYBHOME/stdlib VYB_PFB=2 VYB_GEN=128 VYB_PROMPT="The capital of France is" \
  $VYBHOME/build/vyb native/host/ridge_decode.vyb --module-path native/config \
  --module-path native/json --module-path native/tensor --module-path native/dtype --module-path native/llm
# prints PREFILL_TOK_S=, DECODE_TOK_S=, GEN_IDS=, MEM ...
```

### Local inputs (gitignored, generated once)

| File | From |
|---|---|
| `native/out/ridge_tensors.tsv` | `make -C native ridge-tensors` (runs `native/tools/inventory_to_tsv.py` on `native/gguf/ridge-3.7bpw-inventory.tsv`) |
| `native/out/iq2s_grid.bin`, `iq3s_grid.bin` | `python3 native/tools/iq2_s_ref.py`, `iq3_s_ref.py` |
| `native/out/ridge_invfreq.bin` | `python3 native/tools/gen_invfreq.py <rope_base> <n_rot_dims> native/out/ridge_invfreq.bin` |
| `native/out/ridge_meta.txt` | `python3 native/gguf/ridge_inventory.py <GGUF> --meta native/out/ridge_meta.txt` |
| `artifacts/ridge-tokenizer/{vocab,merges}` | `hf download` (the command is printed by `native/tools/ridge_encoder_check.py`) |

If the tokenizer directory is missing, the prompt encodes to `-1` ids and the output is garbage.
Check for this first when a fresh checkout produces nonsense.

## Performance (tg128 / pp512, tok/s)

RTX 3090 (godzilla, CC 8.6, 24 GB). Vyb main, 2026-10-27 (pp512 last measured 2026-10-26; the
changes since are decode-only):

| Path | tg128 | pp512 |
|---|---|---|
| Vyb default | **39.7** | **791** |
| Vyb `VYB_EXACT=1` | ~20.8 | ~753 |
| llama.cpp (CUDA, same GGUF) | 47.35 | 1258.5 |

DGX Spark (GB10), measured 2026-10-27 with `VYB_GDN32`/`VYB_DSG` and `VYB_Q8SK` on (the GB10
defaults). `VYB_EXACT=1` on GB10 is gold-exact too:

| Path | tg128 | pp512 |
|---|---|---|
| Vyb default | 14.65 | ~640 |
| llama.cpp | 16.64 | 818 |

Peak memory on the 3090 is about 14.2 GB of VRAM.

## Exactness and accuracy gates

- **`VYB_EXACT=1`** keeps the f64 reference math everywhere. Its token stream is bit-identical
  to the gold ids (`GOLD_EXACT`). Every change must keep this.
- **Default (lossy) path**: q8_1 activations + `dp4a` for the IQ2_S / IQ3_S / Q4_K / Q5_K / Q8_0
  GEMVs, an f32 conv and an f32 GDN state (`delta_step_g32`). On 510 wikitext tokens vs the exact
  path: mean KLD **0.00394**, p99 0.044, top-1 agreement 98.04 %, PPL **3.2448** (before the f32
  GDN state: 0.00407 / 3.2568). The rule for a lossy default is mean KLD ≤ ~0.005
  and PPL within ~0.01 of exact.
- **Qwen3-4B resident gate**: `make -C native resident-bench` must PASS (gold stream MATCH).

## Env flags of `ridge_decode.vyb` (`VYB_` prefix)

Flags marked "default on" are enabled on CC ≥ 8 unless `VYB_EXACT=1`. Set one to `0` to switch it off.

| Flag(s) | Meaning |
|---|---|
| `EXACT` | 1 = exact f64 path (turns off all lossy defaults) |
| `RGMAP`, `RGQ8`, `RGQ8_IQ3` | IQ2_S / IQ3_S GEMV: mapped grid, q8_1 activations + dp4a (default on) |
| `Q4Q8`, `Q5Q8`, `Q8SK` | Q4_K / Q5_K / Q8_0 GEMVs on q8_1 + dp4a, Q8_0 split-K (default on for sm_8x; `Q8SK` also on for CC >= 10 / GB10 unless `VYB_EXACT=1`) |
| `CONV32` | f32 short conv in the GDN block (default on) |
| `RMSG` | multi-block RMSNorm `rms_blkg`, exact (default on) |
| `RMSQ8` | `rms_blkg` also writes the q8_1 copy (int8 + per-32 f32 scale) that the Q4_K / Q5_K GEMVs read, instead of a separate `xq8` launch, for the attention norm and (when gate/up use the q8_1 path) the FFN norm (default 1, bit-identical; needs `RMSG`; off when the hidden size is not a multiple of 32) |
| `GDNF` | GDN decode: fused `sg_ag` (sigmoid + alpha gate) and `conv_step` (shift + interleave + conv1d + silu) kernels, 4 fewer launches per GDN layer (default 1, bit-identical; 0 = the separate kernels) |
| `SWQ8` | FFN: `silu32q` fuses SwiGLU with the q8_1 quant of its output for the `ffn_down` GEMV, one launch instead of `silu32` + `xq8` per layer (default 1, bit-identical; needs the q8_1 path and FF a multiple of 32). `VYB_DBG=1` prints `SWQ_LAYER_STEPS` |
| `EMBG` | Decode token embedding (`q6kdeq_ix`, reads the token id from device memory) runs inside the decode CUDA graph instead of as an eager launch (default 1 when the graph is on, bit-identical) |
| `Q6KPTX`, `Q6KR` | Q6_K decode GEMV module and rows per warp. Default `rq6kr2.ptx` with 2 rows per warp on CC < 10 (bit-identical), `rq6k8.ptx` with 1 on GB10; `Q6KR` must match the module |
| `PROMPT`, `PROMPT_IDS`, `FORCE_IDS`, `GEN`, `CHAT`, `CHAT_N`, `CHAT_SYS`, `THINK`, `STREAM`, `TOKTEXT`, `MAXS` | prompt / chat / output control |
| `MODEL`, `TSV`, `INVFREQ`, `LLM_DIR` | input paths (GGUF, tensor TSV, rope inv-freq, tokenizer dir) |
| `PFB`, `PFB64`, `PFBM`, `PFG32`, `PFQ4`, `PFQ4H`, `PFQ5H`, `PFXH`, `PFG128PTX` | batched-prefill variants (`PFB=2` is the bench setting) |
| `GDN32`, `DSG` | f32 GDN recurrent state (`GDN32`) stepped by `delta_step_g32` (`DSG`, `native/tools/gen_rdeltag.py`): state held in registers, 8 warps x 16 rows per 32-column block, one read + one write of the state. Default 1 unless `VYB_EXACT=1`; lossy (KLD above). `GDN32=0` = f64 `DSX`; `DSG=0` = `delta_step_u32` |
| `Q4KU`, `Q5KU` | Q4_K / Q5_K q8_1 GEMVs with the super-block loop unrolled x4 (`gemv_q4k_qu`, `gemv_q5k_qu`; default 1, bit-identical; 0 = the plain loop) |
| `DSX` | Tiled f64 GDN state step `delta_step_x` (default 1, bit-identical to `DSU`; `VYB_DSX=0` falls back to `delta_step_u`). `DSXPTX` overrides the module path |
| `CG` | Decode step (layers + lm_head) replayed as one CUDA graph (default 1, bit-identical). Re-captured when the attention grid bucket (64 positions) grows; 0 = eager launches |
| `LKD` | Kernels launched by a direct `cuLaunchKernel` Vyb extern on the decode stream (default 1; 0 = the `cuda_launch4i` shim, which also disables `CG`) |
| `PSA` | Per-token attention params written through pinned slots + `cuMemcpyAsync` (default 1; required by `CG`) |
| `DSU`, `DSV`, `DSW`, `DST`, `DSC`, `DSF`, `DSP`, `DSPNW`, `DSVJB`, `DSPPTX`, `DSWPTX`, `ACX` | Gated-DeltaNet state-step / attention kernel variants (experimental) |
| `RGPTX`, `RG6PTX`, `RG7PTX`, `RG6PERM`, `RG6TPB`, `RG6WPR`, `RGTPB`, `RGR`, `RGFUSE`, `RK1`, `XPERM`, `Q4KPTX`, `Q4KR` | GEMV kernel selection and tuning (experimental) |
| `DBG`, `LOGDUMP` | debug output / dump logits |

## Kernels

`make -C native ridge-kernels` runs `native/tools/build_ridge_kernels.sh`. The script holds the
`--kernel-shared` size for each kernel that needs one. Several kernels are generated:
`native/tools/gen_rgemv{5,6,7,8}.py`, `gen_rpf*.py`, `gen_rdelta2.py`, `gen_rdelta2x.py` and `gen_rdeltag.py` write the `.vyb`
source, which is committed.
