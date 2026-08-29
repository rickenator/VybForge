# Vyb-native inference (Qwen3-4B configurator) — phase roadmap

Tracked here so phases can be worked serially and, where independent, in
parallel. All GPU work lives under `native/`; the production pipeline is Vyb
only (Python appears solely as a *reference-verification* harness).

Status legend: [x] done · [~] in progress · [ ] not started

## Done
- [x] **P0 substrate on-GPU** — GEMM / RMSNorm / native exp·sin·cos verified
      (`GEMM_OK`, `RMSNORM_OK`, `VMATH_OK`); re-ran Vyb reference GPU tests.
- [x] **One transformer layer forward** — RMSNorm→QKV→RoPE→GQA causal-softmax→
      o_proj→residual→SiLU MLP→residual on the RTX 3090, verified
      (`LAYER_VERIFY: OK`, ~5e-5). = handoff P0 go/no-go gate.
- [x] **GGUF v3 reader** (header + metadata KV + tensor index) — Vyb-native,
      verified on a synthetic fixture (`GGUF_PARSE_VERIFY: OK`).
- [x] **G-gguf-data** (serial): GGUF tensor DATA read by offset → device buffer
      → `q4_0` dequant on-GPU via `deq_q4_0` (`DEQUANT_OK`, 32/32). This is the
      loader's weight path.

## In flight / planned
- [x] **G-json** — generic JSON value parser in Vyb (verified: parses
      `config/mock-system.json` and a `vocab.json` slice with real `build/vyb`).
- [x] **G-tokenizer** — Qwen3 byte-level BPE in Vyb (real 151k vocab + merges).
      Pre-tokenizer is a faithful algorithmic reproduction of the Qwen2 regex
      (letters/digits/punct segmentation + space-before-punct); **EXACT match,
      13/13** vs transformers on plain words, contractions, leading/trailing
      apostrophes, punctuation, and whitespace runs. Non-ASCII words still
      approximate (no Unicode-class regex in Vyb).
- [x] **G-tokenizer detokenizer** — `decode`/`decode_ids` (id → text, reverse
      vocab + glyph→byte). Verified exact in-process round-trip
      encode→decode on the full config-interview prompt batch (18/18 `bad=0`).
      The old `decode_ids` "runtime segfault" was **vyb-lang #205** (native-stack
      overflow mono'ing two generic HashMap types when encode+decode coexist),
      FIXED at bf46304 — not tokenizer logic.
- [~] **G-tensor** — **complete for the vertical slice**: multi-layer stack
      (`STACK_VERIFY: OK`), greedy autoregressive decode (`DECODE_VERIFY: OK`),
      and **stochastic sampling** (temperature + top_k + top_p + seeded LCG;
      `SAMPLER_VERIFY: OK` — kept set/probs/12 draws exact vs numpy). Polish:
      **`tensor::` wrapper module** done & verified for the core ops (context /
      PTX-load / dev-mem / copy / **gemm + rmsnorm** via single-arg
      `cuLaunchKernel`, `TENSOR_GEMM_AND_RMSNORM_OK`); remaining polish is
      moving rope/attn/silu/resid (and the final embed/sample) behind it.
- [~] **G-decode** (final): schema-driven config-interview decode → emits
      `{path, op, value, reason}` SystemSpec JSON; verify byte-compatible with
      the Python/Ollama path on a few interviews. **PROGRESS**: agent-response
      contract **emission** built & verified — `native/gdecode/contract.vyb`
      (question/summary/proposal) via Vyb auto-serialization, all three
      **schema-valid** (`CONTRACT_VERIFY: ALL_OK`), and the module-struct
      `from_string()` round-trip is restored (blocker `rickenator/Vyb#206`
      fixed at `9bada99`). **VERTICAL-SLICE END-TO-END WIRED & VERIFIED**:
      `native/gdecode/run_pipeline.sh` composes prompt → tokenizer encode →
      on-GPU decode (`VYBFORGE_DECODE_PROMPT`) → detokenizer decode →
      agent-response contract emission + jsonschema check — pure Vyb runtime,
      contract `ALL_OK` + round-trip OK (`make gdecode-pipeline`). Slice is the
      inference stand-in; swap the real Qwen3-4B in later. **VERTICAL SLICE
      otherwise complete**:
      substrate → GGUF/JSON/tokenizer loaders → stack → decode → stochastic
      sampling, all Vyb-native on-GPU + reference-verified.

## Serial dependency chain
G-json → (independent) · G-tokenizer → (independent) · G-gguf-data →
G-tensor → G-decode. G-json/G-tokenizer and G-gguf-data can run in parallel.

## Next: swap-in (slice → real Qwen3) — phased
Central model store `~/Models/qwen3/Qwen3-4B-Q4_K_M.gguf` (2.5GB, 24GB VRAM; see
`MODELS.md`). Ground truth captured by `native/gguf/dump_qwen3_meta.py` (gguf-py):
arch qwen3, 36 layers, D=2560, FF=9728, 32/8 heads (GQA 4), eps 1e-6, ctx
262144, file_type 15 = Q4_K_M, 398 tensors (`blk.N.attn_q/k/v/output`,
`attn_q_norm/attn_k_norm/attn_norm/ffn_norm`, `ffn_gate/up/down`, output_norm,
token_embd; Q4_K=12 / Q6_K=14 / F32=0). Saved to `native/out/qwen3_4b_tensors.tsv`.
1. **Real-model GGUF loader — DONE** `native/gguf/read_real_meta.vyb` (uses
   stdlib `io::read_at`, #207, to stream the header/info prefix of the live
   2.5 GB file). Vyb parse == llama.cpp gguf-py on all 398 tensors
   (name/shape/type/offset) + config (`GGUF_META_VERIFY: OK`, `make gguf-real`).
2. **Weights on-GPU + q4_K dequant — DONE** `native/kernels/q4k.vyb` (port of
   llama.cpp `block_q4_K`/`get_scale_min_k4`; signed-byte masking gotcha). Exact
   on a real model weight block vs llama.cpp formula (`Q4K_VERIFY: OK`, bad=0,
   `make q4k`).
3. **Streaming real-weight loader — DONE (issue #5 storage boundary)** —
   `host/q4k_load_driver.vyb` streams real Q4_K bytes via stdlib `read_at` (#207)
   and bulk-dequants with `q4k` to a device buffer; packed bytes stay
   canonical, only the requested slice expands, packed vs expanded memory is
   reported. Verified on a 2048-block real `attn_q` slice (294KB packed →
   4.2MB f64 output; the dequant kernels write f64, so that is the honest figure;
   `ATTNQ_SLICE_VERIFY: OK` bad=0, `make q4k-load`). H2D upload is chunked
   (8 bytes/CUDA call) and read_at length is checked. Whole-model f64 ≈
   32GB > 24GB VRAM → per-tensor/per-layer expansion only (one Qwen3 layer ≈
   404MB f32 / 808MB f64).
   **q6_K dequant — DONE** (attn_v / ffn_down / token_embd bodies):
   `kernels/q6k.vyb`, streamed via read_at (`make q6k`), exact on a real `attn_v`
   Q6_K slice (`Q6K_VERIFY: OK`, bad=0, maxerr ~5e-7). Quant set for a full
   layer is now complete (Q4_K, Q6_K, F32).
4. **Real 1-layer forward — DONE (layer-0 match gate passed).** One-layer weight
   budget (issue #5): blk.0 = 11 tensors, packed 63.9MB → f32 403.7MB → f64
   807.5MB (all fit). Qwen3 layer-0 semantics PINNED (transformers modeling_qwen3
   + real GGUF metadata): D=2560, H=32, KVH=8, HD=128, FF=9728, eps 1e-6,
   `qwen3.rope.freq_base=5e6`. Order: input_layernorm(D) → q/k/v proj →
   **q_norm/k_norm = RMSNorm over HD post-projection** → RoPE (split-half; base
   5e6; identity at pos 0) → causal **GQA** attn (scaling=HD^-0.5, fp32 softmax)
   → o_proj → +residual; then post_attention_layernorm(D) → SiLU(gate)⊗up · down
   → +residual. Weights are GGUF [in,out] row-major (dequant → reshape).
   - `native/gguf/layer0_ref.py` numpy reference (verification-only): vectorized
     full-tensor q4k/q6k dequant (bit-exact vs llama.cpp), implements the pinned
     math for S=2 (tokens 0,1 → exercises nonzero-pos RoPE), writes the contract
     text + `layer0_input.bin`/`layer0_invfreq.bin` (f64, read back by Vyb).
   - `native/kernels/qwen3.vyb` — **qwen3rope** (split-half, base 5e6; k-index
     fix: write `Kd+g2`, not `Kd+gid`) and **f32expand** (norm weights f32→f64).
   - `native/host/layer0_driver.vyb` — tensor-index-wired (parses the tsv, no
     hardcoded offsets): streams all 11 real blk.0 tensors via read_at, bulk
     dequants Q4_K/Q6_K/F32→f64 on-GPU, runs the full layer reusing layer.vyb
     gemm/rmsnorm/attn/silu/resid. Reports packed=63888384 / expanded_f64=
     807446528. (4 module loads need a bounded retry — cuModuleLoadData is flaky;
 filed rickenator/Vyb#209 (String→CString NUL hazard, intermittently
 CUDA_ERROR_INVALID_PTX 218). NOTE: `&&` works fine in Vyb conditions — my
 earlier attribution was wrong; the module-load sweep failure was #209.)
   - `native/tools/verify_layer0.py` + `make layer0` gate: GPU out vs numpy ref
     at tol 2e-3 → **LAYER0_MATCH: OK, maxrel ~5e-6, bad=0** (Phase-3 gate 4).
5. Full 36-layer decode → real text → contract. **PREFILL fully DONE & VERIFIED**:
   `native/gguf/prefill_ref.py` (numpy full-36 prefill) + `native/host/model_driver.vyb`
   (per-layer staged loop, ping-pong, tied lm_head via `emb_dequant` chunked dequant
   + `logits_slice` chunked kernel + `argmax_row`) → `make prefill` →
   **PREFILL_HIDDEN_MATCH: OK (maxrel ~5e-6)** AND **PREFILL_TOP1_MATCH: OK
   (31784, both tokens)**. The real Qwen3-4B next-token argmax is now correct
   on-GPU, pure Vyb. NOTE: the earlier "launch-drop" (#210) was MY bug — a
   `freedom{... return 0}` inside the chunk loop returned after chunk 0, so only
   ~26 token_embd rows dequantized; fixed by returning only after the loop. #209
   (cuModuleLoadData/CString) was a real compiler bug, fixed by Rick.
   **DECODE (greedy autoregressive) FULLY DONE & VERIFIED**: `native/host/decode_driver.vyb`
   keeps all 36 layers packed-resident, dequants per layer per step (handling MIXED
   per-layer quant — attn_v/ffn_down are Q6_K on 0-3 & 31-35, Q4_K elsewhere) and loads
   each layer's RMSNorm weights, runs the full 36-layer forward + tied lm_head argmax →
   `make decode-real` → **DECODE_REAL_MATCH: OK** `[0,1,31784,73305,43790,1888,86805,31784]`
   == numpy. Fixes: vexp NaN→1e14 loop clamp (native/kernels/vmath.vyb), per-layer mixed
   quant offsets+kernels, per-layer norms, and a greedy re-embed bug in `decode_ref.py`
   (was carrying hidden forward instead of re-embedding the full prefix).
6. **kv-cache decode DONE & VERIFIED**: `native/host/kv_driver.vyb` builds the seed forward once via
   `run_layer` (caching each layer's roped-K/V into per-layer `[MAXS,NKV]` cache slabs), then each gen
   step `run_kv` computes ONLY the new position (k/v cached, attention attends the new q over cached
   K/V). `make decode-kv` → **DECODE_REAL_MATCH: OK**, same tokens as the recompute decode and numpy.
   `qwen3rope` gained a +224 POS (base-position) param for single-row roping; decode/model drivers set
   +224=0 (ROP alloc 232). Gotcha hit: single-row rmsnorm/rope `S` is 1 (using 0 launches zero threads,
   collapses the stream to a constant token).
7. Qwen3-8B dogfood (fetch).
Vyb-native **training** is a separate, later research-scale plan.

## Hard-won Vyb gotchas (see native/README.md for detail)
- libm math (`exp/sin/cos/tan`) doesn't lower to device code → pure-arithmetic
  device fns (`vexp/vsin/vcos/vsqrt` in `native/kernels/vmath.vyb`).
- `X * (Y)` parses as `ptr<X>` and silently nulls → always expand to a plain
  multiply chain. Avoid `&&` / `not` (use nested if/else). No scientific float
  literals (`1e-6`); write `0.000001`. Ranges are inclusive (`0..n-1`).
- `import mod::{a, b}` (braced) for flat modules; `--module-path` to resolve.
- Fixed-array element writes now work (Vyb #201 fixed) — but Vec+push still
  fine host-side.
