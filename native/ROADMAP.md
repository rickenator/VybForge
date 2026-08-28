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
   and bulk-dequants with `q4k` to an f32 device buffer; packed bytes stay
   canonical, only the requested slice expands, packed vs expanded memory is
   reported. Verified on a 2048-block real `attn_q` slice (294KB packed →
   2.1MB f32, `ATTNQ_SLICE_VERIFY: OK` bad=0, `make q4k-load`). Whole-model f64 ≈
   32GB > 24GB VRAM → per-tensor/per-layer expansion only (one Qwen3 layer ≈
   404MB f32).
   **q6_K dequant — DONE** (attn_v / ffn_down / token_embd bodies):
   `kernels/q6k.vyb`, streamed via read_at (`make q6k`), exact on a real `attn_v`
   Q6_K slice (`Q6K_VERIFY: OK`, bad=0, maxerr ~5e-7). Quant set for a full
   layer is now complete (Q4_K, Q6_K, F32).
4. **Real 1-layer forward — IN PROGRESS.** One-layer weight budget (issue #5):
   blk.0 = 11 tensors, packed 63.9MB → f32 403.7MB → f64 807.5MB (all fit).
   Staging plan: stream each tensor via read_at + bulk q4k/q6k dequant into an
   f32/f64 device buffer (norms F32 direct). Then the Qwen3 layer kernels:
   - qk_norm: RMSNorm over the 128-dim head (attn_q_norm / attn_k_norm)
   - rope: Qwen3 rotary (pin exact scheme/freq from Qwen3 + llama.cpp before coding)
   - qkv attention: GQA (32 Q / 8 KV), causal softmax, 4096 out
   - mlp: SiLU(gate)⊗up → down; reuse gemm / rmsnorm / resid / gemm kernels
   Reference: python layer-0 using real dequantized weights (verification-only).
5. Full 36-layer decode → real text → contract.
6. LoRA merge (r16; q/k/v/o/gate/up/down) → configurator behavior.
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
