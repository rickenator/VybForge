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
1. Real-model GGUF loader: full GGUF v3 metadata-spec reader (all KV types +
   arrays) → header config + tensor index matching the inventory above.
2. Weights on-GPU + q4_K dequant (only q4_0 today).
3. Real 1-layer forward vs llama.cpp/python reference.
4. Full 36-layer decode → real text → contract.
5. LoRA merge (r16; q/k/v/o/gate/up/down) → configurator behavior.
6. Qwen3-8B dogfood (fetch).
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
