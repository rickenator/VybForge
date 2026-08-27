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
- [~] **G-tensor** — **complete for the vertical slice**: multi-layer stack
      (`STACK_VERIFY: OK`), greedy autoregressive decode (`DECODE_VERIFY: OK`),
      and **stochastic sampling** (temperature + top_k + top_p + seeded LCG;
      `SAMPLER_VERIFY: OK` — kept set/probs/12 draws exact vs numpy). Remaining
      polish: kernel-side embed/sample + the tensor:: module wrapper.
- [ ] **G-decode** (final): schema-driven config-interview decode → emits
      `{path, op, value, reason}` SystemSpec JSON; verify byte-compatible with
      the Python/Ollama path on a few interviews. **VERTICAL SLICE otherwise
      complete**: substrate → GGUF/JSON/tokenizer loaders → stack → decode →
      stochastic sampling, all Vyb-native on-GPU and reference-verified.

## Serial dependency chain
G-json → (independent) · G-tokenizer → (independent) · G-gguf-data →
G-tensor → G-decode. G-json/G-tokenizer and G-gguf-data can run in parallel.

## Hard-won Vyb gotchas (see native/README.md for detail)
- libm math (`exp/sin/cos/tan`) doesn't lower to device code → pure-arithmetic
  device fns (`vexp/vsin/vcos/vsqrt` in `native/kernels/vmath.vyb`).
- `X * (Y)` parses as `ptr<X>` and silently nulls → always expand to a plain
  multiply chain. Avoid `&&` / `not` (use nested if/else). No scientific float
  literals (`1e-6`); write `0.000001`. Ranges are inclusive (`0..n-1`).
- `import mod::{a, b}` (braced) for flat modules; `--module-path` to resolve.
- Fixed-array element writes now work (Vyb #201 fixed) — but Vec+push still
  fine host-side.
