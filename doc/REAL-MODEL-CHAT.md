# Real-model `llm.chat` — GPU 36-layer forward composed with stdlib/vllm CPU head

> Status: Checkpoint A + A2 DONE, hardware-verified (2026-09-11).
> Checkpoint B DONE, hardware-verified (2026-09-11): `qwen3.vyb` gained an
> `embed` kernel and `model_driver.vyb` now builds the layer-0 input via GPU
> `embed(DE, prids, XA)` from the real token_embd instead of the baked
> `layer0_input.bin`; the gather reproduces the [0,1] gold (max abs diff 5e-8)
> and `make chat-real` PASSES (top1 [31784,31784], prefill hidden preserved).
> Arbitrary prompts: `model_driver` reads `VYB_PROMPT`, tokenizes via
> stdlib/vllm `llm_encode`, embeds those ids, runs a variable-S forward, and
> `make chat-prompt` (default "The capital of France is" = 5 tokens
> `[785,6722,315,9625,374]`) gates the GPU per-position lm_head top1 against the
> numpy gold from `prompt_ref.py` ([87054,62868,71961,27051,19151]) — MATCHED
> token-for-token (CHAT-PROMPT: PASS). Regenerate gold via `make chat-prompt-ref`.
> `model_driver.vyb` now composes `llm::` (stdlib/vllm CPU decode) on the real
> Qwen3-4B: full 36-layer forward -> tied lm_head argmax `[31784,31784]` ->
> stdlib/vllm CPU decode -> response `<CogCog>` (the model's deterministic
> 2-token continuation decoded to text). `make -f native/Makefile chat-real`
> gates it (CHAT-REAL: PASS; prefill hidden preserved maxrel 4.9e-6).
> Follows doc/LLM-FORWARD.md status (F1 + F2 DONE).

## Goal
A real-model `llm_chat(prompt) -> String` on the actual Qwen3-4B GGUF:

    response = llm_load(dir)            # stdlib/vllm, real BPE {-vocab.json, merges.txt}
             -> llm_encode(prompt)      # CPU token ids (stdlib/vllm)
             -> [ embed ids -> hidden ] # token_embd gather (GPU)
             -> [ 36-layer forward ]    # model_driver's verified machinery (GPU)
             -> [ output_norm + tied lm_head -> logits ]  (GPU)
             -> llm_pick_argmax / llm_decode  # CPU head (native/llm/llm.vyb)
             -> response String

Closes the CPU(stdlib/vllm) <-> GPU(bindings) composition loop on the REAL
model — the F2 generalization of llm_gpu.vyb (which proved the same contract on
a fixture echo-head).

## What's already verified (do NOT regress)
- `make -f native/Makefile prefill` -> 36-layer hidden maxrel 4.9e-6; tied
  lm_head top1 `[31784, 31784]` == numpy (`model_driver.vyb` + `verify_prefill.py`).
- decode_driver.vyb gold tokens `[0,1,31784,73305,...]`, `make decode-real`.
- CPU head `native/llm/llm.vyb` 7/7; F1 GPU wiring `llm_gpu.vyb` 4/4.

## Seam discipline (constraint discovered this session)
`extern "C"` + `freedom{}` FFI appear ONLY in programs (demos/examples/fixtures/
drivers); no stdlib module contains them. So the GPU forward CANNOT be lifted
into a `share(all)` stdlib-style module — it stays in a *driver* (consistent
with LLM-FORWARD.md: "The GPU forward is NOT in this module … live in a main
driver"). "Reusable main-less module" therefore = a driver-local forward
function, parameterized over S and embedded-input, reused by driver programs
via the llm facade. Shared reuse lives on the CPU side (llm.vyb), as already
established.

## Checkpoints (each hardware-verified, prefill stays green)
- **A. Generalize the verified forward into a callable driver function.**
  Factor model_driver.vyb so the 36-layer forward + lm_head is a standalone
  `model_forward(embeddedHidden, S, …) -> logits/top1` (extern/freedom stay
  driver-side), identical semantics. Gate: refactored model_driver reproduces
  `prefill` (hidden + top1 unchanged).
- **A2. First closed-loop real head.** With the S=2 embedded input already
  verified, run model_forward -> argmax 31784 -> llm_decode -> the decoded
  String == stdlib/vllm decode of 31784. First real llm.chat output (1 token).
- [x] **B. Arbitrary prompts. DONE.** Added `embed` kernel (in qwen3.vyb, the
  already-loaded forward module) so any `llm_encode` ids become the GPU hidden;
  `model_driver` reads `VYB_PROMPT`, tokenizes, embeds, runs variable-S forward.
  `make chat-prompt` ("The capital of France is") gates per-position top1 == numpy
  gold `prompt_ref.py` — matched token-for-token (CHAT-PROMPT: PASS).
- **C. wire into a `llm_chat` driver + Makefile target** (`make chat-real`),
  gate on tokens matching decode-driver gold; update plan status.

## Risks / notes
- Full prefill is ~minutes on the 3090 for S=2 in this driver; keep S small for
  iteration. Bigger S (S=83 corpus) is the training loop's proven regime.
- Embed gather: confirm whether qwen3.ptx needs an `embed` kernel or reuse
  train.ptx (single-module rule: one .ptx per process).
- Keep model_driver.vyb byte-stable until Checkpoint A gate is green; regen
  prefill artifacts only against committed numpy refs.
