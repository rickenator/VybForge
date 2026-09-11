# llm:: — GPU-composing forward facade (CPU(vllm) + GPU(bindings)) — mapped plan

> Status: F1 DONE, hardware-verified (2026-09-11). `native/llm/llm.vyb` +
> `native/llm/llm_probe.vyb` (CPU leg) + `native/llm/llm_gpu.vyb` (F1 wiring):
> on the RTX 3090 a REAL `gemm` forward (layer.ptx) computes logits -> `llm_head`
> -> decoded response == host reference (A->'A', B->'B', input-dependent), 4/4
> PASS. CPU leg 7/7 on the real tokenizer. F2 (real Qwen3-4B via model_driver,
> needs the GGUF on the box) remains the only open item.
> Steering: VybOS issue #9.

## Goal
A typed `llm::` facade with `llm_chat(prompt) -> response`:
  1. tokenize via stdlib/vllm `model_encode` (CPU, real Qwen3 BPE vocab)
  2. embed + GPU transformer forward (bindings/cuda + native kernels), logits
  3. sample via stdlib/vllm sampler (temp + top_k + top_p)
  4. decode via stdlib/vllm `model_decode` -> response String

## Reuse (no new kernels)
The GPU substrate is proven: 1-layer forward (verify_layer.py), 4-layer
weight-tied stack (verify_stack.py), full 36-layer Qwen3-4B prefill + tied
lm_head (model_driver.vyb -> verify_prefill). The real 4B GGUF is NOT on this
host, so F1 verifies on a small hand-wired fixture model through the REAL GPU
kernels (layer.vyb gemm/rmsnorm/attn/silu/resid + qwen3.vyb rope).

## Checkpoints (each hardware-verified on the RTX 3090)
- **F1. Facade contract + fixture-model round trip.** `native/llm/llm.vyb`:
  `llm_chat` composed as encode -> GPU forward (fixture weights) -> sample ->
  decode. Driver builds a tiny deterministic 1-layer model (host `mk_llm_fixture`
  writes weights + a host reference), tokenize/sample/decode through the real
  stdlib/vllm tokenizer, GPU forward on the 3090, verify the decoded response
  string == host reference. **[THE hard boundary: proves the real
  CPU(stdlib/vllm) <-> GPU(bindings) composition end to end on silicon]**
- **F2. Real-Qwen3-4B wiring.** Point the same facade at the 36-layer
  model_driver machinery with the GGUF present (hardware-verifiable only when
  the model file is on the box; cf. verify_prefill).
