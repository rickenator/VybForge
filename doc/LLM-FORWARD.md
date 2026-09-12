# llm:: — GPU-composing forward facade (CPU(vllm) + GPU(bindings)) — mapped plan

> Status: F1 + F2 DONE, all hardware-verified (2026-09-11).
> F1: `native/llm/*` facade (CPU(vllm) head, native/llm/llm.vyb + probes) — CPU
> leg 7/7 on the real tokenizer; F1 wiring (llm_gpu.vyb) 4/4 on GPU
> (real gemm -> llm_head -> decoded response == host).
> F2: REAL Qwen3-4B forward gate PASSES on GPU — 36-layer prefill
> hidden maxrel 4.9e-6, tied lm_head top1 [31784,31784] exact vs the numpy oracle
> (model_driver + verify_prefill; as tested on an RTX 3090). Model at ~/Models/qwen3/.
> A true real-model `llm.chat` (facade over the full 36-layer forward) is a
> further extraction of model_driver's forward into a reusable main-less module
> (next slice; the forward itself is verified).
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

## Checkpoints (each hardware-verified on GPU)
- **F1. Facade contract + fixture-model round trip.** `native/llm/llm.vyb`:
  `llm_chat` composed as encode -> GPU forward (fixture weights) -> sample ->
  decode. Driver builds a tiny deterministic 1-layer model (host `mk_llm_fixture`
  writes weights + a host reference), tokenize/sample/decode through the real
  stdlib/vllm tokenizer, GPU forward on GPU, verify the decoded response
  string == host reference. **[THE hard boundary: proves the real
  CPU(stdlib/vllm) <-> GPU(bindings) composition end to end on silicon]**
- **F2. Real-Qwen3-4B wiring.** Point the same facade at the 36-layer
  model_driver machinery with the GGUF present (hardware-verifiable only when
  the model file is on the box; cf. verify_prefill).
