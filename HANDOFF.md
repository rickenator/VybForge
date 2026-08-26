# VybAIConf Handoff

## Current state

There are two paths: `src/main.vyb` directly calls local Ollama without Hermes,
and `artifacts/vybos-configurator-lora/` contains a Qwen3-4B LoRA adapter.
The Vyb executable is CPU-oriented; adapter training/inference needs CUDA.

## Reference training result

- Base: `Qwen/Qwen3-4B`
- Hardware: ARM64 NVIDIA GB10, CUDA 13
- Corpus: 216 train / 24 eval deterministic records
- Method: QLoRA NF4, rank 16, alpha 32, dropout 0.05
- Run: 3 epochs, 42 steps, batch 1, accumulation 16
- Final logged evaluation loss: 0.05948377028107643

The successful reference run completed all 42 steps and saved a 66 MB final
adapter, committed directly in this private repository with its tokenizer.
Training/inference downloads only the public base model from Hugging Face. The
reference adapter was loaded with the base model and, with Qwen3 thinking
disabled, returned valid configurator JSON for an ARM64 appliance interview
prompt.

## Limitations

This seed corpus is small and template-heavy, so low held-out loss primarily
measures format/template fit. It does not establish broad VybOS design skill.
Read `training/AGENTS.md` before extending the corpus: add only verified
VybOS behavior, include boundary/regression examples, and use a
capability-separated human-reviewed evaluation set.

## Repository hygiene

Keep this final adapter and tokenizer. Do not commit resumable checkpoints,
base-model caches, Python environments, optimizer state, raw training logs, or
host data. Base weights are retrieved from Hugging Face at training/inference
time.
