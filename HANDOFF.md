# VybAIConf Handoff

## Current state

There are three model paths: `src/main.vyb` directly calls local Ollama without
Hermes; `run.sh` supports GPU Ollama, OpenAI Responses, and OpenAI-compatible
Chat Completions gateways; and `artifacts/vybos-configurator-lora/` contains a
Qwen3-4B LoRA adapter. The Vyb executable is CPU-only fallback, not the
recommended interview runtime. Adapter training/inference needs CUDA.

For useful, responsive configuration interviews, provide GPU-backed local
inference or a hosted model. Hermes is supported as a gateway protocol, not as
a required profile: point `VYBAICONF_BACKEND=openai-chat` at its `/v1` base URL
and put its credential only in `VYBAICONF_API_KEY`. Use
`openai-responses` for an OpenAI API model. Do not embed service URLs, keys, or
host-specific assumptions in the repository. A Codex-capable model, when
authorized for the user's OpenAI API project, uses the same Responses path;
VybAIConf never attempts to borrow an interactive Codex login.

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
