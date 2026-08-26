# VybAIConf

VybAIConf is a schema-guided configuration interviewer for VybOS. It ships a
portable Vyb executable for local Ollama, a backend-neutral Python launcher,
and source for reproducing a small Qwen3-4B LoRA adapter that follows the same
review-first JSON contract.

It drafts desired state only. It never builds, realizes, activates, deploys,
or modifies VybOS or the host.

## Included

- `src/main.vyb` — portable Linux Vyb client for local Ollama.
- `config/` — complete mock-system and response schemas.
- `data/` — deterministic VybOS seed corpus: 216 train / 24 eval records.
- `training/` — generator, QLoRA code, explicit job launcher, and handoff
  rules.
- The final 66 MB LoRA adapter and tokenizer are committed directly under
  `artifacts/vybos-configurator-lora/`. The much larger public base model is
  pulled from Hugging Face when training or using the adapter.

## Model backing: firm recommendation

Do not treat CPU-only local inference as the normal deployment. The portable
Vyb executable is useful as a dependency-light fallback, but CPU latency and a
small quantized model make it a poor configuration interviewer. Use one of the
following for real sessions:

- GPU-backed Ollama, with a capable local model, through `./run.sh`.
- A hosted OpenAI Responses model through `./run.sh`.
- Any trustworthy OpenAI-compatible Chat Completions gateway, including a
  Hermes-style local or remote gateway, through `./run.sh`.

For a Codex-capable OpenAI API model authorized on your project, use the
Responses path. This is API-key based; it does not attempt to reuse an
interactive Codex login or browser session.

The launcher sends only the mock desired-state fixture and the conversation to
the selected provider. It never writes credentials, modifies VybOS, or applies
a configuration.

## Quick start

```sh
git clone git@github.com:rickenator/VybAIConf.git
cd VybAIConf
ollama pull qwen3:8b
./run.sh
```

For hosted models, keep the API key in the environment, never in this
repository:

```sh
# OpenAI Responses API
export VYBAICONF_BACKEND=openai-responses
export VYBAICONF_MODEL='your-model-name'
export OPENAI_API_KEY='...'
./run.sh

# Hermes or another OpenAI-compatible Chat Completions gateway
export VYBAICONF_BACKEND=openai-chat
export VYBAICONF_ENDPOINT='https://model-gateway.example/v1'
export VYBAICONF_MODEL='provider-model-name'
export VYBAICONF_API_KEY='...'
./run.sh
```

Some compatible gateways do not support JSON Schema. Set
`VYBAICONF_STRUCTURED_OUTPUT=json_object` (or, as a last fallback, `prompt`) to
relax transport enforcement; VybAIConf will still parse the returned JSON.

`./run-vyb.sh` and the compatible `./vyb-run.sh` name invoke the Vyb/Ollama
CPU fallback. They print a warning deliberately. Their stdout is the raw
Ollama response envelope; nested `message.content` is the configurator JSON.

## Training

```sh
./training/generate-data.sh
./training/start-training.sh --start-training <host>
```

The second command is the only submission gate. It creates an isolated target
environment, verifies CUDA, and launches QLoRA in the background. Read
[REPRODUCING.md](REPRODUCING.md) and [HANDOFF.md](HANDOFF.md) before changing
the corpus or running training.
