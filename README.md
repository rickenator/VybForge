# VybAIConf

VybAIConf is a local, schema-guided configuration interviewer for VybOS. It
ships a Vyb executable for local Ollama and source for reproducing a small
Qwen3-4B LoRA adapter that follows the same review-first JSON contract.

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

## Quick start

```sh
git clone git@github.com:rickenator/VybAIConf.git
cd VybAIConf
export VYB_BIN="$HOME/Projects/Vyb/build/vyb"
mkdir -p bin
"$VYB_BIN" src/main.vyb --build bin/vyb-configurator -O2
ollama pull qwen3:4b
./run-vyb.sh
```

The executable sends a local request to Ollama. Its stdout is the Ollama JSON
envelope; nested `message.content` is the configurator JSON. Set
`VYBAICONF_MODEL=qwen3:8b` for the larger local model.

## Training

```sh
./training/generate-data.sh
./training/start-training.sh --start-training <host>
```

The second command is the only submission gate. It creates an isolated target
environment, verifies CUDA, and launches QLoRA in the background. Read
[REPRODUCING.md](REPRODUCING.md) and [HANDOFF.md](HANDOFF.md) before changing
the corpus or running training.
