# Reproducing VybAIConf

## Clone source, data, and final adapter

```sh
git clone git@github.com:rickenator/VybAIConf.git
cd VybAIConf
```

## Run a GPU or hosted interviewer (recommended)

The CPU-only Vyb executable is intentionally not the normal runtime. Use a
GPU-backed Ollama instance or a hosted model with `run.sh`.

```sh
# GPU-backed Ollama
ollama pull qwen3:8b
./run.sh

# OpenAI Responses
VYBAICONF_BACKEND=openai-responses \
VYBAICONF_MODEL='your-model-name' \
OPENAI_API_KEY='...' ./run.sh

# Hermes or an OpenAI-compatible gateway
VYBAICONF_BACKEND=openai-chat \
VYBAICONF_ENDPOINT='https://model-gateway.example/v1' \
VYBAICONF_MODEL='provider-model-name' \
VYBAICONF_API_KEY='...' ./run.sh
```

The launcher supports strict JSON Schema by default. If the selected compatible
gateway lacks it, set `VYBAICONF_STRUCTURED_OUTPUT=json_object` or `prompt`.
No launcher path calls VybOS.

## Build the Vyb/Ollama CPU fallback

Install Vyb and Ollama on Linux. Ollama must listen locally at `127.0.0.1:11434`.

```sh
export VYB_BIN="$HOME/Projects/Vyb/build/vyb"
mkdir -p bin
"$VYB_BIN" src/main.vyb --build bin/vyb-configurator -O2
ollama pull qwen3:4b
./run-vyb.sh
```

This path sets Qwen3 thinking off, uses an 8192-token context and 256-token
generation limit. It is retained for portable development only and never calls
VybOS.

## Recreate the corpus

```sh
export VYB_BIN="$HOME/Projects/Vyb/build/vyb"
./training/generate-data.sh
python3 - <<'PY'
import json
from pathlib import Path
for path in Path('data').glob('*.jsonl'):
    for line in path.read_text().splitlines():
        json.loads(json.loads(line)['messages'][-1]['content'])
    print(path, 'valid')
PY
```

Expected split: 216 training and 24 evaluation records. The generator is
deterministic Vyb code; it filters Vyb's numeric `main()` return line to retain
strict JSONL.

## Recreate QLoRA

Reference run: `Qwen/Qwen3-4B`, one ARM64 NVIDIA GB10 with CUDA 13, 4-bit NF4
loading, rank-16 LoRA, three epochs, batch size 1, accumulation 16, and 2048
maximum tokens. The final adapter/tokenizer are already in this repository.
`training/train_lora.py` pulls the public base model from Hugging Face when
retraining and overwrites the local adapter path.

```sh
./training/start-training.sh --start-training <host> --model Qwen/Qwen3-4B
```

The launcher copies only `data/` and `training/`, creates a target `.venv`,
installs the pinned Python dependencies, checks CUDA, and writes logs/artifacts
under the target's `~/Projects/VybAIConf/`. It runs only with this explicit flag.

To test the included adapter on CUDA:

```sh
.venv/bin/python training/smoke_adapter.py --prompt \
  'I want an ARM64 VybOS appliance image. What should we decide first?'
```
