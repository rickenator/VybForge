#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  echo "Usage: $0 --start-training <host> [--model <hf-model>]"
  echo "This is the only command that submits training. Generating data never starts a job."
}

if [[ ${1:-} != --start-training || -z ${2:-} ]]; then
  usage
  exit 2
fi
host=$2
shift 2
model='Qwen/Qwen3-4B'
if [[ ${1:-} == --model && -n ${2:-} ]]; then
  model=$2
  shift 2
fi
[[ $# -eq 0 ]] || { usage; exit 2; }
[[ $model =~ ^[A-Za-z0-9._/-]+$ ]] || { echo 'Unsafe model identifier' >&2; exit 2; }

"$root/training/generate-data.sh"
ssh "$host" 'command -v python3 >/dev/null && mkdir -p ~/Projects/VybAIConf'
tar -C "$root" -czf - data training | ssh "$host" 'tar -xzf - -C ~/Projects/VybAIConf'
ssh "$host" "cd ~/Projects/VybAIConf && \
  python3 -m venv .venv && \
  .venv/bin/python -m pip install --upgrade pip && \
  .venv/bin/python -m pip install torch && \
  .venv/bin/python -m pip install -r training/requirements.txt && \
  .venv/bin/python -c 'import torch; assert torch.cuda.is_available(), \"CUDA is unavailable\"; print(\"CUDA\", torch.version.cuda, torch.cuda.get_device_name(0))' && \
  mkdir -p artifacts && \
  nohup .venv/bin/python training/train_lora.py --model '$model' >artifacts/train.log 2>&1 < /dev/null & echo TRAINING_PID=\$!"
