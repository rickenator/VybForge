# VybForge central model store

Real model weights are **never committed** to this repo. They live in a single
central location on the machine, referenced by path (never copied in):

```
~/Models/<family>/<Name>-Q<K>_M.gguf
```

Example (present on this host, used for the G-decode swap-in):

```
~/Models/qwen3/Qwen3-4B-Q4_K_M.gguf      (2.5 GB, Q4_K_M)   <-- 4B base
```

Convention:
- One directory per model family (`~/Models/qwen3/`, `~/Models/qwen3-8b/`, …).
- GGUF filename = canonical model id + quant tag (e.g. `Qwen3-4B-Q4_K_M.gguf`).
- The fine-tuned LoRA adapter + tokenizer **are** committed, under
  `artifacts/vybos-configurator-lora/`; only the large public *base* model is
  fetched centrally.
- Reference runner available: `llama-cli` / `llama-server` (used only for
  baseline verification of a Vyb-native read/decode — never in the Vyb runtime).

Targets for swap-in / dogfood:
- Qwen3-4B  — present (`~/Models/qwen3/Qwen3-4B-Q4_K_M.gguf`).
- Qwen3-8B  — to fetch (no local copy yet).
- Qwen3.8-27B / Flash-Next — already central under `~/checkpoints/` (other work).

Anything that loads a base model reads it from `~/Models/...` and says so in
its header comment — no model is ever looked up inside tool-specific caches.
