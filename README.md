# VybForge

VybForge is the shop floor where the Vyb language stops being a compiler demo
and starts doing work on the machine. Two products share one repo: **Spec**, a
desired-state configuration interviewer for VybOS with a deterministic applier,
and **Infer**, a Vyb-native, on-GPU model-inference and compute substrate.

Everything here **drafts desired state only.** It never builds, realizes,
activates, deploys, or edits VybOS or the host, and it never touches the
`~/Projects/VybOS` tree.

## Spec — desired-state interviewer

Spec turns a human OS-build goal into a reviewable **SystemSpec**. An LLM is
optional; the deterministic backbone is not:

```
config/default-state.json   (a known-good SystemSpec baseline)
        + confirmed proposed_changes  ({path, op, value, reason})
        ── tools/apply.vyb (Vyb 0.7.3, runs locally) ──>
        out/spec.json    (validated, merged machine contract)
        out/system.vyb   (a self-contained config-as-program that reproduces it)
```

`tools/apply.vyb` validates each patch against the contract (path/op/value),
merges add/replace/remove onto the baseline, and renders `out/spec.json` +
`out/system.vyb`, then compiles and runs the rendered program to prove it
reproduces the spec. Schema, generator, interview client, and applier share one
contract: `{path, op: add|replace|remove, value, reason}` targeting
`system | hostname | pkgs | services` (the real VybOS `SystemSpec` shape).

```sh
VYB_BIN="$HOME/Projects/Vyb-vybos/build/vyb" ./tools/apply_interview.py patches.jsonl
```

Run the interviewer with `./run.sh` (GPU Ollama / OpenAI Responses /
OpenAI-compatible Chat Completions including Hermes) or, as a dependency-light
CPU fallback, `./run-vyb.sh` / `./vyb-run.sh` (Vyb + local Ollama).

## Infer — Vyb-native inference / compute substrate

Under `native/`, the part of this repo that is the 6-month center of gravity:
a **Vyb-native** decode of the Qwen3-4B configurator on an RTX 3090, with zero
Python in the production pipeline (Python is reference-verification only).
See `native/README.md` for the verified kernel table (GEMM, RMSNorm, exp/sin/cos,
one transformer layer, GGUF reader + q4_0 dequant, JSON parser, Qwen3 BPE
tokenizer, multi-layer stack, autoregressive decode, stochastic sampler, and the
`tensor::` wrapper). CUDA is today's device backend; host code talks to
buffers/kernels, not to a specific vendor as a product — a future sponsor can
change the backend without renaming the shop.

## Configuration

Env vars are `VYBFORGE_*`. The legacy `VYBAICONF_*` names still work as a
deprecated fallback for one cycle; `VYBFORGE_*` wins when both are set.

```sh
# OpenAI Responses API
export VYBFORGE_BACKEND=openai-responses
export VYBFORGE_MODEL='your-model-name'
export OPENAI_API_KEY='...'
./run.sh

# Hermes or another OpenAI-compatible Chat Completions gateway
export VYBFORGE_BACKEND=openai-chat
export VYBFORGE_ENDPOINT='https://model-gateway.example/v1'
export VYBFORGE_MODEL='provider-model-name'
export VYBFORGE_API_KEY='...'
./run.sh
```

`VYBFORGE_API_KEY` is optional for `openai-chat`, which permits an
unauthenticated trusted-LAN vLLM listener. Set it whenever the gateway requires
bearer authentication. Some gateways lack JSON Schema; set
`VYBFORGE_STRUCTURED_OUTPUT=json_object` (or `prompt`) to relax transport
enforcement — VybForge still parses the returned JSON. API keys live in the
environment, never in the repo.

## Included

- `tools/apply.vyb` + `tools/apply_interview.py` — deterministic desired-state
  applier (Vyb core, Python plumbing).
- `app/configurator.py` + `run.sh` — backend-neutral interviewer launcher.
- `src/main.vyb` — portable Linux Vyb client for local Ollama.
- `config/` — default-state (real SystemSpec baseline) and response schemas.
- `data/` — deterministic VybOS seed corpus: 216 train / 24 eval records.
- `native/` — the Vyb-native on-GPU inference substrate (see `native/README.md`).
- `training/` — generator, QLoRA code, explicit job launcher, and handoff
  rules. Retrained/inferred on godzilla's RTX 3090.
- The final LoRA adapter and tokenizer are committed under
  `artifacts/vybos-configurator-lora/`; the much larger public base model is
  pulled from Hugging Face when training or using the adapter.

## Training

```sh
./training/generate-data.sh
./training/start-training.sh --start-training <host>
```

The second command is the only submission gate: it creates an isolated target
environment, verifies CUDA, and launches QLoRA in the background. Read
[REPRODUCING.md](REPRODUCING.md), [HANDOFF.md](HANDOFF.md), and
`training/AGENTS.md` before changing the corpus or running training. `make -f
native/Makefile verify` is the native-suite entry point.

## Boundaries

- Draft-only: no VybOS builds/realizes/execs/generations, no host edits.
- No secrets in git. No personal host URLs. Machine paths appear only as
  documented `$HOME/Projects/...` examples the owner already uses.
- Python is reference-verification only in the shipped decode/interview runtime.
- Ask the owner before: the GitHub-side repo rename, force-push/rewriting
  history, deleting `artifacts/vybos-configurator-lora/`, or running a training
  job.
