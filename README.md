# VybForge

VybForge is the shop floor where the Vyb language stops being a compiler demo
and starts doing work on the machine. Two products share one repo: **Spec**, a
desired-state configuration interviewer for VybOS with a deterministic applier,
and **Infer**, a Vyb-native, on-GPU model-inference and compute substrate.

Everything here **drafts desired state only.** It never builds, realizes,
activates, deploys, or edits VybOS or the host, and it never touches the
`<VybOS checkout>` tree.

## Environment — one knob for the toolchain (VybForge#15)

Set `VYBHOME` to your Vyb checkout; nothing in this repo needs a path literal.
`vybenv.sh` — sourced by `run.sh`, every gate under `native/legit/`,
`native/gdecode/run_pipeline.sh` and `training/generate-data.sh` — resolves

| var | value |
| --- | --- |
| `VYB` | `$VYBHOME/build/vyb` |
| `VYB_STDLIB` | `$VYBHOME/stdlib` |
| `VYBOS` / `VYBOSHOME` | the sibling VybOS checkout (repair + ledger gates) |

Resolution order, the same rule as Vyb's own `SOURCEME_VYB` (rickenator/Vyb#424,
shipped at the checkout toplevel): `$VYBHOME/SOURCEME_VYB` is sourced when
`VYBHOME` already names a checkout; then `VYBHOME` itself — canonical, with `VYB`
and `VYB_STDLIB` derived from it. A stale exported `VYB` deliberately does *not*
win, so one checkout's binary can never be mixed with another's stdlib. Then an
explicit `$VYB_BIN`/`$VYB`, used only to *derive* the home, then
`$HOME/Projects/Vyb`. With none of those the scripts stop with the exact `export`
line to run — never a silent build against the wrong toolchain. One line in your
shell replaces all of it:

```sh
. "$VYBHOME/SOURCEME_VYB"
```

Inside Vyb drivers the toolchain is resolved at use (`tools/apply_interview.vyb`
re-invokes the compiler; `native/host/chat_server.vyb` spawns the decode driver) —
same precedence, same failure mode.

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
# patches come from $VYBFORGE_PATCHES (default out/patches.jsonl). The toolchain is
# located from $VYBHOME (vybenv.sh; VybForge#15 — Vyb publishes it in
# $VYBHOME/SOURCEME_VYB, rickenator/Vyb#424).
export VYBHOME="$HOME/Projects/Vyb"        # or: . "$VYBHOME/SOURCEME_VYB"
VYBFORGE_PATCHES=patches.jsonl "$VYBHOME/build/vyb" tools/apply_interview.vyb --module-path native/json
```

Run the interviewer with `./run.sh` (GPU Ollama / OpenAI Responses /
OpenAI-compatible Chat Completions including Hermes) or, as a dependency-light
CPU fallback, `./run-vyb.sh` / `./vyb-run.sh` (Vyb + local Ollama).

## Infer — Vyb-native inference / compute substrate

Under `native/`, the part of this repo that is the 6-month center of gravity:
a **Vyb-native** decode of the Qwen3-4B configurator on GPU, with zero
Python in the production pipeline (Python is reference-verification only).
See `native/README.md` for the verified kernel table (GEMM, RMSNorm, exp/sin/cos,
one transformer layer, GGUF reader + q4_0 dequant, JSON parser, Qwen3 BPE
tokenizer, multi-layer stack, autoregressive decode, stochastic sampler, and the
`tensor::` wrapper). CUDA is today's device backend; host code talks to
buffers/kernels, not to a specific vendor as a product — a future sponsor can
change the backend without renaming the shop. Real base-model weights are kept
central (never committed): see `MODELS.md`.

## Configuration

Env vars are `VYBFORGE_*`.

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

- `tools/apply.vyb` + `tools/apply_interview.vyb` — deterministic desired-state
  applier (Vyb core; the Python plumbing is gone).
- `native/legit/run_phase1_battery.sh` + `run_*_gate.sh` — the verification
  entry point: one command runs every Phase-1 acceptance check (schema, applier,
  repair boundary, configurator, interview, coerce, GGUF fixture, JSON unit,
  contract verifier, production-path audit) and prints one line per step. The
  per-step gates are described in `doc/PYTHON-CLEANUP.md`.
- `app/configurator.vyb` + `run.sh` — backend-neutral interviewer launcher
  (Vyb-native; the Python original is gone). stdout carries one JSON contract per
  answer, the banner/prompt go to stderr.
- `src/main.vyb` — portable Linux Vyb client for local Ollama.
- `config/` — default-state (real SystemSpec baseline) and response schemas.
- `data/` — deterministic VybOS seed corpus: 216 train / 24 eval records.
- `native/` — the Vyb-native on-GPU inference substrate (see `native/README.md`).
- `native/legit/forge_buildrecord.vyb` + `run_forge_ledger.sh` — **shared-ledger
  build-record posting**: the forge computes a deterministic artifact hash and
  seals an AUTHENTIC signed build record per package, then POSTS it to the
  shared VybOS registry ledger (`VybOS/modules/ledger.vyb`, consumed
  cross-repo via `--module-path`). Run `./native/legit/run_forge_ledger.sh`
  (see also `native/legit/run_forge_legit_signed.sh` for the forge-local signed
  provenance path).
- `tools/propose_repair.vyb` + `tools/validate_proposal.vyb` (`run_repair_proposal.sh`)
  — **#7 model boundary**: `propose_repair` emits a schema-constrained
  `PatchProposal` (mock/dead deterministic; live via ollama/openai-chat/
  openai-responses); `validate_proposal.vyb` runs it through the REAL VybOS
  repair core (apply → gates → guardrail → promote → seal) and reports
  ACCEPT / REJECT / HUMAN-REQUIRED. Full boundary provable with no model/GPU.
  Vyb-native driver: flags are `VYBFORGE_*` env vars (the JIT has no argv).
- `native/legit/run_configurator_gate.sh` — **P1.4 port-fidelity gate** for
  `app/configurator.vyb`: per-backend request path/Authorization/body are
  byte-compared against the frozen Python-driver baseline
  (`native/legit/fixtures/configurator-baseline/`) through the recording stub
  `native/legit/stub_backend.py`, the emitted contract is compared by value and
  validated against `config/agent-response.schema.json`.
- `tools/configurator_repl.vyb` + `tools/interview_infer.vyb`
  (`native/legit/run_interview_gate.sh`) — **P1.5**: the interview REPL and the
  one-shot `proposed_changes` → JSONL appender, both Vyb-native over a served
  model (the Python originals loaded base+LoRA with torch). The gate checks the
  emitted patch lines are byte-identical to `json.dumps(..., separators=(",", ":"))`
  and that `tools/apply_interview.vyb` consumes them into a spec.
- `tools/{http_client,json_util,chat_request,reply_parse}.vyb` — the shared
  driver plumbing (HTTP/TLS client with the drivers' 180 s timeout, JSON text
  shaping, the backend-neutral chat request builder, one reply-unwrapping rule).
  Drivers run with `--module-path tools --module-path native/json`.
- `native/gdecode/coerce_contract.vyb` (`native/legit/run_coerce_gate.sh`) —
  **P1.6**: the tuned-decode post-filter/repair that coerces whatever JSON the
  decode produced (drifted kind, extra keys, truncated mid-object) into a
  schema-valid agent-response contract, emitting
  `native/out/contract_loradec.json`. The gate compares the port against frozen
  Python baselines byte-for-byte on stdout, contract bytes and exit code over a
  16-fixture corpus, cross-checks its own schema verdict against the jsonschema
  oracle, and asserts `make loradec-contract` is all-Vyb with the whole contract
  set still reporting `CONTRACT_VERIFY: ALL_OK`. Emission goes through
  `native/json/json_emit.vyb` (a `json.dumps`-compatible emitter), so the artifact
  is byte-identical to the Python original's.
- `training/` — generator, QLoRA code, explicit job launcher, and handoff
  rules. Retrained/inferred on GPU (as tested on godzilla's RTX 3090).
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

Phase 2 (torch → Vyb) is in progress; `doc/P2-TRAINER-PLAN.md` is the step map and
the source of each step's gate. `native/legit/run_phase2_battery.sh` runs the Phase-2
gates, and `native/legit/run_phase1_battery.sh` remains the regression gate.

- `training/generate-data.sh` is Python-free: the corpus generator and the
  train/eval split are both Vyb (`training/generate_dataset.vyb`,
  `training/split_dataset.vyb`). The splitter reproduces the committed
  `data/vybos-configurator-{train,eval}.jsonl` byte-for-byte, and the regenerated
  corpus matches the committed `all.jsonl` byte-for-byte
  (`native/legit/run_split_gate.sh`).

## Boundaries

- Draft-only: no VybOS builds/realizes/execs/generations, no host edits.
- No secrets in git. No personal host URLs. Machine paths appear only as
  documented `$HOME/Projects/...` examples the owner already uses.
- Python is reference-verification only in the shipped decode/interview runtime.
- Ask the owner before: the GitHub-side repo rename, force-push/rewriting
  history, deleting `artifacts/vybos-configurator-lora/`, or running a training
  job.
