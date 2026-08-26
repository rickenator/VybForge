# Training Data and Fine-Tuning Rules

## Mission

Maintain a trustworthy seed corpus for a VybOS configuration interviewer. The
model must distinguish demonstrated VybOS capabilities from design goals and
must never claim that a desired state was applied to a host.

## Ground Truth

- Base factual examples on the current VybOS `doc/STATUS.md`, not assumptions
  or an older README alone.
- Preserve the distinction between composed `SystemSpec`/derivations/generations
  and a user-authorized host deployment.
- Mark architecture-specific work, production policy, and any unverified
  hardware path as an open decision or proposal.

## Coverage Handoff: Expand With VybOS

The seed corpus is intentionally small. Expand it whenever VybOS gains a
verified capability, changes a supported contract, resolves a known gap, or
adds a new configuration/module/build path.

For each meaningful VybOS change:

1. Read the changed implementation, its acceptance test or reproduction
   command, and the current `doc/STATUS.md`; do not train from a commit message
   or aspirational plan alone.
2. Add reviewed examples for: the capability's normal configuration interview,
   a correct proposal with explicit paths/values, a boundary or unsupported
   request, and a regression-sensitive question that checks the previous
   contract still holds.
3. Include counterexamples for stale assumptions. When a feature moves from
   planned to proven, replace or relabel old “not implemented” examples rather
   than teaching both as current facts.
4. Preserve a held-out evaluation slice by configuration topic and capability,
   not merely a random copy of a training template. Avoid near-duplicate train
   and evaluation records.
5. Regenerate JSONL, validate both outer chat JSON and inner response-contract
   JSON, record the VybOS source revision in dataset metadata, and inspect a
   sample of every new topic before retraining.

Do not treat a low evaluation loss on repetitive templates as capability proof.
Evaluate the adapter against novel, human-reviewed VybOS configuration
interviews before promoting it for normal use.

## Dataset Contract

- `generate_dataset.vyb` is the canonical deterministic source.
- JSONL records use chat messages; assistant content must itself be valid JSON
  with exactly `kind`, `message`, `missing_fields`, `proposed_changes`, and
  `requires_confirmation`.
- `question` responses ask one high-information question and propose no
  changes. `proposal` responses list explicit path/value/reason objects.
- Keep training and evaluation examples deterministically split. Do not put
  secrets, private paths, credentials, or model-generated unreviewed claims in
  the corpus.

## Training Gate

- `generate-data.sh` may compile the local Vyb generator and write local data;
  it never submits a job.
- `start-training.sh --start-training <host>` is the sole submission path.
  Do not invoke it until the user explicitly says the target is ready.
- Do not install packages or start a CUDA job automatically. Only the explicit
  `start-training.sh --start-training <host>` path may provision its isolated
  `.venv`, verify CUDA, and launch QLoRA in the background.
