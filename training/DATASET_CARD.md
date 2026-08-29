# VybOS Configurator Seed Dataset

This deterministic seed corpus teaches an interviewer to conduct a safe
configuration interview for a real, mid-stream VybOS project. Companion design
doc: `training/CORPUS-STRATEGY.md` (facts-in-context manifest, goal-driven
whole-stack interviews, no-retrain-per-update).

## Grounding

Examples reflect current VybOS status (typed `SystemSpec` composition;
content-addressed derivations and closures; a derived C toolchain, Linux kernel,
and QEMU; QEMU/rootfs boot proofs; byte-determinism proofs) and are grounded in
`doc/STATUS.md`. They never claim arbitrary hardware targets or host deployment
are supported.

## Data Contract

Each record is a chat JSONL object: system prompt embeds a **capabilities
manifest** for that record; assistant content is a JSON object with `kind`,
`message`, `missing_fields`, `proposed_changes`, and `requires_confirmation`,
schema-valid against `config/agent-response.schema.json`.

The generator (`training/generate_dataset.vyb`, deterministic, pure Vyb) emits
720 records across 20 topic classes — single-field edits plus goal-driven
whole-stack decompositions:

- 4 **manifest variants** (0–3) teach manifest-*dependence*: variant 3 (a "wayland
  desktop stack derived" future) appears **only in eval**, so eval is a transfer
  test that updating the manifest changes answers without retraining.
- Topics include goal decompositions (desktop, kiosk, CI-runner, gateway,
  rescue) that stay manifest-grounded and surface un-buildable subsystems as
  open decisions rather than fabricating them.
- Deterministic split: train = manifests 0–2; eval = every 10th record plus all
  of the unseen manifest-3 set.

## Safety

No credentials, host paths, private sources, or unreviewed model claims. Teaches
the review-only boundary: draft and validate desired state, never claim a host
change was applied.

## Verification

- `make -f native/Makefile schemacheck` (Vyb) validates mock-system domains, the
  agent-response schema contract, and every corpus record (kind + exact
  `{path,op,value,reason}` items).
- Reference verifier `tests/test_schema.py` (Python) must stay green.

## Training

`generate-data.sh` only builds the Vyb generator and writes JSONL.
`start-training.sh --start-training <host>` is the explicit submission gate
(background GPU QLoRA for now; the target is Vyb-native LoRA training on the
3090 per CORPUS-STRATEGY.md).
