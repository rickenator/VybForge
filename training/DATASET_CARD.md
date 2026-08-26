# VybOS Configurator Seed Dataset

This deterministic seed dataset teaches a small instruct model to conduct a
safe configuration interview for a real, mid-stream VybOS project.

## Grounding

The examples reflect the current VybOS status snapshot: typed `SystemSpec`
composition; content-addressed derivations and closures; generations and
rollback bookkeeping; rootfs staging; a derived C toolchain; a derived Linux
kernel; a derived QEMU runtime; and a QEMU boot proof. They do **not** claim
that arbitrary hardware targets or host deployment are already supported.

## Data Contract

Each record is a chat JSONL object. Assistant content is a JSON object with:
`kind`, `message`, `missing_fields`, `proposed_changes`, and
`requires_confirmation`.

The generator emits 240 deterministic records, split 216 train / 24 eval.
They are a seed corpus, not sufficient by themselves for a capable model.
Before training, expand it with reviewed real configuration interviews and
negative cases checked by a deterministic schema validator.

## Safety

The dataset contains no credentials, host paths, private package sources, or
model-generated assertions. It teaches a review-only boundary: the agent may
draft and validate desired state, but never claims to apply a host change.

## Training

`generate-data.sh` only builds the Vyb generator and writes JSONL.
`start-training.sh --start-training <host>` is the explicit submission gate.
It generates the data, copies only `data/` and `training/` to that host, checks
for the required Python packages, and then begins QLoRA. It never runs unless
that exact flag is supplied.
