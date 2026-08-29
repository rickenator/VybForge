# VybForge Interviewer Training — Strategy (stable behavior, volatile facts in context)

**Goal:** the configurator interviewer must *never* require a full retrain just because
VybOS gains a capability, a package version bumps, or the supported target set changes.
The model should learn the **behavior** of the interview once; the **facts** it reasons
about are supplied as context at inference time and updated independently.

## The core decoupling

Split everything the model needs into two kinds:

| Kind | Examples | Where it lives | Changes when VybOS changes? |
|------|----------|----------------|------------------------------|
| **Behavior (trained once)** | emitting the `{kind,message,missing_fields,proposed_changes,requires_confirmation}` contract; when to question vs propose vs summarize; how to reason about a *given* manifest + baseline; the safety boundary (never claim host apply) | the LoRA **weights** | No — stable across VybOS evolution |
| **Facts (injected, updatable)** | current status: which targets boot to READY (rootfs bwrap/docker, x86_64 QEMU kernel), the derived toolchain (gcc 13.2.0, kernel 6.6 bzImage), available packages/services, open decisions (ARM64 not proven) | a **capabilities manifest** embedded in the system prompt (+ the `config/default-state.json` baseline referenced there) | **Edit the manifest → no retrain** |

## Manifest mechanics (how "no retrain per update" actually happens)

1. **The system prompt carries the facts.** It embeds a compact *capabilities manifest*
   (a deterministic string built from the same facts `config/default-state.json` + the
   VybOS status snapshot reference).
   The model is told: *"Use only the manifest + baseline in context; do not invent
   packages, versions, or claimed boot support that are not listed."*
2. **The corpus trains on manifest-*dependence*, not manifest *memorization*.** The
   deterministic generator emits **several manifest variants** (e.g. toggling newer
   capabilities, bumping a version) across records, so the model learns to *read the
   manifest it is given* rather than memorize one fixed fact set. A held-out eval set
   uses a manifest **not present in training** (a future capability) — a direct test
   that updating the manifest transfers without retraining.
3. **New VybOS capability = add a fact to the manifest (+ maybe one reviewed example of
   the new topic class), not a full retrain.** Because the model never independently
   asserts an unlisted fact, the manifest alone is enough to change its answers.

## Corpus expansion rules (extends training/AGENTS.md)

- Every record's system prompt includes the manifest for *that* record; the user/assistant
  messages reference facts only via that manifest.
- Add boundary + stale-assumption counterexamples: e.g. "Can VybOS boot on ARM64 yet?"
  must answer via the manifest ("not listed as proven → open decision"), never a cached guess.
- Keep the deterministic train/eval split by topic; the eval split includes topics and a
  manifest variant unseen in training.
- Regenerate = deterministic; validate every assistant payload against
  `config/agent-response.schema.json` with `jsonschema` (system python has it).

## Why this makes the *complete-OS* future tractable

Training a "complete OS" interviewer doesn't mean training every fact about the OS into
weights. It means training the handful of stable behaviors (eliciting goals, mapping to
SystemSpec patches, gating proposals on the baseline+manifest, refusing host apply) once,
then describing each new subsystem (toolchain, kernel, runtime, userspace, services) as
facts in the manifest. The corpus therefore grows by topic *class* occasionally, not by
every VybOS commit.

## Goal-driven whole-stack interviews (the shape to train toward)

The entry point is a **user goal**, not a field edit: *"I want a Hyprland desktop"* /
*"a hardened CI runner"* / *"an offline kiosk"*. The interviewer must turn that goal into
a **complete desired state across the whole SystemSpec surface** (packages + the stack they
imply, services that run it, users/login, network, boot-to-graphical, storage), coordinated
by dependencies — not a single add.

Train two stable behaviors here:

1. **Decompose the goal into the supporting stack.** E.g. a desktop goal implies:
   the compositor, its Wayland/display deps, input/seat, graphics driver/firmware, a
   session/display service, an interactive user account, fonts/theming, xdg portal, and a
   boot path that reaches a graphical session. Each becomes a reviewed
   `proposed_changes` item or an explicit decision.
2. **Stay manifest-grounded.** The manifest lists what VybOS can actually *derive/build*
   (today: base rootfs, toolchain, kernel, QEMU, core packages). If a goal needs a stack
   VybOS cannot yet build — e.g. the Hyprland/mesa graphics stack is *not* a derived
   capability — the correct answer is **not** to fabricate it: propose the parts that are
   real, and surface the missing subsystem as an open decision / capability question.
   This keeps the contract honest AND doubles as the no-retrain test (a future manifest
   that *does* list Hyprland transfers without retraining).

The desired-state model these interviews produce will outgrow the four patch paths
(`system|hostname|pkgs|services`); the deeper contract lives on the Spec side
(AGENTS.md's full checklist + `config/mock-system.json`) while the interviewer keeps the
same `{path,op,value,reason}` patch language. Corpus records should include a mix of
single-field edits (existing) and these whole-stack goal decompositions.
