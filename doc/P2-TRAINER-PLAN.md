# Phase 2 — the torch pipeline → Vyb (execution plan)

Status: IN PROGRESS (started 2026-10-02). Phase 1 is complete; its battery
(`native/legit/run_phase1_battery.sh`) is the regression gate for every step below.

Companion docs: `doc/PYTHON-CLEANUP.md` (Phase 1 record + Oracle policy),
`doc/SPIKINGBRAIN.md` (what this substrate is ultimately for).

## Scope — what actually gets replaced

| Python still in a shipped path | Replaced by |
|---|---|
| `training/generate-data.sh` (inline split heredoc) | `training/split_dataset.vyb` (P2.5a) |
| `native/train/gen_kv_train.py` | `native/train/gen_kv_train.vyb` (P2.1a) |
| `native/train/build_fullmanifest.py` | `native/train/build_fullmanifest.vyb` (P2.1b) |
| `native/train/_build_kv.py` | `native/train/_build_kv.vyb` (P2.1c) |
| `native/train/rbuild.py` | decision (P2.1d) |
| `training/train_lora.py` (NF4 QLoRA, LoRA r16/a32, AdamW, 3 epochs) | `native/train/lora_train.vyb` (P2.2) |
| `training/smoke_adapter.py` | `training/smoke_adapter.vyb` (P2.5) |
| `training/start-training.sh` (venv + torch bootstrap over ssh) | a run script driving the Vyb trainer on the GPU host (P2.5) |

Not in scope, per the Oracle policy: the ~50 `native/train/*_ref.py` and
`native/train/verify_*.py` numpy/torch references stay — they are the correctness
oracle for the trainer, and `.venv` stays because they need it.

## Baseline reality (measured 2026-10-02, before any port)

| generator | committed artifact | oracle status |
|---|---|---|
| `gen_kv_train.py 93 9` | `native/train/kvresp_train_kv.vyb` (107,836 B) | **byte-identical** — this is the regression gate |
| `gen_kv_train.py 513 429` | `native/train/kvresp_train_p429.vyb` (104,685 B) | **DIFFERS** today (108,070 B): p429 was not produced by this generator, or was edited afterwards. A second parameterization can therefore only be compile-checked, not byte-gated |
| `_build_kv.py`, run against today's `kvresp_train.vyb` | `kvresp_train_kv.vyb` | **DIFFERS**: today's run emits 1,247 lines vs the committed 1,291 — the committed file came from an older source (or was edited). Gate = frozen baseline of the current Python output (Phase-1 pattern) |
| `build_fullmanifest.py` | none (`native/out/` is untracked) | Gate = frozen baseline (manifest text + `<i8` ids), with the ids cross-checked against the tokenizer oracle |
| `rbuild.py` | `kvresp_train_r.vyb` | self-described **partial**: "dattn combined-staging + buffer allocs still to splice; THIS IS A PARTIAL BUILD for compile-checking the S->R transform only" |

Both divergences are recorded rather than reconciled: they are generator-vs-artifact
drift, and under the Oracle policy an oracle's disagreement is data, not something to
silently "fix".

## Steps — each one a checkpoint

Rule: implement → its own gate green → Phase-1 battery green → commit. No step lands
on a red suite.

**P2.5a — dataset split in Vyb** (smallest, and it takes Python out of a *shipped* script) — **DONE 2026-10-02**
`training/split_dataset.vyb` reads `data/vybos-configurator-all.jsonl`, keeps rows by
`metadata.split`, and re-emits compact JSON.
Gate: splitting the committed `all.jsonl` reproduces the committed `train.jsonl` +
`eval.jsonl` **byte-for-byte** (468 train / 252 eval records; the Python original
reproduces them too, so the committed corpus is a real oracle), the regenerated
corpus is byte-identical to the committed `all.jsonl` (1,532,026 B), and
`training/generate-data.sh` contains no interpreter invocation.
Gate script: `native/legit/run_split_gate.sh` (also the first line of
`native/legit/run_phase2_battery.sh`).

Finding worth keeping: the corpus round-trip needs `json_emit::json_text_tight`, not
`json_util::json_text`. The latter escapes only the five control characters, so
non-ASCII bytes go out raw — which matched on the train split (ASCII) and broke only
on eval (byte 5,911, line 3). `json_emit`'s escaping is CPython's `ensure_ascii`.

**P2.1a — `gen_kv_train.vyb`**: the (S, NCTX) parameterizer — 12 substitution families
plus the 25-field ASLB byte-layout table.
Gate: (93,9) byte-identical to the committed `kvresp_train_kv.vyb`; (513,429)
byte-identical to a frozen baseline of today's Python output; both outputs pass
`--no-execute`.

**P2.1b — `build_fullmanifest.vyb`**: manifest text → tokenizer ids
(`native/tokenizer`) → `native/out/fullmanifest.txt` + `<i8` ids binary.
Gate: byte-identical to frozen baselines; ids equal the tokenizer oracle's.

**P2.1c — `_build_kv.vyb`**: the three-part source transform (helper insertion,
CK/CV/RSHI allocations, per-token forward swap).
Gate: byte-identical to a frozen baseline of today's Python output; output compiles.

**P2.1d — `rbuild.py`**: decision — port as-is (gate: frozen baseline, partial-ness
documented) or retire as superseded by P2.2's direct trainer. Blocks nothing else.

**P2.2 — the trainer** (the real project): frozen base (GGUF q4_0 dequant path), LoRA
bf16 parameters, AdamW, multi-epoch over the 216-record corpus, per-step loss to disk.
Reuses `train.ptx`, the bwd-lora probe, the fwd-cache/kvctx builders,
`encode_corpus.vyb`.
Gate: per-step loss matches the numpy oracle (`train_full_loop_ref` family) at S=93,
then scales to the full corpus.

**P2.3 — adapter save/load** in a Vyb-native `.bin` (manifest + raw bf16).
Gate: `loradec_driver.vyb` consumes a Vyb-trained adapter end-to-end → kvresp decode
gate + schema-valid contract.

**P2.4 — behavior parity** against the committed torch-trained adapter on the
24-record eval.
Gate: eval CE within tolerance; top-1 match rate on eval records.

**P2.5 — `smoke_adapter.vyb`** (prompt → adapter decode) + the training run script.
Gate: the `start-training` equivalent runs end-to-end on the GPU with no `python3`.

**P2.6 — delete** `training/*.py` and the torch scripts, then re-run the repo-wide
audit.
Gate: `grep -rnw "python\|python3"` hits only oracle/gate invocations and legacy docs.

## Feasibility notes (checked before committing to the approach)

- Vyb's String API has everything the source transforms need: `replace(old,new)`,
  `substr`, `index_of`, `char_at`, `starts_with`, `ends_with`, `trim`. There is no
  `split` — scan with `index_of`. **No new compiler feature is required for P2.1.**
- Compact JSON emission uses `native/json/json_emit.vyb` (`json_text_tight`), which
  P1.6 verified against CPython's `json.dumps(separators=(',',':'))`; integer
  formatting agrees, non-integral floats are a documented deviation.
- Binary output: `Vec<UInt8>` + `push(x as UInt8)` + `io::write_bytes` (P1.7 lesson).
- Drivers take no argv, so paths arrive through env vars, as everywhere else here.

## Risks

1. **P2.2 is the project**; the rest is plumbing. NF4→q4_0 dequant parity, the
   chat-template render and AdamW numerics are where this converges or turns into a
   compiler issue — in which case: file it upstream and fix it, per the dogfooding
   rule, rather than carrying a workaround.
2. The torch run used 3 epochs, per-device batch 1, gradient-accumulation 16. That
   schedule's semantics need writing down before P2.4 makes any parity claim.
3. Because of the drift recorded above, "regenerate the committed artifacts" is true
   for exactly one parameterization of one generator; the Phase-2 gate wording is
   corrected here accordingly.
