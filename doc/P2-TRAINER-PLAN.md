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

**P2.1a — `gen_kv_train.vyb`**: the (S, NCTX) parameterizer — the ordered substitution
families plus the 25-field ASLB byte-layout table. — **DONE 2026-10-02**
`native/train/gen_kv_train.vyb` mirrors `gen_kv_train.py` operation for operation, in
the same order, with the same all-occurrences `replace` semantics — including the
substitutions that are no-ops at S=93 (which is exactly why the regression round-trips)
and the `+<old offset>` sweep that tangles at other S. Arguments and paths come from
the environment (`VYBFORGE_KV_S`/`_NCTX`/`_SRC`/`_OUT`) since the JIT has no argv;
defaults reproduce `gen_kv_train.py 93 9`.
Gate: `native/legit/run_kvgen_gate.sh` — (93,9) rewrites the committed template back to
itself byte-for-byte (107,836 B, sha256 `7f738e41…`); (513,429) equals the frozen
baseline `native/legit/fixtures/kvresp_train_p429_S513_N429.vyb` (108,070 B, sha256
`158a3402…`); an optional `python3` cross-check re-derives both and requires agreement;
and the template hash is unchanged after the run.

Measured: SLAYOUT = S × 67598 × 8 (50,292,912 at S=93, 277,422,192 at S=513) — that is
where the committed `50292912` literal comes from, so the layout table is independently
checkable by hand.

**P2.1b — `build_fullmanifest.vyb`**: manifest text → tokenizer ids
(`native/tokenizer`) → `native/out/fullmanifest.{txt,_ids.bin}`. — **DONE 2026-10-02**
The manifest literal is generated from the oracle's own output file rather than
transcribed, so the prose cannot silently drift; the ids go out 8 bytes each,
least-significant byte first (`numpy.array(ids, dtype="<i8").tobytes()` for
non-negative ids).

**This step found a real tokenizer bug** — and it is the reason P2.1b is worth more than
its size suggests. `native/out/` is untracked, so the baseline was captured from the
oracle: 429 tokens. The port produced **430**. The text was byte-identical, so it was the
tokenizer: at token 51 transformers merges `):\n` into one token (id 982) and Vyb split it
into `):` (1648) + `\n` (198), re-converging immediately after. Cause: the Qwen2
pre-tokenizer's punctuation alternative is ` ?[^\s\p{L}\p{N}]+[\r\n]*` — a punctuation run
absorbs the CR/LF run after it — but `segment_word` classified the remapped CR/LF
byte-chars (`Ċ` = 0xC4 0x8A) as *letters* (their UTF-8 lead byte 196 is in the
letter range), so the piece was cut at the newline and the merge `):` + `\n` → `):\n` was
unreachable. Fixed in `native/tokenizer/tokenizer.vyb`: CR/LF/TAB/VT/FF glyphs get their
own class, punctuation runs absorb trailing CR/LF, and a leading tab attaches to the
following letter run (Qwen's `[^\r\n\p{L}\p{N}]?\p{L}+`). Verified against transformers on
22 boundary cases (all match) and the full manifest now reproduces the oracle exactly.

Why the existing oracle stayed green: `native/train/verify_encode_corpus.py` samples four
corpus records, none of which contains a punctuation-then-newline boundary. The new
`native/tokenizer/test_pretok_boundary.vyb` pins that class directly (expectations are
transformers ids, frozen in the test) and runs in this gate. Worth doing at the model
sweep: widen the corpus oracle's sample, or have it sweep every record's assistant text.

Gate: `native/legit/run_fullmanifest_gate.sh` — both fixture files byte-for-byte plus the
429-token count, the pre-tokenizer boundary test, and an optional `.venv` transformers
cross-check that re-derives the fixture.

**Blocker (filed as Vyb#432, now RESOLVED)** — during P2.1b the `encode-corpus` make target
turned out to be unrunnable: `native/train/encode_corpus.vyb` imports `tokenizer::{encode}`
and `import json_parse`, and both modules define `hexval`, so the whole-module import failed
with `Duplicate symbol after splice: 'hexval'` (pre-existing; the committed revisions of both
modules failed the same way on an unmodified `build/vyb`). Filed as **Vyb#432** with a 3-file
minimal repro and the import matrix, and fixed in Vyb by PR #450 (per-module identity for
spliced declarations whose names collide). The consumer-side one-liner
(`import json_parse::{parse}`) was deliberately never applied and is not needed.

Verified against Vyb `74220c3` on 2026-10-03: `make -C native encode-corpus` now runs end to
end — the driver emits the corpus and the independent oracle re-derives it,
`ENCODE_CORPUS_VERIFY: OK  (4/4 exact match, ids+lables derived)`. The tokenizer's coverage
therefore goes through `encode_corpus.vyb` as well as
`native/tokenizer/test_pretok_boundary.vyb` and the fullmanifest oracle cross-check.

**P2.1c — `_build_kv.vyb`**: the three-part source transform (helper insertion,
CK/CV/RSHI allocations, per-token forward swap). — **DONE 2026-10-02**
`native/train/_build_kv.vyb` mirrors `_build_kv.py` step for step: the uniqueness asserts
become explicit checks with distinct exit codes (2–8) instead of `AssertionError`, and
`str.index(end, si)` becomes an `index_of` on the tail slice.

Two details worth keeping:

- **The ~7.8 KB helper block is extracted from the Python's own triple-quoted literal**, not
  transcribed — the `.vyb` file was generated from it, so the driver code cannot drift by a
  typo. `new_block` is the Python's f-string body with its four interpolations turned into
  placeholders (`@@SLAYOUT@@`, `@@SARGS@@`, `@@SLOS@@`, `@@XOCUR@@`) filled at run time from
  the same 25-entry `OFF` table, so the port keeps the Python's structure rather than baking
  constants.
- **No in-tree oracle exists for this one**: today's generator emits 1,247 lines and the
  committed `kvresp_train_kv.vyb` has 1,291 (older source or edited afterwards). The baseline
  is therefore pinned as a hash fixture, and the gate re-derives it from the live oracle when
  `python3` is available — the oracle is stdlib-only, so that needs no venv.

Gate: `native/legit/run_kvbuild_gate.sh` — byte-identical output vs the fixture hash
(`e05a7cff…`, 103,817 B, 1,247 lines) plus the four stdout lines, `--emit-llvm` on the
generated driver (semantic analysis + codegen; `--check` turned out to be a *formatting*
check, not a semantic one), the live oracle cross-check, and a check that the committed driver
is untouched.

**P2.1d — `rbuild.py`**: decision — **leave un-ported** (recorded 2026-10-02; STATUS header on
the script + "Status notes on the Python that stays" in `doc/PYTHON-CLEANUP.md`). It assembles
`kvresp_train_r.vyb`, a driver that has never existed in the repo; the script is self-declared
partial, nothing builds it, and it belongs to the from-scratch `kvresp_train` line rather than to
P2.2's QLoRA port. Its brittle raw-literal S→R substitution is not worth carrying into Vyb; the
real deliverable is the transform spec in `native/train/FULLMANIFEST-MILESTONE.md`. Blocks
nothing else.

**P2.2 — the trainer** (the real project): frozen base (GGUF q4_0 dequant path), LoRA bf16
parameters, AdamW, multi-epoch over the corpus, per-step loss to disk. Reuses `train.ptx`, the
bwd-lora probe, the fwd-cache/kvctx builders, `encode_corpus.vyb`.
Gate: per-step loss matches the numpy oracle (`train_full_loop_ref` family) at S=93, then scales
to the full corpus.

Corpus reality (checked 2026-10-02): `data/vybos-configurator-all.jsonl` is 720 records, split
468 train / 252 eval (all.jsonl == train+eval). The "216-record" figure this section used to carry
was stale — same class as the `gen_kv_train` (513,429) drift.

Environment reality (checked 2026-10-02): the repo's `.venv` has transformers, numpy, safetensors
and llama_cpp, but **not torch** — `training/train_lora.py` and the torch-side oracles cannot run
in this checkout at all; they need the GB10 host. The oracles that *do* run here (numpy, llama.cpp
Jinja) are therefore the ones the gates should lean on.

**P2.2a — corpus → training sequence** — *part 1 DONE 2026-10-02*
Part 1: `native/train/render_chat.vyb` renders each record through the Qwen3 chat template, which
is not guessed but read from the GGUF metadata (`tokenizer.chat_template`, 4,049 chars, sha256
`3802169b…`, pinned as `native/legit/fixtures/qwen3_chat_template.jinja`). It implements the
template literally — including the multi-step-tool query-index scan and the `</think>` reasoning
split the corpus never exercises — and **refuses** what it does not implement (tools/tool_calls,
non-string content, roles other than system/user/assistant) with a named reason rather than
rendering something plausible. Output is raw text plus a cumulative byte-offset index; both pinned
in `native/legit/fixtures/chat_render_baseline.sha256`. Gate `native/legit/run_chat_render_gate.sh`:
baseline hash, 8 boundary cases (think split, multi-turn think-on-last-assistant-only,
system-not-first, unicode, newline padding), the three refusal cases, and — when `.venv` and the
GGUF are present — a live byte-for-byte cross-check against llama.cpp's own Jinja rendering of the
same template (`native/train/render_chat_ref.py`).
Part 2 (next): token ids per record, reusing `native/tokenizer`; gate vs the oracle's ids. Note the
corpus contains 36 records with non-ASCII content, so the byte-level BPE path is exercised.

**P2.2b — training blocks**: per-record response mask (loss on response tokens only, reusing the
`kvresp` masking), `max_length` 2048, batch 1, gradient accumulation 16.

**P2.2c — base + LoRA init**: frozen q4_0 dequant path, LoRA r=16/alpha=32/dropout=0.05. Dropout
makes bitwise parity impossible, so parity gates run dropout=0 and the trained run uses 0.05 from a
seedable RNG; say which mode a gate ran in, every time.

**P2.2d — optimizer + loop**: AdamW (`adamw_repro.vyb` exists), lr 2e-4, 3 epochs, logging every 5,
eval every 25, save every 25, per-step loss to disk. Needs a GPU run (see P2.4/P2.5).

*Torch reference banked 2026-10-02* (fixture `native/legit/fixtures/qlora_ref468_metrics.txt`): the
repo's own `train_lora.py`, unchanged, run against today's 468/252 corpus on godzilla (RTX 3090,
`~/Projects/VybAIConf/.venv`, torch 2.6.0+cu124 / peft 0.20.0 / trl 0.29.1 / bnb 0.50.1, base model
already in `HF_HOME=/usr/export/LLM/hf`).

What that run pins down, and what it does not:

- **Deterministic and assertable**: 90 optimizer steps (`ceil(468/16)*3`), evals at steps 25/50/75,
  739.28 s wall, ~7.3 s/step, **4.5 GB VRAM peak** — so training fits beside other work far more
  comfortably than the 23.7 GB the 27B server holds.
- **Not byte-assertable**: `train_lora.py` sets no seed and runs LoRA dropout 0.05, so a repeat run
  will not reproduce the digits. This is a curve/tolerance oracle. For a tight per-step gate, re-run
  with dropout 0 and an explicit seed and pin that instead.
- **The eval number to match is ~0.86, not the 0.069 in `artifacts/train-v2.log`.** Train loss
  collapses to ~0.009 in 3 epochs while eval_loss sits at 0.78–0.88: this recipe memorises the
  468-record split. Anyone gating on the old 0.069 would be gating on a different corpus.
- **`eval_num_tokens` is cumulative, not the eval set's size** — it equals the train `num_tokens` of
  the same step (184207 / 362711 / 540965). Don't read it as an eval token count.
- **The committed adapter is a 216-record-era artifact.** 42 = `ceil(216/16)*3` steps in
  `train-v2.log` versus 90 = `ceil(468/16)*3` now, which is what actually proves it. Decide
  deliberately at P2.4 whether to retrain the torch baseline on 468 or compare against the old one.

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
