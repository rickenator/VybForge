# Python cleanup → Vyb-only production

Status: **Phase 1 COMPLETE** (2026-10-02) — P1.1–P1.8 done and gated; Phase 2 (torch QLoRA → Vyb) not started.

Goal: **zero Python in any production path** of VybForge. Python remains only as
reference *oracles* (numpy/torch reference + verification gates) and — until
Phase 2 lands — the legacy torch training pipeline, which is itself being
converted (per Rick's direction 2026-10-01: convert now, not deferred).

Repo convention (native/README.md): "Zero Python in the production pipeline.
Python is used only for reference *verification*, and never at runtime."

## Inventory verdict (107 .py files, 7,277 lines)

| Class | Files | Action |
|---|---|---|
| Oracles (refs + gates) | ~91 | KEEP — gguf refs (7), gguf gates (5), sampler/tokenizer gates (2), tools gates (10) + emit_prompt_ids, gdecode gates (2), train/ refs+gates (~64), train lr-sweep |
| Production tooling | 7 | CONVERT (Phase 1) |
| Superseded / ported | 7 | DELETE |
| Torch training pipeline | 5 (+ start-training.sh) | CONVERT (Phase 2) |

## Phase 1 — production tooling → Vyb (checkpoint gates)

| # | From (Python) | To (Vyb) | Gate |
|---|---|---|---|
| P1.1 | — (deletions) | — | **DONE** — `tests/test_schema.py` (Vyb port exists: `tests/test_schema.vyb`), `native/gguf/dump_qwen3_meta.py` (superseded by `read_real_meta.vyb`; TSV committed), `step1_topk.py`/`topk_step0.py`/`step1c_topk.py` (unreferenced dev steps). No behavior change; `make verify` + `schemacheck` green |
| P1.2 | `tools/apply_interview.py` (99) | `tools/apply_interview.vyb` | **DONE** — identical `out/` artifacts vs Python driver on same patches; rendered program compiles + reproduces spec.json |
| P1.3 | `tools/propose_repair.py` (170) | `tools/propose_repair.vyb` | **DONE** — mock proposal ACCEPTs through the REAL VybOS repair core (`validate_proposal.vyb`), and REJECT / HUMAN-REQUIRED paths verified; live backends exercised for real: `ollama` (127.0.0.1:11434, qwen3:8b) and `openai-chat` (godzilla:8081/v1, qwen3.8-27b) return valid proposals that ACCEPT through the core; `openai-responses` reaches and parses both real endpoints. Python driver deleted; `run_repair_proposal.sh` is now all-Vyb |
| P1.4 | `app/configurator.py` (139) | `app/configurator.vyb` (extends the `src/main.vyb` client via the shared `tools/http_client.vyb`) | **DONE** — gate: `native/legit/run_configurator_gate.sh`. Contract gate: one full interview round → schema-valid contract (live round against godzilla:8081/qwen3.8-27b, validated with the repo's jsonschema oracle). Port-fidelity gate: for all three backends the request **path, Authorization header and body are byte-identical** to the Python driver's (frozen baseline + recording stub), and both drivers emit the same contract JSON value. `run.sh` now drives the Vyb binary; `app/configurator.py` and `tests/test_backends.py` deleted (their three guards are asserted by that gate) |
| P1.5 | `tools/configurator_repl.py` (70), `tools/interview_infer.py` (73) | `tools/configurator_repl.vyb`, `tools/interview_infer.vyb` | **DONE** — gate: `native/legit/run_interview_gate.sh`. `interview_infer` appends one JSONL patch line per proposed_change, **byte-identical** to the Python emission rule (`json.dumps({...}, separators=(",", ":"))`) including object-valued `value`s, and `tools/apply_interview.vyb` consumes those lines into `out/spec.json` + `out/system.vyb` (rendered program reproduces spec). `configurator_repl` prints one contract per Q/A round. Both verified live against godzilla:8081/qwen3.8-27b. The Python originals were torch/GPU-bound (base+LoRA load); the ports talk to a served model through the shared `tools/chat_request.vyb` — no torch in the path. Python drivers deleted |
| P1.6 | `native/gdecode/coerce_contract.py` (171) | `native/gdecode/coerce_contract.vyb` (+ new `native/json/json_emit.vyb`) | **DONE** — gate: `native/legit/run_coerce_gate.sh`. (1) Port parity over a 16-fixture corpus: the port matches the frozen Python baselines **byte-for-byte on stdout, on the emitted contract, and on the exit code** — including the truncated-JSON tolerance path (the real tuned decode, which stops mid-object), kind drift (`"Configuration"` → `proposal`), the model's `changes` alias, non-list `missing_fields`, string/number `requires_confirmation`, nested values, `\u` escapes and both `NO_JSON` paths. (2) The independent jsonschema oracle agrees with the port's own structural `SCHEMA_OK`/`SCHEMA_FAIL` verdict on every emitted contract. (3) `loradec-contract` is now 3 Vyb stages with no python in the target, and the whole set reports **CONTRACT_VERIFY: ALL_OK** (the `loradec` entry is now covered by `verify_contract.py`). Python driver deleted |
| P1.7 | `native/gguf/mk_fixture.py` (76) | `native/gguf/mk_fixture.vyb` | **DONE** — gate: `native/legit/run_gguf_gate.sh`. The port's `test.gguf` is **byte-identical** (208 bytes, sha256 `f23fc384…`) to the frozen Python-generated fixture and its report line is byte-identical too; the Python verifier that asserts the expected layout (header, kv, tensor index, offsets `t0@170 t1@174 t2@190`) still passes on the port's file, and the Vyb q4_0 dequant slice runs green over it. `make gguf` generates through the port and is green; the fixture is now a make prerequisite (it is git-ignored, so `make verify` on a fresh clone previously failed at the GGUF section — it now regenerates). Python generator deleted |
| P1.8 | `tests/test_applier.py`, `tests/test_backends.py` | delete (guards folded into P1.2/P1.4 gates) | **DONE** — both files gone (their guards are asserted by `run_interview_gate.sh` / `run_configurator_gate.sh`); README/ROADMAP/HANDOFF swept for references to retired files; the final audit below classifies every `python3` hit. Result: **0 Python on any production path**, 70 Makefile invocations of 56 scripts all reference verification (kept per the Oracle policy), training-side bootstrap deferred to Phase 2 |

## Porting log — findings worth carrying into P1.4+

**Defects fixed in shared Vyb code while porting** (both were latent; P1.2's
patch corpus happens to contain no string escapes, so its gate did not catch the
first one):

| Where | Defect | Fix |
|---|---|---|
| `native/json/json_parse.vyb` (`parse_string_raw`) | the JSON escape table was `"\"\\/bfnrt"` — *character names*, not the control bytes, so a `\n` inside any JSON string decoded to the letter `n` (and `\t`→`t`, `\r`→`r`, `\b`→`b`, `\f`→`f`). Any model response with multi-line content failed to re-parse. | table now holds the real bytes: `"` `\` `/` `String::from_byte(8)` `String::from_byte(12)` `\n` `\r` `\t` |
| `tools/propose_repair.vyb` (`http_post`) | the endpoint's path prefix was dropped, so `ENDPOINT=http://host:8888/v1` + `/responses` went to `/responses` — llama-server tolerated it, vLLM answered 404. | prefix preserved (`base + path`), matching the Python driver's `endpoint.rstrip("/") + path` |

**Shared driver modules created during the ports** (both under `tools/`, so drivers
run with `--module-path tools --module-path native/json`):

| Module | Contents | Why |
|---|---|---|
| `tools/http_client.vyb` | `PostResp`, `http_post` (HTTP + TLS) | extracted from `propose_repair.vyb` so the driver family shares one client instead of inlining a copy each. Also carries the Python drivers' **180 s socket timeout** (`socket_set_timeout`) — without it a hung backend blocks the CLI forever |
| `tools/json_util.vyb` | `json_compact(text)`, `json_compact_tight(text)`, `json_text(doc, node)`, `json_escape(value)` | normalizes JSON text to `json.dumps`'s shapes — default (`", "` / `": "`) and tight (`separators=(",", ":")`, the applier's JSONL lines) — and emits any parsed value as text. This is what makes a Vyb driver's request body and patch lines **byte-identical** to the Python driver's |
| `tools/chat_request.vyb` | `post_chat(...)`, `api_key_from_env(...)`, message/transcript rendering, structured-output fragments | the one backend-neutral chat request builder (ollama / openai-chat / openai-responses) shared by `app/configurator.vyb`, `tools/configurator_repl.vyb` and `tools/interview_infer.vyb`. An empty `schema_text` omits `format` on the ollama path (that is how `interview_infer` mirrors its schema-less Python original) |
| `tools/reply_parse.vyb` | `reply_text(backend, raw_http_body)` | one unwrapping rule for all reply shapes, including *skip non-`output_text` items* on Responses — reasoning-first endpoints (llama-server, vLLM) otherwise hand back the model's thinking instead of its answer |

**stdlib defect found while porting P1.5** (filed upstream, not worked around silently):

| Where | Defect | Status |
|---|---|---|
| `stdlib/io/mod.vyb` (`open_append`) | composed `FileFlag::APPEND \| FileFlag::CREATE` with **no access mode**, so the descriptor was opened read-only: the file *was* created, the open succeeded, and every `write_str` failed with `EBADF` ("Bad file descriptor"). `tools/interview_infer.vyb` spelled out `open(path, WRITE\|CREATE\|APPEND)` as the documented workaround while the port was being built | **Vyb#419 → FIXED upstream**, commit `17270c7` / PR #422 (`open_append` now sets `WRITE`); workaround removed and the P1.5 gate re-run green |

**JSON layer changes made for P1.6** (the coerce's fidelity rests on these):

| Change | Why |
|---|---|
| new `native/json/json_emit.vyb` | one `json.dumps`-compatible emitter over a parsed `JsonDoc`: `json_text_py` (default `", "` / `": "`), `json_text_tight` (`separators=(",", ":")`), `json_pretty` (`indent=2`, also exposed per-level for the contract's nested `value`s), and `json_escape_py` — escaping with `ensure_ascii=True` semantics (control chars, `\b`/`\f`, non-ASCII as `\uXXXX`, astral codepoints as surrogate pairs). Verified byte-identical to CPython's `json.dumps` across all three shapes on an ASCII, a non-ASCII/astral and a nested-value case |
| `native/json/json_parse.vyb` — `\uXXXX` now **decodes** (surrogate pairs combined) | it used to pass `\uXXXX` through as literal text, so a re-emitter produced `\\uXXXX` where CPython produces `\uXXXX` — silent corruption of any string carrying an escape |
| `native/json/json_parse.vyb` — new `json_parse_strict` | the lenient parser is right for model drivers (it recovers from a model's sloppy escaping) but wrong when the candidate span must parse or not *exactly* as CPython's `json.loads` decides it does. Strict mode fails on unknown escapes, truncated/malformed `\u`, lone surrogates and raw control bytes inside strings. The coerce's candidate scan uses it; the drivers keep the lenient entry point |

Known deviations of the port (measured, not assumed): numbers are stored as `Float` by `json_parse`, so a literal CPython reads as a float and prints non-trivially differs — `1e3` → Vyb `1000` vs CPython `1000.0`, `3.0` → Vyb `3` vs CPython `3.0` (plain integer literals agree); lone-surrogate escapes are rejected by the strict parser where CPython keeps them as lone surrogates (no real decode output contains one); an input file with invalid UTF-8 is not U+FFFD-replaced the way the Python's `errors="replace"` read did.

**Pre-existing pipeline wiring bugs hit while gating P1.6** (both fixed, both unrelated to the port):

| Where | Defect |
|---|---|
| `native/gdecode/run_pipeline.sh` stage B | invoked `native/host/decode_driver.vyb` without `--module-path native/llm`, so `import llm` could not resolve (the module lives at `native/llm/llm.vyb`) and `make gdecode-pipeline` failed outright. Every other caller in the Makefile passes that flag |
| `native/gdecode/run_pipeline.sh` stage D | called bare `python3 native/gdecode/verify_contract.py`, which has no `jsonschema` (the repo installs it in `.venv`), so the pipeline's own verification step died with `ModuleNotFoundError`. Now prefers the repo venv, as the repo's other oracle invocations do |

**The GGUF fixture generator (P1.7)** — the small port that needed real binary output:

| Point | Detail |
|---|---|
| Byte building | `Vec<UInt8>()` + `push(x as UInt8)` + `write_bytes(f, v)` (the #213 surface) is the whole writer. Since a by-value `Vec` parameter can't be mutated, each packer (`le32`/`le64`/`gguf_str`/`ascii_bytes`/`u8list`) returns its own buffer and the caller `concat`s them — `b = b.concat(le32(3))`. Element casts must be explicit (`x as UInt8`) |
| Layout stays computed, not hand-guessed | the port keeps the original's two-phase shape: build the tensor bodies, compute `data_base = 24 + kv.len() + Σ(8 + namelen + 4 + nd*8 + 4 + 8)` from the *actual* byte lengths, then patch the offsets sequentially, then assert the info region really ends at `data_base` (the original's `assert len(b) == data_base`, kept as a loud `exit(2)` and paired with a `write_bytes` length check that exits non-zero) |
| Floats | no float-bits builtin, so the f32 tensor body is written as the IEEE-754 single bits of 1.0/2.0/3.0/4.0 through `le32` (a comment names each constant) rather than a byte soup — the gate's byte comparison is what proves it |
| Report parity | the Python printed `os.path.join(os.path.dirname(__file__), "test.gguf")`, which the Makefile's absolute invocation made an absolute path; the port reproduces the same string with `(env_get("PWD") else "") + "/native/gguf/test.gguf"` so the report line is byte-identical too — run it from the repo root like every other tool |
| Rust-style structs need commas | `struct TInfo { name<String>, dims<Vec<Int>>, ty<Int>, oset<Int> }` — the parser says `Expected comma or closing brace after struct field` otherwise |
| Makefile | the fixture is now a real make prerequisite (`$(ROOT)/native/gguf/test.gguf: .../mk_fixture.vyb`), so `make verify` regenerates it instead of silently depending on a file that only a previous `make gguf` left behind — the fixture is git-ignored, so a fresh clone used to fail that section |

**Vyb language notes that cost time** (all verified on this build):

- string literals interpret `\n \r \t \" \\ \/` but **not `\b` / `\f`** — those
  pass through as backslash+letter. Build control bytes with
  `String::from_byte(n)` (this is what `stdlib/vllm/mod.vyb` does; flagged to
  Rick as a possible language gap).
- the JIT has **no argv** — driver flags are `VYBFORGE_*` env vars (and the
  `VYB_BIN`-style env indirection used elsewhere in the repo).
- **main's return value is printed, not used as the exit code** — the JIT always
  exits 0, so a failing driver run must be detected from its output, not `$?`.
  For a driver whose stdout is *data* (a JSON contract), declare `main() ->`
  (Void) and use bare `return`: a typed `main()<Int>` appends its return value to
  stdout and corrupts the stream.
- the configurator's `response_text` **skips non-`output_text` output items**, so
  reasoning-first endpoints (`/responses` returning a `reasoning` item at
  `output[0]`) work; `app/configurator.vyb` mirrors that rule.
  `tools/propose_repair.vyb` still mirrors its older Python original, which took
  `output[0].content[0].text` blindly — candidate for alignment in a later pass.
- `struct` by-value arguments owning a `Vec` were unsafe (double-free at exit;
  chained method call on the result mis-types) — filed as **Vyb#412** and fixed
  upstream (closed 2026-10-02, fix `8a32202` "a call-result receiver is not a
  storage location"); both repro modes verified clean on the rebuilt binary.
  The `their<T>` aspect-method route (`d.get`/`d.len`/`d.elem`/`d.hstr`) remains
  json_parse's primary documented API and is what these ports use.

**Live-backend caveat (openai-responses)**: both this driver and the Python
original read `output_text`, else `output[0].content[0].text`. On the endpoints
available here (qwen3.8 on llama-server, DSv4.1 on vLLM) `output[0]` is a
`reasoning` item, so that slot holds reasoning prose and the proposal parse
fails — identically in both drivers, so parity holds. Making the responses
backend succeed on reasoning-first endpoints means skipping non-`message`
output items, i.e. a deliberate deviation from the Python original: decide
before P1.5 touches this path.

**Native-decode note (P1.5)**: the plan's gate for these two tools named the native
decode path (`:8888` chat server / loradec). `native/host/chat_server.vyb` is up on
`:8888` but is the *untuned* slice — its own header says "tuned (LoRA) wiring
deferred (next slice)" and it generates `GEN = 3` tokens per request, i.e. it
cannot yet produce a contract. The tuned path is the `loradec` / `loradec-contract`
Makefile targets (P1.6 territory). So the ports talk to a served model through
`tools/chat_request.vyb` — the same backend set as every other driver — and are
verified live against `godzilla:8081` (llama-server, qwen3.8-27b). When the tuned
native decode lands, pointing these tools at it is an endpoint/env change, not a
code change.

## Phase 2 — Vyb-native QLoRA trainer (replaces torch pipeline)

**Execution plan, per-step gates and progress live in `doc/P2-TRAINER-PLAN.md`** —
read that first; the table below is the original shape of the work. Its P2.1 gate
wording is superseded by the measured baselines in that doc: only the `(93, 9)`
parameterization of `gen_kv_train.py` reproduces its committed driver byte-for-byte
(`(513, 429)` and a fresh `_build_kv.py` run do not — pre-existing generator/artifact
drift, recorded not reconciled), so the `(513, 429)` case is gated against a frozen
baseline of the oracle's own output instead.

Landed so far: **P2.5a** `training/split_dataset.vyb` (the inline split heredoc is gone;
`training/generate-data.sh` is Python-free) and **P2.1a** `native/train/gen_kv_train.vyb`
(driver parameterizer). Both gated byte-for-byte; `native/legit/run_phase2_battery.sh`
runs them, `native/legit/run_phase1_battery.sh` stays the regression gate.

Replaces: `training/train_lora.py` (QLoRA, NF4 base), `training/smoke_adapter.py`,
`training/start-training.sh`, `native/train/{build_fullmanifest,_build_kv,gen_kv_train,rbuild}.py`.

Existing Vyb building blocks (verified): `train.ptx` (layer fwd/bwd/AdamW),
bwd-lora probe, fwd-cache + kvctx builders, `encode_corpus.vyb` (corpus → ids),
corpus-CE gate, GGUF q4_0 reader + dequant, `loradec_driver.vyb` (adapter decode).

| # | Step | Gate |
|---|---|---|
| P2.1 | Vyb training-data builders: `_build_kv.vyb`, manifest builder (replaces `_build_kv.py`, `build_fullmanifest.py`, `gen_kv_train.py`, `rbuild.py`) | regenerated data byte-matches committed Vyb drivers (S=93/9 regression identity); corpus ids match `encode_corpus.vyb` |
| P2.2 | Full-model LoRA training driver: bf16/GGUF base weights frozen (q4_0 dequant path), LoRA bf16 params + AdamW, multi-epoch over 216-record corpus, loss to disk | per-step loss matches numpy oracle (reuse `train_full_loop_ref` family) at S=93; then scale to full corpus |
| P2.3 | Adapter save/load in Vyb-native .bin format (manifest + raw bf16) | `loradec_driver.vyb` consumes a Vyb-trained adapter: kvresp decode gate + schema-valid contract |
| P2.4 | Behavior parity: Vyb-trained adapter vs committed torch-trained artifacts on 24-record eval | eval CE within tolerance; kvresp behavioral gate (top-1 match on eval records) |
| P2.5 | `smoke_adapter.vyb` (prompt → adapter decode), drop torch path from scripts | `start-training` equivalent runs end-to-end on GPU, no python3 |
| P2.6 | Delete `training/*.py` + torch training scripts; `.venv` REMAINS (oracles need numpy/torch refs) | final repo-wide audit: `grep python3` hits only oracle/gate targets + legacy docs (the oracles are deliberately kept — see **Oracle policy** below) |

## P1.8 audit — every Python hit classified (2026-10-02)

Method: `grep -rnw "python\|python3"` over `native/Makefile` and every `*.sh` in the
repo, plus a `.py` reference sweep of README, `native/ROADMAP.md`, `native/README.md`,
`doc/` and the handoff notes. Classification is the Oracle policy below: reference
verification stays, production Python goes.

| Where | Hits | Class | Action |
|---|---|---|---|
| `native/Makefile` | **70 invocations, 56 distinct scripts** | reference verification, all of them: 55 match `verify_*.py` / `*_ref.py` (numpy/torch references and oracle verifiers), plus `native/tools/emit_prompt_ids.py`, which turns the GPU encode probe's own output into the ids the numpy gold embeds — oracle input | **kept** |
| `native/legit/run_*.sh` gates (P1.1–P1.8) | verifier calls only, venv-preferred | reference verification | kept |
| `native/gdecode/run_pipeline.sh` | stage D verifier (venv-preferred since P1.6) | reference verification | kept |
| `run.sh`, `run-vyb.sh`, `vyb-run.sh` | **0** | production entry points | Vyb-only (`run.sh` drives `app/configurator.vyb` since P1.4) |
| `training/generate-data.sh` | inline `python3 -` heredoc that synthesizes the train/eval JSONL | training path, not the Phase-1 production path | **DONE in P2.5a** — replaced by `training/split_dataset.vyb`, byte-verified against the committed splits and the regenerated corpus (`native/legit/run_split_gate.sh`); only the `start-training` venv bootstrap is still Python |
| `training/start-training.sh` | `python3 -m venv` + torch bootstrap on the training host | training path | deferred to Phase 2 |
| CI | this repo has no `.github/` workflows | — | nothing to update |

Deleted across Phase 1, verified absent: the 7 converted drivers (`apply_interview`,
`propose_repair`, `configurator`, `configurator_repl`, `interview_infer`,
`coerce_contract`, `mk_fixture`) and the 7 removals (`test_schema`, `test_applier`,
`test_backends`, `dump_qwen3_meta`, `step1_topk`, `topk_step0`, `step1c_topk`).
`tests/` now holds only `test_schema.vyb`, which carries `make schemacheck`.

Stale doc references found and fixed by the sweep (each named a deleted file):

| Doc | Was | Now |
|---|---|---|
| `README.md` | the applier example called `./tools/apply_interview.py patches.jsonl` with a `Vyb-vybos` binary | the Vyb invocation, with `VYBFORGE_PATCHES` / `VYB_BIN` (the toolchain that compiles the rendered program) spelled out |
| `native/ROADMAP.md` | ground truth "captured by `native/gguf/dump_qwen3_meta.py`" | marked retired in P1.1: read directly by `read_real_meta.vyb`, TSV committed as the record |
| `HANDOFF-NEXT-SESSION.md` | a historical log still promising the `test_backends.py` port as a follow-up and the `test_schema.py` original as green | banner added: historical, read this doc first, and the two stale claims named |

Every other `.py` mentioned in the docs names a living oracle.

**Verdict: Phase 1 is complete.** Zero Python on any production path in VybForge —
drivers, make targets and run scripts are Vyb. What remains is the reference set
(70 Makefile invocations of 56 scripts) deliberately kept under the Oracle policy,
plus the Phase-2 training bootstrap.

Re-runnable evidence: `native/legit/run_phase1_battery.sh` runs every P1.x
acceptance check plus the production-path checks and prints one line per step. It
also prints the toolchain's HEAD and the `build/vyb` mtime, because that binary is
rebuilt mid-session in the Vyb checkout — a half-finished rebuild makes Vyb programs
fail in ways that read as repo bugs (see the script header).

## Oracle policy

**Decided 2026-10-02 (Rick). The Python oracles stay.** They are kept as-is through
Phase 1 and Phase 2 — they are not conversion targets, no matter how small or how
mechanical they look.

Why: an oracle is a *second implementation*, not a specification, so it can be
wrong — and we may yet find a bug in one. That is the point of keeping it: it is
the thing that has caught the real defects during this cleanup (the `io::open_append`
descriptor bug, the `json_parse` `\uXXXX` pass-through, the two `run_pipeline.sh`
wiring faults), and it can only keep doing that while it is genuinely independent
of the Vyb code it checks.

Hold period: **until we are done testing a few models** (the model sweep through
SpikingBrain-7B and the fleet rigs). Revisit only after that.

When an oracle is eventually replaced by a Vyb implementation, **leave a comment in
the replacement naming the oracle it replaced** — which file it was and what it
asserted — so the bug surface of the retired oracle stays traceable from the code
that took over. (Recorded here so the eventual replacement doesn't just silently
absorb a possibly-wrong expectation.)

Practical consequence for the audits: a `python3` hit in a make target or run
script is judged by *what it does*, not that it exists. Reference verification
(an oracle asserting a Vyb artifact against an independent implementation) is
sanctioned and stays; production work done in Python is what the audit removes.

## Final invariant

`make verify`, `schemacheck`, gdecode pipeline, apply pipeline, repair boundary,
chat + loradec + training all green; the only `python3` invocations left are the
reference-verification gates (and the oracle tooling they need) — see the oracle
policy above for why they are deliberately kept, and how their eventual Vyb
replacements should record what they replaced.
