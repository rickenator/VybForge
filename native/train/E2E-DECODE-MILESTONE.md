# E2E DECODE → agent-response JSON (Milestone b) — status + build plan

Goal: the interviewer RUNTIME — given a user goal (+ manifest), generate the {kind, message,
missing_fields, proposed_changes, requires_confirmation} JSON via on-GPU decode and validate it. This
is "the fast responsive machine" (inference), independent of training throughput.

## VERIFIED FOUNDATION (done, committed earlier)
- `native/gdecode/contract.vyb` — Vyb agent-response struct + doc_mk_response/doc_to_json emission,
  field order matches config/agent-response.schema.json. `make -f native/Makefile gdecode` ->
  CONTRACT_VERIFY: ALL_OK (question/summary/proposal serialize to schema-valid JSON, roundtrip OK).
- `native/gdecode/run_pipeline.sh` = E2E vertical slice: pipeline_encode (tokenize prompt) ->
  decode_driver (base on-GPU decode) -> pipeline_emit (detokenize + EMIT A CANNED SUMMARY contract,
  not parsed from the model's text) -> verify_contract. It exercises the path but is NOT a real
  interviewer emission (base model, no LoRA, canned summary).

## THE REAL BUILD (what "emit the agent-response JSON" needs)
1. **PERSIST trained adapters**: the per-token trainer (kvresp_train_kv, ad56579) updates LSLB in GPU
   memory and only dumps L0-Uq. Add per-layer U/V dump at the end of a training run -> a decode-loadable
   adapter set (or use the seed m2e_l*.bin to begin).
2. **LoRA-applied decode driver**: extend kv_driver.vyb (KV-cache greedy decode, base weights) so the
   per-token layer forward applies proj_lor (a@W + alpha*(a@U@V)) on all 7 projections using the loaded
   adapters. Reuse the per-token forward from kvrespfwd.vyb's resp_layer_kv (verified LoRA forward) +
   kv_driver's autoregressive loop (predict-next-token, append, stop on EOS/newline/N tokens).
3. **Parse generated text -> AgentResponse**: a strict-ish JSON extractor + contract deserialization
   (contract module) -> {kind,...}; validate with verify_contract.py's jsonschema.
4. **E2E gate**: run_pipeline-style flow on the goal-desktop prompt with LoRA; assert generated JSON is
   schema-valid AND kind/message coherent. Honest caveat: teacher-forced CE training (descends) does NOT
   guarantee fluent self-consistent generation — first run establishes whether the adapters actually
   emit the contract; if not, a short RL-style/next-token fine-tune produces a generation-capable set.

## Sequencing (chosen: b then a)
- b1) Persist trained adapters + LoRA-applied KV decode of the goal prompt -> eyeball/interview-trend.
- b2) Parse generated -> contract + schema gate.
- a)  R-restricted backward (per FULLMANIFEST-MILESTONE.md) for 429-manifest training scale.

## b1+b2 STATUS (2026-09-01) — decode machinery VERIFIED on GPU; contract emission NOT yet
The full LoRA-autoregressive decode pipeline is now built and RUNS cleanly on the 3090,
and the honest caveat is now empirically confirmed:
- **b1 DONE (machinery):** `native/host/loradec_driver.vyb` (derived from the verified
  kvrespfwd.vyb LoRA-loading + per-token resp_layer_kv) + autoregressive gen loop: first
  response token predicted from the last context hidden, then per-position forward into the
  combined CK/CV cache + tied lm_head argmax (`lm_next`, logits_slice+argmax_row from
  qwen3.ptx). `make -f native/Makefile loradec` -> loads LSLB (m2e_l*.bin seed adapters) +
  PKD weights, builds the 9-token goal-context cache, generates 49 tokens ->
  `native/out/loradec_tokens.txt`. VERIFIED: `LORADEC_DONE`, EXIT=0, GPU 99%.
- **b2 DONE (harness):** `native/gdecode/loradec_emit.vyb` detokenizes the decode ->
  `native/out/loradec_text.txt`; `native/gdecode/verify_loradec.py` extracts any JSON object
  and gates it against config/agent-response.schema.json.
- **RESULT: `LORADEC_VERIFY: NO_JSON_CONTRACT`** — the 49 decoded tokens detokenize to
  incoherent text (`观念勃久垒招焕发erberbrobeallet...`), NOT a schema-valid agent response.
  As the caveat predicted: teacher-forced CE-trained seed adapters do NOT emit fluent
  self-consistent generation of the interview contract.
- **IMPLICATION:** the Decode MACHINERY is done and reproducible, but a GENERATION-CAPABLE
  adapter set is REQUIRED to close milestone b. Next step (as anticipated): a short
  NEXT-TOKEN autoregressive fine-tune (train on the contract target with the decode loop, so
  adapters learn to emit `{kind,message,...}`), then re-run `loradec` + `verify_loradec.py`.
  This is the "short RL-style/next-token fine-tune" the plan named.

## b1+b2 STATUS-2 (2026-09-01/02) — ROOT CAUSE + PERSISTENCE FIX (now unblocked by Vyb #213)
The reason b1 decoded incoherently: loradec_driver loaded the SEED `m2e_l*.bin` adapters, but the
CONTRACT-TRAINED adapters live only in GPU memory (LSLB) and kvresp_train_kv NEVER persisted them
(only 128-wide text slices of L0/17/35 for the parity gate). Filed **`rickenator/Vyb#213`** = stdlib/io
raw-binary write (write_bytes/write_at/read_bytes/read_bytes_at, Vec<UInt8> carrier). **CLOSED + rebuilt**
(toolchain = Vyb 0.7.4; verified `IO_RAW_BINARY_ROUNDTRIP_OK`, LE f64 == numpy '<f8').
- **kvresp_train_kv.vyb now PERSISTS** all 36 layers x 14 U/V tensors from LSLB -> `kvresp_l{L}_{T}.bin`
  (raw LE f64, pure Vyb via dump_bin + write_bytes), after the training steps. Base 917504 bytes/layer
  matches loradec's LSLB layout exactly.
- **loradec_driver.vyb now LOADS `kvresp_l*.bin`** (the trained set) instead of m2e_l*.bin.
- RUN: `make -f native/Makefile kvresp-train-kv` (training + persist), then `make loradec` +
  `loradec_emit.vyb` + `verify_loradec.py` (jsonschema gate). If 4 NSTP isn't enough for fluent
  contract emission, bump NSTP and re-run.

## b1+b2 STATUS-3 (2026-09-02) — persistence DONE + verified; generation still NOT fluent
- Pure-Vyb adapter persistence is now VERIFIED: kvresp_train_kv dumps all 36x14 trained U/V ->
  `kvresp_l{L}_{T}.bin` (raw LE f64 via #213 write_bytes; dump_bin matches numpy '<f8' byte-for-byte).
  Trainer still descends 15.93->13.95 == oracle (`KVRESP_TRAIN_VERIFY: OK`). loradec_driver now loads
  the trained `kvresp_l*.bin`.
- EMPIRICAL RESULT: decode with the 4-step-trained adapters STILL yields `LORADEC_VERIFY:
  NO_JSON_CONTRACT` (different garbage than the seed, but no schema-valid JSON).
- WHY: 4 teacher-forced CE steps move the adapters only ~1e-4 abs (relative % looks big only because
  adapter values are tiny near zero) — not enough to reshape the model into emitting fluent,
  self-consistent JSON from scratch. This confirms the plan's deeper caveat: teacher-forced next-token
  CE on one goal->84-token response, run a few steps, does NOT yield generation-capable decoding.
- REAL GAP for closing b: either (a) many more training steps so the adapter actually learns to
  reproduce the contract response sequence ARGMAX-wise, or (b) a DECODE-CONDITIONED objective — feed
  the model's OWN sampled/argmax previous token into the context during training (scheduled sampling /
  student-forcing) so it learns to recover from its own prefixes, matching the autoregressive decode
  distribution. (b) is the faithful "RL-style/next-token fine-tune" the plan named and is the likely
  correct fix. It changes the RESPONSE forward's next-token input from ground-truth to model-sample
  (with annealing prob), a new but bounded edit to kvresp_train_kv.

## b1+b2 STATUS-4 (2026-09-02) — decode-conditioned oracle: OPEN-LOOP injection does NOT descend
Built `kvresp_train_dm_ref.py` (numpy oracle for the decode-conditioned objective, deterministic
no-RNG schedule) and ran it 8 steps:
  dm-oracle loss: [15.9566, 15.4951, 16.1554, 15.6201, 16.2223, 15.8696, 15.6099, 15.5742]
Steps 1-2 are teacher-forced (descend 15.96->15.50, the known good floor). Step 3+ turns on OPEN-LOOP
student injection (~80% of response positions fed the model's own argmax token) — and the loss RISES
back to ~16.2 and oscillates, NEVER going below the step-2 teacher-forced floor.
WHY: the injected argmax tokens are STILL model-garbage (the adapter isn't fluent yet), so the model
gets nonsense prefixes and the CE target becomes inconsistent -> training destabilizes. Open-loop
injection (argmax from the PREVIOUS truth-conditioned forward) is NOT the right student-forcing form.
REAL form of (b): CLOSED-LOOP student-forcing — the response forward must become GENUINELY
AR autoregressive (decode-shaped, token-major: feed the model's own PREVIOUS-STEP generated token into
the next position through the actual decode loop), not a one-shot batched forward that injects the
previous forward's argmax. That is a real re-architecture of the trainer's response forward + a
matching closed-loop oracle, a much larger build with 40-60 min GPU cycles.
DECISION POINT (for Rick): (i) closed-loop student-forcing (the faithful fix, big new
forward/backward + oracle + many GPU runs, uncertain convergence), (ii) many more teacher-forced steps
(option a, cheap, keeps verified gate, heuristic: helps argmax-reproduce the single contract),
(iii) accept that tiny-LoRA teacher-forced fine-tune is not the generation path and pivot to a
different training objective/scale for the interviewer runtime.

## STATUS-5 (2026-09-03) — open-loop scheduled-sampling: TWO forms both fail; running option-(ii) diagnostic
The v2 oracle `kvresp_train_dm_ref.py` was corrected to the STANDARD Bengio inverse-anneal
(teacher_prob 0.95 -> 0.05 across 8 steps; earlier v1 wrongly forced ~80% student from step 3).
Ran to completion; trajectory:
  v2 dm loss: [15.9566, 16.1877, 15.7636, 15.6989, 15.1875, 15.6352, 15.7201, 16.1439]
  teacher_p:  0.95   0.82   0.69    0.56    0.44    0.31    0.18   0.05
STEP 1 = 15.956607 EXACTLY matches the committed v1 step1 -> base machinery is faithful (not a bug).
BUT: min 15.1875 @ step5, FINAL = 16.1439 > STEP-1 15.9566 -> the run OVERALL does NOT descend.
Both open-loop forms now fail: v1 (fixed ~80% student) and v2 (properly annealed 5%->95% student).
MECHANISM (why open-loop scheduled sampling cannot work here, not a tuning bug):
  * the injected "student" tokens are the model's OWN argmax, which are GARBAGE before the adapter
    is fluent. Feeding garbage prefixes + a teacher CE target is self-inconsistent -> the loss
    oscillates around ~15.6-16.2 and never settles below the ~15.5 teacher floor.
  * the dm open-loop loss is a CONTAMINATED metric: as the student fraction rises the INPUT
    distribution worsens, so CE goes UP even if the model were improving — it is not a clean
    "is the model learning" signal. Scheduled sampling's precondition (a student that already
    approximates the target) is not met by a near-zero 4-step adapter.
IMPLICATION: open-loop / batched student-forcing is NOT the path. The faithful form (i) is CLOSED-LOOP
student-forcing (a genuinely autoregressive decode-shaped response forward), which is a large
re-architecture. Before committing to that build, a CHEAP decisive diagnostic for option (ii) is
running: `kvresp_tf_sweep_ref.py` (pure teacher-forced CE, 20 steps, NO injection) tracking
  (a) teacher-forced CE and (b) ARGMAX-MATCH rate over the 83 response labels — the direct
  "can the adapter actually reproduce the contract token-by-token" measure. If argmax-match stays
  ~0 even as CE descends, teacher-forced fine-tune does NOT yield an emitting adapter (reinforces
  pivot/iii or the closed-loop build); if it climbs, more steps is enough (cheap option ii).





