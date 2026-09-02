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

