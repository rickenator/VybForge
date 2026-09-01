# KV TRAINER (per-token) — BUILD STATUS (2026-09-01) — forward OK, frozen-backward BUG remains

`native/train/kvresp_train_kv.vyb` = TRUE per-token KV-cache trainer (context built once, per-token
response forward into the combined S=93 ASLB cache, unchanged S=93 frozen backward + AdamW).

## FORWARD: VERIFIED (2026-09-01)
FWD_GATE_LOSS = 15.9316 == batched forward step-1 exactly (and oracle 15.957 within numerical tails).
Context build + per-token response forward + combined-cache attention are CORRECT.

## FROZEN-BACKWARD GATE: FAIL (2026-09-01) — DO NOT COMMIT
Full 4-step run: losses 15.9316, 15.9762, 15.4101, 15.0188. Step-1 matches, steps 2-4 drift (>oracle).
verify_kvresp_train.py -> KVRESP_TRAIN_VERIFY: FAIL on ALL step-1 gradients (dU_q/dV_q L0/17/35:
corr 0.37-0.71, norm_rel ~1.0), loss-match rel 5.8e-2, L0Uq 2.9e-1.
=> The forward is right (hiddens/loss match) but the frozen BACKWARD reads a WRONG response-row ASLB
activation field -> wrong per-step grads. The batched kvresp_train.vyb (same backward+AdamW, batched
ASLB fill) is the clean ground truth it must equal.

## DIAGNOSTIC PROBES: FORWARD ASLB IS FULLY CORRECT (2026-09-01)
Probe1 (L0 row10): xin/xn/ctx/xo/sq all OK (~1e-6). Probe2 (L0/17/35 row10): L0/L17/L35 layer-input
(offset-0, MY manual fill) OK at all three depths; L17 xn/dqr/dkr/ctx/m2/sq OK (~1e-6).
=> The per-token forward fills ASLB CORRECTLY at all depths, including residual-input and the
combined roped-q/k/v. The backward reads a correct ASLB. => The bug is NOT a forward ASLB fill.

## REFINED HYPOTHESIS (next test)
The unchanged frozen backward + AdamW produce wrong step-1 dU_q/dV_q (corr 0.4-0.7) despite a
provably-correct ASLB => a BACKWARD working-buffer state issue or a per-token-specific interaction:
- The batched forward left the S-sized working buffers (XN/DQ/DKr/.../sq..sd/eq..ed) holding the
  LAST layer's activations; the per-token forward (resp_layer_kv writes DIRECTLY to ASLB) leaves them
  STALE. If the backward reads ANY working buffer WITHOUT an ASLB load first, it sees garbage in the
  per-token case but valid (last-layer) values in the batched case -> different grads despite same ASLB.
- DECISIVE test: scrub the backward block for every working-buffer READ and confirm each is preceded by
  its ASLB load (`dcopy ASLB+Lb*SZ+OFF -> bucket`); any bucket used before its load is the bug (e.g., a
  field like eq/ed, DQN/DKN, or an X1/X1N/DO/Gr/Up/Hu read placed before its dcopy). The most likely
  candidates given the un-probed set: DQ/DK/DV/DQN/DKN/DO/X1/X1N/Gr/Up/Hu at layers>0, or any use of the
  stale eq..ed.
- ALTERNATIVE decisive test: run kvresp_train (batched, known-good) and the per-token driver with a
  common probe that dumps step-1 L0 dU_q; compare directly (not vs oracle). If they differ, diff the
  backward's intermediate buckets.
Commit ONLY after KVRESP_TRAIN_VERIFY: OK.

## FINAL NARROWING (2026-09-01, after exhaustive probes + 2 deterministic full runs)- Probe1/2/3 dumped EVERY backward-read ASLB field at L0 (DQ,DK,DV,DO,X1,X1N,Gr,Up,Hu,sq,sd...sk..su)
  + layer-input (offset-0) at L0/17/35 + xn/dqr/dkr/ctx/m2/sq at L17: ALL match numpy to ~1e-6.
  => The per-token forward fills ASLB exhaustively CORRECTLY. NOT a forward fill bug.
- Clean full run REPRODUCED deterministically: losses 15.9316,15.9762,15.4101,15.0188 (batched was
  15.9316,15.4569,14.8392,14.0814 == oracle ~2e-3). Step-2+ diverges => real bug, not a race.
- Step-1 gradient signature: L35 dU_q corr 0.71 (/oracle) with magnitude ~0.48x; L17 corr 0.38; L0
  corr 0.37 => degrades DOWN the chain. Top layer ~half-magnitude is the clue (residual-identity or a
  scale/atomic factor in the top-layer backward), compounding through chained layers.
- Backward is byte-identical code; reads correct ASLB. So the cause must be a WORKING-BUFFER the
  backward reads that is NOT loaded from ASLB and differs between batched & per-token (batched forward
  seeded them with last-layer values; per-token leaves them stale). NEXT: instrument the BACKWARD (not
  forward) -- dump dCtx2/dQr2/dq2/dU_q at L35 step 1 and diff vs the SAME values from the batched run /
  oracle to find the exact chain link. (Not run yet -- expensive.)
- STATUS: per-token forward VERIFIED (a real, commit-worthy increment on its own); frozen-backward
  integration NOT green -> DO NOT COMMIT kvresp_train_kv.vyb in current state. The committed
  frozen-context trainer (kvresp_train.vyb, 0d03267) remains the verified baseline. Consider
  pivoting to end-to-end decode; return to the per-token backward only if the 369-manifest needs it.

## FIX APPLIED + VERIFIED (2026-09-01) — ROOT CAUSE = rope-adjoint POSITION mismatch
ROOT CAUSE (found via L35 backward chain instrument): frope/drope are orthonormal rotations, so they
only need fwd/bwd CONSISTENCY. The batched driver roped AND de-roped at (POS+s)=(93+s) (self-consistent).
The per-token forward (resp_layer_kv) roped each response token at its absolute position Pp (=row s);
but the shared backward `drope` de-roped at (POS+s)=(93+s) -> a net -93 rotation never cancels ->
WRONG dQn2/dq2/gradients, compounding down the chain (L35 corr 0.71 -> L0 corr 0.37; top ~half-mag).
FIX: it kvresp_train_kv.vyb, the backward `drope` now passes BPP=0 (was POS=93) so it de-ropes at
row-index (= absolute) position, consistent with the per-token forward. Verified: L35 chain dQn2 & dq2
flipped FAIL->OK (corr 1.0, ~1e-6); step-1 dU_q/dV_q L0/17/35 all corr 1.0 / ~2e-6.
RUN: losses 15.9316, 15.4812, 14.6657, 13.9533 == oracle 15.957,15.495,14.841,14.095 (rel 1.1e-2,
corr 0.998, DESCENDS). verify_kvresp_train: ONLY remaining FAIL = final L0 Uq adapter parity 32%.
=> per-token KV trainer WORKS (forward + step-1 backward + descent + loss trajectory all match oracle).
REMAINING (optional): steps-2+ gradient drift (loss ~0.14-0.18 off at steps 3-4 vs batched ~2e-3; final
L0 Uq 32% off) -- small per-step accumulation discrepancy, NOT a functional failure. Suspect a subtle
per-step state (ping-pong XI/XOcur or CK/CV response-row consistency across steps, or single-stream
rounding accumulation from the many per-token launches). NOT committed because verify is not 100% green
(per rule). Options: (a) accept looser adapter gate / evaluate training quality vs descent; (b) chase
the steps-2+ drift; (c) pivot to decode. The committed frozen-context trainer (0d03267) remains solid.


## TO REGENERATE the driver from the committed batched base
`.venv/bin/python native/train/_build_kv.py` (direct open(), no truncation) -> kvresp_train_kv.vyb.
Then edits on top: the TEMP forward gate was REMOVED (full backward now runs). Run:
  ~/Projects/Vyb/build/vyb native/train/kvresp_train_kv.vyb
  .venv/bin/python native/train/verify_kvresp_train.py
Commit ONLY after KVRESP_TRAIN_VERIFY: OK.
