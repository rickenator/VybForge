# KV TRAINER (per-token) — RESOLVED: verified on GPU

**Status 2026-10-02: `kvresp_train_kv.vyb` is verified. `KVRESP_TRAIN_VERIFY: OK`** on a fresh
oracle → driver → verify run (12:39–14:01, toolchain `~/Projects/Vyb/build/vyb` at Vyb HEAD),
committed in `880f4d6` ("TRUE PER-TOKEN KV TRAINER VERIFIED — KVRESP_TRAIN_VERIFY: OK"). The earlier
header of this file ("frozen-backward BUG remains", "DO NOT COMMIT") described a state that no longer
exists; it had been stale since 2026-09-01 and contradicted the committed, verified driver. Removed.
(Filename kept as-is for link stability.)

`native/train/kvresp_train_kv.vyb` = TRUE per-token KV-cache trainer: the 9 context tokens' roped k/v
and activations are built ONCE, the response is forwarded per token into the combined S=93 ASLB cache,
then the unchanged S=93 frozen backward + AdamW runs.

## The bug that lived here — rope-adjoint position mismatch (fixed)

The batched driver roped AND de-roped at `(POS+s)` = `(93+s)`, self-consistent. The per-token forward
(`resp_layer_kv`) ropes each response token at its absolute position (= row index), but the shared
backward `drope` de-roped at `POS+s` — a net −93 rotation that never cancels, giving wrong
`dQn2`/`dq2` and gradients that degraded down the chain (L35 corr 0.71, magnitude ~0.48x → L17 0.38 →
L0 0.37). **Fix:** the driver passes `BPP = 0` to `drope` (was `POS = 93`) so the adjoint de-ropes at
row-index position, consistent with the per-token forward. In the file today: `POS<Int> = 93;
BPP<Int> = 0`, and the launch writes `ROP+224 ← BPP`.

Confirmed by a comment-stripped diff of the two drivers: their backward/optimizer regions are
**byte-identical except that single argument** (plus one added `cuCtxSynchronize()`), so the shared
backward and AdamW are demonstrably the same code in both — the difference is not in them.

## Gate result (fresh run, 2026-10-02)

```
step-1 gradients   L0/17/35 dU_q & dV_q: corr = 1.000000, norm_rel 2.0e-5 .. 5.4e-5   (tol 1e-3)
per-step CE loss   ref [9.338425 7.498938 6.591507 5.991288]
                   gpu [9.33844  7.52178  6.61066  5.99206 ]  maxrel 2.4e-3 (tol 5e-2), corr 0.999968
DESCENT            OK   9.33844 -> 5.99206
final L0 Uq parity maxrel 3.6e-2   (informational)
KVRESP_TRAIN_VERIFY: OK
```

## Why steps 2+ differ from the batched trainer at all — design, not a bug

The two drivers implement different semantics, verified in the code:

- **per-token** (`kvresp_train_kv.vyb:758`, comment *"CONTEXT BUILD (ONCE)"*): the 9 context tokens are
  forwarded **once, before the step loop**, with the adapters applied (`UqP..VdP` from `LSLB`), caching
  their roped k/v into `CK`/`CV` rows 0..8 and their activations into `ASLB` rows 0..8. Every step
  thereafter the response tokens attend over that prefix, frozen at step-1 adapter values.
- **batched** (`kvresp_train.vyb`): no pre-loop phase; `run_layer(S=93, …)` with the adapters runs
  *inside* the step loop and rewrites all 93 rows each step, so its context rows are re-derived from
  the current adapters.

At step 1 the adapters are identical, so the gradients agree exactly (corr 1.000000 — the gate's
primary check). From step 2 the prefixes differ by however much the adapters moved, so the losses
diverge by ~2.4e-3 relative and the final adapter parity lands at 3.6e-2. **It does not compound.**

**Correction to the earlier record here.** This file previously attributed the steps-2+ residual to
"single-stream rounding accumulation" and carried 0.14–0.18 loss drift / 32% adapter parity as the
remaining problem. Those figures were measured *before* `7adc585` (2D-weight orientation across the
training drivers) and `89da0d8` (bulk HtoD + `deq_cached` orientation `inz`) landed — mis-oriented
dequant weights inflate exactly those numbers. With those fixes in, the residual is 2.4e-3 / 3.6e-2.
The mechanism above is the standing explanation of what remains; the "rounding" account was wrong.

## Reproducibility caveat: absolute loss numbers are not pinned

The oracle and driver read gitignored `native/out/` dumps (`m2e_l*_Vd.bin`, currently the 36 files
regenerated 2026-09-12, plus the S=93 input/label set). The **relative** gate is sound because the
oracle regenerates its reference files every run — but absolute losses move with those inputs (9.338
now, 15.9316 on 2026-09-01) *and* with the weight-orientation fixes, so no historical absolute figure
is a stable baseline.

## Running it

```
make -f native/Makefile kvresp-train-kv     # oracle -> driver -> verify
```
~80 min wall (oracle ~12, driver ~33, the rest verify/IO). It writes the **same** `native/out/` files
as the batched `kvresp-train`, so the two must never run concurrently. To regenerate the driver from
the committed batched base: `.venv/bin/python native/train/_build_kv.py` — then re-apply the `BPP = 0`
rope fix, which the generator does not carry. Not wired into `run_phase2_battery.sh`: it is a heavy GPU
gate, unlike the battery's fast port-parity checks.

---

## Historical log (2026-09-01) — kept for provenance

### FORWARD: VERIFIED (2026-09-01)
FWD_GATE_LOSS = 15.9316 == batched forward step-1 exactly (and oracle 15.957 within numerical tails).
Context build + per-token response forward + combined-cache attention are CORRECT.

### FROZEN-BACKWARD GATE: FAIL (2026-09-01)
Full 4-step run: losses 15.9316, 15.9762, 15.4101, 15.0188. Step-1 matches, steps 2-4 drift (>oracle).
verify_kvresp_train.py -> KVRESP_TRAIN_VERIFY: FAIL on ALL step-1 gradients (dU_q/dV_q L0/17/35:
corr 0.37-0.71, norm_rel ~1.0), loss-match rel 5.8e-2, L0Uq 2.9e-1. The forward was right
(hiddens/loss matched) while the frozen backward produced wrong per-step grads.

### DIAGNOSTIC PROBES: FORWARD ASLB IS FULLY CORRECT (2026-09-01)
Probe1 (L0 row10): xin/xn/ctx/xo/sq all OK (~1e-6). Probe2 (L0/17/35 row10): L0/L17/L35 layer-input
(offset-0, manual fill) OK at all three depths; L17 xn/dqr/dkr/ctx/m2/sq OK (~1e-6). Probe3 dumped EVERY
backward-read ASLB field at L0 (DQ,DK,DV,DO,X1,X1N,Gr,Up,Hu,sq,sd...sk..su) + layer-input (offset-0) at
L0/17/35 + xn/dqr/dkr/ctx/m2/sq at L17: ALL match numpy to ~1e-6. => NOT a forward ASLB fill bug.
- Clean full run REPRODUCED deterministically: losses 15.9316, 15.9762, 15.4101, 15.0188.
- Step-1 gradient signature: L35 dU_q corr 0.71 with magnitude ~0.48x; L17 corr 0.38; L0 corr 0.37
  => degrades DOWN the chain.

### FIX APPLIED + VERIFIED (2026-09-01)
Root cause found via the L35 backward chain instrument: frope/drope are orthonormal rotations, so they
only need fwd/bwd CONSISTENCY — see the fix section at the top. Verified: L35 chain dQn2 & dq2 flipped
FAIL->OK (corr 1.0, ~1e-6); step-1 dU_q/dV_q L0/17/35 all corr 1.0 / ~2e-6. RUN at the time: losses
15.9316, 15.4812, 14.6657, 13.9533 vs oracle 15.957,15.495,14.841,14.095 (rel 1.1e-2, corr 0.998,
DESCENDS). The residual it recorded then (steps-2+ drift, 32% adapter parity) has since been reduced
by the orientation fixes to 2.4e-3 / 3.6e-2 — see above.
