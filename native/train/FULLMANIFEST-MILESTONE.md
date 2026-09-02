# FULL-MANIFEST (369→429 token) CONTEXT — MILESTONE (in progress)

Goal (KVCACHE-CONTEXT.md item 4): condition the whole-stack response training on the FULL VybOS
capabilities manifest (system prompt + manifest + user goal), the real KV-cache payoff the per-token
trainer (ad56579) is built for.

## Material built
- `native/train/build_fullmanifest.py` + `native/out/fullmanifest.txt` + `native/out/fullmanifest_ids.bin`
  = the capabilities manifest + system prompt + goal-desktop user goal, tokenized = **429 context ids**
  (near the ~369 target). Trains the response (84 tokens) conditioned on this context.

## VALIDATION REALITY (read before building)
The numpy frozen-context oracle's attention-backward is O(S^2*H*HD) pure-Python. S=93 -> ~6 min; S=513
(429 ctx + 84 resp) -> HOURS (impractical as a per-run gate). So the 369-scale milestone CANNOT reuse the
exact numpy oracle cheaply. Validation shifts to GPU-based:
  1. 429-token context cache (CK/CV rows 0..428) == batched full-S forward roped-K/V (GVCTX-style).
  2. Per-token response forward conditioned on the full manifest -> masked CE loss is sensible (no NaN,
     finite) and, when training, DESCENDS over steps.
  3. Mechanism already anchored at S=93 by the committed numpy oracle (ad56579). Scaling to 429 checks
     parameterization correctness (sizes, cache, positions), not the math (already proven).

## Driver parameterization (S=93 -> S=513, NCTX=9 -> NCTX=429)
kvresp_train_kv.vyb hardcodes S=93, NCTX=9. To scale: set S = NCTX + RESP = 513, NCTX = 429; ALL
ASLB field offsets + buffer sizes + CK/CV sizes + grid counts are S-proportional, so regenerate from the
committed S=93 base by substituting the S-derived literals (the per-layer SLAYOUT, per-field byte offsets
[XI..sd], CK/CV [36][S][NKV], XI/XO/XA/XIN S*D, working buffers S*NQ etc, ASLB total). The context build
loop iterates NCTX=429 tokens (P=0..428), response forward 84 tokens (P=429..512). Backward stays S-cost
until the R-restricted response-row backward is built (separate sub-milestone).

Cleanest path: a parameterized generator (extends _build_kv.py) that emits kvresp_train_kv.vyb for any
(S,NCTX) from the committed S=93 per-token driver, recomputing the ASLB offset table + alloc sizes from
S. Verify: (a) it reproduces the committed S=93 result (regression), (b) S=513 compiles + context cache
matches a batched full-S forward, (c) response training descends conditioned on the full manifest.

## Notes
- The design's "369" was an estimate; the real manifest is 429 tokens — the milestone is "full manifest
  context," exact count is secondary.
- True per-step = response cost requires the R-restricted backward (forward is already response-only).
- Not committed; commit only when (b)+(c) above are verified.

## KEY FINDING (2026-09-01): R-restricted backward is REQUIRED, not optional
At S=513 the combined S-ASLB = 36*layers * SLAYOUT(277422192) = **9.99 GB**, and the S=513 batched
backward is ~4.5 h/step. So scaling to 429 with the current S-ASLB+batched-backward is IMPRACTICAL on
memory AND time. The R-restricted design is the enabler AND is memory-lean:
  - Context cache: CK/CV = [36][S][NKV] = 36*513*1024*8 = 151 MB each (tiny) — holds context rows 0..428
    (from the context build) + response rows 429..512 (per-token forward appends).
  - Response activation cache: per-layer [R=84] compact (25 fields) — ~100 MB total. NOT the 10 GB S-ASLB.
  - Total training memory ~6-7 GB (EMB 3.1 GB + packed weights 2.5 GB + caches) -> fits 24 GB easily.
  - Per-step cost ~ R*NCTX (attention) + R (rest) for fwd AND bwd -> training the 429-manifest is feasible.
=> BUILD the R-restricted per-token trainer (kvresp_train_r.vyb): compact [R] response activation cache +
combined CK/CV + R-size frozen backward + combined-cache dattn. Validate at S=93 first (R=84) against the
committed oracle (KVRESP_TRAIN_VERIFY), then scale S to 513/NC 429. This is the milestone's core build.
Numpy 429 forward target (kvresp429_fwd_ce = 16.754, kvresp429_hidden_gpu_target.bin) is the scaled
forward gate (committed d012b42).

## CONCRETE BUILD PLAN for the R-restricted backward (kvresp_train_r.vyb)
Reliable method = MECHANICAL S->R transform of the COMMITTED (verified) kvresp_train_kv.vyb backward,
NOT a hand-written string builder (that was error-prone). Apply to the real committed backward text:
  1. cp kvresp_train_kv.vyb -> kvresp_train_r.vyb (forward/head verbatim).
  2. In the backward `for (Li2..Lb..)` body:
     a. Replace the S-ASLB full-width load `dcopy(ASLB+Lb*+LayOUT+OFF, bucket, S*rowsize)` lines with
        RESPONSE-ROW compact loads: dcopy(ASLB+Lb*SLAYOUT+OFF + NCTX*rowsize*8, bucket, RESP*rowsize).
        (OFF per field from the same layout table.)
     b. Replace S batch dims: SD 238080->RB 215040, SNQ 380928->RNQ 344064, SNKV 95232->RNKV 86016,
        SFF 904704->RF 817152, S*R 186->RR 168; loc(S)->loc(RESP) in dgemm/dattn batch params;
        SHP->RHP(2688), SKP->RKP(672).
     c. dattn -> COMBINED cache: Q (response roped-q at combined rows NCTX..S-1) + K/V = CK/CV[Lb] slab
        (holds context 0..NCTX-1 + response NCTX..S-1 already) + dC (response dctx at combined rows);
        run dattn(S=NCTX+R); ZERO context-region (rows 0..NCTX-1) dQ/dK/dV (frozen); extract response
        rows -> [R] dQr2/dKr2/dVu2. Requires a compact->combined staging for Q/dC and combined->compact
        extraction for dQ/dK/dV.
     d. AdamW unchanged (all 14 U/V; context LoRA never gets grads).
  3. Compile (vyb) -> fixed: drop the combined-S ASLB, use compact [R] cache + CK/CV only (solves the
     10 GB @ S=513: CK/CV=[36][513][NKV]=151MB). Forward writes response rows to [R] cache + appends
     roped-K/V to CK/CV; context build fills CK/CV rows 0..NCTX-1 once.
  4. Validate @ S=93: (a) forward CE == committed 15.9316; (b) step-1 dU_q/dV_q == oracle (corr 1 at
     ~1e-6, reuse verify_kvresp_train). Then scale S=513/NC=429 with gen_kv_train.py and gate forward
     CE == kvresp429_fwd_ce (16.754) + descent.
VALIDATED tools already committed/working: gen_kv_train.py (S=93 regression == committed driver),
build_fullmanifest.py, kvresp429_fwd_ref.py. NOT yet done: the R-backward body + combined dattn.
