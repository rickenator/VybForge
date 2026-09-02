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
