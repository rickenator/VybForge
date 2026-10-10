# MTP draft path — measured baseline and the chained-draft harvest

Status: design + cost model. The chain is **not implemented yet**; P4.13 remains the single-step
teacher-forced test. Written to be acted on: each step below is a concrete change with a measurement.

## 1. What is measured today

| thing | value | source |
| --- | --- | --- |
| our draft acceptance, 4-step capital fixture | 4/4 exact | `run_ridge_mtp_gate.sh` |
| our draft acceptance, 19-step counting fixture | 17/19 (89.5%) | same, rule-aware |
| our main-pass hidden agreement | cos 0.999609 | P4.12 |
| oracle's own MTP acceptance | **not measurable** | see below |

The oracle gives no usable baseline. At llama.cpp `4df29be`: `LLAMA_CONTEXT_TYPE_MTP = 1` is public
(`llama.h`, settable via `cparams.ctx_type`) and `qwen35.cpp` has `graph_mtp`, and `common/speculative.cpp`
plus `tools/server/server-context.cpp` both reference the MTP context — but `/props` reports
`"speculative.types":"none"` and no flag drives it. Its speculation framework is built around a *separate
draft model* (the entire `--spec-draft-*` family is draft-model config). So llama.cpp ships the head and
does not use it: matching its acceptance is not a target, and the bar must come from our own measurement.

## 2. Why chaining is the real lever (the cost model)

A token's worth of drafting costs, per pass:

- **the draft-head walk**: the chunked lm_head walk over `output.weight` (10.2 GB at 3.7 bpw). This is
  **bandwidth-bound** and it is a single gemm over ALL draft rows — it does **not** scale with the number
  of rows. This is the dominant term, by far.
- **one block-forward per drafted token** (attention + FFN of one block), i.e. ~1/64 of a verification
  pass over the same positions.
- **one verification pass** (64 layers) to check the drafted run.

Therefore N chained drafts per verify cost `1 walk + N blocks + 1 verify`, against a single-token MTP
loop's `1 walk + 1 block + 1 verify` per token. The walk is shared, so the per-accepted-token cost of the
dominant term falls ~N×. That is the "great vs mid" gap, and it is a property of the path, not of the
weights.

## 3. What the chain requires (the structural change)

Today's MTP mode pre-builds **all** S rows of the block input `XA` in one loop (driver ~1280-1299) and
then runs the block once over those rows. A chain cannot be pre-built: step k's input depends on step
k-1's **output**.

1. Run the draft rows **serially**, one row per step: `(h'_k-1, e(d_k-1)) -> d_k`, with `h'_0` = the
   main pass's final normalised hidden and `d_0` = the first drafted token.
2. Feed the block's own **output hidden** forward as the next step's `h`. The block is a transformer
   block, so this is exactly what it is for; no new weights, no new kernel.
3. Attention extent for the chain: row k must see rows `0..k` (its own prefix), not itself alone. This is
   **the risky part** — the teacher-forced runs validate none of it, because each of their steps consumes
   the oracle's own hidden row and is therefore self-consistent by construction.
4. Stage the block's weights **once** for the whole chain (they do not change between steps).

## 4. The measurement that decides whether it pays

Ground truth costs nothing: each fixture already records the oracle's greedy pick at every position, with
its top-2 margin and its `exact`/`membership` rule.

- **acceptance vs N**, N = 1..4, per fixture: fraction of chained drafts that equal the oracle's pick at
  their position (rule-aware: equality where margin >= `margin_bar`, membership of `{top1,top2}` below).
- Acceptance is expected to decay with k; the point of the measurement is the *product*, i.e. the expected
  number of accepted tokens per verify, against the cost model in §2.
- Caveat, stated plainly: our 89.5% is the teacher-forced rate at each step's own position, i.e. the
  best case for k=1. A chained run's k=2..N rates are unknown until measured, and every step of the chain
  is fed our own hidden rather than the oracle's.

## 5. The instrumentation already exists for the offline version

`VYB_MTP_HIDDEN` already takes the draft rows from a **file** and `VYB_PROMPT_IDS` takes the token ids, so
the chain can be measured without new engine code: dump the block's output hidden rows from run k, feed
them back as run k+1's `VYB_MTP_HIDDEN` with our own drafted tokens as the ids, and compare each run's
drafts against the fixture. The production version is the serialised loop in §3; the offline version is
how the acceptance-vs-N curve gets measured first.

## 6. Order of work

1. Dump the block's output hidden per draft step (one small addition to the MTP mode; the probe's stage
   dumper is the existing pattern).
2. Offline chain: iterate runs, measuring acceptance vs N against both fixtures.
3. Serialised chain in the engine (`eng_mtp()`), staging weights once, with the chain's attention extent
   as the one genuinely new correctness question.
4. Set P4.13's bar from §4's numbers, replacing the exact-match criterion.
