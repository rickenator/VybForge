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

## 2. Why chaining is the real lever (the cost model) — CORRECTED

An earlier version of this section claimed the chain amortises the draft-head walk because the walk is a
single gemm over all rows and does not scale with row count. **That is wrong, and it mattered:** the walk
is incurred *per draft step*, because step k's input is the embedding of the token drafted at step k-1,
which exists only after the head's argmax. The head is on the critical path once per step, and no amount
of batching changes that. The "shared walk" only ever held for the teacher-forced pre-build, where all
rows are known up front — which is exactly the freedom a real chain does not have.

The arithmetic that actually holds (bytes of weight traffic, order-of-magnitude, not yet measured):

| term | traffic | note |
| --- | --- | --- |
| one draft step | ~1 block (~1.3 GB) + ~1 head walk (~1.0 GB) | ≈ 2.3 GB |
| one verification pass | 64 blocks (~84 GB) + head (~1.0 GB) | the token's real cost |

So a drafted token costs ~2-3% of the work of computing that token outright, and **accepting a draft saves
~97% of that token's cost**. The chain's *additional* benefit is that one verification pass covers the
whole accepted run, so the pass — dominated by the 64 block-forwards — is amortised over N accepted
tokens rather than one. Both effects are real; they are just not the one originally stated.

Consequence for the target: driving the built-in head *at all* is the bulk of the win (llama.cpp does not
drive it), and chaining adds the verify-pass amortisation on top. That ordering is what the numbers
support, and it is the opposite of the earlier claim's implication that batching was the lever.

## 3. What the chain requires (the structural change) — as implemented-shape

1. Build the draft rows **incrementally, one per step**, into consecutive rows of `XA`: row k uses
   `h'_k-1` (our own hidden from step k-1, or the fixture's row 0 for k = 0) and the embedding of `d_k-1`,
   our own drafted token. Consecutive rows, not a rebuilt row 0 — the batch must accumulate.
2. Run the block over rows `0..k` (i.e. batch length `k+1`) to get row k's output. **This needs no
   persistent KV cache**: causal attention makes rows `0..k-1` recompute identically from their unchanged
   inputs, which are the ones already in `XA`. The cost is O(N²) block-forwards, which is negligible
   against the head walk (~10 total block-forwards for N = 4, versus 64 for a single verify pass).
   `S` is mutable, so the batch length can be varied per step without touching the block body — but the
   buffers must be sized from the original length *before* the loop, and the ids/hidden/head lengths must
   come from a captured full-length value, never from the varying `S`.
3. Run the head **per step**, over row k only, to get `d_k` so row k+1 can be built. This is the part
   that forces a restructure: the head is currently a large inline block wired to run once over S rows,
   so it has to become callable (a function) before a chain can use it. That refactor is also what the
   main path wants.
4. Stage the block's weights once for the whole chain (they do not change between steps).

Still true from the earlier version: the chain's attention extent is the one correctness question the
teacher-forced runs cannot validate — though §7 below now identifies what it actually consists of.


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

## 5. Instrumentation: CORRECTED — the reading half exists, the writing half does not

An earlier version of this section claimed the offline chain needed no new code. That is **wrong** on the
dump side and the error is worth keeping here:

- **What exists:** `VYB_MTP_HIDDEN` takes the draft rows from a file and `VYB_PROMPT_IDS` takes the ids,
  so a *run* can be fed chained input with no engine change.
- **What does not exist:** the driver's `io` import is text-only (`open_read`, `read_at`, `read_all`,
  `open_write`, `write_str`, `close`) and a tree-wide search finds no binary or stage-dump helper
  (`write_f64` / `dump_stage` / `save_stage`, or any `.f64` writer). The W1 "ten stages" were **printed**,
  not dumped to binary. So producing the h′ rows requires building a writer first.
- **The trap:** the obvious workaround — print the f64 rows and convert in Python — depends on
  `to_string`'s float precision, which is unverified. A silently lossy dump yields a confidently wrong
  acceptance curve, which is the worst outcome for a measurement whose only purpose is to decide whether
  to build something. Do not use it without checking that precision first.

## 6. Order of work

1. Offline chain (needs the writer from §5) — measure acceptance vs N against both fixtures.
2. Serialised chain in the engine (`eng_mtp()`), staging weights once, with the chain's attention extent
   as the one genuinely new correctness question.
3. Set P4.13's bar from §4's numbers, replacing the exact-match criterion.

## 7. Does the chain need its own KV cache? — no, corrected

An earlier version of this section concluded the chain requires persistent per-layer K/V for blk.NTL plus
the kernel path that reads it. That conclusion was **wrong**, and the correction is the useful part:

- The claim was that step k must attend over rows computed in *earlier launches*, which a cache would
  supply.
- In fact the rows can simply be **accumulated in `XA` and the batch re-run at each step** with length
  `k+1`. Causal attention means rows `0..k-1` recompute identically from their own unchanged inputs, so
  row k's output is correct, and no cross-launch state is needed at all. The engine already runs
  full-sequence forwards and `S` is mutable, so this needs no new kernel and no new buffer.
- The cost is O(N²) block-forwards: 10 for N = 4, against 64 for a single verification pass — negligible,
  and dwarfed by the per-step head walk either way.

What remains genuinely hard is §3.3: the head must become callable so it can run per step. That, not a KV
cache, is the real work item.

Consequences for the plan:

- The "attention extent" risk flagged in §3 reduces to "the batch must accumulate rows rather than rebuild
  row 0", which is a property of the loop, not a new subsystem. It is still the one thing the
  teacher-forced runs cannot validate, since each of their steps consumes the oracle's hidden and never
  chains.
- The dump detour stays the wrong first move: the head refactor is on the production route; the dump is not.
- The single-step mode is the only validated evidence for blk.64 and must stay untouched: any chain work
  goes behind a new knob, default off.

