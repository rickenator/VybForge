# MTP draft path — measured baseline and the chained-draft harvest

Status: the chain is IMPLEMENTED and MEASURED (§9, W4b), and P4.13's acceptance criterion is now a
rule-aware floor set from that measurement (§10) — the design and cost model below are the reasoning it
was built from, kept because the predictions are worth checking against the result.

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

**Superseded by §9 (W4b):** the chain was built IN THE ENGINE, so no h′-row writer was ever needed — the
h chain never leaves the device (each step reads the previous step's output row by pointer), and the only
file the chain writes is the drafted ids, as text, for the verifier. The analysis below stands for the
offline route, which was not taken.

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

## 6. Order of work — all three DONE

1. ~~Offline chain (needs the writer from §5) — measure acceptance vs N against both fixtures.~~ DONE
   differently and better: the chain is in the engine (§9), so the offline writer was never needed.
2. ~~Serialised chain in the engine (`eng_mtp()`), staging weights once, with the chain's attention extent
   as the one genuinely new correctness question.~~ DONE as `VYB_MTP_CHAIN` (§9). The attention extent is
   no longer a question: the batch IS the accumulated rows, re-run at length k+1, and the mechanism's
   tooth (§9) shows it reproduces the teacher-forced picks bit-for-bit.
3. ~~Set P4.13's bar from §4's numbers, replacing the exact-match criterion.~~ DONE as a rule-aware floor
   (§10).

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

## 8. The head extraction — exact spec (from the code, ready to execute)

Read from the driver's untied path (head block, ~1766-1824). The extraction is mechanical if done from this
spec; it was **not** attempted yet, deliberately, rather than left half-written.

**Why it is required:** the chain needs the head's argmax once per step (the drafted token builds the next
step's input), and the head today is a large inline block wired to run once over `S` rows. There is no way
to call it per step without extracting it.

**Function to extract** (`head_untied`), mirroring the inline walk exactly:

```
head_untied(hid, rowStart, nrows, bv, bo, out_off, VOCAB, D, CH,
            MODEL, PF, DPK, DCH, DCL, q6fn, lsfn, accfn, GM, A4) -> Int   # 0 ok, else an error code
```

Invariants to preserve, all load-bearing — each one a bug already paid for:

- **Seed `bv`/`bo` per row with `-1.0e308` / 0 before any argmax.** `argmax_acc` does NOT seed its running
  best; unseeded, the buffer's zeros win and every logit below zero is ignored.
- **Row offset in bytes:** `out_off + vs0 * nblk2 * 210` with `nblk2 = D / 256` — the row-size factor is
  not optional (omitting it read 5120x too few bytes and produced garbage logits while the hidden was
  already cos 0.9996).
- **Dequant orientation `inz = 0`** (row-major `[rows, D]`), matching `logits_slice`'s `Emb[vs*D + d]`.
  `inz = D` transposes and multiplies wrong pairs.
- **`logits_slice` param block (`GM`, 64 B):** `+0 hid, +8 chunk table, +16 logits out, +24 row index
  (position within `hid`), +32 D, +40 Vstart=0, +48 Vend=rows, +56 V=rows` — chunk-local, so its write
  lands in the chunk's own logits row.
- **`argmax_acc` param block (`A4`, 48 B):** `+0 logits, +8 best values, +16 best indices, +24 rows to
  score, +32 stride=rows, +40 base=vs0`; grid `(rows + 255) / 256`.
- **`cuCtxSynchronize()` after the walk**, before anything reads the winners.

**Per-step caller duty (the chain's step k):** normalise row k of the block's output (`XI + k*D*8`) into
`hid` with the head's own `nextn.shared_head_norm` (the `rfn` launch, 1 row), then call
`head_untied(hid=DH, rowStart=0, nrows=1, ...)` and take the single winner as `d_k`.

**Verification of the refactor (this is what makes it safe):** the main path must produce unchanged
results — after extracting, P4.13 must still read 4/4 capital and 17/19 counting, and P4.12's hidden cos
0.999609 unchanged. If either moves, the extraction changed behaviour and is wrong.

## 9. W4b — the chained draft: implemented and measured

Status: IMPLEMENTED and MEASURED. Knob `VYB_MTP_CHAIN=<N>`, default off; the teacher-forced path is
untouched and the chain path returns before the teacher-forced head walk, so a chain run never writes
`prefill_top1_vyb.txt` / `prefill_hidden_vyb.txt` (it writes `native/out/ridge_mtp_chain.txt`).

**Shape (driver `native/host/model_driver.vyb`).** The block's own `for (L in LST..LEP-1)` loop is
wrapped in an outer per-step loop. Step k: build row k into `XA` from the previous step's OUTPUT ROW
(passed as a device pointer — the driver has no device->device copy and needs none, because the row is
read at build time, before the block overwrites that buffer) and from the previous step's own draft;
set the batch length to k+1; re-seed `XI = XA; XO = XB` (the block swaps them); run the block; normalise
row k with `nextn.shared_head_norm`; score it with `head_untied(DH, 0, 1, ...)` and take `DAO[0]`.
Rows accumulate in `XA` and the batch re-runs, so earlier rows' K/V is rebuilt from unchanged inputs —
§7's "no KV cache needed" holds, and it is measured, not argued: see the tooth below. RoPE needs no
per-step knob: `rope_nrot`'s angle is `(POS + row index)`, so row k sits at exactly the position the
teacher-forced batch gives that row.

**The mechanism's instrument (`VYB_MTP_CHAIN_TF=1`).** The chain's per-step batching fed the
teacher-forced inputs must reproduce `prefill_top1_vyb.txt` BIT-FOR-BIT. It does: 19 of 19 steps on the
counting fixture, and 17/19 acceptance with the same two misses at steps 0 and 1. So the accumulation,
the per-row rope position and the per-row head scoring are right, and a divergence in a real chain
belongs to the draft head rather than to the loop. Without this, "the head cannot chain" and "the chain
loop is wrong" would look identical.

**Acceptance vs N** (one chain from step 0 — the literal measurement asked for):

| fixture | N=1 | N=2 | N=3 | N=4 | N=19 |
| --- | --- | --- | --- | --- | --- |
| capital (4 steps) | 1/1 | 2/2 | 2/3 | 2/4 | — |
| counting (19 steps) | 0/1 | 0/2 | 0/3 | 0/4 | 0/19 |

That number is nearly uninformative, and the reason is structural rather than a defect: a chain consumes
its own draft, so the FIRST disagreement poisons every later step. On the counting fixture step 0 is
already a disagreement — our top1 there is not in the oracle's top-2 and the margin is 0.066, and the
teacher-forced mode misses it too — so every longer N reads 0. Acceptance-vs-N answers "was the whole
run right", never a per-step rate, and no head with a per-step accuracy below 1 can score above zero
there.

**Conditional survival** — the number §2's cost model actually needs — comes from
`VYB_MTP_CHAIN_START=<k0>`: rows before k0 are truth-fed (that is what a verified prefix amounts to
here), so depth 0 is the head's first draft from a fresh verify boundary and depth 1 onwards are the
chained ones. One run per anchor position; depth j = P(hit at depth j | alive through depth j-1).

counting, 19 anchors:

```
depth 0   17/19 (89.5%)   <- equals the teacher-forced rate exactly: the instrument reproduces P4.13
depth 1   15/16 (93.8%)
depth 2   13/14 (92.9%)
depth 3    9/12 (75.0%)
depth 4    5/8  (62.5%)
depth 5    4/4      depth 6  3/4      depth 7  2/3      depth 8  0/1
mean survival 3.58 drafts, expected 4.28 accepted drafts per verify
```

capital, 4 anchors: depth 0 4/4, depth 1 1/3, depth 2 0/1; mean survival 1.25, expected 1.33.

So the head chains for several tokens on the counting prompt (up to 8 in a row at two anchors) and only
about one on the capital prompt: there, one chained step survives 1 time in 3 (depth 1) against 15 of 16
on the counting prompt, and every anchor still gets its own first draft right (depth 0 4/4). The sample
is 4 anchors, so that gap is indicative rather than a rate — the counting prompt's periodicity is also
why its chained drafts recover after a miss. Against §2: ~4.3 accepted drafts per verify on the counting
prompt means ~4.3x fewer verification passes than no speculation, at ~2.3 GB of draft traffic per drafted
token (~2-3% of computing it) — and this is the first measurement of the chain against this head. Note
what it does NOT say: the per-step rate after depth 4 rests on 8 then 4, 3, 2, 1 anchors, so the tail of
the curve is indicative only.

Gate: `native/legit/run_ridge_mtp_chain_gate.sh`. Its PASS/FAIL criterion is the TF tooth (bit-for-bit
agreement with the teacher-forced picks); the chain's acceptance numbers are REPORTED, not judged — a bar
belongs on P4.13's criterion, and §10 sets it from the depth-0 rate measured here.

### h sensitivity (measured while choosing §10's tooth)

The floor's tooth has to be a wrong input the head actually feels, so the head's dependence on the hidden
row was probed directly on the counting fixture (19 steps; the correct input scores 17/19):

| hidden given to the head | agreement |
| --- | --- |
| the fixture's own rows (correct) | 17/19 |
| the same rows shifted FORWARD by one | 19/19 |
| the same rows shifted BACKWARDS by one | 10/19 |
| zeroed | 1/19 |

So h is load-bearing (zeroing destroys the head) but not an exact-position lookup (a one-row shift
survives it). The forward shift's two flips are NOT a signature of an off-by-one pairing: it does not
reproduce on the capital fixture (4/4 correct, 4/4 shifted forward), and 2 flips at near-tie positions out
of 19 is inside what luck can do. It is recorded as a hypothesis with no support, not as a finding.
§10's teeth are the ZEROED and the BACKWARD-shifted input, both of which land far below the floor.

## 10. The acceptance bar — SET, from §9's depth-0 rate

P4.13's criterion WAS exact-match against the oracle's greedy pick. That is **wrong for a draft head**: an
approximate separate head reusing the main model's `output.weight` is not a copy of the main path, so its
agreement with the oracle is below 100% by construction, and the criterion could only ever report FAIL on
a correct engine (it did, for weeks — see the old §5 of `HANDOFF-MTP-CHAIN.md`).

It is now a **rule-aware FLOOR**, pinned per fixture and DERIVED from measurement rather than chosen
(`FLOORS` in `native/tools/ridge_mtp_verify.py`):

| fixture | steps | measured | floor | slack |
| --- | --- | --- | --- | --- |
| `the_capital_of_france_is` | 4 | 4/4 (100%) | 3/4 | 1 miss |
| `1_2_3_4_5_6_7` | 19 | 17/19 (89.5%) | 15/19 | 2 misses |

The measurement behind it is §9's depth-0 survival rate — the same quantity by a different route, and it
reproduces the teacher-forced number exactly, which is what makes it usable as a bar. The rule per step is
unchanged (equality where the oracle's top-2 margin clears `margin_bar`, membership of {top1,top2} below
it); only the verdict relaxed, and every disagreement is still printed with its position, its pick, the
oracle's pick and its margin.

What the slack is calibrated against, both measured:

- **It must absorb a legitimately re-picked near-tie.** The counting fixture's pos-2 pick sits at a
  0.066-nat margin, inside the ~0.02-0.05-nat spread the oracle shows against itself (CPU vs GPU capture
  of the same GGUF — the same spread recorded in `HANDOFF-MTP-CHAIN.md`). The floor relaxes the verdict,
  never the evidence.
- **It must not absorb a real defect, and it does not.** Both teeth are run by
  `native/legit/run_ridge_mtp_gate.sh` BEFORE it judges anything, and both must land below the floor: the
  hidden ZEROED gives **1/19**, the hidden rows shifted BACKWARDS by one give **10/19**. That is the
  difference between a gate and a printout.

Consequences, recorded rather than assumed:

- The oracle still cannot supply a bar (§6: its qwen35 MTP graph aborts on this GGUF), so this one is
  ours, and the derivation above is its evidence.
- The floor is tied to the fixture's step count: if a fixture changes shape the gate FAILS with
  "re-derive the floor" instead of comparing against a stale bar.
- Four steps cannot carry a rate bar — one miss of slack is all a 4-step fixture can express — so the
  19-step counting fixture is the one that does, and the capital fixture is a correlated second sample.
- The floors are calibrated on a DEFAULT-OFF engine state (no chain, `VYB_MTP_POS` inert): any change to
  what the draft head consumes re-derives them, it does not inherit them.


