# HANDOFF — MTP chained draft (the harvest's remaining piece)

Repo: `~/Projects/VybForge`, branch `main`, tip `a95a989`, working tree CLEAN. Everything below was
verified this session; nothing here is a plan that was never tested unless it says so explicitly.

Read this with `doc/QWEN35-MTP-HARVEST.md` (the harvest's mechanism, cost model and the head-extraction
spec) and `HANDOFF-PHASE4.md` (the phase context: W1-W3 landed, W4 in flight, W5 blocked on it).

---

## 1. The one task

Implement the **chained draft** in the driver's MTP mode, behind a new default-off knob
`VYB_MTP_CHAIN=<N>`, then measure **acceptance vs N** against the two fixtures.

Today's MTP mode is teacher-forced: it pre-builds all S rows of the block input (each row = the ORACLE's
hidden row + the prompt's own next token) and runs the block once. A chain is different: step k's input is
the PREVIOUS STEP'S OWN OUTPUT, so rows must be built and run one at a time.

Per step k (k = 0 .. N-1):

1. `mtp_build_row(XA + k*D*8, hrow, tok, ...)` — the row's h is `MHID` (which holds the fixture's row 0 at
   k = 0, then our own previous output), and `tok` is the next token (the fixture's `prompt_ids[1]` at
   k = 0, then OUR OWN previous draft).
2. Run the block over rows `0..k`: set the sequence length to `k+1` before the L loop.
3. The DOLM block then normalises (with `nextn.shared_head_norm`, `MSN`) and calls
   `head_untied(DH, 0, S, DBV, DAO, ...)` — with `S = k+1` this scores rows 0..k, and the existing
   `DBV -> AO` copy means **`AO[k]` is step k's draft**. No rowStart juggling is needed.
4. Read `AO[k]` (a scalar `cuMemcpyDtoH_v2`, the `freedom`-wrapped pattern already in the file), print it
   as `MTP_CHAIN_DRAFT k=<k> tok=<id>`, and make it the next step's `tok`.
5. Copy row k's block output back so it becomes the next step's `h`. See §3 — this is the one part with no
   ready-made mechanism.

Then restore the length and skip the final multi-row head walk (each step already scored its row).

---

## 2. What is already landed and verified (do not redo)

| commit | what | verified by |
| --- | --- | --- |
| `2c164ab` | `head_untied()` extracted | compiles; now exercised (below) |
| `415d6e3` | `mtp_build_row()` extracted and USED by the teacher-forced path | gate: 4/4, 17/19 |
| `a95a989` | DOLM unified onto `head_untied()` — the main path calls it | gate: 4/4, 17/19 |
| `a9d523e` | `native/tools/ridge_mtp_oracle.cpp` + the llama.cpp MTP finding | see §6 |

Both helpers live at top level, before `stage_one`:

```
head_untied(hid, rowStart, nrows, bv, bo, out_off, VOCAB, D, CH,
            MODEL, PF, DPK, chunk, lg, q6fn, lsfn, accfn, GM, A4) <Int>
mtp_build_row(dst, hrow, tok, D, MODEL, te_off, te_ty, MEH, MEN, MHN, MC2, MTMP,
              DPK, q6fn, rfn, gfn, GP1, GP2, EPS) <Int>
```

Their header comments list the invariants each one paid for (caller-seeding of `bv`/`bo`, the `nblk=D/256`
row offset, `inz=0` dequant orientation, the chunk-local `logits_slice` framing, the concat order). Do not
"simplify" any of them.

**Vyb rule that bit twice:** foreign (CUDA) calls are only allowed inside a `freedom { }` block in a
function body. `main` is exempt, which is why 2,000 lines of inline CUDA never showed it. Declare locals
outside the block, assign inside, read after.

---

## 3. The hard constraints (all established from the code, not assumed)

- **No device-to-device copy exists.** The driver's CUDA surface is `cuMemcpyHtoD_v2`,
  `cuMemcpyHtoDAsync_v2`, `cuMemcpyDtoH_v2` only. Any device->device move must round-trip through host
  memory. The established workaround for a bulk move is per-element scalar copies (the top1 dump already
  loops `S*D` scalars), which is workable but adds a copy layer.
- **`XI` aliases `XA`** (`XI<Int> = XA`): the block writes its output back into the buffer that holds the
  input rows, so a chain would clobber the earlier rows' inputs — which step k+1's batch needs for its
  K/V. The chain therefore needs its own preserved input rows (§1.5).
- **Per-step rope position.** The attention branch reads the position base from `ROP + 72`; the MTP
  branch was fixed to use `MPOS` (`if (MTPM == 1) { ... }`). For the chain, set `MPOS = k` before each
  step so row k sits at its own position. (RoPE is relative, but the K/V layout is not.)
- **Structural anchors NOT yet located:** the end of `for (L in LST..LEP - 1) { ... }` and the extent of
  the `if (DOLM == 1) { ... }` block. The wrap needs both insertion points, so locate them first — the
  chain loop header goes just before the L loop and its closing brace after the DOLM block.

---

## 4. How to run things

```
cd ~/Projects/VybForge
. ./vybenv.sh                      # sets $VYB and VYB_STDLIB

# semantic check (must print: Semantic analysis completed successfully)
$VYB native/host/model_driver.vyb --semantic-only \
     --module-path native/config --module-path native/json --module-path native/tensor \
     --module-path native/dtype --module-path native/llm

# the MTP gate (capital 4/4 + counting 17/19 is the expected, correct result)
bash native/legit/run_ridge_mtp_gate.sh
```

A driver run needs (see the existing gate scripts for the full env):
`VYB_MODEL=~/Models/qwen38-27b-ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf`,
`VYB_TSV=native/out/ridge_tensors.tsv`, `VYB_INVFREQ=native/out/ridge_invfreq.bin`,
`VYB_PROMPT_IDS="..."`, `VYB_MTP=1`, `VYB_MTP_HIDDEN=<fixture>.hidden.f64`.

**The gate prints `FAIL` and that is expected** — its criterion is exact-match, which is wrong for a draft
head (§5). Read the per-step numbers, not the verdict. The gate takes ~8 minutes: run it in the background
with notification rather than waiting in the foreground.

Verification rule for any change to this path: **P4.13 must still read 4/4 (capital) and 17/19 (counting)**
with the same per-step picks, and P4.12's hidden cos must stay 0.999609. The chain is default-off, so it
cannot affect those numbers; if they move, the change broke the main path.

---

## 5. The bar is still unset (open decision)

P4.13's criterion is exact-match against the oracle's greedy pick. That is **wrong for a draft head**: a
draft head is an approximate separate head, not a copy of the main path, so its agreement rate is below
100% by construction. Measured: 4/4 on the 4-step capital fixture, 17/19 (89.5%) on the 19-step counting
one, with the two misses at step 0 -> pos 2 (margin 0.066, `membership`) and step 1 -> pos 3 (margin 1.342,
`exact`).

Evidence gathered toward setting a real bar:

- The oracle's own acceptance is **not obtainable**, for two independent reasons: no CLI surface for
  `speculative.types` in this build, and llama.cpp's qwen35 MTP path **aborts** on this GGUF (§6).
- The oracle's cross-implementation spread (CPU vs GPU, same GGUF, same prompt) is ~0.02-0.05 nats, so a
  ~1.3-nat disagreement (step 1) is far outside numeric slack and is a genuine draft disagreement — while
  a position whose top-2 margin is 0.066 is inside that slack.

Items deliberately NOT done: converting P4.13 to acceptance and marking it PASS (the evidence shows the
criterion is wrong but does not establish the threshold), and the chained-draft measurement that would be
the strongest input to that threshold.

---

## 6. llama.cpp's MTP is non-functional for this GGUF (do not plan around it)

`native/tools/ridge_mtp_oracle.cpp` (build command in its header) is the harness that established this. It
builds and loads correctly (`n_embd=5120`, `n_vocab=248320`, fixture parsed), then aborts:

```
qwen35.cpp:502: GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj") failed
create_tensor: loading tensor blk.64.nextn.eh_proj.weight
model has unused tensor blk.64.nextn.eh_proj.weight ... -- ignoring
   (same for .enorm, .hnorm, .shared_head_norm)
```

This build's qwen35 arch drops the nextn tensors it then requires, so `graph_mtp` asserts during
`llama_init_from_model`. The field exists, the server wires `draft-mtp` up, the graph aborts. Consequently
the bar cannot come from llama.cpp, and **ours is the only MTP implementation of this head that runs**.

Also settled: the GGUF carries NO separate `nextn.shared_head_head` (blk.64 has only `eh_proj`, `enorm`,
`hnorm`, `shared_head_norm`), so reusing the main `output.weight` for the draft head is correct — a
candidate explanation for the 2/19 misses was retired there.

---

## 7. Smaller open items

- **Skill update (high value, minutes).** Lessons worth recording: `freedom` scoping for extracted
  functions; the two kernel-contract traps (`logits_slice` reads and writes through ONE `vs`, so a global
  index must ride on the log base; `argmax_acc` does not seed its running best); the oracle-spread
  technique for judging whether a disagreement is real; no-DtoD in the CUDA surface; that llama.cpp's MTP
  is non-functional here; and that `pkill -f` patterns can match the agent's own shell and kill it (hit
  twice this session).
- **Commit message:** `1efd17f` ("W4 impl") has a junk message; amending it needs Rick's explicit OK.
- **Stability fixture:** `native/out/oracle_gpu/1_2_3_4_5_6_7.fix` is a GPU capture of the same prompt
  (gitignored, unused). Promoting it as a recorded cross-implementation stability fixture would formalise
  the ~0.05-nat spread measured above.
- **W5** stays unflipped: flipping `eng_gdn()`/`eng_mtp()` and re-running `run_caps_gate.sh` is blocked on
  W4's status, which in turn depends on the §5 bar decision.
