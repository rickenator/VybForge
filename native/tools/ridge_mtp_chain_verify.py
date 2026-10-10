#!/usr/bin/env python3
"""W4b — the CHAINED draft head against the oracle: acceptance vs N (VybForge#10 phase 4).

The teacher-forced mode (P4.13, ridge_mtp_verify.py) feeds the draft head the ORACLE's hidden row and
the prompt's own next token, so its per-step agreement is an UPPER BOUND. A real speculative decode
cannot do that: step k's input is the previous step's OWN output and the previous step's OWN draft.
This verifier drives that — `VYB_MTP_CHAIN=<N>` runs the block once per step over an accumulating
batch — and measures, per fixture and per N, how many of the N chained drafts equal the oracle's own
greedy pick at their position, under the fixture's rule (equality where the top-2 margin clears
`margin_bar`, membership of {top1, top2} below it).

What it reports, and why:
  - `accept` per N: the raw agreement of the chained drafts.
  - `expected accepted`: sum_{k<N} prod_{j<=k} p_j with p_j = our per-step agreement at j. That is the
    number of tokens one verification pass would be expected to carry, i.e. the quantity the cost
    model in doc/QWEN35-MTP-HARVEST.md §2 amortises the 64-block verify pass over.
  - the per-step detail (ours vs oracle, margin, rule) so a disagreement can be judged: the oracle's
    own cross-implementation spread is ~0.02-0.05 nats, so a disagreement at a position whose top-2
    margin is inside that band is a coin flip, while one at a wide margin is a genuine draft error.

Knobs: VYBFORGE_MTP_CHAIN_N (default "1 2 3 4"), VYBFORGE_MTP_CHAIN_SKIP_RUN=1 to re-check the last
log, VYBFORGE_MTP_CHAIN_LOG, VYBFORGE_MTP_FIXTURES.

Nothing here writes prefill_top1_vyb.txt / prefill_hidden_vyb.txt: the chain run returns before the
teacher-forced head walk, so P4.13 and P4.12 keep their evidence.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ridge_forward_verify as rf                                              # noqa: E402

CHAINLOG = os.environ.get("VYBFORGE_MTP_CHAIN_LOG", os.path.join(rf.OUTD, "ridge_mtp_chain.log"))
CHAINOUT = os.path.join(rf.OUTD, "ridge_mtp_chain.txt")
SKIP_RUN = os.environ.get("VYBFORGE_MTP_CHAIN_SKIP_RUN", "") != ""
NS = [int(x) for x in os.environ.get("VYBFORGE_MTP_CHAIN_N", "1 2 3 4").split()]
SWEEP = os.environ.get("VYBFORGE_MTP_CHAIN_SWEEP", "") != ""
TOOTH = os.environ.get("VYBFORGE_MTP_CHAIN_TOOTH", "") != "" or "--tooth" in sys.argv


def run_chain(ids, hidden_file, n, start=0, tf=False):
    env = dict(os.environ, VYB_STDLIB=rf.STDLIB, VYB_MODEL=rf.MODEL, VYB_TSV=rf.TSV,
               VYB_INVFREQ=rf.INVFREQ, VYB_LLM_DIR=rf.TOKDIR, VYB_MTP="1", VYB_MTP_CHAIN=str(n),
               VYB_MTP_CHAIN_START=str(start),
               VYB_PROMPT_IDS=" ".join(str(i) for i in ids),
               VYB_MTP_HIDDEN=hidden_file)
    env.pop("VYB_PROMPT", None)
    env.pop("VYB_PROMPT_RAW", None)
    if tf:
        env["VYB_MTP_CHAIN_TF"] = "1"
    else:
        env.pop("VYB_MTP_CHAIN_TF", None)
    cmd = [rf.VYB, "native/host/model_driver.vyb",
           "--module-path", "native/config", "--module-path", "native/json",
           "--module-path", "native/tensor", "--module-path", "native/dtype",
           "--module-path", "native/llm"]
    r = subprocess.run(cmd, cwd=rf.REPO, capture_output=True, text=True, env=env)
    out = r.stdout + r.stderr
    with open(CHAINLOG, "w") as fh:
        fh.write(out)
    return out


def run_teacher(ids, hidden_file):
    """The teacher-forced MTP run (no chain): the path P4.13 gates, whose per-step picks are the
    reference this file's tooth compares the chain mechanism against. Same env as run_chain, minus the
    chain knobs."""
    env = dict(os.environ, VYB_STDLIB=rf.STDLIB, VYB_MODEL=rf.MODEL, VYB_TSV=rf.TSV,
               VYB_INVFREQ=rf.INVFREQ, VYB_LLM_DIR=rf.TOKDIR, VYB_MTP="1",
               VYB_PROMPT_IDS=" ".join(str(i) for i in ids),
               VYB_MTP_HIDDEN=hidden_file)
    for k in ("VYB_PROMPT", "VYB_PROMPT_RAW", "VYB_MTP_CHAIN", "VYB_MTP_CHAIN_TF",
              "VYB_MTP_CHAIN_START"):
        env.pop(k, None)
    cmd = [rf.VYB, "native/host/model_driver.vyb",
           "--module-path", "native/config", "--module-path", "native/json",
           "--module-path", "native/tensor", "--module-path", "native/dtype",
           "--module-path", "native/llm"]
    r = subprocess.run(cmd, cwd=rf.REPO, capture_output=True, text=True, env=env)
    with open(CHAINLOG, "w") as fh:
        fh.write(r.stdout + r.stderr)
    return r.stdout + r.stderr


def tooth_fixture(name, want_ids, hid, steps_max):
    """The mechanism's own instrument: feed the chain's per-step batching the TEACHER-FORCED inputs
    (the oracle's hidden row and the prompt's own next token) and require its per-step top1 to be
    IDENTICAL to the teacher-forced mode's own picks — same rows, but accumulated in a growing batch
    and scored one row at a time instead of built up front and scored in one go.

    This is what separates "the draft head cannot chain" from "the chain loop is wrong": they look the
    same in the acceptance numbers, and only this comparison tells them apart. Bit-for-bit equality is
    the right criterion here (unlike acceptance) because both runs embed the same inputs in the same
    arithmetic; a mismatch is a defect in the loop, not a plausible drafting difference."""
    out_t = run_teacher(want_ids, hid)
    if "MODEL_PREFILL_DONE" not in out_t:
        print("RIDGE_MTP_CHAIN_TOOTH_FAIL the teacher-forced run did not complete: %s" % out_t[-600:])
        return 1
    if not os.path.exists(rf.TOPF):
        print("RIDGE_MTP_CHAIN_TOOTH_FAIL the teacher run wrote no %s" % rf.TOPF)
        return 1
    teach = [int(v) for v in rf.read_floats(rf.TOPF).astype("int64")]
    out_c = run_chain(want_ids, hid, steps_max, start=0, tf=True)
    if "MTP_CHAIN_DONE" not in out_c:
        print("RIDGE_MTP_CHAIN_TOOTH_FAIL the chain run did not complete: %s" % out_c[-600:])
        return 1
    got = chain_drafts(out_c)
    if len(got) != len(teach):
        print("RIDGE_MTP_CHAIN_TOOTH_FAIL %d chain steps vs %d teacher-forced steps"
              % (len(got), len(teach)))
        return 1
    bad = [(i, a, b) for i, (a, b) in enumerate(zip(got, teach)) if a != b]
    for i, a, b in bad:
        print("RIDGE_MTP_CHAIN_TOOTH   step %d: chain-mechanism=%d teacher-forced=%d" % (i, a, b))
    if bad:
        print("RIDGE_MTP_CHAIN_TOOTH_FAIL %s: the chain mechanism does NOT reproduce the teacher-forced "
              "picks (%d of %d differ) — the per-step batching is wrong, so the acceptance numbers "
              "below would be meaningless" % (name, len(bad), len(teach)))
        return 1
    print("RIDGE_MTP_CHAIN_TOOTH_OK %s: all %d steps identical to the teacher-forced mode"
          % (name, len(teach)))
    return 0


def chain_drafts(out):
    """Step order, one drafted id per step, as the driver printed them."""
    got = []
    for line in out.splitlines():
        if line.startswith("MTP_CHAIN_DRAFT "):
            f = dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
            got.append((int(f["k"]), int(f["tok"])))
    got.sort()
    return [t for _, t in got]


def hits_of(ours, by_k):
    """Per-step hit flags, in step order, under the fixture's own rule."""
    hits = []
    for i, got in enumerate(ours):
        r = by_k.get(i + 2)
        if r is None:
            continue
        t1, t2, rule = int(r[1]), int(r[3]), r[5]
        hits.append((got == t1) if rule == "exact" else (got in (t1, t2)))
    return hits


def sweep_fixture(name, want_ids, by_k, hid, steps_max):
    """Per-anchor survival: the chain's CONDITIONAL acceptance, which is the only state a speculative
    decode is ever in after a verify pass ("a verified prefix — how many drafts in a row does this head
    get right?"). One run per anchor position p:

      rows 0..p are built from the ORACLE's hidden and the prompt's own tokens — those are the rows a
      verify pass has already settled, so the anchor row p is the state after it (the main model's own
      hidden at p, plus the true token p+1) — and from row p+1 on the chain carries its own output and
      its own previous draft. Draft p (the anchor's own prediction) is therefore the fresh-start FIRST
      draft and draft p+1 the first genuinely chained one.

    Reports the leading-hit run per anchor, the conditional acceptance by depth (depth 0 = that first,
    fully truth-fed draft; depth 1 onwards = the chained ones), and the expected accepted drafts
    sum_j P(alive at depth j) — the standard speculation formula, and the number the harvest doc's cost
    model multiplies the verify pass by. The anchors overlap in position, so their depth-j samples are
    correlated (they are the same positions seen from different anchors); the per-anchor lines are the
    raw evidence for that reason."""
    runs = []
    for p in range(0, steps_max):
        n = steps_max                                        # every anchor gets a full runway
        out = (open(CHAINLOG, encoding="utf-8", errors="replace").read() if SKIP_RUN
               else run_chain(want_ids, hid, n, start=p + 1))
        if "MTP_CHAIN_DONE" not in out or "MODEL_PREFILL_DONE" not in out:
            print("RIDGE_MTP_CHAIN_FAIL anchor=%d the driver did not complete (log: %s): %s"
                  % (p, os.path.relpath(CHAINLOG, rf.REPO), out[-900:]))
            return 1
        ours = chain_drafts(out)
        if len(ours) != n:
            print("RIDGE_MTP_CHAIN_FAIL anchor=%d produced %d drafts, wanted %d" % (p, len(ours), n))
            return 1
        allhits = hits_of(ours, by_k)
        if len(allhits) != n:
            print("RIDGE_MTP_CHAIN_FAIL anchor=%d: %d of %d steps have a recorded oracle pick "
                  "(check that the fixture records pos_top1 through position %d)"
                  % (p, len(allhits), n, steps_max + 1))
            return 1
        hits = allhits[p:]                                   # from the anchor's own draft onward
        alive = 0
        for h in hits:
            if h:
                alive += 1
            else:
                break
        runs.append((p, hits, alive))
        print("RIDGE_MTP_CHAIN_SWEEP anchor=%-3d survival=%d  %s"
              % (p, alive, "".join("1" if h else "0" for h in hits)))

    depth = max(len(h) for _, h, _ in runs)
    print("RIDGE_MTP_CHAIN_SWEEP conditional acceptance by depth (depth 0 = the anchor's own draft, "
          "still fully truth-fed; from depth 1 the h and the token are the chain's own):")
    probs = []
    for j in range(depth):
        den = num = 0
        for _, h, _ in runs:
            if len(h) <= j or not all(h[i] for i in range(j)):
                continue
            den += 1
            num += 1 if h[j] else 0
        probs.append((num, den))
        print("RIDGE_MTP_CHAIN_SWEEP   depth %-3d %d/%d (%.1f%%)"
              % (j, num, den, 100.0 * num / den if den else 0.0))

    surv = 1.0
    exp = 0.0
    for num, den in probs:
        surv *= (num / den) if den else 0.0
        exp += surv
    mean_alive = sum(a for _, _, a in runs) / float(len(runs))
    print("RIDGE_MTP_CHAIN_SWEEP expected=%.2f accepted drafts per verify, mean survival=%.2f over "
          "%d anchors" % (exp, mean_alive, len(runs)))
    print("RIDGE_MTP_CHAIN_SWEEP_SUMMARY fixture=%s anchors=%d mean_survival=%.2f expected=%.2f"
          % (name, len(runs), mean_alive, exp))
    return 0


def verify_fixture(name):
    fx = os.path.join(rf.FIXDIR, name if name.endswith(".fix") else name + ".fix")
    for path, what in ((rf.MODEL, "the Ridge model"), (rf.INVENTORY, "the inventory"),
                       (rf.TSV, "the engine tensor index"), (rf.INVFREQ, "the rope table")):
        if not os.path.exists(path):
            print("RIDGE_MTP_CHAIN_SKIP no %s at %s" % (what, path))
            return None
    if not os.path.exists(fx):
        print("RIDGE_MTP_CHAIN_SKIP no fixture %s" % fx)
        return None
    if not os.access(rf.VYB, os.X_OK):
        print("RIDGE_MTP_CHAIN_SKIP no Vyb toolchain at %s" % rf.VYB)
        return None

    want_ids = [int(x) for x in (rf.field(fx, "prompt_ids") or "").split()]
    pos = rf.fields(fx, "pos_top1")
    kbar = float(rf.field(fx, "margin_bar") or "0.5")
    hid = os.path.join(rf.FIXDIR, rf.field(fx, "hidden_file") or "")
    if len(want_ids) < 3 or not pos or not os.path.exists(hid):
        print("RIDGE_MTP_CHAIN_SKIP %s lacks prompt_ids/pos_top1/hidden" % name)
        return None
    by_k = {int(r[0]): r for r in pos}
    steps_max = len(want_ids) - 1

    print("RIDGE_MTP_CHAIN fixture=%s S=%d margin_bar=%.2f" % (name, steps_max, kbar))
    if TOOTH:
        return tooth_fixture(name, want_ids, hid, steps_max)
    if SWEEP:
        return sweep_fixture(name, want_ids, by_k, hid, steps_max)
    rows = {}
    for n in NS:
        if n < 1:
            continue
        if n > steps_max:
            print("RIDGE_MTP_CHAIN   N=%d SKIPPED (the fixture has only %d draft steps)" % (n, steps_max))
            continue
        out = (open(CHAINLOG, encoding="utf-8", errors="replace").read()
               if SKIP_RUN else run_chain(want_ids, hid, n))
        if "MODEL_PREFILL_DONE" not in out or "MTP_CHAIN_DONE" not in out:
            print("RIDGE_MTP_CHAIN_FAIL N=%d the driver did not complete (log: %s): %s"
                  % (n, os.path.relpath(CHAINLOG, rf.REPO), out[-900:]))
            return 1
        ids_line = [l for l in out.splitlines() if l.startswith("PROMPT_IDS=")]
        got_ids = [int(x) for x in ids_line[0].split("=", 1)[1].split()] if ids_line else []
        if got_ids != want_ids[1:]:
            print("RIDGE_MTP_CHAIN_FAIL the run embedded ids %s, wanted the fixture's tokens 2..N %s"
                  % (got_ids[:8], want_ids[1:9]))
            return 1
        ours = chain_drafts(out)
        if len(ours) != n:
            print("RIDGE_MTP_CHAIN_FAIL N=%d the run produced %d drafts" % (n, len(ours)))
            return 1

        ok = 0
        detail = []
        for i, got in enumerate(ours):
            k = i + 2                                       # the position this step drafts
            r = by_k.get(k)
            if r is None:
                continue
            t1, lp1, t2, lp2, rule = int(r[1]), float(r[2]), int(r[3]), float(r[4]), r[5]
            margin = lp1 - lp2
            hit = (got == t1) if rule == "exact" else (got in (t1, t2))
            detail.append((i, k, got, t1, t2, margin, rule, hit))
            ok += 1 if hit else 0
        rows[n] = (ok, len(ours), detail, hits_of(ours, by_k))

    if not rows:
        print("RIDGE_MTP_CHAIN_FAIL no N in %s fits this fixture" % NS)
        return 1

    for n in sorted(rows):
        ok, tot, _, _ = rows[n]
        print("RIDGE_MTP_CHAIN   N=%-3d accept=%d/%d (%.1f%%)" % (n, ok, tot, 100.0 * ok / tot))

    # The chain is a PREFIX process: step k's input depends only on steps < k, so the first n steps of
    # the longest run ARE the run at N=n (byte for byte — same rows, same batch lengths). On a long
    # fixture that makes one N=max run enough for the whole acceptance-vs-N curve, and the curve below
    # is derived from the longest run for exactly that reason. Say which run it came from.
    ln = max(rows)
    hits = rows[ln][3]
    print("RIDGE_MTP_CHAIN acceptance vs N (from the N=%d run's %d steps, a prefix process):" % (ln, len(hits)))
    for n in range(1, len(hits) + 1):
        k = sum(hits[:n])
        print("RIDGE_MTP_CHAIN   N=%-3d accept=%d/%d (%.1f%%)" % (n, k, n, 100.0 * k / n))

    ok, tot, detail, _ = rows[ln]
    print("RIDGE_MTP_CHAIN per-step detail (N=%d, k = the step, pos = the position it drafts):" % ln)
    for i, k, got, t1, t2, margin, rule, hit in detail:
        print("RIDGE_MTP_CHAIN   step %2d pos %2d ours=%-7d oracle=%-18s margin %7.3f %-10s %s"
              % (i, k, got, t1 if rule == "exact" else "%d|%d" % (t1, t2), margin, rule,
                 "hit" if hit else "MISS"))

    # Expected accepted tokens per verification pass: sum_k prod_{j<=k} p_j, with p_j our measured
    # per-step agreement at step j. Per-step rates beyond the longest measured N are not measured and
    # the estimate therefore stops there — say so rather than extrapolating.
    surv = 1.0
    exp = 0.0
    for r in [1 if h else 0 for h in hits]:
        surv *= r
        exp += surv
    print("RIDGE_MTP_CHAIN expected=%.2f accepted tokens per verify at N=%d (steps %s; the estimate "
          "stops at the longest measured N, it is NOT extrapolated)"
          % (exp, len(hits), "".join("1" if h else "0" for h in hits)))
    print("RIDGE_MTP_CHAIN_SUMMARY fixture=%s N=%d accept=%d/%d" % (name, ln, ok, tot))
    return 0


def main():
    want = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not want:
        want = (os.environ.get("VYBFORGE_MTP_FIXTURES", "the_capital_of_france_is 1_2_3_4_5_6_7")).split()
    bad = []
    ran = 0
    for w in want:
        rc = verify_fixture(w[:-4] if w.endswith(".fix") else w)
        if rc is None:
            continue
        ran += 1
        if rc != 0:
            bad.append(w)
    if not ran:
        print("RIDGE_MTP_CHAIN_SKIP nothing ran (no model/index/rope table/fixture or no toolchain)")
        return 1
    if bad:
        print("RIDGE_MTP_CHAIN_FAIL %s did not complete" % ", ".join(bad))
        return 1
    print("RIDGE_MTP_CHAIN_DONE the chained draft ran and was scored for %d fixture(s)" % ran)
    return 0


if __name__ == "__main__":
    sys.exit(main())
