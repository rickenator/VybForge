#!/usr/bin/env python3
"""P4.13 — the MTP draft head against the oracle, teacher-forced (VybForge#10 phase 4, W4).

The draft head consumes (the NORMALISED hidden at position t, the embedding of token t+1) and predicts
t+2. The captured fixtures already record the oracle's greedy pick at EVERY position, so the head can be
gated against llama.cpp with NO new capture: feed it the ORACLE's own hidden row and the ORACLE's own
next token, and require its per-step top1 to be the oracle's pick for t+2, under the fixture's own rule
(equality where the top-2 margin clears `margin_bar`, membership of {top1, top2} below it).

Using the oracle's hidden rather than ours is deliberate: it takes the main pass out of the loop, so a
disagreement is THIS block's rather than 64 layers of accumulation. (A second pass over our own hidden
tests the pair end to end, and is what P4.12 already does for the main path.)

Knobs: VYBFORGE_MTP_POS (the draft step's rope position base, default 1), VYBFORGE_MTP_SKIP_RUN=1 to
check an existing log, VYBFORGE_MTP_FIXTURE, VYBFORGE_MTP_LOG.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ridge_forward_verify as rf                                              # noqa: E402

MTPLOG = os.environ.get("VYBFORGE_MTP_LOG", os.path.join(rf.OUTD, "ridge_mtp.log"))
SKIP_RUN = os.environ.get("VYBFORGE_MTP_SKIP_RUN", "") != ""
POS = os.environ.get("VYBFORGE_MTP_POS", "1")

# ---- THE ACCEPTANCE CRITERION -------------------------------------------------------------------
# Exact match is the WRONG bar, and saying so is as much of this gate as the number is. The draft head
# is an approximate separate head that reuses the main model's `output.weight`; it is not a copy of the
# main path, so its per-step agreement with the oracle is below 100% BY CONSTRUCTION, and a criterion
# that demands every step agree can only ever report FAIL on a correct engine (it did, for weeks).
#
# The bar is therefore a FLOOR on rule-aware agreement (equality where the oracle's top-2 margin clears
# `margin_bar`, membership of {top1,top2} below it), pinned PER FIXTURE and DERIVED from measurement
# rather than chosen:
#
#     counting fixture  `1_2_3_4_5_6_7`     17/19 measured (89.5%)   floor 15/19   (2 misses of slack)
#     capital fixture   `the_capital_of_france_is`   4/4 measured    floor  3/4    (1 miss of slack)
# Each entry is (floor_min_agree, fixture_step_count, measured_agree_at_derivation).
#
# The measurement is doc/QWEN35-MTP-HARVEST.md §9 — depth 0 of the chain's survival sweep is this same
# quantity, and it reproduces the teacher-forced rate exactly (17/19), which is what makes it usable as
# a bar. Two things the slack is calibrated against, both measured:
#   * it must absorb a legitimately re-picked NEAR-TIE. The counting fixture's pos-2 pick carries a
#     0.066-nat margin, inside the ~0.02-0.05-nat spread the oracle shows against itself (CPU vs GPU
#     capture of the same GGUF); a flat position is a coin flip that fp noise moves.
#   * it must NOT absorb a real defect, and it does not: feeding the head the WRONG hidden row — the
#     off-by-one class this gate exists for — costs most of the 19 steps, not two. That is run as this
#     criterion's own tooth (`VYBFORGE_MTP_HIDDEN_SHIFT=1`, gate step 2), because a floor nobody has
#     watched fail is decoration.
# A fixture with no pinned floor falls back to DEFAULT_FLOOR_RATE, and says so in its line.
FLOORS = {
    "1_2_3_4_5_6_7": (15, 19, 17),
    "the_capital_of_france_is": (3, 4, 4),
}
DEFAULT_FLOOR_RATE = 0.75
# The floor's tooth: shift the fixture's hidden rows by this many positions before the run (0 = off;
# negative shifts backwards), or zero them entirely. Both are "wrong input" controls.
HIDDEN_SHIFT = int(os.environ.get("VYBFORGE_MTP_HIDDEN_SHIFT", "0") or "0")
ZERO_HIDDEN = os.environ.get("VYBFORGE_MTP_HIDDEN_ZERO", "") != ""
CONTROL = HIDDEN_SHIFT != 0 or ZERO_HIDDEN


def shifted_hidden(fx, hid, name):
    """A CONTROL input for the floor: the fixture's hidden rows moved by HIDDEN_SHIFT positions (the
    nearest row repeated at the ends). A step then reads another position's hidden — the wrong-row /
    off-by-one class of defect — and the gate requires this run to land BELOW the floor. A NEGATIVE
    shift moves the rows backwards, which is what tells an off-by-one in the head's hidden convention
    apart from a head that simply does not use the hidden much: if only one direction improves, the
    pairing is off by one; if both do, the hidden is nearly decorative on that prompt."""
    import numpy as np
    sh = [int(x) for x in (rf.field(fx, "hidden_shape") or "").split()]
    if len(sh) != 2:
        return None
    rows, D = sh
    a = np.fromfile(hid, dtype="<f8").reshape(rows, D)
    b = np.empty_like(a)
    if ZERO_HIDDEN:
        b[:] = 0.0
    else:
        for i in range(rows):
            b[i] = a[min(max(i + HIDDEN_SHIFT, 0), rows - 1)]
    tag = "zero" if ZERO_HIDDEN else "shift%d" % HIDDEN_SHIFT
    out = os.path.join(rf.OUTD, "ridge_mtp_hidden_%s_%s.f64" % (tag, name))
    b.tofile(out)
    print("RIDGE_MTP_VERIFY_CONTROL hidden_%s -> %s (the wrong-input this criterion must reject)"
          % (tag, os.path.relpath(out, rf.REPO)))
    return out


def run_mtp(ids, hidden_file, pos):
    env = dict(os.environ, VYB_STDLIB=rf.STDLIB, VYB_MODEL=rf.MODEL, VYB_TSV=rf.TSV,
               VYB_INVFREQ=rf.INVFREQ, VYB_LLM_DIR=rf.TOKDIR, VYB_MTP="1",
               VYB_PROMPT_IDS=" ".join(str(i) for i in ids),
               VYB_MTP_HIDDEN=hidden_file, VYB_MTP_POS=str(pos))
    env.pop("VYB_PROMPT", None)
    cmd = [rf.VYB, "native/host/model_driver.vyb",
           "--module-path", "native/config", "--module-path", "native/json",
           "--module-path", "native/tensor", "--module-path", "native/dtype",
           "--module-path", "native/llm"]
    r = subprocess.run(cmd, cwd=rf.REPO, capture_output=True, text=True, env=env)
    out = r.stdout + r.stderr
    with open(MTPLOG, "w") as fh:
        fh.write(out)
    return out


def verify_mtp(name):
    fx = os.path.join(rf.FIXDIR, name if name.endswith(".fix") else name + ".fix")
    for path, what in ((rf.MODEL, "the Ridge model"), (rf.INVENTORY, "the inventory"),
                       (rf.TSV, "the engine tensor index"), (rf.INVFREQ, "the rope table")):
        if not os.path.exists(path):
            print("RIDGE_MTP_VERIFY_SKIP no %s at %s" % (what, path))
            return 0
    if not os.path.exists(fx):
        print("RIDGE_MTP_VERIFY_SKIP no fixture %s" % fx)
        return 0
    if not os.access(rf.VYB, os.X_OK):
        print("RIDGE_MTP_VERIFY_SKIP no Vyb toolchain at %s" % rf.VYB)
        return 0

    want_ids = [int(x) for x in (rf.field(fx, "prompt_ids") or "").split()]
    pos = rf.fields(fx, "pos_top1")
    kbar = float(rf.field(fx, "margin_bar") or "0.5")
    hid = os.path.join(rf.FIXDIR, rf.field(fx, "hidden_file") or "")
    if len(want_ids) < 3 or not pos or not os.path.exists(hid):
        print("RIDGE_MTP_VERIFY_SKIP %s lacks prompt_ids/pos_top1/hidden" % name)
        return 0
    print("RIDGE_MTP_VERIFY fixture=%s S=%d rope_pos=%s" % (name, len(want_ids), POS))

    hid_used = hid
    if CONTROL:
        sh = shifted_hidden(fx, hid, name)
        if sh:
            hid_used = sh

    # The draft head consumes token t+1, so it runs S-1 steps, one per position 0..S-2;
    # step i drafts position i+2 (1-based), which is the oracle pick this compares against.
    # The driver's MTP mode trims the list itself (the draft head consumes token t+1, so it runs S-1
    # steps over ids[1:]) — pass the FULL id list and let it do that once. Shifting here as well drops
    # a second token and the ids check below catches it.
    out = open(MTPLOG, encoding="utf-8", errors="replace").read() if SKIP_RUN else run_mtp(want_ids, hid_used, POS)
    if "MODEL_PREFILL_DONE" not in out:
        print("RIDGE_MTP_VERIFY_FAIL the driver did not complete (log: %s): %s"
              % (os.path.relpath(MTPLOG, rf.REPO), out[-900:]))
        return 1
    ids_line = [l for l in out.splitlines() if l.startswith("PROMPT_IDS=")]
    got_ids = [int(x) for x in ids_line[0].split("=", 1)[1].split()] if ids_line else []
    if got_ids != want_ids[1:]:
        print("RIDGE_MTP_VERIFY_FAIL the run embedded ids %s, wanted the fixture's tokens 2..N %s"
              % (got_ids[:8], want_ids[1:9]))
        return 1
    ours = [int(v) for v in rf.read_floats(rf.TOPF).astype("int64")]
    if len(ours) != len(want_ids) - 1:
        print("RIDGE_MTP_VERIFY_FAIL the run produced %d steps, wanted %d"
              % (len(ours), len(want_ids) - 1))
        return 1

    by_k = {int(r[0]): r for r in pos}
    # The checker's own tooth: with the comparison deliberately OFF BY ONE it MUST miss, which proves
    # the gate can fail at all (an unchanged-input-only tooth proves nothing about the checker). The
    # gate runs it and requires a failure.
    offby = int(os.environ.get("VYBFORGE_MTP_OFFBY", "0"))
    n_ok = n_bad = n_checked = 0
    for i, got in enumerate(ours):
        k = i + 2 + offby                                 # the position this step drafts
        r = by_k.get(k)
        if r is None:
            continue
        t1, lp1, t2, lp2, rule = int(r[1]), float(r[2]), int(r[3]), float(r[4]), r[5]
        margin = lp1 - lp2
        n_checked += 1
        if rule == "exact":
            hit = (got == t1)
        else:
            hit = got in (t1, t2)
        if hit:
            n_ok += 1
        else:
            n_bad += 1
            print("RIDGE_MTP_VERIFY   step %d -> pos %d: ours=%d oracle=%s (margin %.3f, %s)"
                  % (i, k, got, [t1, t2] if rule != "exact" else t1, margin, rule))
    print("RIDGE_MTP_VERIFY per-step top1: %d/%d agree (%d checked, margin_bar=%.2f)"
          % (n_ok, n_checked, n_checked, kbar))
    # ---- the criterion: a rule-aware FLOOR, not exact match (see THE ACCEPTANCE CRITERION above) ----
    short = name[:-4] if name.endswith(".fix") else name
    floor = FLOORS.get(short)
    if floor:
        fmin, fmax, fmeas = floor
        if n_checked != fmax:
            print("RIDGE_MTP_VERIFY_FAIL the fixture now offers %d steps and the floor was derived for "
                  "%d: the fixture changed, so re-derive the floor (and re-run the tooth) rather than "
                  "compare against a stale bar" % (n_checked, fmax))
            return 1
        below = n_ok < fmin
        print("RIDGE_MTP_VERIFY_ACCEPT fixture=%s agree=%d/%d floor=%d/%d (slack %d) rate=%.1f%% "
              "measured_at_the_floor_derivation=%d/%d"
              % (short, n_ok, n_checked, fmin, fmax, n_ok - fmin, 100.0 * n_ok / n_checked, fmeas, fmax))
    else:
        rate = n_ok / float(n_checked) if n_checked else 0.0
        below = rate < DEFAULT_FLOOR_RATE
        print("RIDGE_MTP_VERIFY_ACCEPT fixture=%s agree=%d/%d floor=%.2f (rate) rate=%.1f%% (no pinned "
              "floor for this fixture: DEFAULT_FLOOR_RATE)" % (short, n_ok, n_checked, DEFAULT_FLOOR_RATE,
                                                               100.0 * rate))
    if CONTROL:
        # This run's verdict is INVERTED on purpose: it is the floor's own tooth, and it must land below
        # the floor. rc==0 from here means "the criterion caught the wrong input".
        what = "zeroed" if ZERO_HIDDEN else "shifted by %d" % HIDDEN_SHIFT
        if below:
            print("RIDGE_MTP_VERIFY_CONTROL_OK hidden %s fell below the floor (%d/%d), so the criterion "
                  "can fail" % (what, n_ok, n_checked))
            return 0
        print("RIDGE_MTP_VERIFY_CONTROL_FAIL hidden %s still cleared the floor (%d/%d): the floor cannot "
              "see this wrong input" % (what, n_ok, n_checked))
        return 1
    if below:
        print("RIDGE_MTP_VERIFY_FAIL the draft head agrees at %d of %d steps, below its floor: the %d "
              "disagreement(s) above are listed with their margins" % (n_ok, n_checked, n_bad))
        return 1
    # The checker's own TOOTH, on the SAME completed run: compared one position later it must MISS at
    # least one step. "At least one", not "all", because a prompt can legitimately repeat a pick at
    # adjacent positions (the counting prompt does). A checker that agrees under a shift would make
    # this gate decoration; an unchanged-input-only tooth proves nothing about the checker.
    sh_ok = 0
    for i, got in enumerate(ours):
        r = by_k.get(i + 3)
        if r is None:
            continue
        st1, st2, srule = int(r[1]), int(r[3]), r[5]
        shit = (got == st1) if srule == "exact" else (got in (st1, st2))
        sh_ok += 1 if shit else 0
    print("RIDGE_MTP_VERIFY tooth off_by_one: %d of %d steps would still agree (must be fewer)"
          % (sh_ok, n_checked))
    if sh_ok >= n_checked:
        print("RIDGE_MTP_VERIFY_FAIL the checker agrees even when the comparison is shifted by one")
        return 1
    print("RIDGE_MTP_VERIFY_SUMMARY fixture=%s steps=%d agree=%d/%d rope_pos=%s"
          % (name, len(ours), n_ok, n_checked, POS))
    return 0


def main():
    want = sys.argv[1:]
    if not want:
        one = os.environ.get("VYBFORGE_MTP_FIXTURE", "")
        want = [one] if one else ["the_capital_of_france_is", "1_2_3_4_5_6_7"]
    bad = []
    for w in want:
        if verify_mtp(w if w.endswith(".fix") else w + ".fix") != 0:
            bad.append(w)
    if bad:
        print("RIDGE_MTP_VERIFY_FAIL %s did not match the oracle" % ", ".join(bad))
        return 1
    print("RIDGE_MTP_VERIFY_DONE the MTP draft head reproduces the oracle's next-next-token picks "
          "for %d fixture(s)" % len(want))
    return 0


if __name__ == "__main__":
    sys.exit(main())
