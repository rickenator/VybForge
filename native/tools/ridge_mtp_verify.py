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


def main():
    name = os.environ.get("VYBFORGE_MTP_FIXTURE", "the_capital_of_france_is")
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

    # The draft head consumes token t+1, so it runs S-1 steps, one per position 0..S-2;
    # step i drafts position i+2 (1-based), which is the oracle pick this compares against.
    # The driver's MTP mode trims the list itself (the draft head consumes token t+1, so it runs S-1
    # steps over ids[1:]) — pass the FULL id list and let it do that once. Shifting here as well drops
    # a second token and the ids check below catches it.
    out = open(MTPLOG, encoding="utf-8", errors="replace").read() if SKIP_RUN else run_mtp(want_ids, hid, POS)
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
    if n_bad or n_ok == 0:
        print("RIDGE_MTP_VERIFY_FAIL the draft head disagrees with the oracle at %d of %d steps"
              % (n_bad, n_checked))
        return 1
    print("RIDGE_MTP_VERIFY_SUMMARY fixture=%s steps=%d agree=%d/%d rope_pos=%s"
          % (name, len(ours), n_ok, n_checked, POS))
    print("RIDGE_MTP_VERIFY_DONE the MTP draft head reproduces the oracle's next-next-token picks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
