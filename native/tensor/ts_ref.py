#!/usr/bin/env python3
"""ts_ref — the authority side of the S0.2b tensor-core gate (validation only; nothing here is
in the runtime path).

numpy IS the definition of shape/stride/broadcast behaviour, so it is the only authority used:

  * strides come from a real float32 array's `.strides // itemsize` (a fresh C-order array),
  * offsets come from `np.ravel_multi_index` — the exact indexing rule,
  * contiguity comes from `.flags["C_CONTIGUOUS"]` on real views (slices, transpose),
  * broadcasting comes from `np.broadcast_shapes` (which raises for incompatible shapes) and
    `np.broadcast_to(...).strides`, whose ZERO strides are the expanded axes.

It also refuses to hand over a toothless table: a table with no refusal case or no
non-contiguous case cannot catch a broken core, so `gen` counts both and fails if either is 0.

Modes:
  gen   <cases.txt> <expected.json>   build the case table + numpy's answers
  check <probe.txt> <expected.json>   compare the Vyb core's answers
"""

import json
import random
import sys

import numpy as np

# Rank 0 through rank 5, plus the degenerate shapes (a 0-dim, all-1 dims, a leading 1).
CORNERS = [(), (5,), (1,), (2, 3), (2, 3, 4), (1, 1), (1, 2, 1, 3), (4, 1, 5),
           (2, 2, 2, 2, 2), (0,), (0, 3), (7, 1, 1, 3)]
BCAST = [(), (1,), (3,), (2, 1), (1, 4), (2, 3), (2, 1, 4), (1, 3, 1), (5, 2, 3, 1), (2, 4)]


def fmt(v):
    """A vector as a case-file token: '-' for an empty (rank 0) vector."""
    v = list(v)
    return ",".join(str(int(x)) for x in v) if v else "-"


def estrides(a):
    return [int(s // a.itemsize) for s in a.strides]


def gen(cases_path, exp_path):
    rng = random.Random(20261002)          # one stream, consumed in a fixed order
    cases, exp = [], {}

    def add(line, _name, expect):
        # The case name IS the line's first token (the probe echoes it back), so this stays
        # consistent however the call site spells it.
        cases.append(line)
        exp[line.split(" ", 1)[0]] = expect

    shapes = list(CORNERS)
    for _ in range(40):
        r = rng.randint(1, 5)
        shapes.append(tuple(rng.randint(1, 6) for _ in range(r)))

    # Case names carry an index: two shapes can format to the same token (a duplicate CORNER, or
    # a random repeat), and a silently-collapsed name would leave the expected-value dict smaller
    # than the case file — the probe would then answer cases nothing checks.
    seq = [0]

    def named(base):
        seq[0] += 1
        return "%s#%d" % (base, seq[0])

    for sh in shapes:
        tok = fmt(sh)
        n = 1 if len(sh) == 0 else int(np.prod(sh))
        add(f"{named('numel')} numel {tok}", None, str(n))
        add(f"{named('nbytes32')} nbytes {tok} 0", None, str(n * 4))
        add(f"{named('nbytes16')} nbytes {tok} 1", None, str(n * 2))
        if 0 not in sh:
            a = np.zeros(sh, dtype=np.float32)
            es = estrides(a)
            add(f"{named('strides')} strides {tok}", None, fmt(es) if sh else "empty")
            add(f"{named('contig')} contig {tok} {fmt(es)}", None, "1")
            if len(sh) > 0:
                for k in range(2):
                    co = tuple(rng.randrange(d) for d in sh)
                    want = int(np.ravel_multi_index(co, sh))
                    add(f"{named('off')} offset {tok} {fmt(es)} {fmt(co)}", None, str(want))
                co_oob = tuple(d + 1 for d in sh)
                add(f"{named('offoob')} offset {tok} {fmt(es)} {fmt(co_oob)}", None, "-1")

    # real views: numpy decides contiguity, and the strides are the view's own
    base = np.zeros((4, 6), dtype=np.float32)
    views = {
        "row": base[1], "col": base[:, 1], "transpose": base.T, "slice2": base[::2],
        "sub": base[1:, 2:5], "flat": base.reshape(-1), "stride_view": base[::2, ::3],
    }
    for nm, v in views.items():
        es = estrides(v)
        want = "1" if v.flags["C_CONTIGUOUS"] else "0"
        add(f"view_{nm} contig {fmt(v.shape)} {fmt(es)}", f"view_{nm}", want)

    # broadcasting, every pair of a small set
    for a in BCAST:
        for b in BCAST:
            nm = f"bcast_{fmt(a)}_{fmt(b)}"
            try:
                r = tuple(np.broadcast_shapes(a, b))
            except ValueError:
                add(f"{nm} bcast {fmt(a)} {fmt(b)}", nm, "refuse")
                continue
            add(f"{nm} bcast {fmt(a)} {fmt(b)}", nm, fmt(r) if r else "empty")
            A = np.zeros(a, dtype=np.float32) if a else np.array(0.0, dtype=np.float32)
            try:
                bs = np.broadcast_to(A, r)
            except ValueError:
                continue
            nm2 = f"bs_{nm}"
            add(f"{nm2} bcaststrides {fmt(a)} {fmt(estrides(A))} {fmt(r)}",
                nm2, fmt(estrides(bs)) if r else "empty")

    open(cases_path, "w").write("\n".join(cases) + "\n")
    json.dump(exp, open(exp_path, "w"), indent=0)

    n_refuse = sum(1 for v in exp.values() if v == "refuse")
    n_noncontig = sum(1 for v in exp.values() if v == "0")
    n_empty = sum(1 for v in exp.values() if v == "empty")
    print("TSREF_CASES=%d refuse=%d noncontig=%d rank0=%d" % (len(exp), n_refuse, n_noncontig, n_empty))
    # a table that cannot catch a broken core is decoration
    if n_refuse == 0 or n_noncontig == 0:
        print("TSREF_FAIL the case table has no %s case — it cannot catch a broken core"
              % ("refusal" if n_refuse == 0 else "non-contiguous"))
        return 1
    print("TSREF_OK")
    return 0


def check(probe_path, exp_path):
    exp = json.load(open(exp_path))
    got = {}
    for line in open(probe_path):
        line = line.strip()
        if line.startswith("TS "):
            parts = line.split(" ", 2)
            if len(parts) == 3:
                got[parts[1]] = parts[2]
    if "TSDONE" not in open(probe_path).read():
        print("TSGATE_FAIL probe did not reach TSDONE")
        return 1
    missing = [k for k in exp if k not in got]
    wrong = [(k, exp[k], got[k]) for k in exp if k in got and got[k] != exp[k]]
    unknown = [k for k in got if k not in exp]
    if missing:
        print("TSGATE_MISSING %d (first: %s)" % (len(missing), ", ".join(missing[:4])))
    if unknown:
        print("TSGATE_UNKNOWN %d (first: %s)" % (len(unknown), ", ".join(unknown[:4])))
    for k, w, g in wrong[:12]:
        print("TSGATE_MISMATCH %s: vyb=%s numpy=%s" % (k, g, w))
    if missing or wrong or unknown:
        print("TSGATE_FAIL %d wrong, %d missing, %d unknown of %d cases"
              % (len(wrong), len(missing), len(unknown), len(exp)))
        return 1
    c = {"refuse": 0, "empty": 0, "noncontig": 0}
    for v in exp.values():
        if v == "refuse":
            c["refuse"] += 1
        elif v == "empty":
            c["empty"] += 1
        elif v == "0":
            c["noncontig"] += 1
    print("TSGATE_CASES_OK %d cases (refuse=%d rank0=%d noncontig=%d)"
          % (len(exp), c["refuse"], c["empty"], c["noncontig"]))
    print("TSGATE_OK")
    return 0


def mutate(exp_path, out_path):
    """Perturb one expectation so the gate can prove its checker actually fails."""
    exp = json.load(open(exp_path))
    for k in sorted(exp):
        if exp[k].isdigit():
            exp[k] = str(int(exp[k]) + 1)
            print("TSREF_MUTATE %s -> %s" % (k, exp[k]))
            break
    json.dump(exp, open(out_path, "w"), indent=0)
    return 0


def main(argv):
    if len(argv) < 2:
        print("usage: ts_ref.py <mode> ..."); return 2
    if argv[1] == "gen":
        return gen(argv[2], argv[3])
    if argv[1] == "check":
        return check(argv[2], argv[3])
    if argv[1] == "mutate":
        return mutate(argv[2], argv[3])
    print("unknown mode %s" % argv[1])
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
