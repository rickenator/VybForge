#!/usr/bin/env python3
"""Gate for wcheck_driver.vyb (make -f native/Makefile wcheck).

The driver GPU-dequants attn_v (L0/L4/L35) and ffn_down (L4/L35) and prints the first
6 f64 values of each buffer; native/gguf/wcheck_ref.py prints the same first 6 values
from numpy's read_weight for the same tensors. Until now those two printouts were
compared by eye, which is not a gate: this compares them by name with an exit code.

Catches the orientation class of bug directly — with a literal 0 in the dequant's
in-dim slot the GPU buffer is in raw GGUF [out,in] order while numpy returns [in,out],
so the two disagree in the first 6 values.
"""
import os, re, sys

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dlog = os.path.join(repo, "native/out/wcheck_driver.log")
rlog = os.path.join(repo, "native/out/wcheck_ref.log")
TOL = 2e-3

for p in (dlog, rlog):
    if not os.path.exists(p):
        print(f"FAIL: missing {p} — run the wcheck target")
        raise SystemExit(2)


def parse_ref(path):
    out = {}
    for line in open(path):
        m = re.match(r"WREF (\S+) L(\d+) -> \[(.*)\]", line.strip())
        if m:
            out[(m.group(1), int(m.group(2)))] = [float(x) for x in m.group(3).split(",")]
    return out


def parse_drv(path):
    out = {}
    for line in open(path):
        m = re.match(r"WCHECK (\S+) L(\d+) .*-> (.*)", line.strip())
        if m:
            vals = [float(x) for x in m.group(3).split()]
            out[(m.group(1), int(m.group(2)))] = vals
    return out


ref, drv = parse_ref(rlog), parse_drv(dlog)
if not ref:
    print(f"FAIL: no WREF lines in {rlog}")
    raise SystemExit(1)
if not drv:
    print(f"FAIL: no WCHECK lines in {dlog}")
    raise SystemExit(1)

bad = []
for key in sorted(drv):
    if key not in ref:
        print(f"  {key[0]} L{key[1]}: no reference line — skipped")
        continue
    g, r = drv[key], ref[key]
    n = min(len(g), len(r))
    if n == 0:
        bad.append(key)
        continue
    rel = max(abs(g[i] - r[i]) / max(abs(r[i]), 1e-6) for i in range(n))
    ok = rel < TOL
    print(f"  {key[0]} L{key[1]}: maxrel={rel:.3e}  gpu[0]={g[0]:.6f} ref[0]={r[0]:.6f}  {'OK' if ok else 'FAIL'}")
    if not ok:
        bad.append(key)

# the union must cover the checks the driver actually performs
missing = [k for k in ref if k not in drv and k[0] == "attn_v"]
print(f"  compared {len(drv) - len([k for k in drv if k not in ref])} checks against numpy")
print("WCHECK_VERIFY:", "OK" if not bad else f"FAIL {bad}")
raise SystemExit(0 if not bad else 1)
