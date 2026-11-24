#!/usr/bin/env python3
"""dsa_scale_report.py — one line per depth: how the DSA layers' cost scales.

Reads `SEC layerbody n=<layer> <ms>` lines and reports the DSA layers (n % 4 == 3) only.
The question it answers: if the cost of a DSA layer depends on how many DSA layers came before it
(state accumulating), the maximum grows with depth. If it depends only on position, it does not.
"""
import os
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sc45.log"
nlay = os.environ.get("NL", "?")
bod = {}
for line in open(path, errors="replace"):
    m = re.match(r"SEC layerbody n=(\d+) (\d+)", line.strip())
    if m:
        bod[int(m.group(1))] = int(m.group(2))
dsa = [(n, v) for n, v in sorted(bod.items()) if n % 4 == 3]
if not dsa:
    print("NLAY=%s : no DSA samples" % nlay)
else:
    vals = [v for _, v in dsa]
    print("NLAY=%s  DSA=%d  max=%d  first=%d  values=%s"
          % (nlay, len(dsa), max(vals), vals[0], vals))
