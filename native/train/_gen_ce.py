#!/usr/bin/env python3
"""Regenerate train_full_ce.vyb for a target sequence length TS, from the S=4 CE driver.

The S=4 train_full_ce.vyb keeps every sequence-scaled size as an S-proportional literal and
the LoRA `s` intermediates as SYMBOLIC `S*R` (they auto-scale). So generalizing 4->TS only
needs to rescale the S=4 numeric literals by f=TS/4:
  - S-dependent element counts (S*D=10240, S*NQ=16384, S*NKV=4096, S*FF=38912)
  - ASLB per-activation byte offsets and per-layer stride 2163136 (all S-proportional)
Use ONLY the S=4 literal set (never global 5120/8192/... which collide with S-independent
adapter numels), and PROTECT S-independent lines (LSLB/MSLB/VSLB/AdamW/adapter loads) whose
40960/8192/19456/... are adapter offsets that must NOT rescale.

Run: python native/train/_gen_ce.py <TS> [NSTP]   [writes native/train/train_full_ce.vyb]
"""
import re, sys
TS = int(sys.argv[1]) if len(sys.argv) > 1 else 84
NSTP = int(sys.argv[2]) if len(sys.argv) > 2 else 4
f = TS / 4.0
BASE = "native/train/train_full_ce.vyb"   # S=4 baseline WITH CE wiring
DST = "native/train/train_full_ce.vyb"
text = open(BASE).read()

# S=4 literals -> TS
rep = {"10240": str(int(10240 * f)), "16384": str(int(16384 * f)),
       "4096": str(int(4096 * f)), "38912": str(int(38912 * f)), "2163136": str(int(2163136 * f))}
rep["77872896"] = str(int(77872896 * f))   # 36 * ASLB per-layer stride (S=4 total)
s4off = [0,81920,163840,294912,327680,360448,491520,524288,655360,688128,819200,901120,
         983040,1064960,1376256,1687552,1998848,2080768,2162688,2162752,2162816,2162880,
         2162944,2163008,2163072]
rep.update({str(o): str(int(o * f)) for o in s4off})
pattern = re.compile(r"(?<!\d)(" + "|".join(sorted(rep, key=len, reverse=True)) + r")(?!\d)")

def protect(line):
    return ("LSLB" in line) or ("MSLB" in line) or ("VSLB" in line) or ("AW+" in line) \
           or ('read_bin("native/out/m2e_l' in line)

def sub_m(m):
    return rep[m.group(1)]

out = []
for line in text.split("\n"):
    out.append(line if protect(line) else pattern.sub(sub_m, line))
new = "\n".join(out)
new = new.replace("S<Int> = 4", "S<Int> = %d" % TS)
new = re.sub(r"NSTP<Int> = \d+", "NSTP<Int> = %d" % NSTP, new)
open(DST, "w").write(new)
print("wrote", DST, "S=%d (f=%.2f) NSTP=%d lines=%d" % (TS, f, NSTP, new.count(chr(10)) + 1))
