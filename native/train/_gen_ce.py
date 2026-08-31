#!/usr/bin/env python3
"""Generate train_full_ce.vyb (S=4 corpus-CE) from the packed S=2 driver.

The S=2 literals 5120(S*D)/8192(S*NQ)/2048(S*NKV)/19456(S*FF) appear BOTH as S-dependent
working-buffer sizes/ASLB offsets AND as S-independent LoRA adapter numels (Uq=5120, Vq=8192,
Uo=8192, Vg=19456, ...) used by AdamW / adapter loads. So we PROTECT S-independent lines
(those touching LSLB/MSLB/VSLB slabs, AW+ AdamW params, or m2e_l* adapter .bin loads) and only
transform S-dependent lines. All numeric replacements use ONE regex pass (no cascading).
ASLB per-layer offsets recomputed for S=4. Also flips S:2->4.
Run: python native/train/_gen_ce.py"""
import re
SRC = "native/train/train_full_loop_pkd.vyb"
DST = "native/train/train_full_ce.vyb"
text = open(SRC).read()

# S-dependent literal -> S=4 value
rep = {"5120": "10240", "8192": "16384", "2048": "4096", "19456": "38912",
       "1081568": "2163136"}
# ASLB per-activation offsets S=2 -> S=4
for a, b in [
    (0,0),(40960,81920),(81920,163840),(147456,294912),(163840,327680),(180224,360448),
    (245760,491520),(262144,524288),(327680,655360),(344064,688128),(409600,819200),
    (450560,901120),(491520,983040),(532480,1064960),(688128,1376256),(843776,1687552),
    (999424,1998848),(1040384,2080768),(1081344,2162688),(1081376,2162752),(1081408,2162816),
    (1081440,2162880),(1081472,2162944),(1081504,2163008),(1081536,2163072),
]:
    rep[str(a)] = str(b)

# longest-first so e.g. 81920 matches before 8192
pattern = re.compile(r"(?<!\d)(" + "|".join(sorted(rep, key=len, reverse=True)) + r")(?!\d)")

def protect(line):
    return ("LSLB" in line) or ("MSLB" in line) or ("VSLB" in line) or ("AW+" in line) \
           or ('read_bin("native/out/m2e_l' in line)

def sub_m(m):
    return rep[m.group(1)]

out = []
for line in text.split("\n"):
    if protect(line):
        out.append(line)
    else:
        out.append(pattern.sub(sub_m, line))
new = "\n".join(out)
# S constant 2 -> 4 (defined in main sig)
new = new.replace("S<Int> = 2", "S<Int> = 4")
# NSTP 5 -> 6 (CE overfit; keep moderate — will regenerate ref to match)
new = new.replace("NSTP<Int> = 5", "NSTP<Int> = 6")
open(DST, "w").write(new)
print("wrote", DST, "lines:", new.count("\n")+1, "| S->4 applied:", "S<Int> = 4" in new)
