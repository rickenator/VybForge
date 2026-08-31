#!/usr/bin/env python3
"""Regenerate train_full_ce.vyb for a target sequence length TS (masks/CE wiring preserved).

Base-S AGNOSTIC: reads the CURRENT S from the file, computes the full ASLB per-layer byte layout
for the current and target S from the activation dims, and rescales every S-proportional literal
(current_S value -> target_S value). The LoRA `s` intermediates and head grids are SYMBOLIC
(S*R / S*D / S*VOCAB) so they auto-scale. PROTECTS S-independent lines (LSLB/MSLB/VSLB slabs,
AW+ AdamW, m2e_l* adapter loads) whose literal sizes are adapter numels that must NOT rescale.

Run: python native/train/_gen_ce.py <TS> [NSTP]
"""
import re, sys
TS = int(sys.argv[1]) if len(sys.argv) > 1 else 93
NSTP = int(sys.argv[2]) if len(sys.argv) > 2 else 4
BASE = "native/train/train_full_ce.vyb"
DST = "native/train/train_full_ce.vyb"
text = open(BASE).read()
cur = int(re.search(r"S<Int> = (\d+)", text).group(1))

# per-activation cache ORDER (elements), for a given S
def layout(S):
    D, NQ, NKV, FF, R = 2560, 4096, 1024, 9728, 2
    seq = [S*D, S*D, S*NQ, S*NKV, S*NKV, S*NQ, S*NKV, S*NQ, S*NKV, S*NQ, S*D, S*D, S*D,
           S*FF, S*FF, S*FF, S*D, S*D, S*R, S*R, S*R, S*R, S*R, S*R, S*R]
    offs = []; acc = 0
    for el in seq:
        offs.append(acc); acc += el * 8
    return offs, acc   # offsets (bytes) + per-layer stride (bytes)

cur_offs, cur_stride = layout(cur)
tgt_offs, tgt_stride = layout(TS)
# element-count literals (S-dependent working-buffer sizes / grids)
rep = {}
for dim, name in [(2560, "D"), (4096, "NQ"), (1024, "NKV"), (9728, "FF")]:
    rep[str(cur * dim)] = str(TS * dim)
# ASLB per-activation offsets + stride + 36*stride
for a, b in zip(cur_offs, tgt_offs):
    rep[str(a)] = str(b)
rep[str(cur_stride)] = str(tgt_stride)
rep[str(36 * cur_stride)] = str(36 * tgt_stride)
pattern = re.compile(r"(?<!\d)(" + "|".join(sorted(rep, key=len, reverse=True)) + r")(?!\d)")

def protect(line):
    return ("LSLB" in line) or ("MSLB" in line) or ("VSLB" in line) or ("AW+" in line) \
           or ('read_bin("native/out/m2e_l' in line)

def sub_m(m):
    return rep[m.group(1)]

out = [line if protect(line) else pattern.sub(sub_m, line) for line in text.split("\n")]
new = "\n".join(out)
# replace ONLY the line-start S constant decl (not LOS<Int>/IDS<Int>, which end in "...S<Int>")
new = re.sub(r"^(\s*)S<Int> = \d+;", r"\1S<Int> = %d;" % TS, new, flags=re.MULTILINE)
new = re.sub(r"NSTP<Int> = \d+", "NSTP<Int> = %d" % NSTP, new)
open(DST, "w").write(new)
print("wrote", DST, "S %d->%d NSTP=%d lines=%d" % (cur, TS, NSTP, new.count(chr(10)) + 1))
