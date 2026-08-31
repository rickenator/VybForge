#!/usr/bin/env python3
import re, subprocess, sys
TS = int(sys.argv[1]) if len(sys.argv) > 1 else 83
f = TS / 4.0
t = open("native/train/train_full_ce.vyb").read()
# recompute expected S=TS ASLB offsets from the S=4 set
s4off = [0,81920,163840,294912,327680,360448,491520,524288,655360,688128,819200,901120,
         983040,1064960,1376256,1687552,1998848,2080768,2162688,2162752,2162816,2162880,
         2162944,2163008,2163072]
exp = {str(int(o * f)) for o in s4off}
exp.add(str(int(2163136 * f)))
# collect ASLB forward/backward offsets
fwd = re.findall(r"ASLB\+L3\*(\d+)\+(\d+)", t)
bwd = re.findall(r"ASLB\+Lb\*(\d+)\+(\d+)", t)
fwd_ok = all(o in exp for _, o in fwd) and len({s for s, _ in fwd}) == 1
bwd_ok = all(o in exp for _, o in bwd)
print("fwd ASLB sites:", len(fwd), "tags all distinct-stride:", fwd_ok, "| all offsets in exp:", all(o in exp for _, o in fwd))
print("bwd ASLB sites:", len(bwd), "all offsets in exp:", bwd_ok)
# no S=4 leftovers
for bad in ["2163136", "81920", "163840", "2162688"]:
    if bad in re.sub(r"LSLB.*", "", t.replace("LSLB", "")):  # crude; real check below
        pass
left = [x for x in ["*2163136", "+163840", "+81920", "+294912"] if x in t.replace("LSLB+", "") and "+106496" not in x]
print("suspicious S=4 leftovers (outside LSLB):", left or "none")
# S constant
print("S<Int> = %d present:" % TS, "S<Int> = %d" % TS in t, "| NSTP:", re.search(r"NSTP<Int> = (\d+)", t).group(1))
# size sanity: XI alloc
m = re.search(r"loc\(XI\),(\d+)\*8", t)
print("XI alloc elems:", m.group(1), "expected:", TS * 2560)
