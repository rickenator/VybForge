#!/usr/bin/env python3
# Verify the real Qwen3-4B Vyb GPU decode token stream against the numpy
# reference (native/gguf/decode_ref.py -> native/out/decode_ids_ref.txt).
# decode_ids_vyb.txt is written by native/host/decode_driver.vyb (2 seed tokens
# + GEN generated tokens). Token-for-token match required.
import os, sys
base = "native/out"
def load(p):
    p = os.path.join(base, p)
    if not os.path.exists(p):
        print(f"DECODE_REAL_VERIFY: FAIL missing {p}"); sys.exit(1)
    return [int(x) for x in open(p).read().split()]

vyb = load("decode_ids_vyb.txt")
ref = load("decode_ids_ref.txt")
if len(vyb) != len(ref):
    print(f"DECODE_REAL_VERIFY: FAIL length vyb={len(vyb)} ref={len(ref)}")
    print(f"  vyb={vyb}")
    print(f"  ref={ref}")
    sys.exit(1)
bad = [i for i,(a,b) in enumerate(zip(vyb,ref)) if a != b]
if bad:
    print(f"DECODE_REAL_VERIFY: FAIL mismatches at indices {bad}")
    print(f"  vyb={vyb}")
    print(f"  ref={ref}")
    sys.exit(1)
print(f"DECODE_REAL_MATCH: OK   tokens={vyb}")
