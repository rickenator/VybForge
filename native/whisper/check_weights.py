#!/usr/bin/env python3
"""check_weights.py — gate: Vyb's inventory of the checkpoint == the reference.

    ~/miniconda/bin/python check_weights.py <vyb_listing> <ref_listing>

Compares name -> (dtype, dims) as a dict (order is not part of the contract) and
the totals, so a dropped, renamed, mis-typed or mis-shaped tensor fails loudly.
"""
import sys


def load(path):
    rows = {}
    tail = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if parts[0] == "TOTAL_BYTES":
                tail["bytes"] = int(parts[-1])
            elif parts[0] == "TENSORS":
                tail["tensors"] = int(parts[-1])
            else:
                rows[parts[0]] = (parts[1], parts[2])
    return rows, tail


def main():
    vyb, vtail = load(sys.argv[1])
    ref, rtail = load(sys.argv[2])
    ok = True

    missing = sorted(set(ref) - set(vyb))
    extra = sorted(set(vyb) - set(ref))
    differ = sorted(k for k in set(ref) & set(vyb) if ref[k] != vyb[k])
    print(f"tensors: vyb={len(vyb)} ref={len(ref)}  missing={len(missing)} "
          f"extra={len(extra)} mismatched={len(differ)}")
    for k in missing[:5]:
        print(f"   missing: {k} {ref[k]}")
    for k in extra[:5]:
        print(f"   extra:   {k} {vyb[k]}")
    for k in differ[:9]:
        print(f"   differs: {k} vyb={vyb[k]} ref={ref[k]}")
    if missing or extra or differ:
        ok = False

    vb = vtail.get("bytes", -1)
    rb = rtail.get("bytes", -2)
    vt = vtail.get("tensors", -1)
    rt = rtail.get("tensors", -2)
    print(f"TOTAL_BYTES: vyb={vb} ref={rb}   TENSORS: vyb={vt} ref={rt}")
    if vb != rb or vt != rt:
        ok = False

    print("WEIGHTS_GATE:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
