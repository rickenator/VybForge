#!/usr/bin/env python3
"""Inventory TSV -> the engine's tensor-index TSV (VybForge#10 phase 4, unit 9c step 2).

The engine loads weights by NAME: `native/host/model_driver.vyb` parses a TSV of
`name \t dims \t ggml_type \t off=<absolute> \t size=0` and stages each tensor it needs. That TSV
exists for Qwen3-4B (`native/out/qwen3_4b_tensors.tsv`); the Ridge model needs its own, and the
honest way to get it is to derive it from the inventory rather than hand-write offsets.

The inventory (`native/gguf/ridge_inventory.py` -> `native/gguf/ridge-3.7bpw-inventory.tsv`) is the
richer file — name, dims, type id, type NAME, role, absolute offset, bytes — so the conversion is a
projection, and this script asserts the two files agree on every tensor's byte count for its type
before writing. Offsets are the inventory's own ABSOLUTE offsets, which is what the engine's
`read_at` wants; nothing is recomputed, so a change in the alignment rule cannot silently shift the
weights by a few bytes.

Usage:
  env -u PYTHONPATH python3 native/tools/inventory_to_tsv.py \
      [--inventory native/gguf/ridge-3.7bpw-inventory.tsv] \
      [--out native/out/ridge_tensors.tsv]
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ggml type id -> bytes per block and elements per block, for the byte-count assertion only.
# A tensor whose byte count disagrees with its type is a tensor the engine would stage with the
# wrong packed length, so it is refused here rather than discovered as garbage activations.
TYPES = {
    0: ("F32", 4, 1), 1: ("F16", 2, 1), 2: ("Q4_0", 18, 32), 8: ("Q8_0", 34, 32),
    12: ("Q4_K", 144, 256), 13: ("Q5_K", 176, 256), 14: ("Q6_K", 210, 256),
    21: ("IQ3_S", 110, 256), 22: ("IQ2_S", 82, 256), 30: ("BF16", 2, 1),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default=os.path.join(REPO, "native/gguf/ridge-3.7bpw-inventory.tsv"))
    ap.add_argument("--out", default=os.path.join(REPO, "native/out/ridge_tensors.tsv"))
    args = ap.parse_args()

    rows, bad = [], []
    for line in open(args.inventory, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        name, dims, tyid, tyname, role, off, nbytes = line.rstrip("\n").split("\t")
        ty, size, per = TYPES.get(int(tyid), (tyname, 0, 0))
        numel = 1
        for d in dims.split("x"):
            numel *= int(d)
        if per:
            want = numel // per * size
            if want != int(nbytes):
                bad.append((name, int(tyid), numel, int(nbytes), want))
        else:
            bad.append((name, int(tyid), numel, int(nbytes), -1))
        rows.append(f"{name}\t{dims}\t{tyid}\toff={off}\tsize=0")

    if bad:
        print(f"INVENTORY_TSV_FAIL {len(bad)} tensor(s) whose byte count does not match their type")
        for b in bad[:8]:
            print("   ", b)
        return 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(rows) + "\n")
    n = len(rows)
    print(f"wrote {args.out}: {n} tensors (offsets = the inventory's absolute offsets)")
    print(f"INVENTORY_TSV_OK tensors={n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
