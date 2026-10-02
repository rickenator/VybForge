#!/usr/bin/env python3
"""Authoritative third opinion for the object-graph layer: torch's own view of a checkpoint.

Loads the shard with `map_location="meta"`, which reads the metadata only (no tensor bytes are
materialized, so this stays cheap on a 1 GB shard) and compares names/dtypes/shapes against the
listing produced by native/torchload/tb_pkl_probe.vyb.

Comparison is BY NAME, not by position: dict order is a property of the file on disk, not of torch,
and a positional comparison would report false failures. Order agreement is still reported as
information.

Usage:  .venv/bin/python native/torchload/tb_torchcheck.py [listing]
Exit:   0 ok, 1 mismatch, 2 torch unavailable (caller decides whether that is fatal).
"""
import os
import sys

LISTING = sys.argv[1] if len(sys.argv) > 1 else "native/out/tb_pkl_listing_vyb.txt"

try:
    import torch
except Exception as exc:  # torch absent -> not a failure of the reader under test
    print(f"TB_TORCHCHECK_UNAVAILABLE ({exc})")
    sys.exit(2)

DT = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32", torch.float64: "F64",
      torch.int8: "I8", torch.uint8: "U8", torch.int16: "I16", torch.int32: "I32",
      torch.int64: "I64", torch.bool: "BOOL"}


def parse_listing(path):
    rows, meta, order = {}, {}, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if "|" not in line:
                k, _, v = line.partition(" ")
                meta[k] = v
                continue
            fld = line.split("|")
            rows[fld[0]] = (fld[1], [int(x) for x in fld[2].split(",") if x != ""])
            order.append(fld[0])
    return meta, rows, order


def main():
    meta, rows, order = parse_listing(LISTING)
    path = os.environ.get("VYBFORGE_TB_FILE", "/home/rick/Models/spikingbrain-v1-7b-base/pytorch_model-00001.bin")
    sd = torch.load(path, map_location="meta", weights_only=True)
    bad = []
    names = [n for n, _ in sd.items()]
    if len(names) != len(rows):
        bad.append(f"count torch={len(names)} listing={len(rows)}")
    for n, t in sd.items():
        if n not in rows:
            bad.append(f"absent from listing: {n}")
            continue
        dt = DT.get(t.dtype, str(t.dtype))
        sh = list(t.shape)
        if dt != rows[n][0]:
            bad.append(f"dtype {n} torch={dt} listing={rows[n][0]}")
        if sh != rows[n][1]:
            bad.append(f"shape {n} torch={sh} listing={rows[n][1]}")
    extra = [n for n in order if n not in sd]
    if extra:
        bad.append(f"listing has {len(extra)} names torch does not (e.g. {extra[0]})")
    if bad:
        for b in bad[:10]:
            print(f"   {b}")
        print(f"TB_TORCHCHECK_FAIL {len(bad)} mismatch(es) vs {LISTING}")
        return 1
    order_note = "same order" if names == order else "order differs (not a criterion)"
    print(f"TB_TORCHCHECK_OK {len(rows)} tensors agree with torch.load(map_location='meta') by name "
          f"(count/dtype/shape); {order_note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
