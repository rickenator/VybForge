#!/usr/bin/env python3
"""oracle_weights.py — independent inventory of a safetensors checkpoint.

    ~/miniconda/bin/python oracle_weights.py [checkpoint]

Writes gold_stt/weights_listing_ref.txt in the same format weights_probe.vyb
emits (name<TAB>dtype<TAB>dims x dims, then TOTAL_BYTES and TENSORS), by parsing
the JSON header itself — not through the Vyb module under test — and cross-checks
against transformers' own state_dict (a third reading) so the reference is not just
one parser agreeing with itself.
"""
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [
    os.environ.get("VYBFORGE_ST_FILE", ""),
    "/usr/export/LLM/hf/hub/models--openai--whisper-tiny.en/snapshots/"
    "87c7102498dcde7456f24cfd30239ca606ed9063/model.safetensors",
]


def find():
    for c in CANDIDATES:
        if c and os.path.exists(c):
            return c
    return None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else find()
    if not path or not os.path.exists(path):
        print("no checkpoint found; pass it as argv[1] or set VYBFORGE_ST_FILE")
        return 2
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode("utf-8"))

    rows = []
    total = 0
    for name, meta in hdr.items():
        if name == "__metadata__":
            continue
        shape = meta["shape"]
        dims = "x".join(str(d) for d in shape) if shape else "scalar"
        rows.append((name, meta["dtype"], dims))
        off = meta["data_offsets"]
        total += off[1] - off[0]

    out_dir = os.path.join(HERE, "gold_stt")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "weights_listing_ref.txt")
    with open(out, "w") as f:
        for name, dt, dims in rows:
            f.write(f"{name}\t{dt}\t{dims}\n")
        f.write(f"TOTAL_BYTES\t\t{total}\n")
        f.write(f"TENSORS\t\t{len(rows)}\n")
    print(f"wrote {out}: {len(rows)} tensors, {total} bytes")

    # third reading: transformers' state_dict keys/shapes
    try:
        from transformers import WhisperForConditionalGeneration
        import torch  # noqa: F401
        m = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        sd = m.state_dict()
        ref = {k: tuple(v.shape) for k, v in sd.items()}
        have = {n: tuple(int(x) for x in d.split("x")) if d != "scalar" else ()
                for n, _, d in rows}
        missing = sorted(set(ref) - set(have))
        extra = sorted(set(have) - set(ref))
        badshape = sorted(k for k in set(ref) & set(have) if ref[k] != have[k])
        print(f"state_dict cross-check: {len(ref)} tensors, "
              f"missing={len(missing)} extra={len(extra)} shape_mismatch={len(badshape)}")
        for k in (missing + extra + badshape)[:8]:
            print("   ", k, ref.get(k), have.get(k))
    except Exception as e:                                   # noqa: BLE001
        print("state_dict cross-check SKIPPED:", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
