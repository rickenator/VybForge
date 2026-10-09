#!/usr/bin/env python3
"""Q4_K dequant reference, cross-checked against the INDEPENDENT `gguf` package.

Q4_K was the one quant type in this repo with no reference and no gate, while being the type of the
27B Ridge model's biggest projections (blk.N.attn_qkv, blk.N.attn_gate, blk.N.ssm_out — 17.7 MB each).
The GDN layer check could not use those weights for exactly that reason: it feeds the authority the
same weights the GPU used, so without an independent reference the comparison would be our numpy
agreeing with our kernel.

Blocks are 144 bytes per 256 values: d (f16), dmin (f16), scales[12] (six-bit scales and mins), qs[128]
(two four-bit quants per byte). The 64-value group g takes qs[32g : 32g+32], the low nibbles belonging
to sub-block 2g and the high nibbles to 2g+1, each scaled by its own 6-bit scale and min:

    value = d * sc[j] * nibble - dmin * m[j]

The authority here is the `gguf` package's own dequantizer (aggregate/quantization/quants.py), the same
route the Q8_0 gate uses — if it is unavailable the run says so instead of implying agreement.

Writes native/out/q4k_ref.txt: one line per tensor, `Q4_K <name>@<off> -> <first 4096 values as the
signed 64-bit integer their f64 bits spell>`, so a GPU dump can be compared bit for bit.

Usage: env -u PYTHONPATH .venv/bin/python native/tools/q4k_ref.py
Env: VYBFORGE_Q4K_TENSORS=<n>  how many tensors to check (default 2, smallest first)
     VYBFORGE_Q4K_NUMS=<n>     values per tensor in the dump (default 4096)
"""
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
INVENTORY = os.path.join(REPO, "native/gguf/ridge-3.7bpw-inventory.tsv")
GGUF = os.environ.get("VYBFORGE_RIDGE_GGUF",
                      os.path.expanduser("~/Models/qwen38-27b-ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf"))
OUT = os.path.join(REPO, "native/out/q4k_ref.txt")
BLOCK, QK = 144, 256


def six_bit(scales):
    """get_scale_min_k4 for all eight sub-blocks, vectorised over the blocks in `scales`.

    j < 4:  d = q[j] & 63,            m = q[j+4] & 63
    j >= 4: d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
            m = (q[j+4] >> 4)   | ((q[j]   >> 6) << 4)
    """
    nb = scales.shape[0]
    sc = np.zeros((nb, 8), dtype=np.uint8)
    mn = np.zeros((nb, 8), dtype=np.uint8)
    for j in range(8):
        if j < 4:
            sc[:, j] = scales[:, j] & 63
            mn[:, j] = scales[:, j + 4] & 63
        else:
            sc[:, j] = ((scales[:, j + 4] & 0x0F) | ((scales[:, j - 4] >> 6) << 4)) & 0xFF
            mn[:, j] = ((scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4)) & 0xFF
    return sc, mn


def dequant_q4k(raw):
    """Our port. Returns an f32 array of the tensor's values, in the file's order."""
    nb = len(raw) // BLOCK
    b = np.frombuffer(raw[:nb * BLOCK], dtype=np.uint8).reshape(nb, BLOCK)
    d = b[:, 0:2].copy().view(np.float16).reshape(-1).astype(np.float32)
    dmin = b[:, 2:4].copy().view(np.float16).reshape(-1).astype(np.float32)
    sc, mn = six_bit(b[:, 4:16])
    qs = b[:, 16:144]
    out = np.empty((nb, 8, 32), dtype=np.float32)
    for j in range(8):
        q = qs[:, 32 * (j // 2): 32 * (j // 2) + 32].astype(np.float32)
        # dequantize_row_q4_K: the LOW nibble is sub-block 2g's own value, the HIGH nibble 2g+1's
        nib = np.where(j % 2 == 0, np.mod(q, 16.0), np.floor_divide(q, 16.0)).astype(np.float32)
        out[:, j, :] = (d[:, None] * sc[:, j, None].astype(np.float32)) * nib \
                       - (dmin[:, None] * mn[:, j, None].astype(np.float32))
    return out.reshape(-1)


def bits(vals):
    return struct.unpack("<%dq" % len(vals), np.asarray(vals, dtype="<f8").tobytes())


def main():
    if not os.path.exists(GGUF):
        print("Q4K_REF_SKIP no model on disk")
        return 0
    if not os.path.exists(INVENTORY):
        print("Q4K_REF_SKIP no inventory")
        return 0

    rows = []
    for line in open(INVENTORY):
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) >= 7 and f[3] == "Q4_K":
            rows.append((int(f[6]), f[0], int(f[5]), int(f[6])))
    rows.sort()                                             # smallest first keeps the run quick
    want = int(os.environ.get("VYBFORGE_Q4K_TENSORS", "2"))
    chosen = rows[:want]
    nums = int(os.environ.get("VYBFORGE_Q4K_NUMS", "4096"))
    if not chosen:
        print("Q4K_REF_SKIP no Q4_K tensors in the inventory")
        return 0

    try:
        from gguf.quants import dequantize
        from gguf.constants import GGMLQuantizationType
    except Exception as e:      # noqa: BLE001
        dequantize = None
        print(f"Q4K_REF_WARN gguf package unavailable ({e.__class__.__name__}); numpy is the only authority")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    agreed = differed = 0
    with open(GGUF, "rb") as fh, open(OUT, "w") as out:
        for nbytes, name, off, _ in chosen:
            fh.seek(off)
            raw = fh.read(nbytes)
            if len(raw) != nbytes:
                print(f"Q4K_REF_FAIL short read {name}")
                return 1
            ours = dequant_q4k(raw)
            indep = "no indep authority"
            if dequantize is not None:
                theirs = dequantize(np.frombuffer(raw, dtype=np.uint8), GGMLQuantizationType.Q4_K)
                same = np.array_equal(ours, theirs)
                if same:
                    agreed += 1
                    indep = "identical to gguf package"
                else:
                    differed += 1
                    bad = int(np.argmax(ours != theirs))
                    indep = f"DIFFERS at {bad}: ours={ours[bad]!r} gguf={theirs[bad]!r}"
            out.write(f"Q4_K {name}@{off} -> {' '.join(str(v) for v in bits(ours[:nums]))}\n")
            print(f"Q4K_REF {name} blocks={nbytes // BLOCK} numel={ours.size} [{indep}]")

    print(f"Q4K_REF_TENSORS={len(chosen)}")
    print(f"Q4K_REF_INDEP identical={agreed} differing={differed}")
    if differed:
        print("Q4K_REF_FAIL our port disagrees with the gguf package")
        return 1
    if agreed == 0:
        print("Q4K_REF_WARN no independent authority available")
    print(f"Q4K_REF_WROTE {OUT}")
    print("Q4K_REF_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
