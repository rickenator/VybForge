#!/usr/bin/env python3
"""Q8_0 dequant reference for the Gated-DeltaNet state path (VybForge#10 phase 3).

Reads REAL Q8_0 tensors out of the Qwen3.8-27B Ridge GGUF using the checked-in inventory
(native/gguf/ridge-3.7bpw-inventory.tsv) and dequantizes each one TWO ways:

  * q8_0_ours()   -- our numpy, straight from the ggml block layout (34 B: d f16, qs[32] int8)
  * gguf.quants.dequantize()  -- the python gguf package, an INDEPENDENT implementation

The two must agree to the last bit. A reference that only agrees with itself would make the
GPU kernel's comparison meaningless, which is the lesson from #22 (compare against something
that shares no code with you).

Writes native/out/q8_0_ref.txt, one line per tensor, for comparison with the kernel:
    Q8_0 <tensor> -> v0 v1 v2 v3 v4 v5
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TSV = os.path.join(REPO, "native/gguf/ridge-3.7bpw-inventory.tsv")
GGUF = os.environ.get(
    "VYBFORGE_RIDGE_GGUF",
    os.path.expanduser("~/Models/qwen38-27b-ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf"),
)
OUT = os.path.join(REPO, "native/out/q8_0_ref.txt")

QK8_0 = 32
BLOCK = 34


def parse_inventory():
    """name -> dict(shape, type, off, bytes) from the checked-in inventory."""
    tens = {}
    with open(TSV) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            toks = line.rstrip("\n").split("\t")
            if len(toks) < 7:
                continue
            name, dims, tid, _tname, _role, off, nbytes = toks[:7]
            tens[name] = dict(shape=[int(x) for x in dims.split("x")],
                              type=int(tid), off=int(off), bytes=int(nbytes))
    return tens


def q8_0_ours(raw):
    """block_q8_0: d f16 @0, qs[32] int8 @2; per-block scale only, signed quants."""
    nb = len(raw) // BLOCK
    b = np.frombuffer(raw, np.uint8).reshape(nb, BLOCK)
    d = b[:, 0:2].copy().view("<f2").astype(np.float64)[:, 0]
    qs = b[:, 2:BLOCK].copy().view(np.int8).astype(np.float64)
    return (qs * d[:, None]).reshape(-1)


def q8_0_gguf(raw):
    """The independent implementation, when the gguf package is importable."""
    try:
        import gguf  # noqa: PLC0415
    except Exception:
        return None
    try:
        q = gguf.GGMLQuantizationType.Q8_0
    except AttributeError:
        return None
    try:
        return np.asarray(gguf.quants.dequantize(np.frombuffer(raw, np.uint8), q),
                          dtype=np.float64).reshape(-1)
    except Exception as exc:  # a version difference must be visible, not silent
        print(f"Q8_0_GGUF_ERR {type(exc).__name__}: {exc}")
        return None


def main():
    if not os.path.exists(GGUF):
        print(f"Q8_0_REF_SKIP model not found: {GGUF}")
        return 0
    if not os.path.exists(TSV):
        print(f"Q8_0_REF_FAIL inventory not found: {TSV}")
        return 1

    tens = parse_inventory()
    names = [n for n, t in tens.items() if t["type"] == 8]
    if not names:
        print("Q8_0_REF_FAIL inventory has no type-8 tensors")
        return 1

    # The GDN state path first (that is what this type is for here), then a couple more so the
    # check does not rest on one shape.
    prefer = [n for n in ("blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight") if n in names]
    rest = [n for n in names if n not in prefer]
    chosen = (prefer + rest)[:4]

    lines = []
    agreed = mismatched = 0
    with open(OUT, "w") as fh:
        for name in chosen:
            t = tens[name]
            with open(GGUF, "rb") as f:
                f.seek(t["off"])
                raw = f.read(t["bytes"])
            if len(raw) != t["bytes"]:
                print(f"Q8_0_REF_FAIL short read {name}: {len(raw)} != {t['bytes']}")
                return 1
            ours = q8_0_ours(raw)
            assert ours.size == int(np.prod(t["shape"])), (ours.size, t["shape"])

            theirs = q8_0_gguf(raw)
            if theirs is None:
                indep = "gguf-package=unavailable"
            else:
                if theirs.size != ours.size:
                    indep = "gguf-package=SIZE-MISMATCH"
                elif np.array_equal(theirs, ours):
                    indep = "gguf-package=IDENTICAL"
                    agreed += 1
                else:
                    d = float(np.max(np.abs(theirs - ours)))
                    indep = f"gguf-package=DIFFERS max|d|={d:.3e}"
                    mismatched += 1

            vals = " ".join(f"{v:.17g}" for v in ours[:6])
            line = f"Q8_0 {name} -> {vals}"
            fh.write(line + "\n")
            lines.append(line)
            print(f"Q8_0_REF {name} type={t['type']} numel={ours.size} bytes={t['bytes']} {indep}")

    print(f"Q8_0_REF_TENSORS={len(chosen)}")
    print(f"Q8_0_REF_INDEP identical={agreed} differing={mismatched}")
    if mismatched:
        print("Q8_0_REF_FAIL our layout disagrees with the gguf package")
        return 1
    if agreed == 0:
        # No independent authority was available: say so rather than implying agreement.
        print("Q8_0_REF_WARN no independent dequantizer available (gguf package missing)")
    print(f"Q8_0_REF_WROTE {OUT}")
    print("Q8_0_REF_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
