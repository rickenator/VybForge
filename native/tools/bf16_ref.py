#!/usr/bin/env python3
"""BF16 dequant reference (VybForge#10 phase 3 — the vision tower's weight type).

BF16 is not a block quant: 2 bytes per element, and the value is exactly the top 16 bits of an
f32. In Ridge it appears only in the mmproj vision tower (110 tensors, 0.85 GiB), so the tensors
come from native/gguf/mmproj-bf16-inventory.tsv.

Our numpy conversion is checked against the INDEPENDENT python `gguf` package (which does
implement BF16), on real tensors, before either is used to judge the GPU kernel. The kernel itself
uses the language's own `ld_bf16`, which is the same conversion.

Writes native/out/bf16_ref.txt:
    BF16 <tensor>@<offset> -> <every value of the checked slice>
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TSV = os.path.join(REPO, "native/gguf/mmproj-bf16-inventory.tsv")
GGUF = os.environ.get(
    "VYBFORGE_MMPROJ_GGUF",
    os.path.expanduser("~/Models/qwen38-27b-ridge/mmproj-Qwen3.8-27B-BF16.gguf"),
)
OUT = os.path.join(REPO, "native/out/bf16_ref.txt")

ELEM_BYTES = 2
SLICE_ELEMS = 4096      # must match native/host/bf16_load_driver.vyb


def bf16_ours(raw):
    """bf16 -> f64: zero-extend the 16 bits, shift into the top half of an f32, reinterpret."""
    u16 = np.frombuffer(raw, "<u2").astype(np.uint32)
    f32 = (u16 << 16).view(np.float32)
    return f32.astype(np.float64)


def bf16_gguf(raw):
    """The independent implementation, when the gguf package is importable."""
    try:
        import gguf
        q = gguf.GGMLQuantizationType.BF16
    except Exception:
        return None
    try:
        return np.asarray(gguf.quants.dequantize(np.frombuffer(raw, np.uint8), q),
                          dtype=np.float64).reshape(-1)
    except Exception as exc:
        print(f"BF16_GGUF_ERR {type(exc).__name__}: {exc}")
        return None


def parse_inventory():
    tens = {}
    with open(TSV) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            toks = line.rstrip("\n").split("\t")
            if len(toks) < 7:
                continue
            name, dims, tid, _tn, _role, off, nbytes = toks[:7]
            tens[name] = dict(shape=[int(x) for x in dims.split("x")], type=int(tid),
                              off=int(off), bytes=int(nbytes))
    return tens


def main():
    if not os.path.exists(GGUF):
        print(f"BF16_REF_SKIP model not found: {GGUF}")
        return 0
    tens = parse_inventory()
    names = [n for n, t in tens.items() if t["type"] == 30]
    if not names:
        print("BF16_REF_FAIL inventory has no type-30 tensors")
        return 1
    chosen = []
    for want in ("v.blk.0.attn_out.weight", "v.blk.0.ffn_down.weight", "v.blk.3.attn_qkv.weight"):
        if want in names:
            chosen.append(want)
    for n in names:
        if len(chosen) >= 3:
            break
        if n not in chosen:
            chosen.append(n)

    agreed = differed = 0
    with open(OUT, "w") as fh:
        for name in chosen:
            t = tens[name]
            nbytes = min(SLICE_ELEMS * ELEM_BYTES, t["bytes"])
            with open(GGUF, "rb") as f:
                f.seek(t["off"])
                raw = f.read(nbytes)
            if len(raw) != nbytes:
                print(f"BF16_REF_FAIL short read {name}")
                return 1
            ours = bf16_ours(raw)

            theirs = bf16_gguf(raw)
            if theirs is None:
                indep = "gguf-package=unavailable"
            elif theirs.size != ours.size:
                indep = f"gguf-package=SIZE {theirs.size} vs {ours.size}"
                differed += 1
            else:
                md = float(np.max(np.abs(theirs - ours)))
                if md == 0.0:
                    indep = "gguf-package=IDENTICAL"
                    agreed += 1
                else:
                    indep = f"gguf-package=DIFFERS maxabs={md:.3e}"
                    differed += 1

            vals = " ".join(f"{v:.17g}" for v in ours)
            fh.write(f"BF16 {name}@{t['off']} -> {vals}\n")
            print(f"BF16_REF {name} numel={ours.size} {indep}")

    print(f"BF16_REF_TENSORS={len(chosen)}")
    print(f"BF16_REF_INDEP identical={agreed} differing={differed}")
    if differed:
        print("BF16_REF_FAIL our conversion disagrees with the gguf package")
        return 1
    if agreed == 0:
        print("BF16_REF_WARN no independent implementation available")
    print(f"BF16_REF_WROTE {OUT}")
    print("BF16_REF_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
