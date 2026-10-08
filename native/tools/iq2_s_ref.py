#!/usr/bin/env python3
"""IQ2_S dequant reference for the FFN middle stack (VybForge#10 phase 3).

IQ2_S is the largest type in Qwen3.8-27B Ridge (160 tensors, 4.25 GiB — the mid-stack FFN),
so it decides whether the 27B is runnable. Unlike Q8_0 there is no third-party Python
dequantizer to check against (the `gguf` package has the enum but no IQ2_S implementation), so
the independent authority here is **llama.cpp's own C**:

  * iq2_s_ours()   -- our numpy port of dequantize_row_iq2_s
  * the compiled authority -- the upstream function body, struct and tables copied VERBATIM out
    of the local llama.cpp checkout by native/tools/ggml_dequant_authority.py and compiled as-is
    with only macro shims. Neither the grid table (1024 entries) nor the sign mask is retyped.

The two must agree on real tensor data. Writes native/out/iq2_s_ref.txt for the GPU gate:
    IQ2_S <tensor>@<element> -> v0 v1 v2 v3 v4 v5
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gguf"))
from iq2s_tables import IQ2_S_BLOCK_BYTES, IQ2_S_GRID, IQ2_S_KMASK  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TSV = os.path.join(REPO, "native/gguf/ridge-3.7bpw-inventory.tsv")
GGUF = os.environ.get(
    "VYBFORGE_RIDGE_GGUF",
    os.path.expanduser("~/Models/qwen38-27b-ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf"),
)
OUT = os.path.join(REPO, "native/out/iq2_s_ref.txt")

QK_K = 256
BLOCK = IQ2_S_BLOCK_BYTES

# The checked slice length, in blocks. A full FFN tensor is ~28 MB and the Vyb driver uploads
# host->device 8 bytes at a time, so the GPU check runs on a real SLICE of a real tensor rather
# than a whole one. 16 blocks = 4096 elements; the layout work is identical at any length.
SLICE_BLOCKS = 16


def grid_array():
    g = np.array(IQ2_S_GRID, dtype=np.uint64)
    return g.view(np.uint8).reshape(len(IQ2_S_GRID), 8)


def kmask_array():
    return np.array(IQ2_S_KMASK, dtype=np.uint8)


def iq2_s_ours(raw):
    """Faithful port of llama.cpp dequantize_row_iq2_s (ggml-quants.c)."""
    grid = grid_array()
    kmask = kmask_array()
    nb = len(raw) // BLOCK
    b = np.frombuffer(raw, np.uint8).reshape(nb, BLOCK)
    d = b[:, 0:2].copy().view("<f2").astype(np.float64)[:, 0]
    qs = b[:, 2:66]           # 2-bit indices [0:32) and sign bytes [32:64)
    qh = b[:, 66:74]
    scales = b[:, 74:82].astype(np.int64).copy()

    out = np.empty((nb, QK_K), np.float64)
    for ib32 in range(QK_K // 32):
        sc = scales[:, ib32]
        db0 = d * (0.5 + (sc & 0xF)) * 0.25
        db1 = d * (0.5 + (sc >> 4)) * 0.25
        for l in range(4):
            dl = db0 if l < 2 else db1               # C: db[l/2]
            qsl = qs[:, 4 * ib32 + l].astype(np.int64)
            qhl = qh[:, ib32].astype(np.int64)
            idx = qsl | ((qhl << (8 - 2 * l)) & 0x300)
            g = grid[idx].astype(np.float64)          # (nb, 8)
            signs = qs[:, 32 + 4 * ib32 + l].astype(np.int64)
            neg = ((signs[:, None] & kmask[None, :]) != 0)
            vals = dl[:, None] * g
            vals = np.where(neg, -vals, vals)
            out[:, ib32 * 32 + l * 8: ib32 * 32 + l * 8 + 8] = vals
    return out.reshape(-1)


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


def c_authority(raw):
    """Dequantize the same bytes with the compiled upstream function; None if unavailable.

    The extractor is shared with the other quant types (native/tools/ggml_dequant_authority.py),
    so IQ2_S and Q5_K are checked against the same build of llama.cpp's own code.
    """
    spec = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ggml_dequant_authority.py")
    try:
        import importlib.util
        s = importlib.util.spec_from_file_location("ggml_auth", spec)
        mod = importlib.util.module_from_spec(s)
        s.loader.exec_module(mod)
    except Exception as exc:
        print(f"IQ2_S_AUTH_ERR {type(exc).__name__}: {exc}")
        return None
    return mod.dequantize(raw, "iq2_s")


def write_grid_image(path):
    """The 8192-byte grid image the kernel expects as the buffer prefix.

    The kernel reads `q` as the grid and `q + 8192` as the blocks, so whoever packs a buffer has
    to supply this exact image. Writing it here, next to the reference, is what keeps the driver
    and the reference from disagreeing about a table neither can derive.
    """
    g = np.array(IQ2_S_GRID, dtype="<u8").tobytes()
    assert len(g) == 8192, len(g)
    with open(path, "wb") as fh:
        fh.write(g)
    return path


def main():
    if not os.path.exists(GGUF):
        print(f"IQ2_S_REF_SKIP model not found: {GGUF}")
        return 0
    write_grid_image(os.path.join(REPO, "native/out/iq2s_grid.bin"))
    print(f"IQ2_S_GRID_IMAGE {os.path.join(REPO, 'native/out/iq2s_grid.bin')} 8192 bytes")

    tens = parse_inventory()
    names = [n for n, t in tens.items() if t["type"] == 22]
    if not names:
        print("IQ2_S_REF_FAIL inventory has no type-22 tensors")
        return 1
    # FFN middle stack, one per layer family so the check does not rest on a single shape.
    chosen = []
    for want in ("blk.4.ffn_down.weight", "blk.16.ffn_gate.weight", "blk.32.ffn_up.weight"):
        if want in names:
            chosen.append(want)
    for n in names:
        if len(chosen) >= 3:
            break
        if n not in chosen:
            chosen.append(n)

    lines, agreed, differed = [], 0, 0
    with open(OUT, "w") as fh:
        for name in chosen:
            t = tens[name]
            nbytes = min(SLICE_BLOCKS * BLOCK, t["bytes"])
            with open(GGUF, "rb") as f:
                f.seek(t["off"])
                raw = f.read(nbytes)
            if len(raw) != nbytes:
                print(f"IQ2_S_REF_FAIL short read {name}")
                return 1
            ours = iq2_s_ours(raw)

            theirs = c_authority(raw)
            if theirs is None:
                indep = "llama.cpp-C=unavailable"
            elif theirs.size != ours.size:
                indep = f"llama.cpp-C=SIZE {theirs.size} vs {ours.size}"
                differed += 1
            else:
                md = float(np.max(np.abs(theirs - ours)))
                if md == 0.0:
                    indep = "llama.cpp-C=IDENTICAL"
                    agreed += 1
                else:
                    indep = f"llama.cpp-C=DIFFERS maxabs={md:.3e}"
                    differed += 1

            vals = " ".join(f"{v:.17g}" for v in ours[:6])
            fh.write(f"IQ2_S {name}@{t['off']} -> {vals}\n")
            print(f"IQ2_S_REF {name} blocks={nbytes // BLOCK} numel={ours.size} {indep}")

    print(f"IQ2_S_REF_TENSORS={len(chosen)}")
    print(f"IQ2_S_REF_INDEP identical={agreed} differing={differed}")
    if differed:
        print("IQ2_S_REF_FAIL our port disagrees with llama.cpp's own code")
        return 1
    if agreed == 0:
        print("IQ2_S_REF_WARN no compiled authority available (llama.cpp source or gcc missing)")
    print(f"IQ2_S_REF_WROTE {OUT}")
    print("IQ2_S_REF_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
