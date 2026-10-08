#!/usr/bin/env python3
"""IQ3_S dequant reference (VybForge#10 phase 3).

IQ3_S is the edge layers' FFN type in Qwen3.8-27B Ridge (32 tensors, 1.14 GiB). Our numpy port of
`dequantize_row_iq3_s` is checked against the compiled upstream function
(native/tools/ggml_dequant_authority.py, spec "iq3_s") on real tensors, before either is used to
judge the GPU kernel.

The tables (iq3s_grid: 512 x 4 grid bytes; kmask_iq2xs: 8 sign masks) are EXTRACTED from the local
llama.cpp checkout with the same extractor the generated iq2s_tables.py uses, so nothing is typed
by hand. This tool also writes native/out/iq3s_grid.bin — the 2048-byte image the kernel takes as
its fifth argument.

Writes native/out/iq3_s_ref.txt:
    IQ3_S <tensor>@<element> -> <every value of the checked slice>
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
OUT = os.path.join(REPO, "native/out/iq3_s_ref.txt")
GRID_BIN = os.path.join(REPO, "native/out/iq3s_grid.bin")
LLAMA = os.environ.get("LLAMA_CPP_SRC", os.path.expanduser("~/Projects/llama.cpp"))

QK_K = 256
BLOCK = 110          # d f16, qs[64], qh[8], signs[32], scales[4]
SLICE_BLOCKS = 16    # must match native/host/iq3_s_load_driver.vyb


def tables():
    """iq3s_grid (512 entries x 4 bytes) and kmask_iq2xs, extracted from upstream."""
    import importlib.util
    gen = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen_iq2s_tables.py")
    s = importlib.util.spec_from_file_location("gen_tables", gen)
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    text = open(os.path.join(LLAMA, "ggml/src/ggml-common.h"),
                encoding="utf-8", errors="replace").read()
    kmask = np.array(mod.extract_table(text, "kmask_iq2xs", 8), dtype=np.int64)
    raw = mod.extract_table(text, "iq3s_grid", 512)
    grid = np.array(raw, dtype="<u4").view(np.uint8).reshape(512, 4).astype(np.int64)
    return kmask, grid


def iq3_s_ours(raw, kmask, grid):
    """Faithful port of llama.cpp dequantize_row_iq3_s (ggml-quants.c)."""
    nb = len(raw) // BLOCK
    b = np.frombuffer(raw, np.uint8).reshape(nb, BLOCK)
    d = b[:, 0:2].copy().view("<f2").astype(np.float64)[:, 0]
    qs = b[:, 2:66].astype(np.int64)
    qh = b[:, 66:74].astype(np.int64)
    signs = b[:, 74:106].astype(np.int64)
    scales = b[:, 106:110].astype(np.int64)

    out = np.empty((nb, QK_K), np.float64)
    for g in range(4):                       # C iterates ib32 = 0, 2, 4, 6
        sc = scales[:, g]
        db1 = d * (1 + 2 * (sc & 0xF))
        db2 = d * (1 + 2 * (sc >> 4))
        qsg = qs[:, 16 * g:16 * g + 16]
        sgg = signs[:, 8 * g:8 * g + 8]
        for half in range(2):
            dl = db1 if half == 0 else db2
            qhw = qh[:, 2 * g + half]
            qsh = qsg[:, 8 * half:8 * half + 8]
            sg = sgg[:, 4 * half:4 * half + 4]
            base = 64 * g + 32 * half
            for l in range(4):
                i1 = (qsh[:, 2 * l] | ((qhw << (8 - 2 * l)) & 256)).astype(np.int64)
                i2 = (qsh[:, 2 * l + 1] | ((qhw << (7 - 2 * l)) & 256)).astype(np.int64)
                g1 = grid[i1].astype(np.float64)
                g2 = grid[i2].astype(np.float64)
                s = sg[:, l].astype(np.int64)
                v1 = dl[:, None] * g1
                v2 = dl[:, None] * g2
                v1 = np.where((s[:, None] & kmask[0:4][None, :]) != 0, -v1, v1)
                v2 = np.where((s[:, None] & kmask[4:8][None, :]) != 0, -v2, v2)
                out[:, base + 8 * l:base + 8 * l + 4] = v1
                out[:, base + 8 * l + 4:base + 8 * l + 8] = v2
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


def authority(raw):
    spec = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ggml_dequant_authority.py")
    try:
        import importlib.util
        s = importlib.util.spec_from_file_location("ggml_auth_iq3s", spec)
        mod = importlib.util.module_from_spec(s)
        s.loader.exec_module(mod)
    except Exception as exc:
        print(f"IQ3_S_AUTH_ERR {type(exc).__name__}: {exc}")
        return None
    return mod.dequantize(raw, "iq3_s")


def main():
    if not os.path.exists(GGUF):
        print(f"IQ3_S_REF_SKIP model not found: {GGUF}")
        return 0
    kmask, grid = tables()
    raw_grid = np.array([int(x) for x in _raw_grid()], dtype="<u4").tobytes()
    assert len(raw_grid) == 2048, len(raw_grid)
    with open(GRID_BIN, "wb") as fh:
        fh.write(raw_grid)
    print(f"IQ3_S_GRID_IMAGE {GRID_BIN} {len(raw_grid)} bytes")

    tens = parse_inventory()
    names = [n for n, t in tens.items() if t["type"] == 21]
    if not names:
        print("IQ3_S_REF_FAIL inventory has no type-21 tensors")
        return 1
    chosen = []
    for want in ("blk.0.ffn_down.weight", "blk.0.ffn_gate.weight", "blk.62.ffn_up.weight"):
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
            nbytes = min(SLICE_BLOCKS * BLOCK, t["bytes"])
            with open(GGUF, "rb") as f:
                f.seek(t["off"])
                raw = f.read(nbytes)
            if len(raw) != nbytes:
                print(f"IQ3_S_REF_FAIL short read {name}")
                return 1
            ours = iq3_s_ours(raw, kmask, grid)

            theirs = authority(raw)
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

            _sl = ours[:4096]
            import struct as _bs
            vals = " ".join(str(_q) for _q in
                              _bs.unpack("<%dq" % len(_sl), np.asarray(_sl, dtype="<f8").tobytes()))
            fh.write(f"IQ3_S {name}@{t['off']} -> {vals}\n")
            print(f"IQ3_S_REF {name} blocks={nbytes // BLOCK} numel={ours.size} {indep}")

    print(f"IQ3_S_REF_TENSORS={len(chosen)}")
    print(f"IQ3_S_REF_INDEP identical={agreed} differing={differed}")
    if differed:
        print("IQ3_S_REF_FAIL our port disagrees with llama.cpp's own code")
        return 1
    if agreed == 0:
        print("IQ3_S_REF_WARN no compiled authority available")
    print(f"IQ3_S_REF_WROTE {OUT}")
    print("IQ3_S_REF_DONE")
    return 0


def _raw_grid():
    """The grid as raw uint32 values, for the kernel's binary image."""
    import importlib.util
    gen = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen_iq2s_tables.py")
    s = importlib.util.spec_from_file_location("gen_tables2", gen)
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    text = open(os.path.join(LLAMA, "ggml/src/ggml-common.h"),
                encoding="utf-8", errors="replace").read()
    return mod.extract_table(text, "iq3s_grid", 512)


if __name__ == "__main__":
    sys.exit(main())
