#!/usr/bin/env python3
"""dtcvt_ref.py — independent reference for the S0.1 dtype-conversion gate.

Two jobs, and it is deliberately NOT a wrapper around the Vyb path:

  gen <case> <outdir>          write <case>.in + <case>.exp
  explain <case> <outdir> <vyb-output-file>
                               classify the mismatches the Vyb side reported

The authority for a conversion is torch (`Tensor.to(torch.bfloat16|torch.float16)`), which
is the framework whose numerics the model has to match. numpy does the bit machinery
(view/frombuffer) because it is the trustworthy way to reinterpret bytes. torch is checked
for RNE at import time (`self_check`) rather than trusted: an oracle that has drifted must
say so instead of silently redefining "correct". When torch is unavailable the narrow
expectations fall back to a software round-to-nearest-even implemented here from the IEEE
rule (documented, and independent of the compiler under test); the chosen authority is
named in the output either way.

Cases (mode is the kernel's mode):
  widen_f16_all    0  every one of the 65536 f16 encodings -> f32
  widen_bf16_all   1  every one of the 65536 bf16 encodings -> f32
  narrow_f16_struct 2 f32 built from every (sign,exp,mtop10) x the five low-13 classes
                      that decide the rounding, so every rounding branch is covered
  narrow_bf16_struct 3 the same construction for bf16's 7-bit mantissa (low-16 classes)
  narrow_rand      2/3  seeded random f32 (both narrow modes)
  real_bf16        1  the first bytes of a REAL bf16 tensor from a real checkpoint
  real_f16         0  real f16 bytes from a real checkpoint (skipped when absent)
"""

import json
import os
import struct
import sys

import numpy as np

try:
    import torch
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

TENSOR_PATH = os.environ.get(
    "VYBFORGE_DTCVT_REAL_BF16",
    os.path.expanduser("~/Models/spikingbrain-v1-7b-base/pytorch_model-00001.bin"))
F16_PATH = os.environ.get(
    "VYBFORGE_DTCVT_REAL_F16",
    "/usr/export/LLM/qwen2.5-coder-32B-instruct-awq/model-00001-of-00005.safetensors")

LOW13 = (0x0000, 0x0FFF, 0x1000, 0x1001, 0x1FFF)   # bf16-ish decision classes for f16
LOW16 = (0x0000, 0x7FFF, 0x8000, 0x8001, 0xFFFF)   # every decision class for bf16
REAL_BYTES = 8 << 20


def self_check():
    """Assert the authority really is round-to-nearest-even before it is used as truth."""
    if not HAVE_TORCH:
        print("DTCVT_REF_NOTE authority: software RNE (no torch)")
        return
    # 1 + 2^-8 is exactly halfway between 1 and the next bf16 value (1 + 2^-7):
    # RNE must go to even => 1.0. The value just above it must go up.
    halfway = torch.tensor([1.0 + 2 ** -8], dtype=torch.float32)
    above = torch.tensor([1.0 + 2 ** -8 + 2 ** -20], dtype=torch.float32)
    a = halfway.to(torch.bfloat16).view(torch.uint16).item()
    b = above.to(torch.bfloat16).view(torch.uint16).item()
    even_bits = torch.tensor([1.0], dtype=torch.float32).to(torch.bfloat16).view(torch.uint16).item()
    if a != even_bits or b == even_bits:
        print("DTCVT_REF_FAIL: torch bfloat16 cast is not round-to-nearest-even")
        sys.exit(3)
    print("DTCVT_REF_NOTE authority: torch %s, bfloat16 cast verified round-to-nearest-even"
          % torch.__version__)


def rne_bf16_software(bits):
    """Round-to-nearest-even of an f32 bit pattern to bf16, from the IEEE rule."""
    bits = bits.astype(np.uint64)
    lsb = (bits >> 16) & 1
    rounding = 0x7FFF + lsb
    rounded = (bits + rounding) >> 16
    return (rounded & 0xFFFF).astype(np.uint16)


def rne_f16_software(bits):
    """Round-to-nearest-even of an f32 bit pattern to f16 (spec-derived, handles the
    exponent-range cases by arithmetic rather than by table)."""
    x = bits.view(np.float32).astype(np.float64)
    out = np.zeros(x.shape, dtype=np.uint16)
    # normal range
    sign = ((bits >> 31) & 1).astype(np.uint64)
    absx = np.abs(x)
    with np.errstate(over="ignore", invalid="ignore"):
        expo = np.floor(np.log2(np.where(absx > 0, absx, 1.0)))
        mant = absx / np.exp2(expo) - 1.0
    e16 = expo.astype(np.int64) + 15
    m10 = np.rint(mant * 1024.0).astype(np.int64)
    carry = (m10 >= 1024).astype(np.int64)
    m10 = np.where(m10 >= 1024, 0, m10)
    e16 = e16 + carry
    e16 = np.where(m10 == 1024, e16 + 1, e16)
    normal = (e16 >= 1) & (e16 <= 30)
    out = np.where(normal, (e16.astype(np.uint64) << 10) | m10.astype(np.uint64), 0).astype(np.uint16)
    return (out | (sign.astype(np.uint16) << 15)).astype(np.uint16)


def expect_bf16(bits_u32):
    """f32 bits -> bf16 bits."""
    if HAVE_TORCH:
        t = torch.from_numpy(bits_u32.view(np.float32).copy()).to(torch.bfloat16)
        return t.view(torch.uint16).numpy().copy()
    return rne_bf16_software(bits_u32)


def expect_f16(bits_u32):
    """f32 bits -> f16 bits."""
    if HAVE_TORCH:
        t = torch.from_numpy(bits_u32.view(np.float32).copy()).to(torch.float16)
        return t.view(torch.uint16).numpy().copy()
    return rne_f16_software(bits_u32)


def widen_f16(u16):
    """f16 bits -> f32 bits (exact), via numpy's own conversion.

    NOT `u16 << 16`: that is the bf16 widening. f16 needs the exponent rebased (5 bits to
    8) and the 10-bit mantissa shifted by 13. Getting this wrong made an earlier version of
    this reference report 63486 phantom f16-widening failures — the Vyb side was right.
    NaNs are canonicalised differently by different implementations, which is why the gate
    classifies NaN inputs separately instead of demanding byte equality on them.
    """
    return np.frombuffer(u16.astype("<u2").tobytes(), dtype="<f2").astype("<f4").view(np.uint32)


def widen_bf16(u16):
    return u16.astype(np.uint32) << 16


def read_real_bf16():
    """First REAL_BYTES of a real bf16 tensor: shard 1 of the SpikingBrain torch .bin
    checkpoint holds exactly one storage (data/0 = model.embeddings.weight)."""
    import zipfile
    if not os.path.exists(TENSOR_PATH):
        return None
    with zipfile.ZipFile(TENSOR_PATH) as z:
        names = [n for n in z.namelist() if n.endswith("/data/0")]
        if not names:
            return None
        with z.open(names[0]) as f:
            return f.read(REAL_BYTES)


def read_real_f16():
    """Real f16 bytes: first F16 tensor in the AWQ shard, via the safetensors header."""
    if not os.path.exists(F16_PATH):
        return None
    with open(F16_PATH, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            if meta.get("dtype") != "F16":
                continue
            lo, hi = meta["data_offsets"]
            take = min(REAL_BYTES, hi - lo)
            f.seek(base + lo)
            return f.read(take)
    return None


def gen(case, outdir):
    os.makedirs(outdir, exist_ok=True)
    inp = os.path.join(outdir, case + ".in")
    exp = os.path.join(outdir, case + ".exp")
    mode = None

    if case == "widen_f16_all":
        mode = 0
        u16 = np.arange(65536, dtype=np.uint16)
        ib = u16.astype("<u2").tobytes()
        eb = widen_f16(u16).astype("<u4").tobytes()
    elif case == "widen_bf16_all":
        mode = 1
        u16 = np.arange(65536, dtype=np.uint16)
        ib = u16.astype("<u2").tobytes()
        eb = widen_bf16(u16).astype("<u4").tobytes()
    elif case == "narrow_f16_struct":
        mode = 2
        top = np.arange(1 << 19, dtype=np.uint32)
        bits = ((top[:, None] << np.uint32(13)) | np.array(LOW13, dtype=np.uint32)[None, :]).ravel()
        ib = bits.astype("<u4").tobytes()
        eb = expect_f16(bits).astype("<u2").tobytes()
    elif case == "narrow_bf16_struct":
        mode = 3
        top = np.arange(1 << 16, dtype=np.uint32)
        bits = ((top[:, None] << np.uint32(16)) | np.array(LOW16, dtype=np.uint32)[None, :]).ravel()
        ib = bits.astype("<u4").tobytes()
        eb = expect_bf16(bits).astype("<u2").tobytes()
    elif case == "narrow_rand":
        mode = 2
        rng = np.random.default_rng(1234)
        bits = rng.integers(0, 1 << 32, size=2_000_000, dtype=np.uint32)
        ib = bits.astype("<u4").tobytes()
        np.save(os.path.join(outdir, "narrow_rand.bits.npy"), bits)
        eb = expect_f16(bits).astype("<u2").tobytes()
    elif case == "narrow_rand_bf16":
        mode = 3
        p = os.path.join(outdir, "narrow_rand.bits.npy")
        if not os.path.exists(p):
            print("DTCVT_REF_FAIL: run narrow_rand first")
            return 1
        bits = np.load(p)
        ib = bits.astype("<u4").tobytes()
        eb = expect_bf16(bits).astype("<u2").tobytes()
    elif case == "real_bf16":
        mode = 1
        raw = read_real_bf16()
        if raw is None:
            print("DTCVT_REF_SKIP real_bf16: no checkpoint at %s" % TENSOR_PATH)
            return 2
        raw = raw[: len(raw) - (len(raw) % 2)]
        u16 = np.frombuffer(raw, dtype="<u2")
        ib = raw
        eb = widen_bf16(u16).astype("<u4").tobytes()
    elif case == "real_f16":
        mode = 0
        raw = read_real_f16()
        if raw is None:
            print("DTCVT_REF_SKIP real_f16: no f16 tensor found at %s" % F16_PATH)
            return 2
        raw = raw[: len(raw) - (len(raw) % 2)]
        u16 = np.frombuffer(raw, dtype="<u2")
        ib = raw
        eb = widen_f16(u16).astype("<u4").tobytes()
    else:
        print("DTCVT_REF_FAIL: unknown case %s" % case)
        return 1

    with open(inp, "wb") as f:
        f.write(ib)
    with open(exp, "wb") as f:
        f.write(eb)
    n = len(ib) // (2 if mode < 2 else 4)
    print("DTCVT_REF_OK case=%s mode=%d n=%d in=%s exp=%s in_bytes=%d exp_bytes=%d"
          % (case, mode, n, inp, exp, len(ib), len(eb)))
    return 0


def explain(case, outdir, vyb_output):
    """Classify the mismatches against the mechanism, so a red gate names a cause.

    The category TOTALS come from the kernel, which classifies every element on the device.
    This function independently classifies the reported examples against the same rule, so a
    disagreement between the two would show up rather than hide.

    The rule (IEEE 754 leaves NaN payload propagation on a conversion
    implementation-defined, so a payload difference is not a value error):
      nan_payload     input NaN, output NaN, bytes differ      -> information
      nan_to_nonnan   input NaN, output NOT NaN                -> FAIL (a NaN became a number)
      truncation      input not NaN, output == round-toward-zero -> FAIL (st_bf16's mechanism)
      value           input not NaN, any other difference       -> FAIL
    """
    inp = os.path.join(outdir, case + ".in")
    if not (os.path.exists(inp) and os.path.exists(vyb_output)):
        print("DTCVT_EXPLAIN_FAIL: missing inputs")
        return 1
    totals = {}
    for line in open(vyb_output):
        if line.startswith("DTCVT_OK"):
            for key in ("mode", "n", "mismatches", "nan_payload", "nan_to_nonnan", "value_diff"):
                if key + "=" in line:
                    totals[key] = int(line.split(key + "=")[1].split()[0])
    if "mode" not in totals:
        print("DTCVT_EXPLAIN_FAIL: no DTCVT_OK line")
        return 1
    mode = totals["mode"]
    raw = np.frombuffer(open(inp, "rb").read(), dtype="<u4" if mode >= 2 else "<u2")
    categories = {}
    shown = 0
    for line in open(vyb_output):
        if not line.startswith("DTCVT_MISMATCH"):
            continue
        i = int(line.split("i=")[1].split()[0])
        got = int(line.split("got=")[1].split()[0], 16)
        want = int(line.split("exp=")[1].split()[0], 16)
        word = int(raw[i])
        if mode >= 2:
            src_nan = (word & 0x7FFFFFFF) > 0x7F800000
            got_nan = (got & 0x7FFF) > (0x7C00 if mode == 2 else 0x7F80)
            truncated = mode == 3 and got == ((word >> 16) & 0xFFFF)
        else:
            src_nan = (word & 0x7FFF) > (0x7C00 if mode == 0 else 0x7F80)
            got_nan = (got & 0x7FFFFFFF) > 0x7F800000
            truncated = False
        if src_nan and got_nan:
            cat = "nan_payload"
        elif src_nan:
            cat = "nan_to_nonnan"
        elif truncated:
            cat = "truncation"
        else:
            cat = "value"
        categories[cat] = categories.get(cat, 0) + 1
        if shown < 8:
            shown += 1
            print("  i=%d in=0x%04x got=0x%04x exp=0x%04x  [%s]"
                  % (i, word, got, want, cat))
    print("DTCVT_EXPLAIN case=%s mode=%d n=%d mismatches=%d nan_payload=%d nan_to_nonnan=%d "
          "value_diff=%d examples=%s"
          % (case, mode, totals["n"], totals["mismatches"], totals["nan_payload"],
             totals["nan_to_nonnan"], totals["value_diff"], json.dumps(categories, sort_keys=True)))
    # Scan the whole corpus for the hazard class, so the report can name a concrete input
    # rather than a count: a NaN whose only set mantissa bits are in the discarded half
    # truncates to a plain Infinity.
    if mode == 3:
        bits32 = raw.astype(np.uint32)
        src_nan = (bits32 & 0x7FFFFFFF) > 0x7F800000
        trunc = (bits32 >> 16) & 0xFFFF
        trunc_nan = (trunc & 0x7FFF) > 0x7F80
        hazard = src_nan & (~trunc_nan)
        idx = np.nonzero(hazard)[0]
        if len(idx):
            i = int(idx[0])
            print("DTCVT_EXPLAIN_HAZARD count=%d example i=%d in=0x%08x -> bf16 0x%04x (%s)"
                  % (len(idx), i, int(bits32[i]), int(trunc[i]),
                     "Inf" if (int(trunc[i]) & 0x7FFF) == 0x7F80 else "number"))
        else:
            print("DTCVT_EXPLAIN_HAZARD count=0")
    return 0


def checktable(vyb_table_file, root):
    """Prove the dtype module's names/widths agree with the container readers' tables.

    The loaders in native/torchload carry their own DTSIZE dicts (they had to, to classify
    before this module existed). They must not drift: one place names dtypes. Reads the
    DTCVT_DT lines the driver printed and compares against every DTSIZE found in the
    loader references.
    """
    import re
    want = {}
    for rel in ("native/torchload/st_ref.py", "native/torchload/tb_pkl_ref.py"):
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            continue
        src = open(path).read()
        m = re.search(r"DTSIZE\s*=\s*\{(.*?)\}", src, re.S)
        if not m:
            continue
        for k, v in re.findall(r'"([A-Za-z0-9_]+)"\s*:\s*(\d+)', m.group(1)):
            want[k] = int(v)
    got = {}
    for line in open(vyb_table_file):
        if not line.startswith("DTCVT_DT "):
            continue
        parts = dict(p.split("=", 1) for p in line.split()[1:])
        got[parts["name"]] = int(parts["bytes"])
    if not want:
        print("DTCVT_TABLE_SKIP: no DTSIZE tables found in the loader references")
        return 2
    missing = sorted(k for k in want if k not in got)
    wrong = sorted((k, want[k], got[k]) for k in want if k in got and got[k] != want[k])
    if missing or wrong:
        print("DTCVT_TABLE_FAIL missing=%s width_mismatch=%s" % (missing, wrong))
        return 1
    print("DTCVT_TABLE_OK %d names agree with the loader tables (dtype module == torchload)"
          % len(want))
    return 0


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    cmd = sys.argv[1]
    if cmd == "gen":
        self_check()
        return gen(sys.argv[2], sys.argv[3])
    if cmd == "explain":
        if len(sys.argv) < 5:
            print(__doc__)
            return 1
        return explain(sys.argv[2], sys.argv[3], sys.argv[4])
    if cmd == "checktable":
        return checktable(sys.argv[2], sys.argv[3])
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
