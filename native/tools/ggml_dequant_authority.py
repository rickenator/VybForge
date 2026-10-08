#!/usr/bin/env python3
"""Compile llama.cpp's own ggml dequant functions as the independent authority for a quant type.

Used by the Ridge refs (`iq2_s_ref.py`, `q5k_ref.py`) to check our numpy ports against the code
llama.cpp actually runs, rather than against a paraphrase of it. For each type a SPEC names:

  * the block struct and its byte size,
  * the functions to extract (a builder function BEFORE its caller, when one is needed),
  * any lookup tables the type needs (extracted verbatim, never retyped).

Nothing but macro shims is added, so a disagreement means OUR port is wrong.

    python3 native/tools/ggml_dequant_authority.py            # build every spec
    dequantize(raw_bytes, "q5_K")                             # -> numpy float64
"""
import hashlib
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.environ.get("LLAMA_CPP_SRC", os.path.expanduser("~/Projects/llama.cpp"))
COMMON_H = os.path.join(SRC, "ggml/src/ggml-common.h")
QUANTS_C = os.path.join(SRC, "ggml/src/ggml-quants.c")
OUTDIR = os.path.join(REPO, "native/out")

QK_K = 256

SPECS = {
    # 2-bit with a 1024-entry codebook; qs carries 2-bit indices then sign bits.
    "iq2_s": dict(
        struct="block_iq2_s", block=82,
        funcs=["dequantize_row_iq2_s"],
        tables=[("kmask_iq2xs", 8), ("iq2s_grid", 1024)],
    ),
    # 6-bit K-quant: scales/mins packed 6-bit in scales[12]; needs the shared bit-unpacker.
    "q5_K": dict(
        struct="block_q5_K", block=176,
        funcs=["get_scale_min_k4", "dequantize_row_q5_K"],
        tables=[],
    ),
}

SHIMS = r"""
// ---- shims only: macros the upstream text expects from ggml's own headers ----
#define QK_K 256
#define K_SCALE_SIZE 12
#define GGML_RESTRICT
#define GGML_EXTENSION              // q5_K's scales/mins live in an anonymous union
// ggml tags its anonymous aggregates for MSVC compatibility; left defined, the union stops being
// anonymous and `x[i].d` no longer resolves — so these must expand to nothing.
#define GGML_COMMON_AGGR_S
#define GGML_COMMON_AGGR_U
typedef uint16_t ggml_half;
// A shim must preserve LAYOUT, not merely compile: this union aliases {half d; half dmin;} with
// a packed pair, so the pair has to be exactly 4 bytes for the member offsets to stay put.
typedef uint32_t ggml_half2;
#define GGML_TABLE_BEGIN(type, name, size) static const type name[size] = {
#define GGML_TABLE_END() };
static float GGML_FP16_TO_FP32(ggml_half h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, bits;
    if (exp == 0) {
        if (man == 0) { bits = sign; }
        else { exp = 127 - 15 + 1; while ((man & 0x400u) == 0) { man <<= 1; exp--; } man &= 0x3FFu;
               bits = sign | (exp << 23) | (man << 13); }
    } else if (exp == 31) { bits = sign | 0x7F800000u | (man << 13); }
    else { bits = sign | ((exp + 112u) << 23) | (man << 13); }
    float f; memcpy(&f, &bits, 4); return f;
}
"""

MAIN = r"""
// ---- harness main: dequantize a raw block dump, print every value ----
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <raw-block-file> <nblocks>\n", argv[0]); return 2; }
    long nb = atol(argv[2]);
    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "open failed\n"); return 3; }
    FILE *sz = NULL; (void)sz;
    size_t nbytes = (size_t)nb * BLOCK_BYTES;
    unsigned char *raw = (unsigned char *)malloc(nbytes);
    if (!raw) return 4;
    if (fread(raw, 1, nbytes, f) != nbytes) { fprintf(stderr, "short read\n"); return 5; }
    fclose(f);
    float *y = (float *)malloc((size_t)nb * QK_K * sizeof(float));
    if (!y) return 6;
    DEQUANT_CALL(raw, y, (int64_t)nb * QK_K);
    for (long i = 0; i < nb * QK_K; ++i) printf("%.17g\n", (double)y[i]);
    return 0;
}
"""


def _extract_table(text, name, count):
    m = re.search(r"GGML_TABLE_BEGIN\([^,]+,\s*" + re.escape(name) + r",\s*" + str(count) + r"\)",
                  text)
    if not m:
        raise RuntimeError(f"table {name} not found")
    end = text.find("GGML_TABLE_END()", m.end())
    if end < 0:
        raise RuntimeError(f"table {name}: no GGML_TABLE_END")
    return text[m.start():end + len("GGML_TABLE_END()")]


def _extract_function(text, signature):
    start = text.find(signature)
    if start < 0:
        raise RuntimeError(f"function {signature} not found")
    brace = text.find("{", start)
    depth = 0
    i = brace
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
        i += 1
    raise RuntimeError(f"unbalanced braces in {signature}")


def _extract_struct(text, name):
    end = text.find("} " + name + ";")
    if end < 0:
        raise RuntimeError(f"struct {name} not found")
    start = text.rfind("typedef struct {", 0, end)
    if start < 0:
        raise RuntimeError(f"struct {name}: no opening typedef")
    return text[start:end + len("} " + name + ";")]


def build(spec_name):
    spec = SPECS[spec_name]
    if not (os.path.exists(COMMON_H) and os.path.exists(QUANTS_C)):
        raise RuntimeError(f"llama.cpp source not found under {SRC}")
    h = open(COMMON_H, encoding="utf-8", errors="replace").read()
    c = open(QUANTS_C, encoding="utf-8", errors="replace").read()

    body = ""
    for tname, tcount in spec["tables"]:
        body += _extract_table(h, tname, tcount) + "\n"
    body += _extract_struct(h, spec["struct"]) + "\n"
    body += f"#define BLOCK_BYTES {spec['block']}\n"
    for fname in spec["funcs"]:
        sig = next((ln.split("(")[0].strip() + "(" for ln in c.splitlines()
                    if fname + "(" in ln and "void" in ln), None)
        if sig is None:
            raise RuntimeError(f"no definition found for {fname}")
        body += _extract_function(c, sig) + "\n"

    dequant = [f for f in spec["funcs"] if f.startswith("dequantize_")][0]
    main = MAIN.replace("DEQUANT_CALL", dequant)
    harness = ("#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n"
               "#include <assert.h>\n" + SHIMS + "\n// ---- verbatim from llama.cpp ----\n"
               + body + main)

    os.makedirs(OUTDIR, exist_ok=True)
    digest = hashlib.sha256(harness.encode()).hexdigest()[:16]
    csrc = os.path.join(OUTDIR, f"ggml_authority_{spec_name}.c")
    binary = os.path.join(OUTDIR, f"ggml_authority_{spec_name}")
    stamp = binary + ".stamp"
    have = open(stamp).read().strip() if os.path.exists(stamp) else ""
    if have != digest or not os.path.exists(binary):
        open(csrc, "w", encoding="utf-8").write(harness)
        r = subprocess.run([os.environ.get("CC", "gcc"), "-O2", "-o", binary, csrc],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"compile failed: {r.stderr.strip()[:400]}")
        open(stamp, "w").write(digest)
    return binary


def dequantize(raw, spec_name):
    binary = build(spec_name)
    block = SPECS[spec_name]["block"]
    nb = len(raw) // block
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(raw)
        path = tf.name
    try:
        r = subprocess.run([binary, path, str(nb)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"authority rc={r.returncode}: {r.stderr.strip()[:200]}")
        return np.array([float(x) for x in r.stdout.split()], dtype=np.float64)
    finally:
        os.unlink(path)


def upstream_provenance():
    try:
        commit = subprocess.run(["git", "-C", SRC, "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = "unknown"
    return f"llama.cpp {commit}"


if __name__ == "__main__":
    try:
        for name in SPECS:
            print(f"built {name}: {build(name)}")
        print(upstream_provenance())
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
