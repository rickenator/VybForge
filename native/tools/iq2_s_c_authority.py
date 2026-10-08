#!/usr/bin/env python3
"""The independent authority for IQ2_S: compile llama.cpp's OWN dequant and run it.

The `gguf` package implements Q8_0 but not IQ2_S, so there is no third-party Python check for
this type. Instead this assembles a C file out of the local llama.cpp checkout —

  * the `block_iq2_s` struct  (ggml-common.h)
  * the `iq2s_grid` and `kmask_iq2xs` tables  (ggml-common.h, verbatim text)
  * `dequantize_row_iq2_s`  (ggml-quants.c, verbatim text)

— with only MACRO SHIMS added (QK_K, GGML_RESTRICT, the table macros, an fp16->fp32 converter),
compiles it, and exposes dequantize(raw)->numpy. Nothing is retyped, so the authority is the
code llama.cpp actually runs, not a paraphrase of it.

Build cache: native/out/iq2s_authority (rebuilt when the harness text changes).
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
BIN = os.path.join(OUTDIR, "iq2s_authority")
HARNESS = os.path.join(OUTDIR, "iq2s_authority.c")
BLOCK_BYTES = 82
QK_K = 256

SHIMS = r"""
// ---- shims only: macros the upstream text expects from ggml's own headers ----
#define QK_K 256
#define GGML_RESTRICT
typedef uint16_t ggml_half;
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
// ---- harness main: dequantize a raw block dump, print the first 6 values ----
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <raw-block-file> <nblocks>\n", argv[0]); return 2; }
    long nb = atol(argv[2]);
    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "open failed\n"); return 3; }
    size_t nbytes = (size_t)nb * sizeof(block_iq2_s);
    block_iq2_s *x = (block_iq2_s *)malloc(nbytes);
    if (!x) return 4;
    if (fread(x, 1, nbytes, f) != nbytes) { fprintf(stderr, "short read\n"); return 5; }
    fclose(f);
    float *y = (float *)malloc((size_t)nb * QK_K * sizeof(float));
    if (!y) return 6;
    dequantize_row_iq2_s(x, y, nb * QK_K);
    // Print the WHOLE buffer, not a sample: the comparison should cover every element of the
    // slice, so a layout mistake in one sub-group cannot hide behind six good values.
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


def build_harness():
    if not (os.path.exists(COMMON_H) and os.path.exists(QUANTS_C)):
        raise RuntimeError(f"llama.cpp source not found under {SRC}")
    h = open(COMMON_H, encoding="utf-8", errors="replace").read()
    c = open(QUANTS_C, encoding="utf-8", errors="replace").read()

    harness = (
        "#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n"
        + "#include <assert.h>\n"      # upstream's dequantize_row_iq2_s asserts
        + SHIMS
        + "\n// ---- verbatim from ggml-common.h ----\n"
        + _extract_table(h, "kmask_iq2xs", 8) + "\n"
        + _extract_table(h, "iq2s_grid", 1024) + "\n"
        + _extract_struct(h, "block_iq2_s") + "\n"
        + "\n// ---- verbatim from ggml-quants.c ----\n"
        + _extract_function(c, "void dequantize_row_iq2_s(") + "\n"
        + MAIN
    )
    os.makedirs(OUTDIR, exist_ok=True)
    digest = hashlib.sha256(harness.encode()).hexdigest()[:16]
    stamp = os.path.join(OUTDIR, "iq2s_authority.stamp")
    have = open(stamp).read().strip() if os.path.exists(stamp) else ""
    if have != digest or not os.path.exists(BIN):
        open(HARNESS, "w", encoding="utf-8").write(harness)
        cc = os.environ.get("CC", "gcc")
        cmd = [cc, "-O2", "-o", BIN, HARNESS]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"compile failed: {r.stderr.strip()[:400]}")
        open(stamp, "w").write(digest)
    return BIN


def dequantize(raw):
    """Dequantize raw block bytes with the compiled upstream function."""
    binpath = build_harness()
    nb = len(raw) // BLOCK_BYTES
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(raw)
        path = tf.name
    try:
        r = subprocess.run([binpath, path, str(nb)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"authority rc={r.returncode}: {r.stderr.strip()[:200]}")
        return np.array([float(x) for x in r.stdout.split()], dtype=np.float64)
    finally:
        os.unlink(path)


if __name__ == "__main__":
    # self-test: dequantize zeros and report, so the tool can be smoke-tested alone
    try:
        b = build_harness()
        print(f"built {b}")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
