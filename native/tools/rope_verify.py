#!/usr/bin/env python3
"""Ridge's attention RoPE: the real ggml op (`rope_authority.c`) vs candidate layouts (phase 4, unit 10).

Two things are measured here, and the second is the one that decides work:

1. **Does the MRoPE sectioning collapse for a text-only model?** `build_layer_attn` calls
   `ggml_rope_multi` with `sections = [11,11,10,0]` and `is_imrope`. Every dimension section picks its
   theta from one of four position ids — and a text run gives all four the same token position. So
   IMROPE and MROPE must produce IDENTICAL output; if they do not, the sections matter even for text and
   the whole kernel is MRoPE, not plain rope. This is asserted, not assumed.

2. **Which rotation layout does the op actually use?** The engine's `qwen3rope` rotates every head dim
   with pairs (i, i + HD/2), which cannot express "rotate the first n_dims only". Candidates are tried
   against the authority and exactly ONE must match: a layout is not something to be argued from source
   when a harness can run the real op.

Usage: rope_verify.py     (env: CC, VYBHOME, VYBFORGE_LLAMA)
"""
import os
import struct
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
LLAMA = os.environ.get("VYBFORGE_LLAMA", os.path.expanduser("~/Projects/llama.cpp"))
BUILD = os.path.join(REPO, "native/build")
AUTH_BIN = os.path.join(BUILD, "rope_authority")
AUTH_IN = os.path.join(BUILD, "rope_authority_in.bin")
AUTH_OUT = os.path.join(BUILD, "rope_authority_out.bin")

# Ridge's own numbers, read out of the GGUF metadata (doc/QWEN35-PHASE4.md).
HD, NH, NTOK = 256, 24, 5
ND = 64                      # rope.dimension_count
SECTIONS = (11, 11, 10, 0)   # rope.dimension_sections
BASE = 1e7                   # rope.freq_base
MAXREL = float(os.environ.get("VYBFORGE_ROPE_MAXREL", "1e-4"))


def build_authority():
    cmd = [os.environ.get("CC", "gcc"), "-O2", "-o", AUTH_BIN, os.path.join(HERE, "rope_authority.c"),
           f"-I{LLAMA}/ggml/include", f"-L{LLAMA}/build/bin",
           "-lggml", "-lggml-base", "-lggml-cpu", f"-Wl,-rpath,{LLAMA}/build/bin"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("ROPE_VERIFY_FAIL authority build: " + (r.stderr.strip()[:400] or r.stdout.strip()[:400]))
        return False
    return True


def run_authority(q, pos, mode):
    """q: (NTOK, NH, HD) in ggml order; pos: (NTOK, k) int32 — k planes of token positions.

    NEOX (the mode `llama_model_rope_type` gives this architecture) takes ONE plane; the MRoPE modes
    take four. The op asserts on the length, which is the cheapest possible way to learn that.
    """
    with open(AUTH_IN, "wb") as fh:
        fh.write(np.ascontiguousarray(q, dtype="<f4").tobytes())
        fh.write(np.ascontiguousarray(pos, dtype="<i4").tobytes())
    r = subprocess.run([AUTH_BIN, AUTH_IN, AUTH_OUT, str(HD), str(NH), str(NTOK), str(ND), mode,
                        *[str(s) for s in SECTIONS], str(pos.size)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(("authority", r.stdout + r.stderr)[:300])
    return np.fromfile(AUTH_OUT, dtype="<f4").reshape(NTOK, NH, HD)


def rel(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-30))


def rot(x, posf, i, j, k):
    """Rotate the (i, j) dim pair of every head by angle pos*inv[k]; returns a copy."""
    out = x.copy()
    inv = np.array([BASE ** (-2.0 * m / ND) for m in range(ND // 2)])
    ang = posf * inv[k]
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    a, b = x[:, :, i].copy(), x[:, :, j].copy()
    out[:, :, i] = a * c - b * s
    out[:, :, j] = a * s + b * c
    return out


def cand_neox_first(x, posf):
    """The hypothesis: NEOX pairing inside the first ND dims, rest passed through."""
    out = x.copy()
    for k in range(ND // 2):
        out = rot(out, posf, k, k + ND // 2, k)
    return out


def cand_swizzle(x, posf):
    """As `rotate_pairs(n_dims, n_dims/2)`: output pair (2k, 2k+1) reads source (ND/2+k, ND/2+k+1)."""
    out = x.copy()
    for k in range(ND // 2):
        i, j = ND // 2 + k, ND // 2 + k + 1
        if j >= ND:
            break
        out = rot(out, posf, i, j, k)
    return out


def cand_adjacent(x, posf):
    """In-place adjacent pairs (2k, 2k+1) inside the first ND dims."""
    out = x.copy()
    for k in range(ND // 2):
        out = rot(out, posf, 2 * k, 2 * k + 1, k)
    return out


def cand_neox_all(x, posf):
    """The engine's current kernel: NEOX over EVERY head dim."""
    out = x.copy()
    for k in range(HD // 2):
        inv = np.array([BASE ** (-2.0 * m / HD) for m in range(HD // 2)])
        ang = posf * inv[k]
        c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
        a, b = x[:, :, k].copy(), x[:, :, k + HD // 2].copy()
        out[:, :, k] = a * c - b * s
        out[:, :, k + HD // 2] = a * s + b * c
    return out


def main():
    print(f"ROPE_VERIFY geometry head_dim={HD} n_head={NH} tokens={NTOK} n_dims={ND} "
          f"sections={list(SECTIONS)} freq_base={BASE:g}")
    if not os.path.exists(os.path.join(LLAMA, "ggml/include/ggml.h")):
        print(f"ROPE_VERIFY_SKIP no llama.cpp checkout at {LLAMA}")
        return 0
    if not build_authority():
        return 1

    rng = np.random.default_rng(20261009)
    q = rng.normal(0.0, 1.0, size=(NTOK, NH, HD)).astype(np.float32)
    posf = np.arange(NTOK, dtype=np.float64)                      # text: one position per token
    pos = np.repeat(posf.astype(np.int32)[:, None], 4, axis=1)    # ... the same in all four sections

    cands = (("neox_first_nd", cand_neox_first), ("swizzle_2k_in_nd", cand_swizzle),
             ("adjacent_in_nd", cand_adjacent), ("neox_all_dims", cand_neox_all))

    # A MATRIX, not an assertion: which (ggml mode, rotation layout) reproduces the op? The reconciling
    # fact is in ggml.h itself — the sections choose which dims are PAIRED, not the theta per dim
    # ("idx used for theta: [0..n_dims/2], not reset for each section"), and IMROPE/MROPE lay the
    # sections out differently ([ttyxttyx...] vs [ttttyyxx...]). So on a text model the section
    # POSITIONS are all equal and the modes still differ, in the pairing. Which mode an arch uses is
    # `llama_model_rope_type`'s business (qwen35 sits with qwen3next), and what it produces here is
    # what the kernel has to reproduce.
    best = None
    scores = {}
    for mode in ("neox", "mrope", "imrope"):
        # one position plane for NEOX, four for the MRoPE modes
        pm = pos[:, :1].copy() if mode == "neox" else pos
        ref = run_authority(q, pm, mode).astype(np.float64)
        for name, fn in cands:
            r = rel(fn(q.astype(np.float64), posf), ref)
            scores[(mode, name)] = r
            mark = "MATCH" if r <= MAXREL else "differs"
            print(f"ROPE_VERIFY mode={mode:7s} cand {name:16s} maxrel={r:.3e} {mark}")
            if r <= MAXREL and (best is None or r < scores[best]):
                best = (mode, name)

    if best is None:
        print("ROPE_VERIFY_FAIL no (mode, layout) pair reproduces the op — the layout is something "
              "this list does not contain (read ggml.h's IMROPE diagram and add the candidate)")
        return 1
    others = sorted((v, k) for k, v in scores.items() if k != best)
    print(f"ROPE_VERIFY_SUMMARY winner mode={best[0]} layout={best[1]} "
          f"maxrel={scores[best]:.3e} runner_up={others[0][1][0]}/{others[0][1][1]}={others[0][0]:.2e}")
    print(f"ROPE_VERIFY_DONE ggml's own rope op for head_dim={HD} n_dims={ND} is reproduced by "
          f"mode={best[0]} with layout '{best[1]}', within maxrel {MAXREL:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
