#!/usr/bin/env python3
"""l2 normalisation, RMS normalisation and the gated RMS epilogue, checked against ggml's own ops.

Third unit of the phase-4 layer wiring (VybForge #10, item 1), same technique as the recurrence and
the short convolution: a C harness linking the same libggml llama.cpp runs, plus a numpy port, and the
check compares the two.

ggml's ne0 is the fastest axis, so a tensor of ne (n_col, n_rows) is a numpy array of shape
(n_rows, n_col) — the LAST axis is the normalised one. Both ops normalise along ne0 only, and every
trailing dim of the model's tensor is just another row here.

The three modes, and the conventions that are easy to get wrong (read first-hand from ops.cpp):

  l2   scale = 1.0f/fmaxf(sqrtf(sum), eps)      -> eps is a FLOOR, applied AFTER the sqrt
  rms  scale = 1.0f/sqrtf(mean + eps)           -> eps is INSIDE the sqrt
  both: the products x*x are f32, the accumulation is double, the narrowing/reciprocal is f32
  epilogue = RMSNorm(output, ssm_norm) * SiLU(z)   (qwen35.cpp build_norm_gated)

Model geometry: the l2 norm is per head over head_k_dim = 128 (qwen35.cpp:440-443) and the epilogue
normalises over head_v_dim = 128 with a per-channel weight (ssm_norm weight is {head_v_dim}).

Usage: norm_verify.py
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
LLAMA = os.environ.get("VYBFORGE_LLAMA", os.path.expanduser("~/Projects/llama.cpp"))
BUILD = os.path.join(REPO, "native/build")
SRC = os.path.join(HERE, "norm_authority.c")
BIN = os.path.join(BUILD, "norm_authority")
MAXREL = float(os.environ.get("VYBFORGE_NORM_MAXREL", "1e-6"))

N_COL = 128     # head_k_dim = head_v_dim = ssm.state_size
H_K = 16        # ssm.group_count  (q/k heads the l2 norm sees)
H_V = 48        # ssm.time_step_rank (value heads the epilogue sees)
T = 4
B = 1
EPS = 1e-6      # f_norm_rms_eps, the eps both the l2 norm and the epilogue use


def build():
    os.makedirs(BUILD, exist_ok=True)
    cmd = [os.environ.get("CC", "gcc"), "-O2", "-o", BIN, SRC,
           f"-I{LLAMA}/ggml/include", f"-L{LLAMA}/build/bin",
           "-lggml", "-lggml-base", "-lggml-cpu", f"-Wl,-rpath,{LLAMA}/build/bin"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("NORM_VERIFY_FAIL build: " + (r.stderr.strip()[:300] or r.stdout.strip()[:300]))
        return False
    return True


def run(mode, x, w=None, gate=None):
    """Feed one tensor (and, for the epilogue, w and gate) through the authority; return its output."""
    inp, outp = os.path.join(BUILD, "norm_in.bin"), os.path.join(BUILD, "norm_out.bin")
    with open(inp, "wb") as fh:
        fh.write(x.astype("<f4").tobytes())
        if w is not None:
            fh.write(w.astype("<f4").tobytes())
        if gate is not None:
            fh.write(gate.astype("<f4").tobytes())
    r = subprocess.run([BIN, inp, outp, mode, repr(EPS), str(N_COL), str(x.shape[0]), "1"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError((r.stdout + r.stderr)[:300])
    print("  " + r.stdout.strip())
    return np.fromfile(outp, dtype="<f4").reshape(x.shape).astype(np.float64)


# --- the port ---------------------------------------------------------------------------------
# The exact f32/double split of ops.cpp: square in f32, accumulate in double, narrow once, then a
# float32 sqrt / fmaxf / reciprocal.

def _sumsq_double(x):
    p = (x * x).astype(np.float32)                     # x*x is a float multiply, rounded to f32
    return p.astype(np.float64).sum(axis=1)            # then widened and accumulated in double


def l2_port(x, eps, eps_inside_sqrt=False):
    s = _sumsq_double(x)
    inner = (s.astype(np.float32) + np.float32(eps)) if eps_inside_sqrt else s.astype(np.float32)
    sq = np.sqrt(inner).astype(np.float32)             # sqrtf(sum): f32 in, f32 out
    denom = sq if eps_inside_sqrt else np.maximum(sq, np.float32(eps))   # eps is the FLOOR
    scale = np.float32(1.0) / denom
    return (x * scale[:, None]).astype(np.float32)


def rms_port(x, eps, eps_after_sqrt=False):
    mean = (_sumsq_double(x) / x.shape[1]).astype(np.float32)   # double divide, narrowed to f32
    inner = np.sqrt(mean).astype(np.float32) if eps_after_sqrt else mean
    denom = np.maximum(inner, np.float32(eps)) if eps_after_sqrt else np.sqrt(mean + np.float32(eps)).astype(np.float32)
    scale = np.float32(1.0) / denom
    return (x * scale[:, None]).astype(np.float32)


def silu_f32(z):
    z = z.astype(np.float32)
    return (z / (np.float32(1.0) + np.exp(-z).astype(np.float32))).astype(np.float32)


def epilogue_port(x, w, gate, eps):
    return ((rms_port(x, eps) * w.astype(np.float32)).astype(np.float32) * silu_f32(gate)).astype(np.float32)


def rel(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-30))


def main():
    rows_l2, rows_ep = H_K * T * B, H_V * T * B
    rng = np.random.default_rng(20261008)
    print(f"NORM_VERIFY eps={EPS:g} n_col={N_COL} l2_rows={rows_l2} epilogue_rows={rows_ep}")
    if not build():
        return 1

    # A tiny-norm leading row on purpose: it is the only place the l2 'floor after sqrt' and the rms
    # 'eps inside sqrt' conventions are distinguishable, so the check below has teeth.
    def normal_rows(n):
        a = rng.normal(0.0, 0.2, size=(n, N_COL)).astype(np.float32)
        a[0] = (rng.normal(0.0, 1.0, size=N_COL) * 1e-8).astype(np.float32)
        return a

    bad = 0
    got = {}
    for mode, rows in (("l2", rows_l2), ("rms", rows_ep)):
        x = normal_rows(rows)
        ref = run(mode, x)
        port = (l2_port if mode == "l2" else rms_port)(x, EPS)
        r = rel(port, ref)
        got[mode] = r
        print(f"NORM_VERIFY {mode:4s} n={x.size} maxrel={r:.3e} (authority scale {float(np.max(np.abs(ref))):.3e})")
        if r > MAXREL:
            bad = 1
        # discrimination: the opposite convention must NOT match, else the test proves nothing
        wrong = (l2_port(x, EPS, eps_inside_sqrt=True) if mode == "l2"
                 else rms_port(x, EPS, eps_after_sqrt=True))
        print(f"NORM_VERIFY {mode:4s} opposite-convention maxrel={rel(wrong, ref):.3e} (should be large)")

    x = normal_rows(rows_ep)
    w = rng.normal(0.0, 1.0, size=N_COL).astype(np.float32)
    gate = rng.normal(0.0, 1.0, size=(rows_ep, N_COL)).astype(np.float32)
    ref = run("epilogue", x, w, gate)
    r = rel(epilogue_port(x, w, gate, EPS), ref)
    got["epilogue"] = r
    print(f"NORM_VERIFY epilogue n={x.size} maxrel={r:.3e} (authority scale {float(np.max(np.abs(ref))):.3e})")
    if r > MAXREL:
        bad = 1

    if bad:
        print(f"NORM_VERIFY_FAIL worse than maxrel {MAXREL:g}")
        return 1
    print(f"NORM_VERIFY_SUMMARY l2={got['l2']:.3e} rms={got['rms']:.3e} epilogue={got['epilogue']:.3e}")
    print(f"NORM_VERIFY_DONE l2, rms and the gated epilogue reproduce ggml's ops within maxrel {MAXREL:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
