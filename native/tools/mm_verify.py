#!/usr/bin/env python3
"""mul_mat and the linear-attention input gating, checked against ggml's own ops.

Fourth unit of the phase-4 layer wiring (VybForge #10, item 1), same technique as the three before:
a C harness linking the same libggml llama.cpp runs, plus a numpy port, and the check compares them.

What the layer actually does (src/models/qwen35.cpp, build_layer_attn_linear), for a 27B Ridge layer
whose geometry is n_embd = 5120, head_k_dim = head_v_dim = 128, n_k_heads = 16, n_v_heads = 48:

    wqkv     = mul_mat(wqkv,      cur)   (5120 -> key_dim*2 + value_dim = 10240)
    z        = mul_mat(wqkv_gate, cur)   (5120 -> value_dim = 6144)
    beta     = sigmoid (mul_mat(ssm_beta,  cur))   (5120 -> 48)
    alpha    = mul_mat(ssm_alpha, cur); alpha = softplus(alpha + ssm_dt) * ssm_a
    ... the recurrence ...
    cur      = mul_mat(ssm_out, final_output)      (6144 -> 5120)

ggml_mul_mat(w, x) is w^T x, so a weight of ne (n_col, n_out) times activations of ne (n_col, n_tokens)
gives ne (n_out, n_tokens) — i.e. numpy (n_tokens, n_out). Getting that backwards is the same trap the
recurrence port fell into, so the verifier reports the mis-read buffer's error alongside the correct
one, exactly as the norm check reports the opposite eps convention.

The two activations are pinned to their ggml forms, which are not their obvious alternatives:
    sigmoid  1.f/(1.f + expf(-x))
    softplus (x > 20.0f) ? x : logf(1.0f + expf(x))     -- threshold at 20, logf(1+expf), not log1p

Usage: mm_verify.py [--tokens T]
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
LLAMA = os.environ.get("VYBFORGE_LLAMA", os.path.expanduser("~/Projects/llama.cpp"))
BUILD = os.path.join(REPO, "native/build")
SRC = os.path.join(HERE, "mm_authority.c")
BIN = os.path.join(BUILD, "mm_authority")
GGUF = os.environ.get(
    "VYBFORGE_RIDGE_GGUF",
    os.path.expanduser("~/Models/qwen38-27b-ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf"))

# The 27B Ridge geometry, read from the GGUF's own metadata.
N_EMBD = 5120
KEY_DIM = 128 * 16      # head_k_dim * ssm.group_count
VALUE_DIM = 128 * 48    # head_v_dim * ssm.time_step_rank
CONV_DIM = KEY_DIM * 2 + VALUE_DIM          # 10240
N_V_HEADS = 48

# name, K (n_col), N (n_out) — the five projections of one linear-attention layer.
PROJECTIONS = [
    ("wqkv",      N_EMBD, CONV_DIM),
    ("wqkv_gate", N_EMBD, VALUE_DIM),
    ("ssm_beta",  N_EMBD, N_V_HEADS),
    ("ssm_alpha", N_EMBD, N_V_HEADS),
    ("ssm_out",   VALUE_DIM, N_EMBD),
]
MAXREL = float(os.environ.get("VYBFORGE_MM_MAXREL", "1e-5"))


def build():
    os.makedirs(BUILD, exist_ok=True)
    cmd = [os.environ.get("CC", "gcc"), "-O2", "-o", BIN, SRC,
           f"-I{LLAMA}/ggml/include", f"-L{LLAMA}/build/bin",
           "-lggml", "-lggml-base", "-lggml-cpu", f"-Wl,-rpath,{LLAMA}/build/bin"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("MM_VERIFY_FAIL build: " + (r.stderr.strip()[:300] or r.stdout.strip()[:300]))
        return False
    return True


def run(mode, w, x, dt=None, a=None):
    """Feed w, x as ggml-ORDER arrays: w is (N, K) and x is (M, K), i.e. the memory of ggml tensors
    of ne (K, N) and ne (K, M). Return the authority output as (M, N) — the memory of its ne (N, M)."""
    inp, outp = os.path.join(BUILD, "mm_in.bin"), os.path.join(BUILD, "mm_out.bin")
    with open(inp, "wb") as fh:
        for arr in (w, x, dt, a):
            if arr is not None:
                fh.write(arr.astype("<f4").tobytes())
    r = subprocess.run([BIN, inp, outp, mode, str(w.shape[1]), str(w.shape[0]), str(x.shape[0]), "1"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError((r.stdout + r.stderr)[:300])
    return np.fromfile(outp, dtype="<f4").reshape(x.shape[0], w.shape[0]).astype(np.float64)


# --- the port ---------------------------------------------------------------------------------
# ggml's ne0 is the fastest axis, so a weight of ne (K, N) has the memory of a numpy (N, K) array —
# that is "w" here, and the maths uses its transpose. Activations of ne (K, M) are numpy (M, K), and
# the output of ne (N, M) is numpy (M, N). Reading the weight the other way round is the same axis
# trap that hid in the recurrence port, so the verifier exercises it as a live check.

def mm_port(w, x):
    return (x.astype(np.float64) @ w.astype(np.float64).T).astype(np.float32)


def sigmoid_port(v):
    return (np.float32(1.0) / (np.float32(1.0) + np.exp(-v.astype(np.float32)))).astype(np.float32)


def softplus_port(v, variant="exact"):
    v = v.astype(np.float32)
    with np.errstate(over="ignore"):
        if variant == "log1p":
            return np.log1p(np.exp(v)).astype(np.float32)
        if variant == "no_threshold":
            return np.log(np.float32(1.0) + np.exp(v)).astype(np.float32)
        return np.where(v > np.float32(20.0), v, np.log(np.float32(1.0) + np.exp(v))).astype(np.float32)


def alpha_port(w, x, dt, a, variant="exact"):
    biased = (mm_port(w, x) + dt.astype(np.float32)).astype(np.float32)
    return (softplus_port(biased, variant) * a.astype(np.float32)).astype(np.float32)


def rel(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-30))


def real_gate_params(h_v):
    """The layer's own ssm_dt / ssm_a from the GGUF when it is on disk, else synthetic.

    ssm_a is stored pre-computed (llama.cpp names it SSM_A_NOSCAN and multiplies by it, and
    qwen35.cpp comments it as -A_log.exp()), so the port must use it as-is; asserting the sign here
    is how the reference knows it is not supposed to re-apply the exp.
    """
    try:
        import gguf
        r = gguf.GGUFReader(GGUF)
        cur = {}
        for name, key in (("blk.0.ssm_dt.bias", "dt"), ("blk.0.ssm_a", "a")):
            t = next(t for t in r.tensors if t.name == name)
            cur[key] = np.array(t.data, dtype=np.float32).reshape(-1)[:h_v].copy()
            assert cur[key].shape == (h_v,)
        return cur["dt"], cur["a"], True
    except Exception as e:      # noqa: BLE001 - any failure means fall back, but say so
        print(f"MM_VERIFY note: real ssm_dt/ssm_a unavailable ({e.__class__.__name__}); using synthetic")
        rng = np.random.default_rng(7)
        return (rng.normal(0.0, 0.5, size=h_v).astype(np.float32),
                -np.exp(rng.normal(0.0, 1.0, size=h_v)).astype(np.float32), False)


def main():
    T = 4
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--tokens":
            T = int(argv[i + 1])
    print(f"MM_VERIFY geometry n_embd={N_EMBD} key_dim={KEY_DIM} value_dim={VALUE_DIM} tokens={T}")
    if not build():
        return 1

    rng = np.random.default_rng(20261008)
    bad = 0

    # ── 1. the five projections: mul_mat(w, x) = w^T x, at the model's real shapes ──────
    worst = 0.0
    for name, k, n in PROJECTIONS:
        w = rng.normal(0.0, 0.05, size=(n, k)).astype(np.float32)   # ggml order: ne (k, n)
        x = rng.normal(0.0, 0.5, size=(T, k)).astype(np.float32)    # ggml order: ne (k, T)
        ref = run("mm", w, x)
        r = rel(mm_port(w, x), ref)
        worst = max(worst, r)
        print(f"MM_VERIFY mm {name:9s} K={k:5d} N={n:5d} n={T * n:6d} maxrel={r:.3e}")
        if r > MAXREL:
            bad = 1
        # the axis lesson, as a live check: read the authority buffer as (N, M) instead of (M, N)
        raw = np.fromfile(os.path.join(BUILD, "mm_out.bin"), dtype="<f4")
        misread = raw.reshape(n, T).astype(np.float64)
        print(f"MM_VERIFY mm {name:9s} transposed-read maxrel={rel(mm_port(w, x).T, misread):.3e} (must be large)")

    # ── 2. beta = sigmoid(mul_mat(ssm_beta, cur)) ────────────────────────────────────────
    k, n = N_EMBD, N_V_HEADS
    w = rng.normal(0.0, 0.05, size=(n, k)).astype(np.float32)
    x = rng.normal(0.0, 3.0, size=(T, k)).astype(np.float32)
    ref = run("beta", w, x)
    got = sigmoid_port(mm_port(w, x))
    r_beta = rel(got, ref)
    print(f"MM_VERIFY beta  sigmoid(mm) maxrel={r_beta:.3e} maxabs={float(np.max(np.abs(got - ref))):.3e} "
          f"(authority scale {float(np.max(np.abs(ref))):.3e})")
    if r_beta > MAXREL:
        bad = 1

    # ── 3. alpha = softplus(mul_mat(ssm_alpha, cur) + ssm_dt) * ssm_a ────────────────────
    w = rng.normal(0.0, 0.05, size=(n, k)).astype(np.float32)
    # One token scaled up hard on purpose: it pushes alpha+dt past softplus's x > 20 threshold (and,
    # for the no-threshold variant, past exp's f32 overflow at ~88) so the conventions are separable.
    x = rng.normal(0.0, 2.0, size=(T, k)).astype(np.float32)
    x[1] = (x[1] * 400.0).astype(np.float32)
    dt, a, real = real_gate_params(n)
    print(f"MM_VERIFY alpha ssm_dt=[{dt.min():.3e},{dt.max():.3e}] ssm_a=[{a.min():.3e},{a.max():.3e}] "
          f"({'real GGUF values' if real else 'synthetic'})")
    ref = run("alpha", w, x, dt, a)
    exact = alpha_port(w, x, dt, a)
    r_alpha = rel(exact, ref)
    print(f"MM_VERIFY alpha exact       maxrel={r_alpha:.3e} maxabs={float(np.max(np.abs(exact - ref))):.3e} "
          f"(authority scale {float(np.max(np.abs(ref))):.3e})")
    if r_alpha > MAXREL:
        bad = 1
    for variant in ("log1p", "no_threshold"):
        print(f"MM_VERIFY alpha {variant:12s} maxrel={rel(alpha_port(w, x, dt, a, variant), ref):.3e} (should be non-zero)")
    if real and not np.all(a < 0):
        print("MM_VERIFY_FAIL ssm_a in the GGUF is not all negative — the -exp(A_log) assumption is wrong")
        bad = 1
    # sanity: the threshold branch really is exercised by this input
    n_hi = int(np.sum((mm_port(w, x) + dt.astype(np.float32)) > 20.0))
    print(f"MM_VERIFY alpha threshold branch exercised on {n_hi}/{T * n} values "
          f"({int(np.sum((exact / a) > 20.0))} of our values came from it)")

    if bad:
        print(f"MM_VERIFY_FAIL worse than maxrel {MAXREL:g}")
        return 1
    print(f"MM_VERIFY_SUMMARY mm_worst={worst:.3e} beta={r_beta:.3e} alpha={r_alpha:.3e}")
    print(f"MM_VERIFY_DONE mul_mat, sigmoid and the softplus gate reproduce ggml's ops within maxrel {MAXREL:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
