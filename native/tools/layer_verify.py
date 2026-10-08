#!/usr/bin/env python3
"""One qwen35 linear-attention block, wired end to end and checked against ggml's own ops.

Fifth unit of the phase-4 layer wiring (VybForge #10, item 1). The four earlier units verified the
pieces in isolation (recurrence, short convolution, the two norms + epilogue, the projections and
gates); this one checks the WIRING — that they are assembled in the order, shapes and residual
structure src/models/qwen35.cpp uses — by comparing every stage against `native/tools/layer_authority.c`,
which builds the same block out of the real ggml ops and runs it in one graph.

Per-stage comparison is the point: a wiring mistake (a residual added in the wrong place, z entering
the epilogue at the wrong shape, q|k|v split at the wrong channel offsets, the conv window on the
wrong side) shows up as ONE stage diverging, which names the mistake instead of merely failing.

Conventions, all of them ggml's (ne0 fastest -> numpy's LAST axis):
    x     (T, n_embd)          ne (n_embd, T)
    wqkv  (qkv_dim, n_embd)    ne (n_embd, qkv_dim)   — a weight "mul_mat(w, x)" is w^T x
    mm(x, w) = x @ w.T
The gated delta net state is zero: this is the FIRST token of a sequence, which is the only geometry
the op's sequential kernel covers (T > 1 uses a chunked kernel that is not yet characterised).

Usage: layer_verify.py
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
LLAMA = os.environ.get("VYBFORGE_LLAMA", os.path.expanduser("~/Projects/llama.cpp"))
BUILD = os.path.join(REPO, "native/build")
SRC = os.path.join(HERE, "layer_authority.c")
BIN = os.path.join(BUILD, "layer_authority")
OUTDIR = os.path.join(BUILD, "layer_stages")
GGUF = os.environ.get(
    "VYBFORGE_RIDGE_GGUF",
    os.path.expanduser("~/Models/qwen38-27b-ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf"))
IN_FILE = os.path.join(BUILD, "layer_in.bin")

# The 27B Ridge geometry, from the GGUF's own metadata.
N_EMBD = 5120
HEAD_DIM = 128
H_K = 16
H_V = 48
D_CONV = 4
T, B = 1, 1
EPS = 1e-6
KEY_DIM = HEAD_DIM * H_K
VALUE_DIM = HEAD_DIM * H_V
QKV_DIM = 2 * KEY_DIM + VALUE_DIM
SCALE = 1.0 / float(np.sqrt(HEAD_DIM))
MAXREL = float(os.environ.get("VYBFORGE_LAYER_MAXREL", "1e-5"))


def build():
    os.makedirs(BUILD, exist_ok=True)
    cmd = [os.environ.get("CC", "gcc"), "-O2", "-o", BIN, SRC,
           f"-I{LLAMA}/ggml/include", f"-L{LLAMA}/build/bin",
           "-lggml", "-lggml-base", "-lggml-cpu", f"-Wl,-rpath,{LLAMA}/build/bin"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("LAYER_VERIFY_FAIL build: " + (r.stderr.strip()[:400] or r.stdout.strip()[:400]))
        return False
    return True


def real_gate_params():
    """The layer's own ssm_dt / ssm_a from the GGUF when it is on disk, else synthetic.

    Using the real ones keeps the alpha gate in the range the model actually produces (its ssm_dt
    bias reaches ~19, close to softplus's x > 20 threshold), which a synthetic draw can miss.
    """
    try:
        import gguf
        r = gguf.GGUFReader(GGUF)
        cur = {}
        for name, key in (("blk.0.ssm_dt.bias", "dt"), ("blk.0.ssm_a", "a")):
            t = next(t for t in r.tensors if t.name == name)
            cur[key] = np.array(t.data, dtype=np.float32).reshape(-1)[:H_V].copy()
        assert cur["dt"].shape == (H_V,) and cur["a"].shape == (H_V,)
        return cur["dt"], cur["a"], True
    except Exception as e:      # noqa: BLE001 - any failure means fall back, but say so
        print(f"LAYER_VERIFY note: real ssm_dt/ssm_a unavailable ({e.__class__.__name__}); synthetic")
        rng = np.random.default_rng(7)
        return (rng.normal(0.0, 0.5, size=H_V).astype(np.float32),
                -np.exp(rng.normal(1.0, 0.5, size=H_V)).astype(np.float32), False)


# --- the exact f32 arithmetic ggml's CPU ops use ------------------------------------------------
# Squares and products are f32, accumulation is f64, the narrowing and the reciprocal are f32 — the
# same split the norm and GDN op bodies use (ops.cpp). Porting these as plain float64 maths would show
# up as a small but constant disagreement, so they are mirrored rather than approximated.

def _sumsq(v):
    return (v * v).astype(np.float32).astype(np.float64).sum(axis=-1)      # f32 products, f64 sum


def rms_norm_last(v, w, eps):
    mean = (_sumsq(v) / v.shape[-1]).astype(np.float32)
    scale = (np.float32(1.0) / np.sqrt(mean + np.float32(eps))).astype(np.float32)
    return (v * scale[..., None]).astype(np.float32) * w.astype(np.float32)


def l2_norm_last(v, eps):
    sq = np.sqrt(_sumsq(v).astype(np.float32)).astype(np.float32)          # eps floors AFTER sqrt
    scale = (np.float32(1.0) / np.maximum(sq, np.float32(eps))).astype(np.float32)
    return (v * scale[..., None]).astype(np.float32)


def sigmoid(v):
    return (np.float32(1.0) / (np.float32(1.0) + np.exp(-v.astype(np.float32)))).astype(np.float32)


def softplus(v):
    v = v.astype(np.float32)
    with np.errstate(over="ignore"):
        return np.where(v > np.float32(20.0), v,
                        np.log(np.float32(1.0) + np.exp(v))).astype(np.float32)


def silu(v):
    v = v.astype(np.float32)
    return (v / (np.float32(1.0) + np.exp(-v))).astype(np.float32)


def mm(x, w):
    """ggml_mul_mat(w, x) where w is ggml-order (N, K) and x is (M, K): result (M, N)."""
    return (x.astype(np.float64) @ w.astype(np.float64).T).astype(np.float32)


def conv1d_causal(conv_in, cw):
    """out[t, ch] = sum_j conv_in[ch, t+j] * cw[ch, j] — ggml-order (qkv_dim, ncs) and (qkv_dim, d_conv)."""
    out = np.zeros((T, conv_in.shape[0]), dtype=np.float64)
    for t in range(T):
        for j in range(D_CONV):
            out[t] += conv_in[:, t + j].astype(np.float64) * cw[:, j].astype(np.float64)
    return out.astype(np.float32)


def delta_rule(q, k, v, gate, beta):
    """The verified recurrence for a zero initial state, one token (T=1), in ggml numpy order."""
    out = np.zeros_like(v, dtype=np.float64)
    for b in range(B):
        for t in range(T):
            for h in range(H_V):
                hk = h % H_K                                   # the mod broadcast, established by data
                kd = k[b, t, hk].astype(np.float64)
                m = np.zeros((HEAD_DIM, HEAD_DIM), dtype=np.float64)
                m *= np.exp(float(gate[b, t, h, 0]))
                delta = (v[b, t, h].astype(np.float64) - m @ kd) * float(beta[b, t, h, 0])
                m += np.outer(delta, kd)
                out[b, t, h] = (m @ q[b, t, hk].astype(np.float64)) * SCALE
    return out


def rel(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-30))


def main():
    print(f"LAYER_VERIFY geometry n_embd={N_EMBD} head_dim={HEAD_DIM} H_k={H_K} H_v={H_V} "
          f"d_conv={D_CONV} T={T} B={B} eps={EPS:g}")
    if not build():
        return 1

    # ---- synthetic weights and a synthetic layer input, in ggml order -------------------------
    rng = np.random.default_rng(20261008)
    s = 1.0 / np.sqrt(N_EMBD)
    x = rng.normal(0.0, 1.0, size=(T, N_EMBD)).astype(np.float32)
    attn_norm = rng.normal(1.0, 0.05, size=(N_EMBD,)).astype(np.float32)
    wqkv = rng.normal(0.0, s, size=(QKV_DIM, N_EMBD)).astype(np.float32)
    wgate = rng.normal(0.0, s, size=(VALUE_DIM, N_EMBD)).astype(np.float32)
    wbeta = rng.normal(0.0, s, size=(H_V, N_EMBD)).astype(np.float32)
    walpha = rng.normal(0.0, s, size=(H_V, N_EMBD)).astype(np.float32)
    ssm_dt, ssm_a, real = real_gate_params()
    conv1d = rng.normal(0.0, 0.2, size=(QKV_DIM, D_CONV)).astype(np.float32)
    ssm_norm = rng.normal(1.0, 0.05, size=(HEAD_DIM,)).astype(np.float32)
    ssm_out = rng.normal(0.0, s, size=(N_EMBD, VALUE_DIM)).astype(np.float32)
    print(f"LAYER_VERIFY ssm_dt=[{ssm_dt.min():.3e},{ssm_dt.max():.3e}] "
          f"ssm_a=[{ssm_a.min():.3e},{ssm_a.max():.3e}] ({'real GGUF values' if real else 'synthetic'})")

    with open(IN_FILE, "wb") as fh:
        for arr in (x, attn_norm, wqkv, wgate, wbeta, walpha, ssm_dt, ssm_a, conv1d, ssm_norm, ssm_out):
            fh.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())

    os.makedirs(OUTDIR, exist_ok=True)
    r = subprocess.run([BIN, IN_FILE, OUTDIR, str(N_EMBD), str(HEAD_DIM), str(H_K), str(H_V),
                        str(D_CONV), str(T), str(B), repr(EPS), "1"], capture_output=True, text=True)
    if r.returncode != 0:
        print("LAYER_VERIFY_FAIL run: " + (r.stdout + r.stderr)[-600:])
        return 1
    print("  " + r.stdout.strip().splitlines()[-1])

    # ---- the reference, one stage at a time, in qwen35.cpp's order ----------------------------
    ref = {}
    ref["xn"] = rms_norm_last(x, attn_norm, EPS)
    ref["qkv"] = mm(ref["xn"], wqkv)
    ref["z"] = mm(ref["xn"], wgate)
    ref["beta"] = sigmoid(mm(ref["xn"], wbeta)).reshape(B, T, H_V, 1)
    alpha = (mm(ref["xn"], walpha) + ssm_dt.astype(np.float32)).astype(np.float32)
    ref["gate"] = (softplus(alpha) * ssm_a.astype(np.float32)).astype(np.float32).reshape(B, T, H_V, 1)

    conv_in = np.zeros((QKV_DIM, D_CONV - 1 + T), dtype=np.float32)
    conv_in[:, D_CONV - 1:] = ref["qkv"].reshape(T, QKV_DIM).T          # window (zero) then tokens
    ref["conv_in"] = conv_in
    conv_silu = silu(conv1d_causal(conv_in, conv1d))
    ref["conv_silu"] = conv_silu
    q = conv_silu[:, :KEY_DIM].reshape(B, T, H_K, HEAD_DIM)
    k = conv_silu[:, KEY_DIM:2 * KEY_DIM].reshape(B, T, H_K, HEAD_DIM)
    v = conv_silu[:, 2 * KEY_DIM:].reshape(B, T, H_V, HEAD_DIM)
    ref["v_conv"] = v
    ref["q_norm"] = l2_norm_last(q, EPS)
    ref["k_norm"] = l2_norm_last(k, EPS)
    ref["gdn_out"] = delta_rule(ref["q_norm"], ref["k_norm"], v, ref["gate"], ref["beta"]).astype(np.float32)
    z2d = ref["z"].reshape(B, T, H_V, HEAD_DIM)
    ref["epi"] = (rms_norm_last(ref["gdn_out"], ssm_norm, EPS) * silu(z2d)).astype(np.float32)
    ref["y"] = mm(ref["epi"].reshape(T, VALUE_DIM), ssm_out).reshape(B, T, N_EMBD)
    ref["layer_out"] = (x + ref["y"].reshape(T, N_EMBD)).astype(np.float32)

    # ---- compare stage by stage ---------------------------------------------------------------
    # Each authority dump is the stage's raw memory, i.e. a numpy array of the reversed ggml ne.
    shapes = {
        "xn": (B, T, N_EMBD), "qkv": (B, T, QKV_DIM), "z": (B, T, VALUE_DIM),
        "beta": (B, T, H_V, 1), "gate": (B, T, H_V, 1), "conv_in": (QKV_DIM, D_CONV - 1 + T, B),
        "conv_silu": (B, T, QKV_DIM), "v_conv": (B, T, H_V, HEAD_DIM),
        "q_norm": (B, T, H_K, HEAD_DIM), "k_norm": (B, T, H_K, HEAD_DIM),
        "gdn_out": (B, T, H_V, HEAD_DIM), "epi": (B, T, H_V, HEAD_DIM),
        "y": (B, T, N_EMBD), "layer_out": (B, T, N_EMBD),
    }
    bad = 0
    order = ["xn", "qkv", "z", "beta", "gate", "conv_in", "conv_silu", "v_conv", "q_norm", "k_norm",
             "gdn_out", "epi", "y", "layer_out"]
    for name in order:
        p = os.path.join(OUTDIR, f"{name}.bin")
        if not os.path.exists(p):
            print(f"LAYER_VERIFY {name:11s} MISSING (no dump — the harness did not reach it)")
            bad = 1
            continue
        got = np.fromfile(p, dtype="<f4").reshape(shapes[name]).astype(np.float64)
        want = np.asarray(ref[name], dtype=np.float64).reshape(shapes[name])
        r_ = rel(got, want)
        flag = "ok" if r_ <= MAXREL else "DIFFERS"
        print(f"LAYER_VERIFY {name:11s} n={got.size:7d} maxrel={r_:.3e} scale={float(np.max(np.abs(want))):.3e} {flag}")
        if r_ > MAXREL:
            bad = 1

    if bad:
        print(f"LAYER_VERIFY_FAIL at least one stage is worse than maxrel {MAXREL:g}")
        return 1

    # ---- the wiring check's teeth ---------------------------------------------------------------
    # Two plausible mis-wirings of this very layer, each recomputed from the verified stages and
    # compared against the authority's final output. If either came out small, the check above would
    # not be measuring the wiring at all.
    base = np.fromfile(os.path.join(OUTDIR, "layer_out.bin"), dtype="<f4").reshape(B, T, N_EMBD).astype(np.float64)

    def tail(gdn, epilogue=True, residual=x):
        e = rms_norm_last(gdn, ssm_norm, EPS)
        if epilogue:
            e = (e * silu(z2d)).astype(np.float32)          # the gated epilogue: RMSNorm * SiLU(z)
        yy = mm(e.reshape(T, VALUE_DIM), ssm_out).reshape(T, N_EMBD)
        return (residual.reshape(T, N_EMBD) + yy).astype(np.float32)

    alt_missing_z = rel(tail(ref["gdn_out"], epilogue=False), base)
    alt_residual = rel(tail(ref["gdn_out"], residual=ref["xn"]), base)
    print(f"LAYER_VERIFY alt epilogue-without-SiLU(z) maxrel={alt_missing_z:.3e} (must be large)")
    print(f"LAYER_VERIFY alt residual-on-normed-input maxrel={alt_residual:.3e} (must be large)")
    if min(alt_missing_z, alt_residual) <= MAXREL:
        print("LAYER_VERIFY_FAIL a mis-wiring matches the authority — the stage check proves nothing")
        return 1

    print(f"LAYER_VERIFY_SUMMARY stages={len(order)} worst_within={MAXREL:g} "
          f"rejected_alternatives={alt_missing_z:.2e}/{alt_residual:.2e}")
    print(f"LAYER_VERIFY_DONE the layer wiring reproduces ggml's own graph within maxrel {MAXREL:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
