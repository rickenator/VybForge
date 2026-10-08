#!/usr/bin/env python3
"""Gated DeltaNet reference, corrected for axis order, checked against ggml's own op.

THE CONVENTION THIS FILE EXISTS TO GET RIGHT: ggml stores ne[0] contiguously (the fastest axis),
while numpy's fastest axis is the LAST one. So an ggml tensor of ne (S, H, T, B) is a numpy array of
shape (B, T, H, S) — the last axis is S. The state, ne (S_v, S_v, H, B), is (B, H, S_v, S_v) with the
last axis being i and the one before it j. Reading those leading axes as if they were ggml's is what
made an earlier reference of mine disagree with the op at model geometry and agree at unit strides.

Self-contained on purpose: it builds the harness (native/tools/gdn_authority.c), generates inputs,
runs the op, runs the port, and reports both broadcasts, in one process.

Usage: gdn_verify.py [--tokens T] [--seqs B]
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
LLAMA = os.environ.get("VYBFORGE_LLAMA", os.path.expanduser("~/Projects/llama.cpp"))
BUILD = os.path.join(REPO, "native/build")
SRC = os.path.join(HERE, "gdn_authority.c")
BIN = os.path.join(BUILD, "gdn_authority")

S = 128        # head_k_dim = head_v_dim = ssm.state_size
H_K = 16       # ssm.group_count
H_V = 48       # ssm.time_step_rank
SCALE = 1.0 / float(np.sqrt(S))
MAXREL = float(os.environ.get("VYBFORGE_GDN_MAXREL", "1e-5"))


def build():
    os.makedirs(BUILD, exist_ok=True)
    cmd = [os.environ.get("CC", "gcc"), "-O2", "-o", BIN, SRC,
           f"-I{LLAMA}/ggml/include", f"-L{LLAMA}/build/bin",
           "-lggml", "-lggml-base", "-lggml-cpu", f"-Wl,-rpath,{LLAMA}/build/bin"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("GDN_VERIFY_FAIL build: " + (r.stderr.strip()[:300] or r.stdout.strip()[:300]))
        return False
    return True


def inputs(T, B, seed=20261008):
    """Deterministic inputs in ggml's memory order: numpy shape (B, T, H, ne0)."""
    rng = np.random.default_rng(seed)
    q = rng.normal(0.0, 0.1, size=(B, T, H_K, S)).astype(np.float32)
    k = rng.normal(0.0, 0.1, size=(B, T, H_K, S)).astype(np.float32)
    v = rng.normal(0.0, 0.1, size=(B, T, H_V, S)).astype(np.float32)
    g = (-rng.uniform(0.05, 1.0, size=(B, T, H_V, 1))).astype(np.float32)
    be = rng.uniform(0.1, 0.9, size=(B, T, H_V, 1)).astype(np.float32)
    st = rng.normal(0.0, 0.05, size=(B, H_V, S, S)).astype(np.float32)   # [b, h, j, i]
    eps = 1e-6
    for arr in (q, k):
        nrm = np.maximum(np.linalg.norm(arr, axis=-1, keepdims=True), eps)   # over i, a floor
        arr /= nrm
    return q, k, v, g, be, st


def rule(q, k, v, g, be, st, bcast="mod"):
    B, T = q.shape[0], q.shape[1]
    m_all = st.astype(np.float64).copy()          # [b, h, j, i]
    out = np.zeros((B, T, H_V, S), dtype=np.float64)
    rep = H_V // H_K
    for b in range(B):
        for t in range(T):
            for h in range(H_V):
                hk = (h % H_K) if bcast == "mod" else (h // rep)
                kd = k[b, t, hk].astype(np.float64)
                m = m_all[b, h]
                m *= np.exp(float(g[b, t, h, 0]))
                delta = (v[b, t, h].astype(np.float64) - m @ kd) * float(be[b, t, h, 0])
                m += np.outer(delta, kd)
                out[b, t, h] = (m @ q[b, t, hk].astype(np.float64)) * SCALE
    return out, m_all


def main():
    T, B = 1, 1
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--tokens":
            T = int(argv[i + 1])
        if a == "--seqs":
            B = int(argv[i + 1])
    if T != 1:
        print("GDN_VERIFY_FAIL multi-token uses the op's chunked path, not established; use --tokens 1")
        return 1
    print(f"GDN_VERIFY geometry S={S} H_k={H_K} H_v={H_V} T={T} B={B}")
    if not build():
        return 1

    q, k, v, g, be, st = inputs(T, B)
    inp, outp, stp = (os.path.join(BUILD, n) for n in ("v_in.bin", "v_out.bin", "v_st.bin"))
    with open(inp, "wb") as fh:
        for a in (q, k, v, g, be, st):
            fh.write(a.astype("<f4").tobytes())
    r = subprocess.run([BIN, inp, outp, stp, str(S), str(H_K), str(H_V), str(T), str(B), "1"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("GDN_VERIFY_FAIL run: " + (r.stdout + r.stderr)[:300])
        return 1
    print("  " + r.stdout.strip())

    # the op's buffer: scores (ne0-fastest) then state (ne0-fastest)
    ref = np.fromfile(outp, dtype="<f4").reshape(B, T, H_V, S).astype(np.float64)
    rsts = np.fromfile(stp, dtype="<f4").reshape(B, H_V, S, S).astype(np.float64)

    verdict = 0
    for mode in ("mod", "tile"):
        o, s = rule(q, k, v, g, be, st, bcast=mode)
        ro = float(np.max(np.abs(o - ref)) / max(float(np.max(np.abs(ref))), 1e-30))
        rs = float(np.max(np.abs(s - rsts)) / max(float(np.max(np.abs(rsts))), 1e-30))
        state = "MATCH" if max(ro, rs) <= MAXREL else "differs"
        print(f"GDN_VERIFY broadcast={mode:4s} maxrel out={ro:.3e} state={rs:.3e}  {state}")
        if max(ro, rs) > MAXREL:
            verdict = 1
        else:
            verdict = 0
            break

    if verdict == 0:
        print(f"GDN_VERIFY_DONE the port reproduces ggml's op within maxrel {MAXREL:g}")
    else:
        print("GDN_VERIFY_FAIL neither broadcast reproduces the op")
    return verdict


if __name__ == "__main__":
    sys.exit(main())
