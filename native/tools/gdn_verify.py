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


def run_op(q, k, v, g, be, st, T, B, tag="v"):
    """One call to ggml's op, returning (scores, new_state) in numpy order."""
    inp, outp, stp = (os.path.join(BUILD, f"{tag}_{n}.bin") for n in ("in", "out", "st"))
    with open(inp, "wb") as fh:
        for a in (q, k, v, g, be, st):
            fh.write(a.astype("<f4").tobytes())
    r = subprocess.run([BIN, inp, outp, stp, str(S), str(H_K), str(H_V), str(T), str(B), "1"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("GDN_VERIFY_FAIL run: " + (r.stdout + r.stderr)[:300])
        return None, None
    return (np.fromfile(outp, dtype="<f4").reshape(B, T, H_V, S).astype(np.float64),
            np.fromfile(stp, dtype="<f4").reshape(B, H_V, S, S).astype(np.float64))


def maxabs(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def multi_token_checks(q, k, v, g, be, st, ref, rsts, T, B):
    """The multi-token characterisation (VybForge#10 phase 4, W6), checked rather than asserted.

    ggml's op has ONE compute path (ggml-cpu/ops.cpp: ggml_compute_forward_gated_delta_net_f32 ->
    ..._one_chunk, whose only loop over tokens is `for (t = 0; t < n_tokens; ++t)` inside the head/seq
    chunk; the `K` op param is a SNAPSHOT-SLOT COUNT, not a kernel selector). Two consequences are
    measurable, and both are required here EXACTLY (bit-for-bit), because they are structural rather
    than numeric:

      1. REPLAY: T single-token calls, carrying the state forward, reproduce the one T-token call —
         same scores and same final state, exactly.
      2. CAUSALITY: changing ONLY the last token's inputs leaves every earlier token's scores
         bit-identical (and does change the last one, so the check can fail).

    An earlier note in this repo claimed a second, differently-arranged kernel for several tokens, on
    the evidence that token 0's output changed when more tokens followed. That observation was an
    artefact of a transposed read, and it is withdrawn; these two checks are what replaced it."""
    ok = True
    stc = st
    outs = []
    for t in range(T):
        ot, stc = run_op(q[:, t:t + 1], k[:, t:t + 1], v[:, t:t + 1], g[:, t:t + 1], be[:, t:t + 1],
                         stc, 1, B, tag="vseq")
        if ot is None:
            return False
        outs.append(ot)
    seq_out = np.concatenate(outs, axis=1)
    d_o, d_s = maxabs(seq_out, ref), maxabs(stc, rsts)
    print(f"GDN_VERIFY replay  {T} x T=1 calls == one T={T} call:  out maxabs={d_o:.3e} "
          f"state maxabs={d_s:.3e}  {'IDENTICAL' if (d_o == 0.0 and d_s == 0.0) else 'DIFFERS'}")
    ok = ok and d_o == 0.0 and d_s == 0.0

    valt = v.copy()
    valt[:, T - 1] *= -0.5                      # only the LAST token differs
    alt, _ = run_op(q, k, valt, g, be, st, T, B, tag="valt")
    if alt is None:
        return False
    d_prev = maxabs(alt[:, :T - 1], ref[:, :T - 1])
    d_last = maxabs(alt[:, T - 1], ref[:, T - 1])
    print(f"GDN_VERIFY causal  only the last token changed: earlier tokens maxabs={d_prev:.3e} "
          f"({'IDENTICAL' if d_prev == 0.0 else 'DIFFERS'}, must be identical), "
          f"last token maxabs={d_last:.3e} ({'differs' if d_last > 0 else 'IDENTICAL'}, must differ)")
    ok = ok and d_prev == 0.0 and d_last > 0.0
    return ok


def main():
    T, B = 1, 1
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--tokens":
            T = int(argv[i + 1])
        if a == "--seqs":
            B = int(argv[i + 1])
    print(f"GDN_VERIFY geometry S={S} H_k={H_K} H_v={H_V} T={T} B={B}")
    if not build():
        return 1

    q, k, v, g, be, st = inputs(T, B)
    ref, rsts = run_op(q, k, v, g, be, st, T, B)
    if ref is None or rsts is None:
        return 1

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

    mt_ok = True
    if T > 1:
        mt_ok = multi_token_checks(q, k, v, g, be, st, ref, rsts, T, B)
        if not mt_ok:
            verdict = 1

    if verdict == 0:
        print(f"GDN_VERIFY_SUMMARY broadcast={mode} T={T} maxrel_out={ro:.3e} maxrel_state={rs:.3e}"
              + (f" multi_token_verified=1" if T > 1 else ""))
        print(f"GDN_VERIFY_DONE the port reproduces ggml's op within maxrel {MAXREL:g}"
              + (f" at T={T}, with the multi-token path characterised" if T > 1 else ""))
    else:
        print("GDN_VERIFY_FAIL neither broadcast reproduces the op"
              + ("" if mt_ok else " (or the multi-token characterisation failed)"))
    return verdict


if __name__ == "__main__":
    sys.exit(main())
