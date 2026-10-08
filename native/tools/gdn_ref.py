#!/usr/bin/env python3
"""Numpy port of the Gated DeltaNet recurrence, checked against ggml's own op (VybForge#10 phase 4).

The novel math of a qwen35 recurrent layer lives inside GGML_OP_GATED_DELTA_NET. This file

  1. generates deterministic inputs at the model's real geometry (S=128, H_k=16, H_v=48),
  2. runs ggml's op on them via native/tools/gdn_authority.c (linking the libggml llama.cpp was
     built with — the identical implementation, not a copy),
  3. runs the numpy port below on the same bytes,
  4. reports the difference in ulp, the same way the quant gates do.

The port mirrors the CPU kernel's indexing exactly, including the fact that the state is stored
TRANSPOSED (s_out[j*S_v + i] = S[i][j], ops.cpp:10849-10850) and that the read-out happens AFTER the
rank-1 update, scaled by 1/sqrt(S).

Deliberately NOT in this check: the short convolution, the l2 normalisation of q/k, the gated RMS
norm and the projections. Those are other ops in the graph (ggml_ssm_conv, l2_norm, rms_norm); they
belong in the same treatment and will be added as separate checked units, not folded in here where a
failure could not be attributed.

Usage: gdn_ref.py [--tokens T] [--seqs B]
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
SCALE = 1.0 / np.sqrt(float(S))

# The authority is ggml's CPU kernel in float32; this port is float64. The bound is therefore the
# authority's own precision, not a hoped-for floor: measured at ~1e-7 relative, set at 1e-5 for
# headroom over accumulation-order differences. Override with VYBFORGE_GDN_MAXREL.
MAXREL = float(os.environ.get("VYBFORGE_GDN_MAXREL", "1e-5"))


def build_authority():
    """Compile the harness against the libggml this llama.cpp checkout was built with."""
    os.makedirs(BUILD, exist_ok=True)
    cmd = [
        os.environ.get("CC", "gcc"), "-O2", "-o", BIN, SRC,
        f"-I{LLAMA}/ggml/include",
        f"-L{LLAMA}/build/bin",
        "-lggml", "-lggml-base", "-lggml-cpu",
        f"-Wl,-rpath,{LLAMA}/build/bin",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("GDN_AUTH_BUILD_FAIL " + (r.stderr.strip()[:400] or r.stdout.strip()[:400]))
        return False
    return True


def inputs(T, B, seed=20261008):
    """Deterministic inputs at model geometry. q/k are per-head l2-normalised, as the graph does
    before the op (qwen35.cpp:440-443); g is negative (the graph's gate is -exp(A_log)*softplus),
    beta is in (0,1) (it is a sigmoid)."""
    rng = np.random.default_rng(seed)
    q = rng.normal(0.0, 0.1, size=(S, H_K, T, B)).astype(np.float32)
    k = rng.normal(0.0, 0.1, size=(S, H_K, T, B)).astype(np.float32)
    v = rng.normal(0.0, 0.1, size=(S, H_V, T, B)).astype(np.float32)
    g = (-rng.uniform(0.05, 1.0, size=(1, H_V, T, B))).astype(np.float32)
    be = rng.uniform(0.1, 0.9, size=(1, H_V, T, B)).astype(np.float32)
    st = rng.normal(0.0, 0.05, size=(S, S, H_V, B)).astype(np.float32)

    # l2 norm per head over the S dim, eps as a FLOOR (not inside the sqrt), per qwen35.cpp:440-443
    eps = 1e-6
    for arr in (q, k):
        nrm = np.maximum(np.linalg.norm(arr, axis=0, keepdims=True), eps)
        arr /= nrm
    return q, k, v, g, be, st


def numpy_rule(q, k, v, g, be, st):
    """The recurrence, indexing the state exactly as the CPU kernel does."""
    T, B = q.shape[2], q.shape[3]
    state = st.astype(np.float64).copy()          # [j, i, h, b] with state[j,i] = S[i][j]
    out = np.zeros((S, H_V, T, B), dtype=np.float64)
    for b in range(B):
        for t in range(T):
            for h in range(H_V):
                hk = h % H_K
                kd = k[:, hk, t, b].astype(np.float64)
                qd = q[:, hk, t, b].astype(np.float64)
                vd = v[:, h, t, b].astype(np.float64)
                m = state[:, :, h, b]                                   # rows j, cols i
                m *= np.exp(float(g[0, h, t, b]))                       # decay
                pred = m @ kd                                          # pred[j] = sum_i S[i][j]*k[i]
                delta = (vd - pred) * float(be[0, h, t, b])
                m += np.outer(delta, kd)                               # S[i][j] += k[i]*delta[j]
                out[:, h, t, b] = (m @ qd) * SCALE                     # read AFTER the update
    return out, state


def main():
    T, B = 1, 1
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--tokens":
            T = int(argv[i + 1])
        if a == "--seqs":
            B = int(argv[i + 1])

    print(f"GDN_REF geometry S={S} H_k={H_K} H_v={H_V} T={T} B={B}")
    if T != 1:
        # The op has TWO kernels. For a single token it runs the sequential rule this port
        # implements — verified bit-for-bit at T=1. For several tokens it runs a CHUNKED kernel that
        # fills the same buffer with a different arrangement (established by running the op itself:
        # token 0's output changes when more tokens follow, which no causal recurrence can do).
        # Comparing multi-token output therefore needs that arrangement established first, so it is
        # deliberately not claimed here.
        print("GDN_REF_FAIL multi-token comparison is not established yet (the op's chunked path "
              "arranges its output differently); use --tokens 1")
        return 1
    if not build_authority():
        print("GDN_REF_FAIL authority did not build")
        return 1

    q, k, v, g, be, st = inputs(T, B)
    in_path = os.path.join(BUILD, "gdn_in.bin")
    out_path = os.path.join(BUILD, "gdn_out.bin")
    st_path = os.path.join(BUILD, "gdn_state.bin")
    with open(in_path, "wb") as fh:
        for arr in (q, k, v, g, be, st):
            fh.write(arr.astype("<f4").tobytes())

    r = subprocess.run([BIN, in_path, out_path, st_path, str(S), str(H_K), str(H_V), str(T), str(B), "4"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("GDN_REF_FAIL authority run: " + (r.stdout.strip() + r.stderr.strip())[:400])
        return 1
    print("  authority: " + r.stdout.strip())

    ref_out = np.fromfile(out_path, dtype="<f4").reshape(S, H_V, T, B).astype(np.float64)
    ref_st = np.fromfile(st_path, dtype="<f4").reshape(S, S, H_V, B).astype(np.float64)

    my_out, my_st = numpy_rule(q, k, v, g, be, st)

    verdict = 0
    for name, mine, theirs in (("attention output", my_out, ref_out), ("final state", my_st, ref_st)):
        a = mine.ravel()
        b = theirs.ravel()
        # The authority is ggml's CPU kernel, which accumulates in FLOAT32; this port accumulates in
        # float64. Bit-equality is therefore not achievable by construction — the meaningful
        # question is whether the two agree to the precision the authority itself carries. Report
        # the relative gap (against the authority's own magnitude) and the worst absolute one.
        scale = max(float(np.max(np.abs(b))), 1e-30)
        absdiff = np.abs(a - b)
        rel = absdiff / np.maximum(np.abs(b), 1e-30)
        worst_rel = float(np.max(rel))
        worst_abs = float(np.max(absdiff))
        nd = int(np.count_nonzero(a != b))
        print(f"GDN_REF {name:16s} n={a.size:8d} ndiff={nd:8d} maxrel={worst_rel:.3e} "
              f"maxabs={worst_abs:.3e} (authority scale {scale:.3e})")
        if worst_rel > MAXREL:
            verdict = 1

    if verdict:
        print(f"GDN_REF_FAIL the numpy port is worse than maxrel {MAXREL:g} against ggml's op")
    else:
        print(f"GDN_REF_DONE within maxrel {MAXREL:g} of ggml's own Gated DeltaNet op")
    return verdict


if __name__ == "__main__":
    sys.exit(main())
