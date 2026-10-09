#!/usr/bin/env python3
"""Ridge's attention stages: ggml's own ops vs the spec the engine intends (phase 4, unit 10 step 3a).

The risk this measures is the JOINT Q+gate projection. `attn_q.weight` is one matrix
(`5120 x 12288`) that carries, per head, a 256-dim q block followed by a 256-dim gate block, and
`native/tools/attn_authority.c` performs the split with `ggml_view_3d` exactly as llama.cpp does. A
re-reading that swapped the halves, or took the RMS norm over the whole projection instead of per head,
would still produce plausible numbers — so each stage is compared, and the alternatives are required to
differ.

Stages: qg (pre-split), q_pre/gate_pre (post-split), q_norm/k_norm (per-head RMS), q_rope/k_rope
(NEOX rope over the first n_dims). Attention, the output gate and `wo` are the next increment.

Usage: attn_verify.py    (env: CC, VYBFORGE_LLAMA, VYBFORGE_ATTN_MAXREL)
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rope_verify as rv

REPO = rv.REPO
LLAMA = rv.LLAMA
BUILD = os.path.join(REPO, "native/build")
BIN = os.path.join(BUILD, "attn_authority")
IN = os.path.join(BUILD, "attn_authority_in.bin")
OUT = os.path.join(BUILD, "attn_authority_out.bin")

MAXREL = float(os.environ.get("VYBFORGE_ATTN_MAXREL", "1e-4"))
EPS = 1e-6
ND = 64                                     # Ridge's rope.dimension_count
SECTIONS = (11, 11, 10, 0)
BASE = 1e7                                  # ridge_theta
# A geometry that keeps the shapes honest while staying small: the interleave is per head, so two
# heads are enough to pin it, and head_dim stays at Ridge's real 256 (the norm and rope are per head).
NH, HD, NKV, S, D = 3, 256, 2, 3, 64


def build_authority():
    cmd = [os.environ.get("CC", "gcc"), "-O2", "-o", BIN, os.path.join(HERE, "attn_authority.c"),
           f"-I{LLAMA}/ggml/include", f"-L{LLAMA}/build/bin",
           "-lggml", "-lggml-base", "-lggml-cpu", f"-Wl,-rpath,{LLAMA}/build/bin"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("ATTN_VERIFY_FAIL build: " + (r.stderr.strip()[:400] or r.stdout.strip()[:400]))
        return False
    return True


def run_authority(Wqg, Wk, nmq, nmk, hid):
    with open(IN, "wb") as fh:
        for a in (Wqg, Wk, nmq, nmk, hid):
            fh.write(np.ascontiguousarray(a, dtype="<f4").tobytes())
    r = subprocess.run([BIN, IN, OUT, str(NH), str(HD), str(NKV), str(S), str(D), str(ND), str(EPS),
                        *[str(x) for x in SECTIONS]], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(("attn authority", r.stdout + r.stderr)[:300])
    shapes = {}
    for line in r.stdout.splitlines():
        if line.startswith("ATTN_STAGE "):
            p = dict(kv.split("=") for kv in line.split()[2:])
            shapes[line.split()[1]] = int(p["n"])
    data = np.fromfile(OUT, dtype="<f4")
    out, off = {}, 0
    for nm in ("qg", "q_pre", "gate_pre", "q_norm", "q_rope", "k_norm", "k_rope"):
        n = shapes[nm]
        out[nm] = data[off:off + n].astype(np.float64)
        off += n
    return out


def rms(x, w, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    r = x / np.sqrt(np.mean(x * x, axis=axis, keepdims=True) + EPS)
    return r * w


def neox(x, pos, n_rot):
    """NEOX pairing inside the first n_rot dims of x's last axis (the P4.6/P4.7-verified layout)."""
    out = x.copy()
    x = x.copy()
    half = n_rot // 2
    inv = np.array([BASE ** (-2.0 * m / n_rot) for m in range(half)])
    for k in range(half):
        ang = pos * inv[k]
        c, s = np.cos(ang), np.sin(ang)
        # broadcast against the SLICE x[..., k], not against x: (S, n_head) needs c shaped (S, 1)
        c = c.reshape((-1,) + (1,) * (x[..., k].ndim - 1))
        s = s.reshape((-1,) + (1,) * (x[..., k].ndim - 1))
        a, b = x[..., k].copy(), x[..., k + half].copy()
        out[..., k] = a * c - b * s
        out[..., k + half] = a * s + b * c
    return out


def spec(Wqg, Wk, nmq, nmk, hid, *, gate_first=False, norm_whole=False, rope_whole=False):
    """The engine's intended reading of the block, stage by stage. `pos` is (S,)."""
    pos = np.arange(S, dtype=np.float64)
    qg = hid @ Wqg.T                                     # (S, nh*2*hd) -- ggml ne = (nqg, S)
    qg = qg.reshape(S, NH, 2 * HD)                       # per head: q block then gate block
    if gate_first:
        gate_pre, q_pre = qg[..., :HD], qg[..., HD:]
    else:
        q_pre, gate_pre = qg[..., :HD], qg[..., HD:]
    if norm_whole:
        # the misreading: one RMS norm across the whole per-head 512, applied to q
        joined = qg.reshape(S, NH * 2 * HD)
        q_pre = rms(joined, np.concatenate([np.repeat(nmq, NH), np.repeat(nmq, NH)]))[:, :NH * HD]
        q_pre = q_pre.reshape(S, NH, HD)
    q_norm = q_pre if norm_whole else rms(q_pre, nmq)
    gate_eff = gate_pre if norm_whole else gate_pre
    q_rope = neox(q_norm, pos, HD if rope_whole else ND)
    kflat = hid @ Wk.T                                   # (S, nkv*hd)
    k_pre = kflat.reshape(S, NKV, HD)
    k_norm = rms(k_pre, nmk)
    k_rope = neox(k_norm, pos, HD if rope_whole else ND)
    return {"qg": qg.reshape(S, -1).ravel(), "q_pre": q_pre.ravel(), "gate_pre": gate_eff.ravel(),
            "q_norm": q_norm.ravel(), "q_rope": q_rope.ravel(), "k_norm": k_norm.ravel(),
            "k_rope": k_rope.ravel()}


def main():
    print(f"ATTN_VERIFY geometry n_head={NH} head_dim={HD} n_kv={NKV} S={S} D={D} n_dims={ND} eps={EPS:g}")
    if not os.path.exists(os.path.join(LLAMA, "ggml/include/ggml.h")):
        print(f"ATTN_VERIFY_SKIP no llama.cpp checkout at {LLAMA}")
        return 0
    if not build_authority():
        return 1

    rng = np.random.default_rng(777)
    Wqg = rng.normal(0, 0.02, size=(NH * 2 * HD, D)).astype(np.float32)
    Wk = rng.normal(0, 0.02, size=(NKV * HD, D)).astype(np.float32)
    nmq = rng.normal(1.0, 0.05, size=(HD,)).astype(np.float32)
    nmk = rng.normal(1.0, 0.05, size=(HD,)).astype(np.float32)
    hid = rng.normal(0, 1.0, size=(S, D)).astype(np.float32)

    ref = run_authority(Wqg, Wk, nmq, nmk, hid)
    print("ATTN_VERIFY stage comparison against ggml's own ops (ggml is f32, the engine f64):")
    mine = spec(Wqg, Wk, nmq, nmk, hid)
    worst, bad = 0.0, []
    for nm in ("qg", "q_pre", "gate_pre", "q_norm", "q_rope", "k_norm", "k_rope"):
        r = rv.rel(mine[nm], ref[nm])
        worst = max(worst, r)
        print(f"ATTN_VERIFY stage {nm:9s} maxrel={r:.3e} {'ok' if r <= MAXREL else 'MISMATCH'}")
        if r > MAXREL:
            bad.append(nm)
    if bad:
        print(f"ATTN_VERIFY_FAIL the spec does not reproduce ggml for: {', '.join(bad)} "
              f"(worst {worst:.3e}) — do not build engine code on this reading")
        return 1

    # The teeth: each alternative reading must MISS, or the check is not measuring the reading.
    alts = (("gate-first split", dict(gate_first=True)), ("norm over the whole projection", dict(norm_whole=True)),
            ("rope over the whole head", dict(rope_whole=True)))
    for nm, kw in alts:
        a = spec(Wqg, Wk, nmq, nmk, hid, **kw)
        r = max(rv.rel(a[k], ref[k]) for k in ("q_norm", "q_rope"))
        ok = r > MAXREL * 100
        print(f"ATTN_VERIFY alt {nm:30s} maxrel={r:.3e} {'rejected' if ok else 'NOT DISTINGUISHED'}")
        if not ok:
            print(f"ATTN_VERIFY_FAIL the alternative '{nm}' is not distinguished from the spec — this "
                  f"check cannot tell them apart, so a pass here proves nothing")
            return 1

    print(f"ATTN_VERIFY_DONE {len(ref)} stages reproduce ggml's ops within {MAXREL:g} (worst {worst:.3e}), "
          f"and all {len(alts)} alternative readings are rejected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
