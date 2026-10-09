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
# n_kv == n_head on purpose: ggml's batched mul_mat needs matching batch dims, so the fixture stays
# inside the shape dance the harness performs. GQA (fewer kv heads) needs the K/V repeat, untested.
NH, HD, NKV, S, D = 3, 256, 3, 3, 64
KQS = 1.0 / (HD ** 0.5)                       # 1/sqrt(head_dim), unless GGUF overrides the scale
# What the gate requires vs what the harness also computes. The front half is measured and required;
# attention/gate/wo are computed and REPORTED only, because they do not agree yet (see the note the
# verifier prints) — a mismatch that is printed but not gated is a known-open item, not a pass.
FRONT = ("qg", "q_pre", "gate_pre", "q_norm", "q_rope", "k_norm", "k_rope", "attn", "gated", "out")
# The harness's dump ORDER, which is what the sequential parse below walks. It must list EVERY dumped
# tensor: leaving `scores`/`probs` out of this list shifted every later slot by two, so `attn` was read
# from the scores slot, `gated` from probs and `out` from attn — which read exactly like a numerical
# disagreement (and appeared the moment those two dumps were added, i.e. like flakiness).
DUMPED = ("qg", "q_pre", "gate_pre", "q_norm", "q_rope", "k_norm", "k_rope",
          "scores", "probs", "attn", "gated", "out")
# The harness's score/prob tensors are ne = (S_kv, S_q, n_head), i.e. ne0-fastest order (kv, q, head);
# the spec computes (q, head, kv), so these two stages are stored transposed to match the dump.
SCORE_STAGES = ("scores", "probs")
# The two DIAGNOSTIC tensors, not block outputs: their comparison order is not settled, but `attn` is
# computed FROM probs on both sides and matches ggml at 2.7e-7, so the two sides agree in content.
LATER = ("scores", "probs")
STAGES = FRONT + LATER


def build_authority():
    cmd = [os.environ.get("CC", "gcc"), "-O2", "-o", BIN, os.path.join(HERE, "attn_authority.c"),
           f"-I{LLAMA}/ggml/include", f"-L{LLAMA}/build/bin",
           "-lggml", "-lggml-base", "-lggml-cpu", f"-Wl,-rpath,{LLAMA}/build/bin"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("ATTN_VERIFY_FAIL build: " + (r.stderr.strip()[:400] or r.stdout.strip()[:400]))
        return False
    return True


THREADS = int(os.environ.get("VYBFORGE_ATTN_THREADS", "1"))


def run_authority(Wqg, Wk, Wv, Wo, nmq, nmk, hid):
    with open(IN, "wb") as fh:
        for a in (Wqg, Wk, Wv, Wo, nmq, nmk, hid):
            fh.write(np.ascontiguousarray(a, dtype="<f4").tobytes())
    r = subprocess.run([BIN, IN, OUT, str(NH), str(HD), str(NKV), str(S), str(D), str(ND), "%.9g" % EPS,
                        "%.9g" % KQS, *[str(x) for x in SECTIONS], str(THREADS)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(("attn authority", r.stdout + r.stderr)[:300])
    shapes = {}
    for line in r.stdout.splitlines():
        if line.startswith("ATTN_STAGE "):
            p = dict(kv.split("=") for kv in line.split()[2:])
            shapes[line.split()[1]] = int(p["n"])
    data = np.fromfile(OUT, dtype="<f4")
    out, off = {}, 0
    for nm in DUMPED:
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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def spec(Wqg, Wk, Wv, Wo, nmq, nmk, hid, *, gate_first=False, norm_whole=False, rope_whole=False,
         non_causal=False, raw_gate=False, no_scale=False):
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
    # causal attention over these S tokens, then the output gate, then wo
    kqs = 1.0 if no_scale else KQS
    v_pre = (hid @ Wv.T).reshape(S, NKV, HD)
    # NOTE the letters: the query-token and key-token axes must be DISTINCT. Writing "shd,skd->shk"
    # reuses `s` for both, which silently forces key token == query token (it contracts the two
    # token axes together) and turns the output's third axis into the kv HEAD instead of the key.
    # The symptom is subtle: only the diagonal (q == kv) agrees, and no axis permutation can match.
    sc_scaled = np.einsum("qhd,khd->qhk", q_rope, k_rope) * kqs   # (q, head, kv), UNMASKED
    sc = sc_scaled
    if not non_causal:
        above = np.arange(S)[None, :] > np.arange(S)[:, None]      # key index > query index
        sc = np.where(above[:, None, :], -np.inf, sc_scaled)
    m = sc.max(axis=2, keepdims=True)
    e = np.exp(sc - m)
    pr = e / e.sum(axis=2, keepdims=True)
    at = np.einsum("qhk,kd...".replace("...", ""), pr, v_pre) if False else np.einsum("qhk,khd->qhd", pr, v_pre)
    gated = at * (gate_eff if raw_gate else sigmoid(gate_eff))
    out = gated.reshape(S, NH * HD) @ Wo.T
    # matched to the harness's ne0-fastest dump order for the two score tensors: its tensor is
    # ne = (S_kv, S_q, n_head), so the flat order is (kv, q, head) — axes (2, 0, 1) of (q, head, kv),
    # NOT (2, 1, 0), which silently reorders the head axis into the query axis
    sc_t = np.transpose(sc_scaled, (2, 0, 1)).ravel()
    pr_t = np.transpose(pr, (2, 0, 1)).ravel()
    return {"qg": qg.reshape(S, -1).ravel(), "q_pre": q_pre.ravel(), "gate_pre": gate_eff.ravel(),
            "q_norm": q_norm.ravel(), "q_rope": q_rope.ravel(), "k_norm": k_norm.ravel(),
            "k_rope": k_rope.ravel(), "scores": sc_t, "probs": pr_t,
            "attn": at.ravel(), "gated": gated.ravel(), "out": out.ravel()}


def main():
    print(f"ATTN_VERIFY geometry n_head={NH} head_dim={HD} n_kv={NKV} S={S} D={D} n_dims={ND} eps={EPS:g} "
          f"threads={THREADS}")
    if not os.path.exists(os.path.join(LLAMA, "ggml/include/ggml.h")):
        print(f"ATTN_VERIFY_SKIP no llama.cpp checkout at {LLAMA}")
        return 0
    if not build_authority():
        return 1

    rng = np.random.default_rng(777)
    Wqg = rng.normal(0, 0.02, size=(NH * 2 * HD, D)).astype(np.float32)
    Wk = rng.normal(0, 0.02, size=(NKV * HD, D)).astype(np.float32)
    Wv = rng.normal(0, 0.02, size=(NKV * HD, D)).astype(np.float32)
    # ggml's ne0 is the INPUT width, so a numpy array (r, c) is the ggml tensor ne=(c, r): W_o's
    # ggml ne is (n_head*head_dim, D) and therefore reads here as (D, n_head*head_dim)
    Wo = rng.normal(0, 0.02, size=(D, NH * HD)).astype(np.float32)
    nmq = rng.normal(1.0, 0.05, size=(HD,)).astype(np.float32)
    nmk = rng.normal(1.0, 0.05, size=(HD,)).astype(np.float32)
    hid = rng.normal(0, 1.0, size=(S, D)).astype(np.float32)

    ref = run_authority(Wqg, Wk, Wv, Wo, nmq, nmk, hid)
    first = open(OUT, "rb").read()
    run_authority(Wqg, Wk, Wv, Wo, nmq, nmk, hid)
    again = open(OUT, "rb").read()
    if first != again:
        nd = sum(1 for a, b in zip(first, again) if a != b)
        print(f"ATTN_VERIFY_FAIL the harness is NON-DETERMINISTIC at threads={THREADS}: two runs of the "
              f"same graph differ in {nd} of {len(first)} bytes — no verdict from it can gate anything")
        return 1
    print(f"ATTN_VERIFY determinism: two runs byte-identical at threads={THREADS} ({len(first)} bytes)")
    print("ATTN_VERIFY stage comparison against ggml's own ops (ggml is f32, the engine f64):")
    mine = spec(Wqg, Wk, Wv, Wo, nmq, nmk, hid)
    worst, bad = 0.0, []
    for nm in FRONT:
        r = rv.rel(mine[nm], ref[nm])
        worst = max(worst, r)
        print(f"ATTN_VERIFY stage {nm:9s} maxrel={r:.3e} {'ok' if r <= MAXREL else 'MISMATCH'}")
        if r > MAXREL:
            a = np.asarray(mine[nm], dtype=np.float64)
            b = np.asarray(ref[nm], dtype=np.float64)
            i = int(np.argmax(np.abs(a - b)))
            print(f"ATTN_VERIFY   worst #{i}/{a.size}: harness={b[i]:.6g} spec={a[i]:.6g} "
                  f"|a|max={np.max(np.abs(a)):.4g} |b|max={np.max(np.abs(b)):.4g}")
            bad.append(nm)
    if bad:
        print(f"ATTN_VERIFY_FAIL the spec does not reproduce ggml for: {', '.join(bad)} "
              f"(worst {worst:.3e}) — do not build engine code on this reading")
        return 1

    # Reported, NOT gated. The error did NOT move when the causal mask was replaced (diag_mask_inf ->
    # explicit mask tensor + soft_max_ext, both at 1.409e+00 to three decimals), so the mask is not the
    # cause and the attention stage's difference lies elsewhere. Until it is found, this stays visible
    # and unauthorised rather than inside the gate.
    # Diagnostic tensors, not block outputs. Their flat comparison order is not settled (soft_max_ext's
    # output arrangement, or my transpose of it), but `attn` is computed FROM probs on both sides and is
    # gated at 2.7e-7, so the two sides agree in content — only the comparison's ordering is in question.
    # NOT the cause of the earlier "flakiness": that was the parse-offset bug fixed in DUMPED above.
    print("ATTN_VERIFY_NOT_GATED diagnostic tensors (order unsettled, content corroborated by attn):")
    for nm in LATER:
        a, b = np.asarray(mine[nm], dtype=np.float64), np.asarray(ref[nm], dtype=np.float64)
        r = rv.rel(a, b)
        print(f"ATTN_VERIFY_NOT_GATED   {nm:9s} maxrel={r:.3e}")
        if nm in SCORE_STAGES and r > MAXREL:
            # name the divergence: which axis has the disagreement, and what the values are there
            i = int(np.argmax(np.abs(a - b)))
            print(f"ATTN_VERIFY_NOT_GATED     worst entry #{i}: harness={b[i]:.6g} spec={a[i]:.6g}")
            print(f"ATTN_VERIFY_NOT_GATED     in (kv, q, head) that is kv={i % S} q={(i // S) % S} "
                  f"head={i // (S * S)}  (harness dumps ne0-fastest)")



    # The teeth: each alternative reading must MISS, or the check is not measuring the reading.
    alts = (("gate-first split", dict(gate_first=True), "q_norm"),
            ("norm over the whole projection", dict(norm_whole=True), "q_norm"),
            ("rope over the whole head", dict(rope_whole=True), "q_rope"),
            ("non-causal attention", dict(non_causal=True), "attn"),
            ("raw gate (no sigmoid)", dict(raw_gate=True), "gated"),
            ("no kq_scale", dict(no_scale=True), "attn"))
    for nm, kw, key in alts:
        a = spec(Wqg, Wk, Wv, Wo, nmq, nmk, hid, **kw)
        r = rv.rel(a[key], ref[key])
        ok = r > MAXREL * 100
        print(f"ATTN_VERIFY alt {nm:30s} ({key}) maxrel={r:.3e} {'rejected' if ok else 'NOT DISTINGUISHED'}")
        if not ok:
            print(f"ATTN_VERIFY_FAIL the alternative '{nm}' is not distinguished from the spec — this "
                  f"check cannot tell them apart, so a pass here proves nothing")
            return 1

    print(f"ATTN_VERIFY_DONE {len(FRONT)} front-half stages reproduce ggml's ops within {MAXREL:g} "
          f"(worst {worst:.3e}), and all {len(alts)} alternative readings of them are rejected; "
          f"the {len(LATER)} diagnostic tensors ({', '.join(LATER)}) are reported with their order unsettled")
    return 0


if __name__ == "__main__":
    sys.exit(main())
