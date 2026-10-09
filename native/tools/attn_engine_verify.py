#!/usr/bin/env python3
"""Ridge's ATTENTION block on the ENGINE's own weight path vs the block built from ggml's ops.

VybForge#10 phase 4, unit 10 step 5.

What this adds over `run_attn_block_gate.sh` (P4.8, the AUTHORITY). P4.8 proved the block's ten
stages against ggml on a SYNTHETIC fixture, so the reading of llama.cpp's attention is right. It
says nothing about whether the ENGINE can run that block: staging the model's own quantized
`attn_q` (a JOINT q+gate matrix, Q5_K), splitting it, the per-head norms, the partial rope, the
causal attention with 24 query heads over 4 kv heads, the output gate and `wo` are all engine
code paths that P4.8 never touched. This runs THE ENGINE'S OWN DRIVER — native/host/model_driver.vyb
in attention-probe mode (`VYB_ATTN_PROBE=<layer>`) — which stages every operand from the live GGUF
by name through its own dequant kernels and multiplies with layer.ptx's `gemm`, and compares its ten
dumped stages against native/tools/attn_authority.c fed the SAME weights (the `gguf` package's
dequantizers, which are the independent reference the Q4_K/Q5_K gates already use).

The GQA question is settled by construction on both sides: the authority REPLICATES each kv group
into its own K/V rows (the form P4.8 verified) and the engine's `attn` kernel groups by
`h / (H/KVH)` — the same function. The verifier replicates the ENGINE's two k stages the same way to
compare them, and the teeth recompute the attention from the ENGINE's own q_rope/k_rope and require
the contiguous grouping to reproduce it and the round-robin one (`h % KVH`) to MISS.

Teeth, all required to be distinguishable:
  * the staged `attn_q`/`attn_output` operands at the exact addresses gemm reads, against the
    model's own dequantised tensors (a good block fed a wrong operand is invisible to stage
    comparison);
  * `attn` recomputed from the engine's own q_rope/k_rope under (a) the contiguous grouping —
    must MATCH, (b) round-robin — must MISS;
  * `gated` with the sigmoid dropped — must MISS.

SKIPs (never PASSes) without the model, the inventory, libggml, the Vyb toolchain or the engine's
tensor index. NaN fails (comparisons are written `not (r <= bar)`).

Usage: attn_engine_verify.py   (env: VYBFORGE_RIDGE_GGUF, VYBFORGE_ATTN_PROBE_LAYER, VYBFORGE_ATTN_S,
                               VYBFORGE_AE_MAXREL, CC, VYBHOME, VYBFORGE_LLAMA)
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

REPO = os.path.dirname(os.path.dirname(HERE))
VYBHOME = os.environ.get("VYBHOME", os.path.expanduser("~/Projects/Vyb"))
VYB = os.environ.get("VYB", os.path.join(VYBHOME, "build", "vyb"))
STDLIB = os.environ.get("VYB_STDLIB", os.path.join(VYBHOME, "stdlib"))
LLAMA = os.environ.get("VYBFORGE_LLAMA", os.path.expanduser("~/Projects/llama.cpp"))
BUILD = os.path.join(REPO, "native/build")
AUTH_BIN = os.path.join(BUILD, "attn_authority")
AUTH_IN = os.path.join(BUILD, "attn_engine_authority_in.bin")
AUTH_OUT = os.path.join(BUILD, "attn_engine_authority_out.bin")
DRV_X = os.path.join(BUILD, "attn_engine_x.bin")
DRV_LOG = os.path.join(BUILD, "attn_engine_driver.log")
TSV = os.path.join(REPO, "native/out/ridge_tensors.tsv")
INVENTORY = os.path.join(REPO, "native/gguf/ridge-3.7bpw-inventory.tsv")
INVFREQ = os.path.join(REPO, "native/out/ridge_invfreq.bin")
MODEL = os.environ.get("VYBFORGE_RIDGE_GGUF",
                       os.path.expanduser("~/Models/qwen38-27b-ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf"))

PROBE_LAYER = int(os.environ.get("VYBFORGE_ATTN_PROBE_LAYER", "3"))   # Ridge's first attention block
S = int(os.environ.get("VYBFORGE_ATTN_S", "4"))
MAXREL = float(os.environ.get("VYBFORGE_AE_MAXREL", "1e-4"))
THREADS = int(os.environ.get("VYBFORGE_ATTN_THREADS", "1"))

# The authority's dump order (attn_authority.c's `stages` table) — the reader walks it POSITIONALLY,
# so it must name every tensor the harness dumps, in order.
DUMPED = ("qg", "q_pre", "gate_pre", "q_norm", "q_rope", "k_norm", "k_rope",
          "scores", "probs", "attn", "gated", "out")
# The ten stages the ENGINE dumps (no scores/probs: the engine's attn kernel does not expose them).
STAGES = ("qg", "q_pre", "gate_pre", "q_norm", "q_rope", "k_norm", "k_rope", "attn", "gated", "out")


def rel(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-30))


# ── the model's own tensors, read from the inventory (absolute offsets) ──────────────────────────
def inv_row(name):
    for line in open(INVENTORY):
        if line.startswith(name + "\t"):
            f = line.split("\t")
            return int(f[2]), f[3], f[1], int(f[5]), int(f[6])   # type_id, type name, dims, offset, bytes
    return None


def real_tensor(name, want_shape):
    """The model's own bytes, dequantised with the independent `gguf` package.

    Same authority chain as the Q4_K/Q5_K gates: `gguf`'s dequantizers are the trusted reference, our
    numpy ports were verified against them, and the GPU kernels were verified bit-exact against our
    ports. The engine stages the raw bytes and dequantises on the GPU; the authority gets these values.
    """
    row = inv_row(name)
    if row is None:
        return None
    type_id, tname, _dims, off, nbytes = row
    with open(MODEL, "rb") as fh:
        fh.seek(off)
        raw = fh.read(nbytes)
    if len(raw) != nbytes:
        return None
    if tname == "F32":
        vals = np.frombuffer(raw, dtype="<f4")
    else:
        import gguf
        vals = gguf.quants.dequantize(np.frombuffer(raw, dtype=np.uint8),
                                      gguf.GGMLQuantizationType(type_id)).astype(np.float32)
    if vals.size != int(np.prod(want_shape)):
        print(f"ATTN_ENGINE_VERIFY_FAIL {name}: {vals.size} values, wanted {int(np.prod(want_shape))}")
        return None
    return vals.reshape(want_shape).copy()


def meta(name, default=None):
    try:
        import gguf
        r = gguf.GGUFReader(MODEL)
        for f in r.fields.values():
            if f.name == name:
                return f.contents()
    except Exception:                                          # noqa: BLE001
        pass
    return default


# ── the authority ───────────────────────────────────────────────────────────────────────────────
def build_authority():
    cmd = [os.environ.get("CC", "gcc"), "-O2", "-o", AUTH_BIN, os.path.join(HERE, "attn_authority.c"),
           f"-I{LLAMA}/ggml/include", f"-L{LLAMA}/build/bin",
           "-lggml", "-lggml-base", "-lggml-cpu", f"-Wl,-rpath,{LLAMA}/build/bin"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("ATTN_ENGINE_VERIFY_FAIL authority build: " + (r.stderr.strip()[:400] or r.stdout.strip()[:400]))
        return False
    return True


def run_authority(Wqg, Wk, Wv, Wo, nmq, nmk, hid, geo):
    """geo = (NH, HD, NKV, S, D, ND, eps, kqs, sections). Geom is passed EXPLICITLY, never read from
    this module's globals: several geometries are exercised and a stale global is indistinguishable
    from a wrong computation."""
    nh, hd, nkv, s, d, nd, eps, kqs, sections = geo
    with open(AUTH_IN, "wb") as fh:
        for a in (Wqg, Wk, Wv, Wo, nmq, nmk, hid):
            fh.write(np.ascontiguousarray(a, dtype="<f4").tobytes())
    r = subprocess.run([AUTH_BIN, AUTH_IN, AUTH_OUT, str(nh), str(hd), str(nkv), str(s), str(d),
                        str(nd), "%.9g" % eps, "%.9g" % kqs, "explicit", *[str(x) for x in sections],
                        str(THREADS)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(("attn authority", r.stdout + r.stderr)[:300])
    shapes = {}
    for line in r.stdout.splitlines():
        if line.startswith("ATTN_STAGE "):
            p = dict(kv.split("=") for kv in line.split()[2:])
            shapes[line.split()[1]] = int(p["n"])
    data = np.fromfile(AUTH_OUT, dtype="<f4")
    out, off = {}, 0
    for nm in DUMPED:
        n = shapes[nm]
        out[nm] = data[off:off + n].astype(np.float64)
        off += n
    return out


def run_driver():
    env = dict(os.environ, VYB_STDLIB=STDLIB, VYB_MODEL=MODEL, VYB_TSV=TSV,
               VYB_ATTN_X=DRV_X, VYB_ATTN_PROBE=str(PROBE_LAYER), VYB_PROMPT_RAW="1",
               VYB_INVFREQ=INVFREQ)
    cmd = [VYB, "native/host/model_driver.vyb",
           "--module-path", "native/config", "--module-path", "native/json",
           "--module-path", "native/tensor", "--module-path", "native/dtype",
           "--module-path", "native/llm"]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
    out = r.stdout + r.stderr
    with open(DRV_LOG, "w") as fh:
        fh.write(out)
    return out


def parse_dumps(text):
    secs, name, want, buf = {}, None, 0, []
    for line in text.splitlines():
        if line.startswith("DUMP "):
            if name is not None:
                secs[name] = buf[:want]
            _, name, count = line.split()
            want, buf = int(count), []
        elif name is not None and len(buf) < want:
            try:
                buf.extend(int(v) for v in line.split())
            except ValueError:
                pass
    if name is not None:
        secs[name] = buf[:want]
    return {k: np.frombuffer(np.array(v, dtype="<i8").tobytes(), dtype="<f8") for k, v in secs.items()}


def rms(x, w, eps, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    return x / np.sqrt(np.mean(x * x, axis=axis, keepdims=True) + eps) * w


def causal_attn(q, k, v, kvh_per_head, scale):
    """(S,nh,hd) q/k/v -> (S,nh,hd) context, causal, each query head reading kv head kvh_per_head[h]."""
    s = q.shape[0]
    kq = np.einsum("qhd,khd->qhk", q, k[:, kvh_per_head, :]) * scale
    above = np.arange(s)[None, :] > np.arange(s)[:, None]
    kq = np.where(above[:, None, :], -np.inf, kq)
    m = kq.max(axis=2, keepdims=True)
    e = np.exp(kq - m)
    pr = e / e.sum(axis=2, keepdims=True)
    return np.einsum("qhk,khd->qhd", pr, v[:, kvh_per_head, :])


def main():
    print(f"ATTN_ENGINE_VERIFY probe_layer=blk.{PROBE_LAYER} S={S} maxrel={MAXREL:g} threads={THREADS}")
    if not os.path.exists(MODEL):
        print(f"ATTN_ENGINE_VERIFY_SKIP the Ridge model is not on disk at {MODEL}")
        return 0
    if not os.path.exists(INVENTORY):
        print(f"ATTN_ENGINE_VERIFY_SKIP no inventory at {INVENTORY}")
        return 0
    if not os.path.exists(TSV):
        print(f"ATTN_ENGINE_VERIFY_SKIP no engine tensor index at {TSV} (run native/tools/inventory_to_tsv.py)")
        return 0
    if not os.path.exists(os.path.join(LLAMA, "ggml/include/ggml.h")):
        print(f"ATTN_ENGINE_VERIFY_SKIP no llama.cpp checkout at {LLAMA}")
        return 0
    try:
        import gguf                                                 # noqa: F401
    except Exception as e:                                         # noqa: BLE001
        print(f"ATTN_ENGINE_VERIFY_SKIP no `gguf` package for the independent dequant ({e.__class__.__name__})")
        return 0
    if not os.path.exists(INVFREQ):
        print(f"ATTN_ENGINE_VERIFY_SKIP no rope table at {INVFREQ} "
              f"(native/tools/gen_invfreq.py <base> <n_dims> <out>)")
        return 0

    p = f"blk.{PROBE_LAYER}."
    # Geometry from the model's OWN tensor table, never from literals. The inventory gives each
    # tensor's ggml dims (ne0 x ne1 — in ggml order, ne0 is the INPUT width for a 2-D weight) and its
    # BYTE SIZE; the element count derived from the bytes must agree with ne0*ne1, so the two
    # independent readings cross-check.
    per256 = {0: 1024, 12: 144, 13: 176, 14: 210}

    def ne(name):
        row = inv_row(name)
        if row is None or row[0] not in per256:
            return None
        d0, _, d1 = row[2].partition("x")
        a, b = int(d0), (int(d1) if d1 else 1)      # a 1-D tensor has ne1 = 1, not ne1 = ne0
        if row[4] * 256 // per256[row[0]] != a * b:
            return None
        return a, b

    qne, kne, nne, one = (ne(p + n) for n in ("attn_q.weight", "attn_k.weight", "attn_q_norm.weight",
                                              "attn_output.weight"))
    if None in (qne, kne, nne, one):
        print(f"ATTN_ENGINE_VERIFY_FAIL blk.{PROBE_LAYER}'s attention tensors are absent from {INVENTORY} "
              f"(or their dims disagree with their bytes)")
        return 1
    D = qne[0]
    HD = nne[0]
    H = qne[1] // (2 * HD)
    KVH = kne[1] // HD
    if HD <= 0 or H <= 0 or KVH <= 0 or qne[1] != 2 * H * HD or kne[0] != D or one != (H * HD, D):
        print(f"ATTN_ENGINE_VERIFY_FAIL the attention tensors do not describe a consistent block "
              f"(attn_q ne={qne} attn_k ne={kne} q_norm ne={nne} out ne={one} -> D={D} H={H} KVH={KVH} HD={HD})")
        return 1
    NROT = int(np.fromfile(INVFREQ, dtype="<f8").size) * 2
    eps = float(meta("qwen35.attention.layer_norm_rms_epsilon", meta("qwen35.attention.layer_norm_epsilon", 1e-6)))
    print(f"ATTN_ENGINE_VERIFY geometry D={D} H={H} KVH={KVH} HD={HD} NROT={NROT} eps={eps:g} S={S}")

    # ── the fixture: the BLOCK INPUT (pre-attn_norm), S tokens of D, f64 ─────────────────────────
    rng = np.random.default_rng(20261009)
    x = rng.normal(0.0, 1.0, size=(S, D)).astype(np.float64)
    with open(DRV_X, "wb") as fh:
        fh.write(np.ascontiguousarray(x, dtype="<f8").tobytes())

    # ── the model's own weights (the authority's operands; the engine stages the same bytes) ─────
    Wqg = real_tensor(p + "attn_q.weight", (2 * H * HD, D))
    Wk = real_tensor(p + "attn_k.weight", (KVH * HD, D))
    Wv = real_tensor(p + "attn_v.weight", (KVH * HD, D))
    Wo = real_tensor(p + "attn_output.weight", (D, H * HD))
    nmq = real_tensor(p + "attn_q_norm.weight", (HD,))
    nmk = real_tensor(p + "attn_k_norm.weight", (HD,))
    nan = real_tensor(p + "attn_norm.weight", (D,))
    if any(t is None for t in (Wqg, Wk, Wv, Wo, nmq, nmk, nan)):
        print(f"ATTN_ENGINE_VERIFY_FAIL could not read one of blk.{PROBE_LAYER}'s attention tensors")
        return 1
    print(f"ATTN_ENGINE_VERIFY weights: {p}attn_q real {Wqg.shape} ({inv_row(p + 'attn_q.weight')[1]}), "
          f"attn_k/attn_v {Wk.shape} ({inv_row(p + 'attn_k.weight')[1]}), attn_output {Wo.shape}, "
          f"q_norm/k_norm/attn_norm real")
    if not build_authority():
        return 1

    hid = rms(x, nan, eps).astype(np.float32)                       # attn_norm, f64 -> f32 for the op
    # The authority expresses GQA by REPLICATING each kv group into its own K/V rows — the form P4.8
    # verified — which is the FUNCTION the engine's `attn` kernel implements as kv = h/(H/KVH).
    # Replicate the whole HD-row BLOCK per group (np.tile), not each row: a per-row repeat would
    # scramble which weights a head reads and the two sides would then be comparing different
    # functions (that mistake is why this fixture's k stages disagreed at 1.37 before it was fixed).
    grp = H // KVH
    k_idx = np.array([h // grp for h in range(H)])
    Wk_rep = np.concatenate([np.tile(Wk[j * HD:(j + 1) * HD], (grp, 1)) for j in range(KVH)])
    Wv_rep = np.concatenate([np.tile(Wv[j * HD:(j + 1) * HD], (grp, 1)) for j in range(KVH)])
    geo = (H, HD, H, S, D, NROT, eps, 1.0 / float(np.sqrt(HD)), (11, 11, 10, 0))
    ref = run_authority(Wqg, Wk_rep, Wv_rep, Wo, nmq, nmk, hid, geo)
    # determinism: the authority must not flip between runs (§33)
    first = open(AUTH_OUT, "rb").read()
    run_authority(Wqg, Wk_rep, Wv_rep, Wo, nmq, nmk, hid, geo)
    if open(AUTH_OUT, "rb").read() != first:
        print("ATTN_ENGINE_VERIFY_FAIL the authority is NON-DETERMINISTIC at threads=%d" % THREADS)
        return 1

    out = run_driver()
    if "ATTN_PROBE_DONE" not in out:
        if "SKIP" in out:
            print("ATTN_ENGINE_VERIFY_SKIP " + [l for l in out.splitlines() if "SKIP" in l][0].strip())
            return 0
        print("ATTN_ENGINE_VERIFY_FAIL driver did not complete (log: "
              f"{os.path.relpath(DRV_LOG, REPO)}): " + out[-900:])
        return 1
    for line in out.splitlines():
        if line.startswith("ATTN_GEOM") or line.startswith("ATTN_STAGED") or line.startswith("ATTN_CFG"):
            print("ATTN_ENGINE_VERIFY " + line)
    # the driver's own report must agree with the geometry read from the file (two readers, one file)
    gl = [l for l in out.splitlines() if l.startswith("ATTN_GEOM")][0]
    gd = dict(kv.split("=") for kv in gl.split()[2:])
    if (int(gd["D"]), int(gd["H"]), int(gd["KVH"]), int(gd["HD"]), int(gd["NROT"])) != (D, H, KVH, HD, NROT):
        print(f"ATTN_ENGINE_VERIFY_FAIL the driver's geometry {gd} disagrees with the tensor table's "
              f"D={D} H={H} KVH={KVH} HD={HD} NROT={NROT}")
        return 1
    got = parse_dumps(out)

    # ── the staged operands, at the addresses gemm reads ────────────────────────────────────────
    probes = []
    for line in out.splitlines():
        if line.startswith("PROBE aq ") or line.startswith("PROBE ao "):
            _, tag, idx, bits = line.split()
            v = np.frombuffer(np.array([int(bits)], dtype="<i8").tobytes(), dtype="<f8")[0]
            probes.append((tag, int(idx), v))
    pbad = []
    for tag, idx, v in probes:
        if tag == "aq":
            k, n = idx // (2 * H * HD), idx % (2 * H * HD)
            want = Wqg[n, k]
        else:
            k, n = idx // D, idx % D
            want = Wo[n, k]
        if not (abs(v - want) <= 1e-5 * max(abs(want), 1e-12)):
            pbad.append((tag, idx, v, want))
    if not probes or pbad:
        print(f"ATTN_ENGINE_VERIFY staged operand FAIL ({len(pbad)}/{len(probes)} slots wrong)")
        for tag, idx, v, want in pbad[:4]:
            print(f"ATTN_ENGINE_VERIFY   {tag} idx {idx} driver={v:.9e} model={want:.9e}")
        print("ATTN_ENGINE_VERIFY_FAIL the staged operand is not the model's own weight")
        return 1
    print(f"ATTN_ENGINE_VERIFY staged operands ok ({len(probes)} slots of B[k*N+n] == the model's own "
          f"dequantised attn_q/attn_output)")

    # ── the ten stages ─────────────────────────────────────────────────────────────────────────
    worst, bad = 0.0, []
    for nm in STAGES:
        key = "a1_" + nm
        if key not in got:
            print(f"ATTN_ENGINE_VERIFY {nm:9s} MISSING (the driver dumped no {key})")
            bad.append(nm)
            continue
        gpu = got[key].astype(np.float64)
        r = np.asarray(ref[nm], dtype=np.float64)
        if nm in ("k_norm", "k_rope"):
            # the engine keeps KVH rows and groups inside the kernel; the authority was handed H
            # replicated rows. Replicate the engine's rows the same way (head h reads its group's
            # copy) so the comparison is of the same function, and say so.
            g = gpu.reshape(S, KVH, HD)
            gpu = np.repeat(g, grp, axis=1).reshape(-1)
        if gpu.size != r.size:
            print(f"ATTN_ENGINE_VERIFY {nm:9s} LENGTH MISMATCH driver={gpu.size} authority={r.size}")
            bad.append(nm)
            continue
        rr = rel(gpu, r)
        worst = max(worst, rr)
        if not (rr <= MAXREL):                                   # NaN must FAIL
            bad.append(nm)
            w = int(np.nanargmax(np.abs(gpu - r)))
            print(f"ATTN_ENGINE_VERIFY {nm:9s} n={gpu.size:7d} maxrel={rr:.3e} DIFFERS")
            print(f"ATTN_ENGINE_VERIFY   worst #{w}: driver={gpu[w]:.9e} authority={r[w]:.9e}")
        else:
            print(f"ATTN_ENGINE_VERIFY {nm:9s} n={gpu.size:7d} maxrel={rr:.3e} ok")
    if bad:
        print(f"ATTN_ENGINE_VERIFY_FAIL the engine does not reproduce ggml's attention block for: "
              f"{', '.join(bad)} (driver log: {os.path.relpath(DRV_LOG, REPO)})")
        return 1

    # ── the teeth: recompute the attention from the ENGINE's own q_rope/k_rope ───────────────────
    kqs = 1.0 / float(np.sqrt(HD))
    qr = got["a1_q_rope"].reshape(S, H, HD)
    kr = got["a1_k_rope"].reshape(S, KVH, HD)
    vv = (hid.astype(np.float64) @ Wv.T.astype(np.float64)).reshape(S, KVH, HD)
    at_ok = causal_attn(qr, kr, vv, k_idx, kqs)
    at_rr = causal_attn(qr, kr, vv, np.array([h % KVH for h in range(H)]), kqs)
    e_ok = rel(at_ok.ravel(), got["a1_attn"])
    e_rr = rel(at_rr.ravel(), got["a1_attn"])
    print(f"ATTN_ENGINE_VERIFY tooth contiguous grouping (h//%d) reproduces the engine's attn: maxrel={e_ok:.3e}"
          % KVH)
    print(f"ATTN_ENGINE_VERIFY tooth round-robin grouping (h%%%d)      (must MISS):          maxrel={e_rr:.3e}"
          % KVH)
    if not (e_ok <= MAXREL):
        print("ATTN_ENGINE_VERIFY_FAIL the engine's attn is not the contiguous-grouping attention")
        return 1
    if not (e_rr > MAXREL * 100):
        print(f"ATTN_ENGINE_VERIFY_FAIL the round-robin alternative is NOT distinguished ({e_rr:.3e}) — "
              f"this fixture cannot tell the groupings apart, so its pass proves nothing")
        return 1
    # the gate
    gated_ok = rel(got["a1_attn"] * (1.0 / (1.0 + np.exp(-got["a1_gate_pre"]))), got["a1_gated"])
    gated_raw = rel(got["a1_attn"] * got["a1_gate_pre"], got["a1_gated"])
    nogate = rel(got["a1_attn"], got["a1_gated"])
    print(f"ATTN_ENGINE_VERIFY tooth gated == attn * sigmoid(gate_pre): maxrel={gated_ok:.3e}")
    print(f"ATTN_ENGINE_VERIFY tooth raw gate (no sigmoid)  (must MISS): maxrel={gated_raw:.3e}")
    print(f"ATTN_ENGINE_VERIFY tooth gate dropped entirely  (must MISS): maxrel={nogate:.3e}")
    if not (gated_ok <= MAXREL) or min(gated_raw, nogate) <= MAXREL * 100:
        print("ATTN_ENGINE_VERIFY_FAIL the output gate is not the measured function")
        return 1

    print(f"ATTN_ENGINE_VERIFY_SUMMARY layer=blk.{PROBE_LAYER} S={S} stages={len(STAGES)} worst={worst:.3e} "
          f"teeth=grouping({e_rr:.2e})/raw_gate({gated_raw:.2e})/no_gate({nogate:.2e})")
    print("ATTN_ENGINE_VERIFY_DONE the engine's own staging, gemm, split, per-head norms, partial rope, "
          "GQA attention, output gate and wo reproduce ggml's graph for the attention block")
    return 0


if __name__ == "__main__":
    sys.exit(main())
