#!/usr/bin/env python3
"""The GPU wiring of one recurrent block vs the block built from ggml's own ops (unit 7).

Two implementations of the SAME layer, compared stage by stage:

  * native/tools/layer_authority.c  — unit 5's authority: the real ggml ops (rms_norm, mul_mat,
    sigmoid, softplus, ssm_conv, gated_delta_net), one graph, f32, dumps every stage;
  * native/host/gdn_layer_driver.vyb — unit 7: the same block out of the Vyb GPU kernels (rmsnorm,
    mm_nt, sigmoid_k, alpha_gate, conv1d_k, silu_k, l2norm, delta_step, norm_gated, add_k), f64.

Agreement therefore means "our GPU wiring reproduces ggml's own graph", not "our code agrees with our
code" — the same standard the earlier units were held to. The fixture is generated once and fed to
both, in their own precisions: the authority computes in f32, the kernels in f64, so the expected
agreement is the f32 rounding of the authority (~1e-6), while a wiring mistake is O(1).

Geometry is small enough to be quick and deliberately NOT degenerate — S=32, H_k=4, H_v=8, d_conv=4,
n_embd=512 — because a wiring bug that only appears once strides exceed 1 is exactly what the earlier
units were bitten by.

Usage: gdn_layer_kernel_verify.py
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
VYBHOME = os.environ.get("VYBHOME", os.path.expanduser("~/Projects/Vyb"))
VYB = os.environ.get("VYB", os.path.join(VYBHOME, "build", "vyb"))
STDLIB = os.environ.get("VYB_STDLIB", os.path.join(VYBHOME, "stdlib"))
LLAMA = os.environ.get("VYBFORGE_LLAMA", os.path.expanduser("~/Projects/llama.cpp"))
BUILD = os.path.join(REPO, "native/build")
AUTH_BIN = os.path.join(BUILD, "layer_authority")
AUTH_DIR = os.path.join(BUILD, "layer_authority_small")
AUTH_IN = os.path.join(BUILD, "gdn_layer_authority_in.bin")
DRV_IN = os.path.join(BUILD, "gdn_layer_in.bin")
DRV_LOG = os.path.join(BUILD, "gdn_layer_driver.log")

NE, S, H_K, H_V, DC, T, B = 512, 32, 4, 8, 4, 1, 1
KEY, VALUE = H_K * S, H_V * S
QKV = 2 * KEY + VALUE
EPS = 1e-6
SCALE = 1.0 / float(np.sqrt(S))
MAXREL = float(os.environ.get("VYBFORGE_GDNL_MAXREL", "1e-4"))


def real_gate_params(n):
    path = os.environ.get("VYBFORGE_RIDGE_GGUF",
                          os.path.expanduser("~/Models/qwen38-27b-ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf"))
    try:
        import gguf
        r = gguf.GGUFReader(path)
        a = next(t for t in r.tensors if t.name == "blk.0.ssm_a")
        dt = next(t for t in r.tensors if t.name == "blk.0.ssm_dt.bias")
        av = np.array(a.data, dtype=np.float32).reshape(-1)
        dv = np.array(dt.data, dtype=np.float32).reshape(-1)
        return dv[:n].copy(), av[:n].copy(), True
    except Exception as e:      # noqa: BLE001
        print(f"GDNL_VERIFY note: real ssm_dt/ssm_a unavailable ({e.__class__.__name__}); synthetic")
        rng = np.random.default_rng(7)
        return (rng.normal(0.0, 0.5, size=n).astype(np.float32),
                (-np.exp(rng.normal(1.0, 0.5, size=n))).astype(np.float32), False)



INVENTORY = os.path.join(REPO, "native/gguf/ridge-3.7bpw-inventory.tsv")


def real_q8_weight(name):
    """The RAW GGUF bytes and the numpy dequant of a Q8_0 tensor from the model, or (None, None).

    The layer check uses the model's own ssm_alpha/ssm_beta here: those two are the Q8_0 tensors of
    the gdn_state path, so the projections run on real quantized weights, dequantized on the GPU by
    the existing q8_0deq kernel (gated separately in P2/S0.5) and in numpy here for the authority.
    """
    try:
        off = size = None
        for line in open(INVENTORY):
            if line.startswith(name + "\t"):
                f = line.split("\t")
                off, size = int(f[5]), int(f[6])
                break
        if off is None:
            raise KeyError(name)
        with open(os.environ.get("VYBFORGE_RIDGE_GGUF",
                                 os.path.expanduser("~/Models/qwen38-27b-ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf")), "rb") as fh:
            fh.seek(off)
            raw = fh.read(size)
        if len(raw) != size:
            raise ValueError("short read")
        blocks = np.frombuffer(raw, dtype=np.uint8).reshape(size // 34, 34)
        d = blocks[:, :2].copy().view(np.float16).reshape(-1).astype(np.float32)      # f16 scale per block
        q = blocks[:, 2:].copy().view(np.int8).astype(np.float32)                     # 32 signed quants
        vals = (d[:, None] * q).reshape(-1)                                          # [n_out, n_in] ggml order
        # The model's tensors are (48, 5120); this fixture is (H_V, NE) = (8, 512) so it can only use
        # them if the geometry matches. Say so rather than feeding a truncated slice.
        if vals.size != H_V * NE:
            print(f"GDNL_VERIFY note: real {name} is {vals.size} values, this fixture's geometry wants "
                  f"{H_V * NE} — synthetic weights here (raise NE/S/H_v to use the model's own)")
            return None, None
        return raw, vals
    except Exception as e:      # noqa: BLE001
        print(f"GDNL_VERIFY note: real {name} unavailable ({e.__class__.__name__}); synthetic")
        return None, None


def fixture():
    """One fixture, both precisions. Weights are ggml-order: a tensor of ne (K, N) is numpy (N, K)."""
    rng = np.random.default_rng(20261008)
    s = 1.0 / np.sqrt(NE)
    x = rng.normal(0.0, 1.0, size=(2, NE)).astype(np.float32)   # two decode steps
    attn_norm = rng.normal(1.0, 0.05, size=NE).astype(np.float32)
    wqkv = rng.normal(0.0, s, size=(QKV, NE)).astype(np.float32)
    wgate = rng.normal(0.0, s, size=(VALUE, NE)).astype(np.float32)
    wbeta = rng.normal(0.0, s, size=(H_V, NE)).astype(np.float32)
    walpha = rng.normal(0.0, s, size=(H_V, NE)).astype(np.float32)
    alpha_raw, alpha_deq = real_q8_weight("blk.0.ssm_alpha.weight")
    beta_raw, beta_deq = real_q8_weight("blk.0.ssm_beta.weight")
    quantised = alpha_raw is not None and beta_raw is not None
    if quantised:
        walpha = alpha_deq.astype(np.float32)     # the authority sees the same values the GPU will
        wbeta = beta_deq.astype(np.float32)
    dt, a, real = real_gate_params(H_V)
    conv1d = rng.normal(0.0, 0.2, size=(QKV, DC)).astype(np.float32)
    ssm_norm = rng.normal(1.0, 0.05, size=S).astype(np.float32)
    ssm_out = rng.normal(0.0, s, size=(NE, VALUE)).astype(np.float32)
    zwindow = np.zeros(DC * QKV, dtype=np.float32)   # the conv buffer, window slots and all (ncs frames/channel)
    state = np.zeros(S * S * H_V, dtype=np.float32)  # a fresh sequence starts at zero
    return dict(x=x, attn_norm=attn_norm, wqkv=wqkv, wqkv_gate=wgate, ssm_beta=wbeta,
                alpha_q8=(alpha_raw if quantised else b""), beta_q8=(beta_raw if quantised else b""),
                ssm_alpha=walpha, ssm_dt=dt, ssm_a=a, ssm_conv1d=conv1d, ssm_norm=ssm_norm,
                ssm_out=ssm_out, zwindow=zwindow, state=state), real, quantised


def build_authority():
    cmd = [os.environ.get("CC", "gcc"), "-O2", "-o", AUTH_BIN,
           os.path.join(HERE, "layer_authority.c"),
           f"-I{LLAMA}/ggml/include", f"-L{LLAMA}/build/bin",
           "-lggml", "-lggml-base", "-lggml-cpu", f"-Wl,-rpath,{LLAMA}/build/bin"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("GDNL_VERIFY_FAIL authority build: " + (r.stderr.strip()[:400] or r.stdout.strip()[:400]))
        return False
    return True


def build_ptx():
    env = dict(os.environ, VYB_STDLIB=STDLIB)
    for src, out in (("native/kernels/gdn.vyb", "native/build/gdn.ptx"),
                     ("native/kernels/rmsnorm.vyb", "native/build/rmsnorm.ptx")):
        r = subprocess.run([VYB, "--kernel", src, "--ptx", out, "--module-path", "native/kernels"],
                           cwd=REPO, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            print("GDNL_VERIFY_FAIL compile: " + (r.stdout + r.stderr)[-400:])
            return False
    return True


def run_authority_step(fx, x_step, window, state, outdir):
    """One T=1 step of the authority with a given conv window and delta-net state."""
    with open(AUTH_IN, "wb") as fh:
        fh.write(np.ascontiguousarray(x_step, dtype="<f4").tobytes())          # x is FIRST in the file
        for k in ("attn_norm", "wqkv", "wqkv_gate", "ssm_beta", "ssm_alpha",
                  "ssm_dt", "ssm_a", "ssm_conv1d", "ssm_norm", "ssm_out"):
            fh.write(np.ascontiguousarray(fx[k], dtype="<f4").tobytes())
        fh.write(np.ascontiguousarray(window, dtype="<f4").tobytes())
        fh.write(np.ascontiguousarray(state, dtype="<f4").tobytes())
    os.makedirs(outdir, exist_ok=True)
    r = subprocess.run([AUTH_BIN, AUTH_IN, outdir, str(NE), str(S), str(H_K), str(H_V),
                        str(DC), str(T), str(B), repr(EPS), "1"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(("authority", r.stdout + r.stderr)[:400])


def run_driver(fx):
    with open(DRV_IN, "wb") as fh:
        for k in ("x", "attn_norm", "wqkv", "wqkv_gate", "ssm_beta", "ssm_alpha", "ssm_dt", "ssm_a",
                  "ssm_conv1d", "ssm_norm", "ssm_out", "zwindow", "state"):
            fh.write(np.ascontiguousarray(fx[k], dtype="<f8").tobytes())
        for k in ("alpha_q8", "beta_q8"):
            fh.write(bytes(fx[k]))                    # raw GGUF bytes, or nothing
    env = dict(os.environ, VYB_STDLIB=STDLIB)
    r = subprocess.run([VYB, "native/host/gdn_layer_driver.vyb"], cwd=REPO,
                       capture_output=True, text=True, env=env)
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


SHAPES = {
    "xn": (B, T, NE), "qkv": (B, T, QKV), "z": (B, T, VALUE), "beta": (B, T, H_V, 1),
    "gate": (B, T, H_V, 1), "conv_silu": (B, T, QKV), "q_norm": (B, T, H_K, S),
    "k_norm": (B, T, H_K, S), "gdn_out": (B, T, H_V, S), "epi": (B, T, H_V, S),
    "y": (B, T, NE), "layer_out": (B, T, NE),
}


def rel(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-30))


def main():
    print(f"GDNL_VERIFY geometry n_embd={NE} S={S} H_k={H_K} H_v={H_V} d_conv={DC} "
          f"key={KEY} value={VALUE} qkv={QKV}, steps=2")
    if not build_authority() or not build_ptx():
        return 1
    fx, real, quantised = fixture()
    print(f"GDNL_VERIFY fixture {'real ssm_a/ssm_dt from blk.0' if real else 'synthetic ssm_a/ssm_dt'}, "
          f"ssm_alpha/ssm_beta {'REAL Q8_0 tensors of blk.0 (dequantised on the GPU)' if quantised else 'synthetic'}, "
          f"state and conv window start at zero")

    # The authority runs the sequence as two single-token steps: step 2 gets step 1's conv window
    # (the last d_conv-1 frames of its conv_input) and step 1's delta-net state — the carry-over a
    # decode loop has to reproduce.
    d1, d2 = os.path.join(AUTH_DIR, "s1"), os.path.join(AUTH_DIR, "s2")
    win0 = np.zeros(QKV * (DC - 1), dtype=np.float32)
    st0 = np.zeros(S * S * H_V, dtype=np.float32)
    run_authority_step(fx, fx["x"][0], win0, st0, d1)
    conv1 = np.fromfile(os.path.join(d1, "conv_in.bin"), dtype="<f4").reshape(QKV, DC)  # ne (ncs, qkv_dim)
    win1 = np.ascontiguousarray(conv1[:, 1:DC]).reshape(-1)
    st1 = np.fromfile(os.path.join(d1, "state_out.bin"), dtype="<f4").reshape(-1)
    run_authority_step(fx, fx["x"][1], win1, st1, d2)

    out = run_driver(fx)
    if "SKIP" in out and "GDNL_DONE" not in out:
        print("GDNL_VERIFY_SKIP " + [l for l in out.splitlines() if "SKIP" in l][0].strip())
        return 0
    if "GDNL_DONE" not in out:
        print("GDNL_VERIFY_FAIL driver: " + out[-700:])
        return 1
    got = parse_dumps(out)

    stages = ("xn", "qkv", "z", "beta", "gate", "conv_silu", "q_norm", "k_norm",
              "gdn_out", "epi", "y", "layer_out", "state_out")
    bad = 0
    for step, d in ((1, d1), (2, d2)):
        for name in stages:
            key = f"s{step}_{name}"
            ap = os.path.join(d, f"{name}.bin")
            if not os.path.exists(ap) or key not in got:
                print(f"GDNL_VERIFY {key:16s} MISSING (authority={'y' if os.path.exists(ap) else 'n'} "
                      f"driver={'y' if key in got else 'n'})")
                bad = 1
                continue
            shape = (B, H_V, S, S) if name == "state_out" else SHAPES[name]
            ref = np.fromfile(ap, dtype="<f4").reshape(shape).astype(np.float64).reshape(-1)
            gpu = got[key].reshape(-1)
            if gpu.size != ref.size:
                print(f"GDNL_VERIFY {key:16s} LENGTH MISMATCH gpu={gpu.size} authority={ref.size}")
                bad = 1
                continue
            r = rel(gpu, ref)
            # NaN fails: `not (r <= bar)`, never `r > bar`.
            if not (r <= MAXREL):
                bad = 1
                print(f"GDNL_VERIFY {key:16s} n={gpu.size:6d} maxrel={r:.3e} DIFFERS")
                w = int(np.nanargmax(np.abs(gpu - ref)))
                print(f"                 worst idx {w}: gpu={gpu[w]:.9e} authority={ref[w]:.9e}")
            else:
                print(f"GDNL_VERIFY {key:16s} n={gpu.size:6d} maxrel={r:.3e} ok")

    if bad:
        print(f"GDNL_VERIFY_FAIL worse than maxrel {MAXREL:g} (driver log: {os.path.relpath(DRV_LOG, REPO)})")
        return 1

    # The check's teeth: mis-wirings of the carry-over, computed from the authority's own stages.
    lo2 = np.fromfile(os.path.join(d2, "layer_out.bin"), dtype="<f4").reshape(-1).astype(np.float64)
    y2 = np.fromfile(os.path.join(d2, "y.bin"), dtype="<f4").reshape(-1).astype(np.float64)
    xn2 = np.fromfile(os.path.join(d2, "xn.bin"), dtype="<f4").reshape(-1).astype(np.float64)
    alt_no_res = rel(y2, lo2)
    alt_norm_res = rel(xn2 + y2, lo2)
    print(f"GDNL_VERIFY alt step-2 no-residual          maxrel={alt_no_res:.3e} (must be large)")
    print(f"GDNL_VERIFY alt step-2 residual-on-attn_norm maxrel={alt_norm_res:.3e} (must be large)")
    if min(alt_no_res, alt_norm_res) <= MAXREL:
        print("GDNL_VERIFY_FAIL an alternative matches — the stage check proves nothing")
        return 1
    print(f"GDNL_VERIFY_SUMMARY steps=2 stages=26 rejected_alternatives={alt_no_res:.2e}/{alt_norm_res:.2e}")
    print(f"GDNL_VERIFY_DONE the GPU layer reproduces ggml's own graph for two chained steps "
          f"within maxrel {MAXREL:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
