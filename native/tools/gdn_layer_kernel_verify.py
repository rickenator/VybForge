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


def fixture():
    """One fixture, both precisions. Weights are ggml-order: a tensor of ne (K, N) is numpy (N, K)."""
    rng = np.random.default_rng(20261008)
    s = 1.0 / np.sqrt(NE)
    x = rng.normal(0.0, 1.0, size=NE).astype(np.float32)
    attn_norm = rng.normal(1.0, 0.05, size=NE).astype(np.float32)
    wqkv = rng.normal(0.0, s, size=(QKV, NE)).astype(np.float32)
    wgate = rng.normal(0.0, s, size=(VALUE, NE)).astype(np.float32)
    wbeta = rng.normal(0.0, s, size=(H_V, NE)).astype(np.float32)
    walpha = rng.normal(0.0, s, size=(H_V, NE)).astype(np.float32)
    dt, a, real = real_gate_params(H_V)
    conv1d = rng.normal(0.0, 0.2, size=(QKV, DC)).astype(np.float32)
    ssm_norm = rng.normal(1.0, 0.05, size=S).astype(np.float32)
    ssm_out = rng.normal(0.0, s, size=(NE, VALUE)).astype(np.float32)
    zwindow = np.zeros(DC * QKV, dtype=np.float32)   # the conv buffer's window slots (ncs frames/channel)
    state = np.zeros(S * S * H_V, dtype=np.float32)
    return dict(x=x, attn_norm=attn_norm, wqkv=wqkv, wqkv_gate=wgate, ssm_beta=wbeta,
                ssm_alpha=walpha, ssm_dt=dt, ssm_a=a, ssm_conv1d=conv1d, ssm_norm=ssm_norm,
                ssm_out=ssm_out, zwindow=zwindow, state=state), real


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


def run_authority(fx):
    with open(AUTH_IN, "wb") as fh:
        for k in ("x", "attn_norm", "wqkv", "wqkv_gate", "ssm_beta", "ssm_alpha",
                  "ssm_dt", "ssm_a", "ssm_conv1d", "ssm_norm", "ssm_out"):
            fh.write(np.ascontiguousarray(fx[k], dtype="<f4").tobytes())
    os.makedirs(AUTH_DIR, exist_ok=True)
    r = subprocess.run([AUTH_BIN, AUTH_IN, AUTH_DIR, str(NE), str(S), str(H_K), str(H_V),
                        str(DC), str(T), str(B), repr(EPS), "1"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(("authority", r.stdout + r.stderr)[:400])


def run_driver(fx):
    with open(DRV_IN, "wb") as fh:
        for k in ("x", "attn_norm", "wqkv", "wqkv_gate", "ssm_beta", "ssm_alpha", "ssm_dt", "ssm_a",
                  "ssm_conv1d", "ssm_norm", "ssm_out", "zwindow", "state"):
            fh.write(np.ascontiguousarray(fx[k], dtype="<f8").tobytes())
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
          f"key={KEY} value={VALUE} qkv={QKV}")
    if not build_authority() or not build_ptx():
        return 1
    fx, real = fixture()
    print(f"GDNL_VERIFY fixture {'real ssm_a/ssm_dt from blk.0' if real else 'synthetic ssm_a/ssm_dt'}, "
          f"state=0 (first token of a sequence)")

    run_authority(fx)
    out = run_driver(fx)
    if "SKIP" in out and "GDNL_DONE" not in out:
        print("GDNL_VERIFY_SKIP " + [l for l in out.splitlines() if "SKIP" in l][0].strip())
        return 0
    if "GDNL_DONE" not in out:
        print("GDNL_VERIFY_FAIL driver: " + out[-700:])
        return 1
    got = parse_dumps(out)

    bad = 0
    for name in ("xn", "qkv", "z", "beta", "gate", "conv_silu", "q_norm", "k_norm",
                 "gdn_out", "epi", "y", "layer_out"):
        ap = os.path.join(AUTH_DIR, f"{name}.bin")
        if not os.path.exists(ap) or name not in got:
            print(f"GDNL_VERIFY {name:11s} MISSING (authority={'y' if os.path.exists(ap) else 'n'} "
                  f"driver={'y' if name in got else 'n'})")
            bad = 1
            continue
        ref = np.fromfile(ap, dtype="<f4").reshape(SHAPES[name]).astype(np.float64).reshape(-1)
        gpu = got[name].reshape(-1)
        if gpu.size != ref.size:
            print(f"GDNL_VERIFY {name:11s} LENGTH MISMATCH gpu={gpu.size} authority={ref.size}")
            bad = 1
            continue
        r = rel(gpu, ref)
        # NOTE: `not (r <= MAXREL)`, not `r > MAXREL` — a NaN comparison is False, so the inverted
        # form would let a NaN output pass silently.
        if not (r <= MAXREL):
            flag = "DIFFERS"
            bad = 1
            print(f"GDNL_VERIFY {name:11s} n={gpu.size:6d} maxrel={r:.3e} "
                  f"scale={float(np.max(np.abs(ref))):.3e} {flag}")
            with np.errstate(invalid="ignore"):
                d = np.abs(gpu - ref)
            w = int(np.nanargmax(d)) if not np.all(np.isnan(d)) else 0
            print(f"           worst idx {w}: gpu={gpu[w]:.9e} authority={ref[w]:.9e}")
        else:
            print(f"GDNL_VERIFY {name:11s} n={gpu.size:6d} maxrel={r:.3e} "
                  f"scale={float(np.max(np.abs(ref))):.3e} ok")

    if bad:
        print(f"GDNL_VERIFY_FAIL worse than maxrel {MAXREL:g} (driver log: {os.path.relpath(DRV_LOG, REPO)})")
        return 1

    # The check's teeth: two mis-wirings of THIS layer, computed from the authority's own stages.
    x = np.fromfile(os.path.join(AUTH_DIR, "xn.bin"), dtype="<f4").reshape(-1).astype(np.float64)
    lo = np.fromfile(os.path.join(AUTH_DIR, "layer_out.bin"), dtype="<f4").reshape(-1).astype(np.float64)
    yv = np.fromfile(os.path.join(AUTH_DIR, "y.bin"), dtype="<f4").reshape(-1).astype(np.float64)
    xraw = np.fromfile(os.path.join(AUTH_DIR, "xn.bin"), dtype="<f4").reshape(-1)
    xin = fx["x"].astype(np.float64)
    alt_norm_res = rel(x + yv, lo)          # residual on the normed input instead of the block input
    alt_no_res = rel(yv, lo)                # residual dropped altogether
    print(f"GDNL_VERIFY alt residual-on-attn_norm maxrel={alt_norm_res:.3e} (must be large)")
    print(f"GDNL_VERIFY alt no-residual           maxrel={alt_no_res:.3e} (must be large)")
    if min(alt_norm_res, alt_no_res) <= MAXREL:
        print("GDNL_VERIFY_FAIL an alternative matches — the stage check proves nothing")
        return 1
    print(f"GDNL_VERIFY_SUMMARY stages=12 rejected_alternatives={alt_norm_res:.2e}/{alt_no_res:.2e}")
    print(f"GDNL_VERIFY_DONE the GPU wiring reproduces ggml's own graph within maxrel {MAXREL:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
