#!/usr/bin/env python3
"""The recurrent-layer kernels on the GPU vs an fp64 numpy reference (phase 4 item 1, unit 6).

Runs `native/host/gdn_driver.vyb` — which loads native/build/gdn.ptx and runs `l2norm`,
`delta_step` and `norm_gated` on one token at the 27B geometry — and compares every element of every
result against a reference computed here.

The reference is fp64, matching the kernels: the point of this gate is the kernels, not a precision
floor, so a wiring error cannot hide inside a "tolerance". (The pure-op gates do the opposite — they
mirror ggml's f32 bodies — because there the authority IS f32.)

Fixture: deterministic fp64 values with the model's shapes and the model's own ssm_dt/ssm_a (so the
gate stays in the range the alpha gate actually produces: a negative decay, softplus'd alpha).

Usage: gdn_kernel_verify.py           (env: VYBFORGE_GDNK_MAXREL to widen the bar while triaging)
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
BUILD = os.path.join(REPO, "native/build")
IN_FILE = os.path.join(BUILD, "gdn_in.bin")
PTX = os.path.join(BUILD, "gdn.ptx")
LOG = os.path.join(BUILD, "gdn_driver.log")
GGUF = os.environ.get(
    "VYBFORGE_RIDGE_GGUF",
    os.path.expanduser("~/Models/qwen38-27b-ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf"))

S, H_K, H_V = 128, 16, 48
EPS = 1e-6
SCALE = 1.0 / float(np.sqrt(S))
MAXREL = float(os.environ.get("VYBFORGE_GDNK_MAXREL", "1e-6"))


def real_gate_params():
    """blk.0.ssm_dt / blk.0.ssm_a when the model is on disk, else a synthetic negative decay."""
    try:
        import gguf
        r = gguf.GGUFReader(GGUF)
        cur = {}
        for name, key in (("blk.0.ssm_dt.bias", "dt"), ("blk.0.ssm_a", "a")):
            t = next(t for t in r.tensors if t.name == name)
            cur[key] = np.array(t.data, dtype=np.float32).reshape(-1)[:H_V].astype(np.float64)
        assert cur["dt"].shape == (H_V,) and cur["a"].shape == (H_V,)
        return cur["dt"], cur["a"], True
    except Exception as e:      # noqa: BLE001
        print(f"GDNK_VERIFY note: real ssm_dt/ssm_a unavailable ({e.__class__.__name__}); synthetic")
        rng = np.random.default_rng(7)
        return (rng.normal(0.0, 0.5, size=H_V), -np.exp(rng.normal(1.0, 0.5, size=H_V)), False)


def fixture():
    """Deterministic inputs, all in ggml order (ne0 fastest) as numpy arrays of the reversed shape."""
    rng = np.random.default_rng(20261008)
    dt, a, real = real_gate_params()
    q = rng.normal(0.0, 0.3, size=(H_K, S))
    k = rng.normal(0.0, 0.3, size=(H_K, S))
    v = rng.normal(0.0, 0.3, size=(H_V, S))
    gate = np.log1p(np.exp(rng.normal(0.0, 1.0, size=H_V))) * a[:H_V]   # softplus(alpha) * ssm_a, as the model builds it
    beta = 1.0 / (1.0 + np.exp(-rng.normal(0.0, 1.0, size=H_V)))
    state = rng.normal(0.0, 0.05, size=(H_V, S, S))      # [h, j, i]
    w = rng.normal(1.0, 0.05, size=S)
    z = rng.normal(0.0, 1.0, size=(H_V, S))
    return q, k, v, gate, beta, state, w, z, real


def reference(q, k, v, gate, beta, state, w, z, readout="exact", decay=True):
    """fp64 reference. `readout`/`decay` exist so the check can also report a REJECTED alternative:
    contracting the read-out on the state's first axis instead of its second is the exact mistake
    this kernel was written with first, and it must not come out small."""
    def l2(x):
        return x / np.maximum(np.sqrt((x * x).sum(axis=-1, keepdims=True)), EPS)

    qn, kn = l2(q), l2(k)
    out = np.zeros((H_V, S))
    state_out = np.zeros_like(state)
    for h in range(H_V):
        hk = h % H_K                                      # the interleaved broadcast, from unit 1
        dec = state[h] * (np.exp(float(gate[h])) if decay else 1.0)   # [j, i]
        delta = (v[h] - dec @ kn[hk]) * float(beta[h])     # [j]
        m = dec + np.outer(delta, kn[hk])
        out[h] = ((m @ qn[hk]) if readout == "exact" else (m.T @ qn[hk])) * SCALE
        state_out[h] = m
    mean = (out * out).sum(axis=-1, keepdims=True) / float(S)
    epi = (out / np.sqrt(mean + EPS)) * w * (z / (1.0 + np.exp(-z)))
    return {"q_norm": qn, "k_norm": kn, "gdn_out": out, "state_out": state_out, "epi": epi}


def build_ptx():
    cmd = [VYB, "--kernel", "native/kernels/gdn.vyb", "--ptx", "native/build/gdn.ptx",
           "--module-path", "native/kernels"]
    env = dict(os.environ)
    env["VYB_STDLIB"] = STDLIB
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
    if r.returncode != 0 or not os.path.exists(PTX):
        print("GDNK_VERIFY_FAIL compile: " + (r.stdout + r.stderr)[-500:])
        return False
    return True


def run_driver():
    env = dict(os.environ)
    env["VYB_STDLIB"] = STDLIB
    r = subprocess.run([VYB, "native/host/gdn_driver.vyb"], cwd=REPO,
                       capture_output=True, text=True, env=env)
    out = r.stdout + r.stderr
    with open(LOG, "w") as fh:
        fh.write(out)
    return out


def parse_dumps(text):
    """Read the driver's DUMP sections: one Int per element (the fp64 bit pattern).

    Each section's header declares its element count, and exactly that many values are taken — the
    program's own trailing output (Vyb prints main's return value) must not leak into the last
    section.
    """
    secs, name, want, buf = {}, None, 0, []
    for line in text.splitlines():
        if line.startswith("DUMP "):
            if name is not None:
                secs[name] = buf[:want]
            _, name, count = line.split()
            want, buf = int(count), []
        elif name is not None and len(buf) < want:
            toks = line.split()
            try:
                vals = [int(v) for v in toks]          # all-or-nothing: a stray token must not shift
                buf.extend(vals)
            except ValueError:
                pass
    if name is not None:
        secs[name] = buf[:want]
    return {k: np.frombuffer(np.array(v, dtype="<i8").tobytes(), dtype="<f8") for k, v in secs.items()}


def main():
    print(f"GDNK_VERIFY geometry S={S} H_k={H_K} H_v={H_V} eps={EPS:g} scale={SCALE:.9f}")
    if not build_ptx():
        return 1
    q, k, v, gate, beta, state, w, z, real = fixture()
    print(f"GDNK_VERIFY gate=[{gate.min():.3e},{gate.max():.3e}] beta=[{beta.min():.3e},{beta.max():.3e}] "
          f"({'real ssm_a' if real else 'synthetic'})")

    with open(IN_FILE, "wb") as fh:
        for arr in (q, k, v, gate, beta, state, w, z):
            fh.write(np.ascontiguousarray(arr, dtype="<f8").tobytes())

    out = run_driver()
    if "SKIP" in out and "GDNK_DONE" not in out:
        print("GDNK_VERIFY_SKIP " + [l for l in out.splitlines() if "SKIP" in l][0].strip())
        return 0
    if "GDNK_DONE" not in out:
        print("GDNK_VERIFY_FAIL driver: " + out[-600:])
        return 1

    got = parse_dumps(out)
    want = reference(q, k, v, gate, beta, state, w, z)
    shapes = {"q_norm": (H_K, S), "k_norm": (H_K, S), "gdn_out": (H_V, S),
              "state_out": (H_V, S, S), "epi": (H_V, S)}
    bad = 0
    for name in ("q_norm", "k_norm", "gdn_out", "state_out", "epi"):
        if name not in got:
            print(f"GDNK_VERIFY {name:10s} MISSING (driver printed no section)")
            bad = 1
            continue
        a = got[name]
        b = np.asarray(want[name], dtype=np.float64).reshape(-1)
        if a.size != b.size:
            print(f"GDNK_VERIFY {name:10s} LENGTH MISMATCH gpu={a.size} ref={b.size}")
            bad = 1
            continue
        scale = max(float(np.max(np.abs(b))), 1e-30)
        r = float(np.max(np.abs(a - b)) / scale)
        flag = "ok" if r <= MAXREL else "DIFFERS"
        print(f"GDNK_VERIFY {name:10s} n={a.size:7d} maxrel={r:.3e} scale={scale:.3e} {flag}")
        if r > MAXREL:
            bad = 1
            w_ = int(np.argmax(np.abs(a - b)))
            print(f"           worst idx {w_}: gpu={a[w_]:.9e} ref={b[w_]:.9e}")

    if bad:
        print(f"GDNK_VERIFY_FAIL worse than maxrel {MAXREL:g} (log: {os.path.relpath(LOG, REPO)})")
        return 1

    # The check's teeth: the two alternatives this kernel was wrong-with or could be wrong-with.
    gpu_out = got["gdn_out"]
    b = np.max(np.abs(np.asarray(want["gdn_out"]).reshape(-1))) or 1e-30
    alt_axis = float(np.max(np.abs(gpu_out - np.asarray(
        reference(q, k, v, gate, beta, state, w, z, readout="transposed")["gdn_out"]).reshape(-1))) / b)
    alt_decay = float(np.max(np.abs(gpu_out - np.asarray(
        reference(q, k, v, gate, beta, state, w, z, decay=False)["gdn_out"]).reshape(-1))) / b)
    print(f"GDNK_VERIFY alt read-out-on-first-axis maxrel={alt_axis:.3e} (must be large)")
    print(f"GDNK_VERIFY alt no-decay                 maxrel={alt_decay:.3e} (must be large)")
    if min(alt_axis, alt_decay) <= MAXREL:
        print("GDNK_VERIFY_FAIL an alternative matches the kernel — the comparison proves nothing")
        return 1

    print(f"GDNK_VERIFY_SUMMARY sections=5 worst_within={MAXREL:g} "
          f"elements={sum(int(np.prod(shapes[n])) for n in shapes)} "
          f"rejected_alternatives={alt_axis:.2e}/{alt_decay:.2e}")
    print(f"GDNK_VERIFY_DONE the three GDN kernels match the fp64 reference within maxrel {MAXREL:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
