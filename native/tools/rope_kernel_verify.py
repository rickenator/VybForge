#!/usr/bin/env python3
"""The rope VARIANT kernel on the GPU vs ggml's own rope op (phase 4, unit 10 step 2).

`rope_verify.py` pinned the spec by measurement: Ridge's attention rope is NEOX over the first
n_dims = 64 dims of each 256-dim head, dims 64..255 passed through — mode=neox/`neox_first_nd` at
8.993e-08, with the engine's current behaviour (every head dim, pairs (i, i+HD/2)) rejected at 1.478.
This check moves that spec onto the GPU: `native/kernels/rope.vyb`'s `rope_nrot` runs through
`native/host/rope_driver.vyb` and is compared against the same authority.

Two runs, and the second is the tooth: n_rot = 64 must match the authority, and n_rot = HD (the whole
head, i.e. what the engine's `qwen3rope` does today) must NOT — otherwise the check would be measuring
something that does not depend on the parameter at all.

Usage: rope_kernel_verify.py    (env: CC, VYBHOME, VYBFORGE_LLAMA)
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rope_verify as rv                     # the harness build + the authority runner + rel()

REPO = rv.REPO
LLAMA = rv.LLAMA
VYBHOME = os.environ.get("VYBHOME", os.path.expanduser("~/Projects/Vyb"))
VYB = os.environ.get("VYB", os.path.join(VYBHOME, "build", "vyb"))
STDLIB = os.environ.get("VYB_STDLIB", os.path.join(VYBHOME, "stdlib"))
BUILD = os.path.join(REPO, "native/build")
DRV_IN = os.path.join(BUILD, "rope_driver_in.bin")
DRV_LOG = os.path.join(BUILD, "rope_driver.log")

HD, NH, NKVH, NTOK = 256, 24, 4, 5
ND = 64
BASE = 1e7
MAXREL = rv.MAXREL


def build_kernel():
    env = dict(os.environ, VYB_STDLIB=STDLIB)
    r = subprocess.run([VYB, "--kernel", "native/kernels/rope.vyb", "--ptx", "native/build/rope.ptx",
                        "--module-path", "native/kernels"], cwd=REPO, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print("ROPE_KERNEL_FAIL compile: " + (r.stdout + r.stderr)[-400:])
        return False
    return True


def authority(q, n_head, mode):
    """Run ggml's op on q of shape (NTOK, n_head, HD); one position plane (NEOX)."""
    with open(rv.AUTH_IN, "wb") as fh:
        fh.write(np.ascontiguousarray(q, dtype="<f4").tobytes())
        fh.write(np.ascontiguousarray(np.arange(NTOK, dtype=np.int32), dtype="<i4").tobytes())
    r = subprocess.run([rv.AUTH_BIN, rv.AUTH_IN, rv.AUTH_OUT, str(HD), str(n_head), str(NTOK), str(ND), mode,
                        *[str(s) for s in (11, 11, 10, 0)], str(NTOK)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(("authority", r.stdout + r.stderr)[:300])
    return np.fromfile(rv.AUTH_OUT, dtype="<f4").reshape(NTOK, n_head, HD).astype(np.float64)


def run_driver(q, k, n_rot):
    # a full HD/2 table, so the same fixture serves any n_rot the driver is asked for
    fr = np.array([BASE ** (-2.0 * m / ND) for m in range(HD // 2)], dtype="<f8")
    with open(DRV_IN, "wb") as fh:
        fh.write(np.ascontiguousarray(q, dtype="<f4").tobytes())
        fh.write(np.ascontiguousarray(k, dtype="<f4").tobytes())
        fh.write(fr.tobytes())
    env = dict(os.environ, VYB_STDLIB=STDLIB, VYB_ROPE_IN=DRV_IN, VYB_ROPE_S=str(NTOK), VYB_ROPE_H=str(NH),
               VYB_ROPE_KVH=str(NKVH), VYB_ROPE_HD=str(HD), VYB_ROPE_NROT=str(n_rot))
    r = subprocess.run([VYB, "native/host/rope_driver.vyb", "--module-path", "native/host"], cwd=REPO,
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
                secs[name] = np.array(buf[:want], dtype="<i8")
            _, name, count = line.split()
            want, buf = int(count), []
        elif name is not None and len(buf) < want:
            try:
                buf.extend(int(v) for v in line.split())
            except ValueError:
                pass
    if name is not None:
        secs[name] = np.array(buf[:want], dtype="<i8")
    return {k: np.frombuffer(v.tobytes(), dtype="<f8") for k, v in secs.items()}


def main():
    print(f"ROPE_KERNEL_VERIFY geometry head_dim={HD} n_head={NH} n_head_kv={NKVH} tokens={NTOK} n_dims={ND}")
    if not os.path.exists(os.path.join(LLAMA, "ggml/include/ggml.h")):
        print(f"ROPE_KERNEL_VERIFY_SKIP no llama.cpp checkout at {LLAMA}")
        return 0
    if not rv.build_authority() or not build_kernel():
        return 1

    rng = np.random.default_rng(4242)
    q = rng.normal(0.0, 1.0, size=(NTOK, NH, HD)).astype(np.float32)
    k = rng.normal(0.0, 1.0, size=(NTOK, NKVH, HD)).astype(np.float32)
    qa = authority(q, NH, "neox").reshape(-1)
    ka = authority(k, NKVH, "neox").reshape(-1)

    bad = 0
    for n_rot, expect_match in ((ND, True), (HD, False)):
        out = run_driver(q, k, n_rot)
        if "SKIP" in out and "ROPE_DRIVER_DONE" not in out:
            print("ROPE_KERNEL_VERIFY_SKIP " + [l for l in out.splitlines() if "SKIP" in l][0].strip())
            return 0
        if "ROPE_DRIVER_DONE" not in out:
            print("ROPE_KERNEL_VERIFY_FAIL driver (n_rot=" + str(n_rot) + "): " + out[-700:])
            return 1
        got = parse_dumps(out)
        assert "q" in got and "k" in got, "driver printed no q/k dumps"
        rq, rk = rv.rel(got["q"], qa), rv.rel(got["k"], ka)
        worst = max(rq, rk)
        ok = worst <= MAXREL
        state = "MATCH" if ok else "differs"
        print(f"ROPE_KERNEL_VERIFY n_rot={n_rot:3d} q maxrel={rq:.3e} k maxrel={rk:.3e} {state} "
              f"({'expected' if ok == expect_match else 'UNEXPECTED'})")
        if ok != expect_match:
            if expect_match:
                print(f"ROPE_KERNEL_VERIFY_FAIL n_rot={n_rot} should reproduce the op but does not "
                      f"(worst {worst:.3e}); driver log {os.path.relpath(DRV_LOG, REPO)}")
            else:
                print(f"ROPE_KERNEL_VERIFY_FAIL n_rot={n_rot} (the whole head) also matches — the "
                      f"check does not depend on n_rot, so it proves nothing about this model")
            bad = 1
    if bad:
        return 1
    print(f"ROPE_KERNEL_VERIFY_DONE the kernel reproduces ggml's rope at n_rot={ND} (and does not at "
          f"n_rot={HD}), maxrel bar {MAXREL:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
