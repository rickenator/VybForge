#!/usr/bin/env python3
"""The recurrent block on the ENGINE's weight path vs the block built from ggml's own ops.

VybForge#10 phase 4, unit 9c step 2.

What this adds over `gdn_layer_kernel_verify.py` (P4.3). That gate proved the block can be WIRED out
of Vyb kernels — but it fed both sides a fixture, and the driver multiplied with `mm_nt`, whose B
operand is [out,in]. The engine does not work that way: it finds each tensor by NAME in the GGUF,
stages the packed types with the same dequant kernels it uses for every layer, and multiplies with
`gemm` (layer.ptx), whose B operand is [in,out] — the layout the dequant kernels' transposing write
produces (VybForge#11). So the numbers below come through a DIFFERENT staging path and a DIFFERENT
matmul from P4.3's, and they are compared against the same authority (unit 5: the real ggml ops), on
the same real weights (blk.0 of the Ridge model).

The fixture is reused from `gdn_layer_kernel_verify` so the authority side is byte-identical to
P4.3's: what changed is the driver under test, not the reference. Only the two step inputs (x) come
from the fixture; every weight is read from the GGUF by the driver itself, which is the point.

Usage: gdn_engine_verify.py    (env: VYBFORGE_RIDGE_GGUF, CC, VYBHOME, VYBFORGE_LLAMA)
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gdn_layer_kernel_verify as gdnl        # the fixture + the authority runner, reused verbatim

REPO = gdnl.REPO
VYB, STDLIB = gdnl.VYB, gdnl.STDLIB
NE, S, H_K, H_V, DC, T, B = gdnl.NE, gdnl.S, gdnl.H_K, gdnl.H_V, gdnl.DC, gdnl.T, gdnl.B
MAXREL = gdnl.MAXREL

BUILD = os.path.join(REPO, "native/build")
DRV_X = os.path.join(BUILD, "gdn_engine_x.bin")
DRV_Z = os.path.join(BUILD, "gdn_engine_zero.bin")
DRV_LOG = os.path.join(BUILD, "gdn_engine_driver.log")
TSV = os.path.join(REPO, "native/out/ridge_tensors.tsv")
MODEL = os.environ.get("VYBFORGE_RIDGE_GGUF",
                       os.path.expanduser("~/Models/qwen38-27b-ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf"))
# which recurrent block the engine driver probes. blk.0 is the first one in the Ridge model (the
# interleave is 3 recurrent, then 1 attention, so any 0/1/2 works; blk.3 is attention).
PROBE_LAYER = int(os.environ.get("VYBFORGE_GDN_PROBE_LAYER", "0"))


def run_driver(fx):
    """The ENGINE's own driver, in probe mode.

    `native/host/model_driver.vyb` is the file the loop lives in; VYB_GDN_PROBE=<layer> makes it run
    ONE recurrent block through the same staging, kernels and state carry-over the loop will use.
    Pointing the check at this driver (rather than at a side harness) is the point: a green probe is
    the loop's block half, verified in the file that will run it.
    """
    with open(DRV_X, "wb") as fh:
        fh.write(np.ascontiguousarray(fx["x"], dtype="<f8").tobytes())
    with open(DRV_Z, "wb") as fh:
        fh.write(b"\x00" * (S * S * H_V * 8))
    env = dict(os.environ, VYB_STDLIB=STDLIB, VYB_MODEL=MODEL, VYB_TSV=TSV,
               VYB_GDN_X=DRV_X, VYB_GDN_Z=DRV_Z, VYB_GDN_PROBE=str(PROBE_LAYER),
               VYB_PROMPT_RAW="1")
    cmd = [VYB, "native/host/model_driver.vyb",
           "--module-path", "native/config", "--module-path", "native/json",
           "--module-path", "native/tensor", "--module-path", "native/dtype",
           "--module-path", "native/llm"]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
    out = r.stdout + r.stderr
    with open(DRV_LOG, "w") as fh:
        fh.write(out)
    return out


def main():
    print(f"GDNL_ENGINE_VERIFY geometry n_embd={NE} S={S} H_k={H_K} H_v={H_V} d_conv={DC}, steps=2")
    if not os.path.exists(TSV):
        print(f"GDNL_ENGINE_VERIFY_FAIL no engine tensor index at {TSV} "
              f"(run native/tools/inventory_to_tsv.py)")
        return 1
    if not os.path.exists(MODEL):
        print(f"GDNL_ENGINE_VERIFY_SKIP the model is not on disk at {MODEL}")
        return 0
    if not gdnl.build_authority():
        print("GDNL_ENGINE_VERIFY_FAIL authority build")
        return 1

    fx, real, quantised, f32_real, q4_real = gdnl.fixture()
    if not (quantised and f32_real and q4_real):
        print("GDNL_ENGINE_VERIFY_SKIP the model's own blk.0 tensors are unavailable — this check is "
              "about the engine's staging of REAL weights, so a synthetic fixture would prove nothing")
        return 0
    # The fixture carries the model's real attn_norm / ssm_conv1d / ssm_a / ssm_dt and the real
    # quantised weights, but ssm_norm is still synthetic there while the driver stages the model's
    # own. Both sides must see the SAME weights or the comparison measures the fixture, not the
    # staging — so the authority gets the real ssm_norm here.
    real_ssm_norm = gdnl.real_f32_weight("blk.0.ssm_norm.weight", (S,))
    if real_ssm_norm is None:
        print("GDNL_ENGINE_VERIFY_SKIP blk.0.ssm_norm.weight is unavailable")
        return 0
    fx["ssm_norm"] = real_ssm_norm
    print("GDNL_ENGINE_VERIFY fixture: real blk.0 weights (Q4_K projections, Q8_0 alpha/beta, F32 "
          "norms/conv/a/dt/ssm_norm) staged by the DRIVER from the GGUF; two decode steps from zero state")

    # The authority: two single-token steps, step 2 carrying step 1's conv window and state.
    d1 = os.path.join(gdnl.AUTH_DIR, "e1")
    d2 = os.path.join(gdnl.AUTH_DIR, "e2")
    win0 = np.zeros(gdnl.QKV * (DC - 1), dtype=np.float32)
    st0 = np.zeros(S * S * H_V, dtype=np.float32)
    gdnl.run_authority_step(fx, fx["x"][0], win0, st0, d1)
    conv1 = np.fromfile(os.path.join(d1, "conv_in.bin"), dtype="<f4").reshape(gdnl.QKV, DC)
    win1 = np.ascontiguousarray(conv1[:, 1:DC]).reshape(-1)
    st1 = np.fromfile(os.path.join(d1, "state_out.bin"), dtype="<f4").reshape(-1)
    gdnl.run_authority_step(fx, fx["x"][1], win1, st1, d2)

    out = run_driver(fx)
    if "SKIP" in out and "GDN_PROBE_DONE" not in out:
        print("GDNL_ENGINE_VERIFY_SKIP " + [l for l in out.splitlines() if "SKIP" in l][0].strip())
        return 0
    if "GDN_PROBE_DONE" not in out:
        print("GDNL_ENGINE_VERIFY_FAIL driver: " + out[-900:])
        return 1
    for line in out.splitlines():
        if line.startswith("GDN_GEOM") or line.startswith("GDN_STAGED"):
            print("GDNL_ENGINE_VERIFY " + line)
    got = gdnl.parse_dumps(out)

    # The staged operand itself, at the exact addresses the engine's gemm reads: B[k*N+n] must be
    # the model's own dequantised weight. Stage-level agreement alone cannot see a good block fed a
    # slightly wrong operand, so this is checked directly (and it is what localised the alpha/beta
    # reuse bug: the projection was right and its OPERAND's consumer was not).
    probes = []
    for line in out.splitlines():
        if line.startswith("PROBE so "):
            _, tag, idx, bits = line.split()
            v = np.frombuffer(np.array([int(bits)], dtype="<i8").tobytes(), dtype="<f8")[0]
            probes.append((int(idx), v))
    so = fx["ssm_out"].astype(np.float64)                 # numpy (NE, VALUE) = [n,k]
    pbad = []
    for idx, v in probes:
        k, n = idx // NE, idx % NE
        want = so[n, k]
        if not (abs(v - want) <= 1e-5 * max(abs(want), 1e-12)):
            pbad.append((idx, v, want))
    if not probes or pbad:
        print(f"GDNL_ENGINE_VERIFY staged ssm_out operand FAIL ({len(pbad)}/{len(probes)} slots wrong)")
        for idx, v, want in pbad[:4]:
            print(f"                   idx {idx} driver={v:.9e} model={want:.9e}")
        print("GDNL_ENGINE_VERIFY_FAIL the staged operand is not the model's own weight")
        return 1
    print(f"GDNL_ENGINE_VERIFY staged ssm_out operand ok ({len(probes)} slots of B[k*N+n] == the "
          f"model's own dequantised tensor)")

    stages = ("xn", "qkv", "z", "beta", "gate", "conv_silu", "q_norm", "k_norm",
              "gdn_out", "epi", "y", "layer_out", "state_out")
    bad = 0
    worst = 0.0
    for step, d in ((1, d1), (2, d2)):
        for name in stages:
            key = f"s{step}_{name}"
            ap = os.path.join(d, f"{name}.bin")
            if not os.path.exists(ap) or key not in got:
                print(f"GDNL_ENGINE_VERIFY {key:16s} MISSING (authority="
                      f"{'y' if os.path.exists(ap) else 'n'} driver={'y' if key in got else 'n'})")
                bad = 1
                continue
            shape = (B, H_V, S, S) if name == "state_out" else gdnl.SHAPES[name]
            ref = np.fromfile(ap, dtype="<f4").reshape(shape).astype(np.float64).reshape(-1)
            gpu = got[key].reshape(-1)
            if gpu.size != ref.size:
                print(f"GDNL_ENGINE_VERIFY {key:16s} LENGTH MISMATCH driver={gpu.size} "
                      f"authority={ref.size}")
                bad = 1
                continue
            r = gdnl.rel(gpu, ref)
            worst = max(worst, r)
            # NaN must FAIL: `not (r <= bar)`, never `r > bar`.
            if not (r <= MAXREL):
                bad = 1
                print(f"GDNL_ENGINE_VERIFY {key:16s} n={gpu.size:6d} maxrel={r:.3e} DIFFERS")
                w = int(np.nanargmax(np.abs(gpu - ref)))
                print(f"                   worst idx {w}: driver={gpu[w]:.9e} authority={ref[w]:.9e}")
            else:
                print(f"GDNL_ENGINE_VERIFY {key:16s} n={gpu.size:6d} maxrel={r:.3e} ok")

    if bad:
        print(f"GDNL_ENGINE_VERIFY_FAIL worse than maxrel {MAXREL:g} "
              f"(driver log: {os.path.relpath(DRV_LOG, REPO)})")
        return 1

    # The check's teeth, computed from the authority's own step-2 stages.
    lo2 = np.fromfile(os.path.join(d2, "layer_out.bin"), dtype="<f4").reshape(-1).astype(np.float64)
    y2 = np.fromfile(os.path.join(d2, "y.bin"), dtype="<f4").reshape(-1).astype(np.float64)
    xn2 = np.fromfile(os.path.join(d2, "xn.bin"), dtype="<f4").reshape(-1).astype(np.float64)
    alt_no_res = gdnl.rel(y2, lo2)
    alt_norm_res = gdnl.rel(xn2 + y2, lo2)
    print(f"GDNL_ENGINE_VERIFY alt step-2 no-residual          maxrel={alt_no_res:.3e} (must be large)")
    print(f"GDNL_ENGINE_VERIFY alt step-2 residual-on-attn_norm maxrel={alt_norm_res:.3e} (must be large)")
    if min(alt_no_res, alt_norm_res) <= MAXREL:
        print("GDNL_ENGINE_VERIFY_FAIL an alternative matches — the stage check proves nothing")
        return 1

    print(f"GDNL_ENGINE_VERIFY_SUMMARY steps=2 stages=26 worst={worst:.3e} "
          f"rejected_alternatives={alt_no_res:.2e}/{alt_norm_res:.2e}")
    print("GDNL_ENGINE_VERIFY_DONE the engine's own staging + gemm reproduce ggml's graph for the "
          "recurrent block, two chained steps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
