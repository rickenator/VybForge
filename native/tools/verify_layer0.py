#!/usr/bin/env python3
"""Verifier: does the GPU Qwen3 layer-0 forward (native/out/layer0_vyb.txt,
written by native/host/layer0_driver.vyb) match the numpy reference
(native/out/layer0_ref.txt)? Gate for the Phase-3 swap-in layer match (step 4)."""
import os, numpy as np

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VYB = os.path.join(repo, "native/out/layer0_vyb.txt")
REF = os.path.join(repo, "native/out/layer0_ref.txt")
TOL = 2e-3   # f64 GPU vs numpy fp32-quantized reference: expect ~1e-3..1e-4

if not os.path.exists(VYB):
    print("LAYER0_VYB_MISSING: run make -f native/Makefile layer0 first")
    raise SystemExit(2)

vyb = np.loadtxt(VYB)
ref = np.loadtxt(REF)
assert vyb.shape == ref.shape, (vyb.shape, ref.shape)
n = ref.size
abserr = np.abs(vyb - ref)
scale = np.maximum(np.abs(ref), 1e-6)
relerr = abserr / scale
max_abs = float(abserr.max())
max_rel = float(relerr.max())
bad = int((relerr > TOL).sum())
worst = int(np.argmax(abserr))
print(f"layer-0 out: n={n}")
print(f"  max abs err = {max_abs:.6e}  at idx {worst} (ref={ref[worst]:.6g} vyb={vyb[worst]:.6g})")
print(f"  max rel err = {max_rel:.6e}  bad(> {TOL}) = {bad}")
print("  vyb range [%.6g, %.6g]  ref range [%.6g, %.6g]" %
      (vyb.min(), vyb.max(), ref.min(), ref.max()))
print("LAYER0_MATCH:", "OK" if bad == 0 else "FAIL")
raise SystemExit(0 if bad == 0 else 1)
