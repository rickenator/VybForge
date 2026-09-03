#!/usr/bin/env python3
"""Verifier for the full-36-layer Qwen3 prefill:
  - prefill_hidden_vyb.txt (GPU) vs prefill_hidden_ref.txt (numpy) -> hidden match
  - prefill_top1_vyb.txt (GPU, tied lm_head argmax) vs prefill_top1_ref.txt -> exact
Run after `make -f native/Makefile prefill`."""
import os, numpy as np

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(repo, "native/out")
TOL = 2e-3

ok = True

hv = os.path.join(OUT, "prefill_hidden_vyb.txt")
hr = os.path.join(OUT, "prefill_hidden_ref.txt")
if not (os.path.exists(hv) and os.path.exists(hr)):
    print("MISSING hidden files")
    raise SystemExit(2)
vyb = np.loadtxt(hv); ref = np.loadtxt(hr)
rel = np.abs(vyb - ref) / np.maximum(np.abs(ref), 1e-6)
bad = int((rel > TOL).sum())
print(f"36-layer hidden: bad(rel>{TOL}) = {bad}  maxrel = {rel.max():.3e}  maxabs = {np.abs(vyb-ref).max():.3e}")
print("PREFILL_HIDDEN_MATCH:", "OK" if bad == 0 else "FAIL")
if bad: ok = False

t1v = os.path.join(OUT, "prefill_top1_vyb.txt")
t1r = os.path.join(OUT, "prefill_top1_ref.txt")
if os.path.exists(t1v) and os.path.exists(t1r):
    v = np.loadtxt(t1v, dtype=int)
    r = np.loadtxt(t1r, dtype=int)
    match = (v == r).all() and v.shape == r.shape
    print(f"  top1 vyb={v.tolist()} ref={r.tolist()} -> {'MATCH' if match else 'MISMATCH'}")
    print("PREFILL_TOP1_MATCH:", "OK" if match else "FAIL")
    if not match: ok = False
else:
    print("PREFILL_TOP1_MATCH: SKIP (no top1 files; DOLM not run)")

raise SystemExit(0 if ok else 1)
