#!/usr/bin/env python3
"""Gate for M2e-2: the GPU full-36-layer LoRA BACKWARD (m2e_L{layer}_{UV}_vyb.txt for the 14
U/V grads + dx) vs the numpy oracle (m2e_L{layer}_{UV}_ref.txt). Checks every projection of
every layer + the chained dx0."""
import numpy as np, os, sys
out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "native", "out"))
tol = 2e-2  # grads near vanishing input layers have tiny magnitudes -> use relative-on-large-elems

UV = ["Uq","Vq","Uk","Vk","Uv","Vv","Uo","Vo","Ug","Vg","Uu","Vu","Ud","Vd"]

def check(name_v, name_r):
    v = np.loadtxt(os.path.join(out, name_v))
    r = np.loadtxt(os.path.join(out, name_r))
    if v.size != r.size:
        return "FAIL", 9.9, 0.0
    m = np.abs(r) > 1e-6
    if not m.any():
        return "OK", 0.0, 1.0
    rel = np.max(np.abs(v - r)[m] / np.abs(r)[m])
    corr = np.corrcoef(v, r)[0, 1]
    flag = "OK" if rel < tol and corr > 0.999 and not np.isnan(rel) else "FAIL"
    return flag, rel, corr

bad = []; tot = 0; ok = 0
for L in range(36):
    # dx is compared at the layer level too (layer's input grads)
    for uv in UV:
        tot += 1
        f, rel, corr = check(f"m2e_L{L}_{uv}_vyb.txt", f"m2e_L{L}_{uv}_ref.txt")
        if f != "OK": bad.append(f"L{L}_{uv} ({rel:.2e})")
        else: ok += 1
    tot += 1
    f, rel, corr = check(f"m2e_L{L}_dx_vyb.txt", f"m2e_L{L}_dx_ref.txt")
    if f != "OK": bad.append(f"L{L}_dx ({rel:.2e})")
    else: ok += 1
print(f"backward grads checked {ok}/{tot}")
print("M2E2_VERIFY:", "OK" if not bad else f"FAIL {bad[:24]}")
sys.exit(0 if not bad else 1)
