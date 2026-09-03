#!/usr/bin/env python3
"""Compare the per-token ASLB layer-0 response-row-10 dumps (diagkv_L0_*.txt) against the
numpy/batched ground truth to localize which backward-read field the per-token fill gets wrong.
Row 10 = response token 1 (absolute position 10)."""
import numpy as np, os
out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "native", "out"))
D, NQ, R = 2560, 4096, 2

def cmp(name, gpath, ref, sl, el):
    g = np.loadtxt(os.path.join(out, gpath))
    r = np.fromfile(os.path.join(out, ref), "<f8")[sl:el]
    n = min(g.size, r.size); g = g[:n]; r = r[:n]
    corr = np.corrcoef(g, r)[0, 1] if r.std() > 1e-30 else 1.0
    rn = float(np.linalg.norm(r)); nr = float(np.linalg.norm(g - r)) / rn if rn > 0 else 0.0
    m = np.abs(r) > 1e-6
    maxrel = np.max(np.abs(g - r)[m] / np.abs(r)[m]) if m.any() else 0.0
    print(f"{name:8s}: corr={corr:.6f} normrel={nr:.3e} maxrel={maxrel:.3e} |r|max={np.abs(r).max():.2e}")
    return max(nr, 1 - corr)

# layer-0 input (offset 0) row 10 = embed[resp_token1]  (kvresp_emb_ref = first 3 response embeds)
c0 = cmp("xin", "diagkv_L0_xin_row10.txt", "kvresp_emb_ref.bin", 1*D, 2*D)
# attn_norm output
c1 = cmp("xn",  "diagkv_L0_xn_row10.txt", "kvresp_L0_xn_ref.bin", 1*D, 2*D)
# attention output DCtx
c2 = cmp("ctx", "diagkv_L0_ctx_row10.txt", "kvresp_L0_ctx_ref.bin", 1*NQ, 2*NQ)
# layer-0 output (XOcur slot) = Ll1 ref row1 (saved l1 = x[NCTX:S] after L0)
c3 = cmp("xo",  "diagkv_L0_xo_row10.txt", "kvresp_Ll1_ref.bin", 1*D, 2*D)
# LoRA sq row10 = xn_row10 @ Uq   (m2e_l0_Uq = [D,R])
xn = np.fromfile(os.path.join(out, "kvresp_L0_xn_ref.bin"), "<f8")[1*D:2*D]
Uq = np.fromfile(os.path.join(out, "m2e_l0_Uq.bin"), "<f8").reshape(D, R)
sq_ref = xn @ Uq
gs = np.loadtxt(os.path.join(out, "diagkv_L0_sq_row10.txt"))
nr = float(np.linalg.norm(gs - sq_ref)) / float(np.linalg.norm(sq_ref))
print(f"sq       : normrel={nr:.3e} (ref xn@Uq) |sq|={np.abs(sq_ref).max():.2e}")
c4 = nr
worst = max(c0, c1, c2, c3, c4)
print("\nworst-normrel (or 1-corr) =", f"{worst:.3e}")
