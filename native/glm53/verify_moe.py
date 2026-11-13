"""P2 gate 1: MOE_ROUTE_VERIFY.

Compares the numpy router (moe_ref.route) against exllamav3's own reference
composition `_routing_ds3_torch` — the implementation the CUDA router kernel is
tested against, i.e. exactly what serves GLM-5.3-Flash — using the REAL layer-3
router weights (native/glm53/fixture).

Gate 2 (MOE_BLOCK_SANITY) runs the full block once with the P1-dequantized
expert-0 weights mapped to every lane (only expert 0 is in the fixture so far;
the 8-distinct-expert case arrives with the loader) and checks numpy vs torch
agree on the same weights, which exercises orientation, clamp and weighting.

Usage: env -u PYTHONPATH ~/venvs/exl3/bin/python native/glm53/verify_moe.py
"""

import sys

import numpy as np
import torch

sys.path.insert(0, "/home/rick/Projects/VybForge/native/glm53")
sys.path.insert(0, "/home/rick/Projects/VybForge/native/exl3")
import moe_ref  # noqa: E402
from exl3_ref import dequant_exl3, load_fixture  # noqa: E402
from exllamav3.modules.block_sparse_mlp_routing import _routing_ds3_torch  # noqa: E402

HIDDEN = moe_ref.HIDDEN
S = 8
rs = np.random.RandomState(3)
y = (rs.randn(S, HIDDEN) * 0.02).astype(np.float32)

gw, bias = moe_ref.load_router()

# --- numpy router ---
idx_np, w_np = moe_ref.route(y, gw, bias)

# --- exllamav3 reference router ---
class Stub:
    pass

cfg = Stub()
cfg.gate_tensor = torch.from_numpy(gw.astype(np.float32).T.copy()).half()
cfg.num_experts = moe_ref.N_EXPERTS
cfg.num_experts_per_tok = moe_ref.TOPK
cfg.e_score_correction_bias = torch.from_numpy(bias.copy())
cfg.routed_scaling_factor = moe_ref.SCALE
cfg.n_group = 1
cfg.topk_group = 1

yt = torch.from_numpy(y)
idx_or, w_or = _routing_ds3_torch(S, cfg, yt, {})
idx_or = idx_or.cpu().numpy()
w_or = w_or.float().cpu().numpy()

set_ok = 0
worst = 0.0
for s in range(S):
    a = dict(zip(idx_np[s].tolist(), w_np[s].tolist()))
    b = dict(zip(idx_or[s].tolist(), w_or[s].tolist()))
    if set(a) == set(b):
        set_ok += 1
        for k in a:
            worst = max(worst, abs(a[k] - b[k]))
    else:
        print(f"  token {s}: index mismatch np={sorted(a)} or={sorted(b)}")
print(f"routing: {set_ok}/{S} tokens with identical expert sets, max |dw| = {worst:.3e}")
gate1 = set_ok == S and worst < 2e-3

# --- gate 2: block sanity, expert-0 weights on every lane ---
fx = load_fixture("/home/rick/Projects/VybForge/native/exl3/fixture")
w_down = dequant_exl3(fx["down_proj_trellis"] if "down_proj_trellis" in fx else fx["trellis"],
                      fx["suh"], fx["svh"]).astype(np.float64)
expert_w = {0: (None, None, None)}

# reload the other two projections through the fixture loader (down/gate/up)
def load_proj(proj):
    d = load_fixture("/home/rick/Projects/VybForge/native/exl3/fixture", proj)
    return dequant_exl3(d["trellis"], d["suh"], d["svh"]).astype(np.float64)

wg_e = load_proj("gate_proj")     # [in 4096, out 2048]
wu_e = load_proj("up_proj")
wd_e = load_proj("down_proj")     # [in 2048, out 4096]
shared = moe_ref.load_shared()

for e in range(moe_ref.N_EXPERTS):
    expert_w[e] = (wg_e, wu_e, wd_e)     # same weights on every lane (see docstring)

out_np, idx2, w2 = moe_ref.moe_block(y[:4], gw, bias, expert_w, shared)

# torch twin of the same math, same weights
def t_ffn(x, wg, wu, wd):
    g = torch.from_numpy(x @ wg)
    u = torch.from_numpy(x @ wu)
    lim = moe_ref.SWIGLU_LIMIT
    g = torch.clamp(g, max=lim)
    u = torch.clamp(u, -lim, lim)
    return (torch.nn.functional.silu(g) * u).numpy() @ wd

out_t = np.zeros_like(out_np)
for s in range(4):
    for lane in range(moe_ref.TOPK):
        out_t[s] += w2[s, lane] * t_ffn(y[s:s + 1].astype(np.float64), wg_e, wu_e, wd_e)[0]
out_t += t_ffn(y[:4].astype(np.float64), *shared)

d = np.abs(out_np - out_t)
denom = np.maximum(np.abs(out_t).max(), 1e-12)
print(f"block: max abs diff numpy vs torch = {d.max():.3e} (out scale {denom:.3e}), "
      f"rel {d.max()/denom:.2e}")
gate2 = d.max() / denom < 1e-9

ok = gate1 and gate2
print("MOE_ROUTE_VERIFY:", "OK" if gate1 else "FAIL")
print("MOE_BLOCK_SANITY:", "OK" if gate2 else "FAIL")
sys.exit(0 if ok else 1)
