"""Prepare the P2 Vyb fixture: f32 arrays both sides see identically, plus the
numpy reference output. Writes to native/glm53/prep/ (gitignored).

Usage: env -u PYTHONPATH ~/venvs/exl3/bin/python native/glm53/prepare_moe_fixture.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, "/home/rick/Projects/VybForge/native/glm53")
sys.path.insert(0, "/home/rick/Projects/VybForge/native/exl3")
import moe_ref  # noqa: E402
from exl3_ref import dequant_exl3, load_fixture  # noqa: E402

HIDDEN, INTER, NE, TOPK = moe_ref.HIDDEN, moe_ref.INTER, moe_ref.N_EXPERTS, moe_ref.TOPK
OUT = "/home/rick/Projects/VybForge/native/glm53/prep"
os.makedirs(OUT, exist_ok=True)

S = 4
rs = np.random.RandomState(3)
# hidden-state scale: big enough that gate logits move the selection off the bias
# (at 0.02 the bias dominated and every token routed identically)
y = (rs.randn(S, HIDDEN) * 0.3).astype(np.float32)

gw, bias = moe_ref.load_router()
gw32 = gw.astype(np.float32)          # load_router already returns f32
bias32 = bias.astype(np.float32)


def deq(proj):
    d = load_fixture("/home/rick/Projects/VybForge/native/exl3/fixture", proj)
    return dequant_exl3(d["trellis"], d["suh"], d["svh"]).astype(np.float32)


def ensure(name, shape, builder):
    """Load a prepared array, or build it once and cache it (the f64 dequant of a
    2048x4096 expert is minutes of numpy; reruns must not repeat it)."""
    p = f"{OUT}/{name}.f32"
    want = int(np.prod(shape)) * 4
    if os.path.exists(p) and os.path.getsize(p) == want:
        return np.fromfile(p, dtype="<f4").reshape(shape)
    a = np.ascontiguousarray(builder().astype(np.float32))
    a.tofile(p)
    print(f"  built {name} {shape}")
    return a


wg_e = ensure("wg_e", (HIDDEN, INTER), lambda: deq("gate_proj"))
wu_e = ensure("wu_e", (HIDDEN, INTER), lambda: deq("up_proj"))
wd_e = ensure("wd_e", (INTER, HIDDEN), lambda: deq("down_proj"))
sg = ensure("wg_s", (HIDDEN, INTER), lambda: moe_ref.load_shared()[0])
su = ensure("wu_s", (HIDDEN, INTER), lambda: moe_ref.load_shared()[1])
sd = ensure("wd_s", (INTER, HIDDEN), lambda: moe_ref.load_shared()[2])

for name, arr in (("y", y), ("gate", gw32), ("bias", bias32),
                  ("wg_e", wg_e), ("wu_e", wu_e), ("wd_e", wd_e),
                  ("wg_s", sg), ("wu_s", su), ("wd_s", sd)):
    arr.tofile(f"{OUT}/{name}.f32")
    print(f"{name}: {arr.shape} {arr.nbytes:,}B")

# reference output with the SAME f32 weights (numpy works in f64 on them)
expert_w = {e: (wg_e.astype(np.float64), wu_e.astype(np.float64), wd_e.astype(np.float64))
            for e in range(NE)}
out, idx, w = moe_ref.moe_block(y.astype(np.float64), gw32, bias32,
                                expert_w, (sg, su, sd))
out.astype(np.float32).tofile(f"{OUT}/ref_out.f32")
idx.astype(np.float32).tofile(f"{OUT}/ref_idx.f32")
w.astype(np.float32).tofile(f"{OUT}/ref_w.f32")
print(f"S={S} ref out {out.shape}, |out| max {np.abs(out).max():.4f}")
