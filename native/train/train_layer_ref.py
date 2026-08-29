#!/usr/bin/env python3
"""M2b reference: ONE real Qwen3 blk.0 forward WITH LoRA on all 7 projections
(q/k/v/o/gate/up/down), teacher-forced S=2, computing+caching every activation the
backward pass will need. Verification-only. LoRA: proj = x@W + alpha_r*((x@U)@V).
Writes LoRA bins (U/V per projection, r=2) + per-activation refs (train_layer_*.txt).
"""
import os, importlib.util, numpy as np
repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = "/home/rick/Models/qwen3/Qwen3-4B-Q4_K_M.gguf"
out = os.path.join(repo, "native", "out")
spec = importlib.util.spec_from_file_location("l0", os.path.join(repo, "native/gguf/layer0_ref.py"))
l0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l0)
D, H, KVH, HD, FF = 2560, 32, 8, 128, 9728
NQ, NKV, S, R, alpha_r = H*HD, KVH*HD, 2, 2, 2.0
tens = l0.parse_tsv()
pre = "blk.0."
def rw(n):
    t = tens[pre+n+".weight"]; return l0.read_weight(t, tuple(t["shape"]))
W = {n: rw(n) for n in ["attn_q","attn_k","attn_v","attn_output","ffn_gate","ffn_up","ffn_down"]}
LN = {n: rw(n) for n in ["attn_norm","attn_q_norm","attn_k_norm","ffn_norm"]}
x = np.fromfile(os.path.join(out, "layer0_input.bin"), "<f8").reshape(S, D)
fr = np.fromfile(os.path.join(out, "layer0_invfreq.bin"), "<f8").reshape(-1)  # HD/2=64
# LoRA factors (deterministic) + forward
rng = np.random.default_rng(53)
proj_shapes = {"q":(D,NQ),"k":(D,NKV),"v":(D,NKV),"o":(NQ,D),"g":(D,FF),"u":(D,FF),"d":(FF,D)}
Lo = {}
for nm,(In,Ot) in proj_shapes.items():
    U = rng.normal(size=(In,R))*0.05; V = rng.normal(size=(R,Ot))*0.05
    U.astype("<f8").tofile(os.path.join(out, f"tl_U{nm}.bin")); V.astype("<f8").tofile(os.path.join(out, f"tl_V{nm}.bin"))
    Lo[nm]=(U,V)

def proj(act, Wm, nm):
    U,V = Lo[nm]; s = act@U; ex = s@V
    return (act@Wm + alpha_r*ex, s)

cache = {}
cache["x"]=x
xn = l0.rmsnorm(x, LN["attn_norm"]); cache["xn"]=xn
cache["q"],c_s_q = proj(xn, W["attn_q"],"q"); cache["s_q"]=c_s_q   # [S,NQ]-ish? store as used
# store s per-proj (used later; keep list)
s_all={"q":c_s_q}
q, s_all["q"] = proj(xn, W["attn_q"], "q"); cache["q"]=q
k, s_all["k"] = proj(xn, W["attn_k"], "k"); cache["k"]=k
v, s_all["v"] = proj(xn, W["attn_v"], "v"); cache["v"]=v
qn = l0.rmsnorm(q.reshape(S,H,HD), LN["attn_q_norm"]); cache["qn"]=qn
kn = l0.rmsnorm(k.reshape(S,KVH,HD), LN["attn_k_norm"]); cache["kn"]=kn
qr, kr = l0.rope(qn, kn, np.arange(S)); cache["qr"]=qr; cache["kr"]=kr
ctx = l0.gqa(qr, kr, v.reshape(S,KVH,HD)); cache["ctx"]=ctx
o, s_all["o"] = proj(ctx.reshape(S,NQ), W["attn_output"], "o"); cache["o"]=o
x1 = x + o; cache["x1"]=x1
x1n = l0.rmsnorm(x1, LN["ffn_norm"]); cache["x1n"]=x1n
gate, s_all["g"] = proj(x1n, W["ffn_gate"], "g"); cache["gate"]=gate
up,   s_all["u"] = proj(x1n, W["ffn_up"], "u"); cache["up"]=up
hu = (gate/(1+np.exp(-gate)))*up; cache["hu"]=hu
m, s_all["d"] = proj(hu, W["ffn_down"], "d"); cache["m"]=m
xo = x1 + m; cache["xo"]=xo

for nm in ["x","xn","q","k","v","qn","kn","qr","kr","ctx","o","x1","x1n","gate","up","hu","m","xo","s_q","s_k","s_v","s_o","s_g","s_u","s_d"]:
    if nm in cache:
        np.savetxt(os.path.join(out, f"tl_{nm}_ref.txt"), cache[nm].reshape(-1), fmt="%.8g")
    elif nm.startswith("s_"):
        np.savetxt(os.path.join(out, f"tl_{nm}_ref.txt"), s_all[nm[2:]].reshape(-1), fmt="%.8g")
print("TRAIN_LAYER_REF_DONE x", x.shape, "xo", xo.shape, "s_q", s_all["q"].shape)
