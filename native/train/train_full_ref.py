#!/usr/bin/env python3
"""M2e-1 reference: full 36-layer Qwen3-4B forward WITH LoRA on all 7 projections
(q/k/v/o/gate/up/down) per layer, teacher-forced S=2, streaming ONE layer's weights at
a time (like prefill_ref). Caches EVERY activation the backward pass will need for each
layer into a per-layer dict (xn, q, k, v, qn, kn, qr, kr, ctx, o, x1, x1n, gate, up, hu,
m, xo + LoRA s-intermediates s_q..s_d) and writes them as native/out/m2e_L{layer}_{t}.txt
for the GPU driver (train_full.vyb) to match. LoRA: proj = a@W + alpha_r*((a@U)@V).
This is the backward-feed cache the M2e-2 backward consumes, so nothing is thrown away.
"""
import os, importlib.util, numpy as np

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = "/home/rick/Models/qwen3/Qwen3-4B-Q4_K_M.gguf"
out = os.path.join(repo, "native", "out")
spec = importlib.util.spec_from_file_location("l0", os.path.join(repo, "native/gguf/layer0_ref.py"))
l0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l0)

D, H, KVH, HD, FF = 2560, 32, 8, 128, 9728
NQ, NKV, S, R, alpha_r = H*HD, KVH*HD, 2, 2, 2.0
seq = np.arange(S)

tens = l0.parse_tsv()
# input = token_embd rows 0..S-1 (real embeddings, same as prefill / M2b input contract)
te = tens["token_embd.weight"]
nblk_emb = D // 256
x = np.empty((S, D), np.float64)
with open(MODEL, "rb") as fh:
    for s in range(S):
        fh.seek(te["off"] + s * nblk_emb * 210)
        x[s] = l0.dequant_q6k(fh.read(nblk_emb * 210)).reshape(-1)
np.savetxt(os.path.join(out, "m2e_input.txt"), x.reshape(-1), fmt="%.17g")
x.tofile(os.path.join(out, "m2e_input.bin"))

INVF = np.fromfile(os.path.join(out, "layer0_invfreq.bin"), "<f8")

# deterministic LoRA factors, per layer (seed derived from layer so each layer is distinct)
proj_shapes = {"q":(D,NQ),"k":(D,NKV),"v":(D,NKV),"o":(NQ,D),"g":(D,FF),"u":(D,FF),"d":(FF,D)}
WK = {"q":"attn_q","k":"attn_k","v":"attn_v","o":"attn_output","g":"ffn_gate","u":"ffn_up","d":"ffn_down"}
Lo_by_layer = {}
for layer in range(36):
    rng = np.random.default_rng(1000 + layer)
    Lo = {}
    for nm,(In,Ot) in proj_shapes.items():
        U = rng.normal(size=(In,R))*0.05; V = rng.normal(size=(R,Ot))*0.05
        U.astype("<f8").tofile(os.path.join(out, f"m2e_l{layer}_U{nm}.bin"))
        V.astype("<f8").tofile(os.path.join(out, f"m2e_l{layer}_V{nm}.bin"))
        Lo[nm]=(U,V)
    Lo_by_layer[layer]=Lo

def layer_weight(tens, layer, n):
    name = f"blk.{layer}.{n}.weight"
    return l0.read_weight(tens[name], tuple(tens[name]["shape"]))

def proj(act, Wm, nm, layer):
    U,V = Lo_by_layer[layer][nm]; s = act@U
    return act@Wm + alpha_r*(s@V), s

def rms(v_, w):
    return l0.rmsnorm(v_, w)

all_cache = {}
for layer in range(36):
    pre = f"blk.{layer}."
    W = {n: layer_weight(tens, layer, n) for n in
         ["attn_norm","attn_q","attn_k","attn_v","attn_q_norm","attn_k_norm",
          "attn_output","ffn_norm","ffn_gate","ffn_up","ffn_down"]}
    c = {}
    xn = rms(x, W["attn_norm"]); c["xn"]=xn; c["x"]=x
    q,sq = proj(xn, W["attn_q"], "q", layer); c["q"]=q; c["s_q"]=sq
    k,sk = proj(xn, W["attn_k"], "k", layer); c["k"]=k; c["s_k"]=sk
    v,sv = proj(xn, W["attn_v"], "v", layer); c["v"]=v; c["s_v"]=sv
    qn = rms(q.reshape(S,H,HD), W["attn_q_norm"]); c["qn"]=qn
    kn = rms(k.reshape(S,KVH,HD), W["attn_k_norm"]); c["kn"]=kn
    qr,kr = l0.rope(qn, kn, seq); c["qr"]=qr; c["kr"]=kr
    ctx = l0.gqa(qr, kr, v.reshape(S,KVH,HD)); c["ctx"]=ctx
    o,so = proj(ctx.reshape(S,NQ), W["attn_output"], "o", layer); c["o"]=o; c["s_o"]=so
    x1 = x + o; c["x1"]=x1
    x1n = rms(x1, W["ffn_norm"]); c["x1n"]=x1n
    gate,sg = proj(x1n, W["ffn_gate"], "g", layer); c["gate"]=gate; c["s_g"]=sg
    up,su   = proj(x1n, W["ffn_up"], "u", layer); c["up"]=up; c["s_u"]=su
    hu = (gate/(1+np.exp(-gate)))*up; c["hu"]=hu
    m,sd = proj(hu, W["ffn_down"], "d", layer); c["m"]=m; c["s_d"]=sd
    xo = x1 + m; c["xo"]=xo
    all_cache[layer]=c
    x = xo   # next layer input
    print(f"layer {layer} xo range [{xo.min():.6g},{xo.max():.6g}]", flush=True)

# write every cached tensor per layer (the GPU activation-cache contract)
kept = ["x","xn","q","k","v","qn","kn","qr","kr","ctx","o","x1","x1n","gate","up","hu","m","xo",
        "s_q","s_k","s_v","s_o","s_g","s_u","s_d"]
for L in range(36):
    c = all_cache[L]
    for t in kept:
        np.savetxt(os.path.join(out, f"m2e_L{L}_{t}_ref.txt"), c[t].reshape(-1), fmt="%.9g")
np.savetxt(os.path.join(out, "m2e_hidden_ref.txt"), all_cache[35]["xo"].reshape(-1), fmt="%.17g")
print("M2E1_REF_DONE layers=36 final hidden norm", np.linalg.norm(all_cache[35]["xo"]))
