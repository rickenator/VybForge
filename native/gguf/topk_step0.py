#!/usr/bin/env python3
"""Print the numpy-reference step-0 logits top-8 for the decode seed [0,1],
so we can see whether the GPU's picked token (41524) is a near-tie with the
reference's 31784 (precision sensitivity) or a real divergence (bug)."""
import os, importlib.util, numpy as np
repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = "/home/rick/Models/qwen3/Qwen3-4B-Q4_K_M.gguf"
spec = importlib.util.spec_from_file_location("l0", os.path.join(repo, "native/gguf/layer0_ref.py"))
l0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l0)
D, H, KVH, HD, FF = 2560, 32, 8, 128, 9728
tens = l0.parse_tsv()

def forward_layer(x, W):
    xn = l0.rmsnorm(x, W["attn_norm"])
    q = l0.rmsnorm((xn @ W["attn_q"]).reshape(x.shape[0], H, HD), W["attn_q_norm"])
    k = l0.rmsnorm((xn @ W["attn_k"]).reshape(x.shape[0], KVH, HD), W["attn_k_norm"])
    v = (xn @ W["attn_v"]).reshape(x.shape[0], KVH, HD)
    pos = np.arange(x.shape[0]); q, k = l0.rope(q, k, pos)
    o = l0.gqa(q, k, v)
    layer = x + (o @ W["attn_output"])
    h = l0.rmsnorm(layer, W["ffn_norm"])
    gate = h @ W["ffn_gate"]; up = h @ W["ffn_up"]
    out = layer + ((gate/(1+np.exp(-gate))*up) @ W["ffn_down"])
    return out

def layer_weights(layer):
    pre = f"blk.{layer}."
    W = {}
    for n in ["attn_k","attn_norm","attn_output","attn_q","attn_v","ffn_down",
              "ffn_gate","ffn_norm","ffn_up","attn_q_norm","attn_k_norm"]:
        shape = tuple(tens[pre+n+".weight"]["shape"])
        W[n] = l0.read_weight(tens[pre+n+".weight"], shape)
    return W

def emb_row(tid):
    te = tens["token_embd.weight"]; nblk = D//256
    with open(MODEL,"rb") as fh:
        fh.seek(te["off"]+tid*nblk*210)
        return l0.dequant_q6k(fh.read(nblk*210)).reshape(-1)

x = np.stack([emb_row(0), emb_row(1)])
for L in range(36):
    x = forward_layer(x, layer_weights(L))
on_shape = tuple(tens["output_norm.weight"]["shape"])
h = l0.rmsnorm(x[-1], l0.read_weight(tens["output_norm.weight"], on_shape))
te = tens["token_embd.weight"]
with open(MODEL,"rb") as fh:
    fh.seek(te["off"]); emb = l0.dequant_q6k(fh.read(te["numel"]*210//256)).reshape(te["shape"][1], D)
lg = h @ emb.T   # 1-D [V] (h is the last-row hidden)
top = np.argsort(-lg)[:8]
print("step0 top8 token:logit")
for t in top:
    print(f"  {t}: {lg[t]:.6f}")
