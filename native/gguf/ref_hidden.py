#!/usr/bin/env python3
"""Dump the numpy-reference step-0 final hidden (rmsnorm of last position)
for the decode seed [0,1] to native/out/decode_hidden_ref.txt, for comparison
with the Vyb GPU decode's native/out/decode_hidden_vyb.txt."""
import os, importlib.util, numpy as np
repo = "/home/rick/Projects/VybAIConf"
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
    return layer + ((gate/(1+np.exp(-gate))*up) @ W["ffn_down"])

def layer_weights(layer):
    pre = f"blk.{layer}."; W = {}
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
np.savetxt(os.path.join(repo, "native/out/decode_hidden_ref.txt"), h, fmt="%.12g")
print("REF_HIDDEN_SAVED", list(h[:8]))
