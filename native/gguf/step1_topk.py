#!/usr/bin/env python3
"""Step-1 logits (prefix [0,1,31784]) top-8 from numpy, to see if the GPU's
step-1 pick (73305) is a near-tie with the reference's (12001) — i.e. greedy
precision sensitivity vs a real forward bug."""
import os, importlib.util, numpy as np
repo = "/home/rick/Projects/VybAIConf"
MODEL = "/home/rick/Models/qwen3/Qwen3-4B-Q4_K_M.gguf"
spec = importlib.util.spec_from_file_location("l0", os.path.join(repo, "native/gguf/layer0_ref.py"))
l0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l0)
D,H,KVH,HD,FF = 2560,32,8,128,9728
tens = l0.parse_tsv()
def fl(x,W):
    xn=l0.rmsnorm(x,W["attn_norm"])
    q=l0.rmsnorm((xn@W["attn_q"]).reshape(x.shape[0],H,HD),W["attn_q_norm"])
    k=l0.rmsnorm((xn@W["attn_k"]).reshape(x.shape[0],KVH,HD),W["attn_k_norm"])
    v=(xn@W["attn_v"]).reshape(x.shape[0],KVH,HD)
    pos=np.arange(x.shape[0]); q,k=l0.rope(q,k,pos)
    o=l0.gqa(q,k,v)
    layer=x+(o@W["attn_output"])
    h=l0.rmsnorm(layer,W["ffn_norm"])
    gate=h@W["ffn_gate"]; up=h@W["ffn_up"]
    return layer+((gate/(1+np.exp(-gate))*up)@W["ffn_down"])
def lw(layer):
    pre=f"blk.{layer}."; W={}
    for n in ["attn_k","attn_norm","attn_output","attn_q","attn_v","ffn_down","ffn_gate","ffn_norm","ffn_up","attn_q_norm","attn_k_norm"]:
        sh=tuple(tens[pre+n+".weight"]["shape"]); W[n]=l0.read_weight(tens[pre+n+".weight"],sh)
    return W
def emb(tid):
    te=tens["token_embd.weight"]; nb=D//256
    with open(MODEL,"rb") as fh:
        fh.seek(te["off"]+tid*nb*210); return l0.dequant_q6k(fh.read(nb*210)).reshape(-1)
x=np.stack([emb(0),emb(1),emb(31784)])   # step-1 prefix
for L in range(36): x=fl(x,lw(L))
onsh=tuple(tens["output_norm.weight"]["shape"])
h=l0.rmsnorm(x[-1],l0.read_weight(tens["output_norm.weight"],onsh))
te=tens["token_embd.weight"]
with open(MODEL,"rb") as fh:
    fh.seek(te["off"]); E=l0.dequant_q6k(fh.read(te["numel"]*210//256)).reshape(te["shape"][1],D)
lg=h@E.T
top=np.argsort(-lg)[:10]
print("step1 numpy top10 token:logit")
for t in top: print(f"  {t}: {lg[t]:.6f}")
print("GPU pick 73305 logit:", lg[73305], " ref pick 12001 logit:", lg[12001])
