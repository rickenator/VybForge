#!/usr/bin/env python3
"""Reference for the T2 training-forward activation cache (native/train/train_fwd.vyb).
Computes the layer-0 activations backward needs (pre-norm XN, roped Q/K, V, attn ctx,
ffn_norm X1N, silu-out Hu) for the SAME layer0_input.bin the Vyb driver reads, and saves
them flat for a tolerance compare. Verification-only.
"""
import os, importlib.util, numpy as np
repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = "/home/rick/Models/qwen3/Qwen3-4B-Q4_K_M.gguf"
spec = importlib.util.spec_from_file_location("l0", os.path.join(repo, "native/gguf/layer0_ref.py"))
l0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l0)
D, H, KVH, HD, FF = 2560, 32, 8, 128, 9728
S = 2; NQ = H*HD; NKV = KVH*HD
tens = l0.parse_tsv()
pre = "blk.0."
W = {}
for n in ["attn_k","attn_norm","attn_output","attn_q","attn_v","ffn_down","ffn_gate",
          "ffn_norm","ffn_up","attn_q_norm","attn_k_norm"]:
    sh = tuple(tens[pre+n+".weight"]["shape"])
    W[n] = l0.read_weight(tens[pre+n+".weight"], sh)
x = np.fromfile(os.path.join(repo, "native/out/layer0_input.bin"), dtype="<f8").reshape(S, D)

xn = l0.rmsnorm(x, W["attn_norm"])
q = l0.rmsnorm((xn @ W["attn_q"]).reshape(S, H, HD), W["attn_q_norm"])
k = l0.rmsnorm((xn @ W["attn_k"]).reshape(S, KVH, HD), W["attn_k_norm"])
v = (xn @ W["attn_v"]).reshape(S, KVH, HD)
pos = np.arange(S)
qr, kr = l0.rope(q, k, pos)
ctx = l0.gqa(qr, kr, v)
layer = x + (ctx.reshape(S, H*HD) @ W["attn_output"])
x1n = l0.rmsnorm(layer, W["ffn_norm"])
gate = x1n @ W["ffn_gate"]; up = x1n @ W["ffn_up"]
hu = (gate/(1.0+np.exp(-gate))) * up

def save(name, a):
    out = os.path.join(repo, f"native/out/train_fwd_ref_{name}.txt")
    np.savetxt(out, a.reshape(-1), fmt="%.8g")
    print(f"saved {os.path.basename(out)} ({a.size})")
save("XN", xn); save("DQr", qr.reshape(S, NQ)); save("DKr", kr.reshape(S, NKV))
save("DV", v.reshape(S, NKV)); save("DCtx", ctx.reshape(S, NQ)); save("X1N", x1n); save("Hu", hu)
print("TRAIN_FWD_REF_DONE")
