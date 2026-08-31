#!/usr/bin/env python3
"""Reference for the KV-cache context build: forwards the goal-context token slice (9 tokens)
through all 36 layers (with the same seed LoRA) and dumps each layer's roped-K (DKr, [S,NKV])
and V (DV, [S,NKV]) that the GPU kvctx context-cache should replicate. Fast (9 tokens)."""
import os, importlib.util, numpy as np
repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out = os.path.join(repo, "native", "out")
import json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.path.join(repo, "artifacts", "vybos-configurator-lora"))
for line in open(os.path.join(repo, "data/vybos-configurator-train.jsonl")):
    d = json.loads(line)
    if "Hyprland desktop workstation" in " ".join(m["content"] for m in d["messages"] if m["role"] == "user"):
        usr = [m["content"] for m in d["messages"] if m["role"] == "user"][0]
        break
ctx_ids = np.array(tok.encode(usr, add_special_tokens=False), dtype=np.int64)
S = len(ctx_ids)

D, H, KVH, HD, FF = 2560, 32, 8, 128, 9728
NQ, NKV, R, alpha_r, EPS = H * HD, KVH * HD, 2, 2.0, 1e-6
l0spec = importlib.util.spec_from_file_location("l0", os.path.join(repo, "native/gguf/layer0_ref.py"))
l0 = importlib.util.module_from_spec(l0spec); l0spec.loader.exec_module(l0)
tens = l0.parse_tsv()
WK = {"q": "attn_q", "k": "attn_k", "v": "attn_v", "o": "attn_output", "g": "ffn_gate", "u": "ffn_up", "d": "ffn_down"}
proj_shapes = {"q": (D, NQ), "k": (D, NKV), "v": (D, NKV), "o": (NQ, D), "g": (D, FF), "u": (D, FF), "d": (FF, D)}
ALL_W = [{nn: l0.read_weight(tens[f"blk.{L}.{nn}.weight"], tuple(tens[f"blk.{L}.{nn}.weight"]["shape"])) for nn in ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]} for L in range(36)]
ALL_LN = [{n: l0.read_weight(tens[f"blk.{L}.{n}.weight"], tuple(tens[f"blk.{L}.{n}.weight"]["shape"])) for n in ["attn_norm", "attn_q_norm", "attn_k_norm", "ffn_norm"]} for L in range(36)]
def rms(v, w):
    return v * (1.0 / np.sqrt((v ** 2).mean(-1, keepdims=True) + EPS)) * w
def load_Lo(L):
    return {nm: (np.fromfile(os.path.join(out, f"m2e_l{L}_U{nm}.bin"), "<f8").reshape(D, R) if proj_shapes[nm][0] == D else np.fromfile(os.path.join(out, f"m2e_l{L}_U{nm}.bin"), "<f8").reshape(proj_shapes[nm][0], R),
                 np.fromfile(os.path.join(out, f"m2e_l{L}_V{nm}.bin"), "<f8").reshape(R, proj_shapes[nm][1])) for nm in proj_shapes}
Lo = {L: load_Lo(L) for L in range(36)}
def lproj(a, nm, L):
    U, V = Lo[L][nm]; return a @ ALL_W[L][WK[nm]] + alpha_r * ((a @ U) @ V)
te = tens["token_embd.weight"]
emb_m = None
with open("/home/rick/Models/qwen3/Qwen3-4B-Q4_K_M.gguf", "rb") as fh:
    fh.seek(te["off"]); emb_m = l0.dequant_q6k(fh.read(te["numel"] * 210 // 256)).reshape(te["shape"][1], D)
INVF = np.fromfile(os.path.join(out, "layer0_invfreq.bin"), "<f8")
freqs = lambda s_: s_ * INVF[:HD // 2]
seq = np.arange(S)
x = emb_m[ctx_ids].astype("<f8")
all_dkr = [None] * 36; all_dv = [None] * 36
for L in range(36):
    xn = rms(x, ALL_LN[L]["attn_norm"])
    k = lproj(xn, "k", L)
    v = lproj(xn, "v", L)
    kn = rms(k.reshape(S, KVH, HD), ALL_LN[L]["attn_k_norm"])
    qn = rms(lproj(xn, "q", L).reshape(S, H, HD), ALL_LN[L]["attn_q_norm"])
    qr, kr = l0.rope(qn, kn, seq)
    all_dkr[L] = kr.reshape(S, NKV)
    all_dv[L] = v.reshape(S, NKV)
    # continue the residual forward to the next layer input
    ctx = l0.gqa(qr, kr, v.reshape(S, KVH, HD))
    o = lproj(ctx.reshape(S, NQ), "o", L)
    x1 = x + o
    x1n = rms(x1, ALL_LN[L]["ffn_norm"])
    g = lproj(x1n, "g", L); up = lproj(x1n, "u", L)
    m = lproj((g / (1 + np.exp(-g))) * up, "d", L)
    x = x1 + m
for L in range(36):
    all_dkr[L].tofile(os.path.join(out, f"kvctx_L{L}_DKr_ref.bin"))
    all_dv[L].tofile(os.path.join(out, f"kvctx_L{L}_DV_ref.bin"))
np.savetxt(os.path.join(out, "kvctx_ctx_ids.txt"), ctx_ids, fmt="%d")
print("kvctx ref done S=", S)
