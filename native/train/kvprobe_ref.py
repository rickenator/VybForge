#!/usr/bin/env python3
"""Numpy refs for the per-token ASLB PROBE2: layer-input (x before layer L) for response token 1
(abs row 10) at L0/17/35, and (at L17) xn, roped-q, roped-k, ctx, m, sq for the same row. Compare
against diagkv_L*_*.txt to localize which ASLB field the per-token fill gets wrong at a deep layer."""
import os, importlib.util, numpy as np, json
repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out = os.path.join(repo, "native", "out")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.path.join(repo, "artifacts", "vybos-configurator-lora"))
for line in open(os.path.join(repo, "data/vybos-configurator-train.jsonl")):
    d = json.loads(line); msgs = d["messages"]
    if "Hyprland desktop workstation" in " ".join(m["content"] for m in msgs if m["role"] == "user"):
        usr = [m["content"] for m in msgs if m["role"] == "user"][0]
        asst = [m["content"] for m in msgs if m["role"] == "assistant"][-1]; break
ctx_ids = np.array(tok.encode(usr, add_special_tokens=False), dtype=np.int64)   # 9
resp_ids = np.array(tok.encode(asst, add_special_tokens=False), dtype=np.int64) # 84
D, H, KVH, HD, FF = 2560, 32, 8, 128, 9728
NQ, NKV, R, alpha_r = H * HD, KVH * HD, 2, 2.0
S = 9 + 84
l0s = importlib.util.spec_from_file_location("l0", os.path.join(repo, "native/gguf/layer0_ref.py"))
l0 = importlib.util.module_from_spec(l0s); l0s.loader.exec_module(l0)
tens = l0.parse_tsv()
WK = {"q": "attn_q", "k": "attn_k", "v": "attn_v", "o": "attn_output", "g": "ffn_gate", "u": "ffn_up", "d": "ffn_down"}
proj_shapes = {"q": (D, NQ), "k": (D, NKV), "v": (D, NKV), "o": (NQ, D), "g": (D, FF), "u": (D, FF), "d": (FF, D)}
ALL_W = [{nm: l0.read_weight(tens[f"blk.{L}.{WK[nm]}.weight"], tuple(tens[f"blk.{L}.{WK[nm]}.weight"]["shape"])) for nm in WK} for L in range(36)]
ALL_LN = [{n: l0.read_weight(tens[f"blk.{L}.{n}.weight"], tuple(tens[f"blk.{L}.{n}.weight"]["shape"])) for n in ["attn_norm", "attn_q_norm", "attn_k_norm", "ffn_norm"]} for L in range(36)]
def rms(v, w):
    return v * (1.0 / np.sqrt((v ** 2).mean(-1, keepdims=True) + 1e-6)) * w
def load_Lo(L):
    return {nm: (np.fromfile(os.path.join(out, f"m2e_l{L}_U{nm}.bin"), "<f8").reshape(proj_shapes[nm][0], R),
                 np.fromfile(os.path.join(out, f"m2e_l{L}_V{nm}.bin"), "<f8").reshape(R, proj_shapes[nm][1])) for nm in proj_shapes}
Lo = {L: load_Lo(L) for L in range(36)}
def lproj(a, nm, L):
    U, V = Lo[L][nm]; return a @ ALL_W[L][nm] + alpha_r * ((a @ U) @ V)
te = tens["token_embd.weight"]
with open("/home/rick/Models/qwen3/Qwen3-4B-Q4_K_M.gguf", "rb") as fh:
    fh.seek(te["off"]); emb = l0.dequant_q6k(fh.read(te["numel"] * 210 // 256)).reshape(te["shape"][1], D)
INVF = np.fromfile(os.path.join(out, "layer0_invfreq.bin"), "<f8")
seq = np.arange(S)
x = emb[np.concatenate([ctx_ids, resp_ids])].astype("<f8")     # [93, D]

def dump(name, a):
    np.asarray(a, dtype=np.float64).reshape(-1).tofile(os.path.join(out, name))

for L in range(36):
    if L in (0, 17, 35):
        dump(f"kvprobe_L{L}_xin_ref.bin", x[10])                 # layer-input, response token 1 (abs row 10)
    xn = rms(x, ALL_LN[L]["attn_norm"])
    q = lproj(xn, "q", L); k = lproj(xn, "k", L); v = lproj(xn, "v", L)
    qn = rms(q.reshape(S, H, HD), ALL_LN[L]["attn_q_norm"])
    kn = rms(k.reshape(S, KVH, HD), ALL_LN[L]["attn_k_norm"])
    qr, kr = l0.rope(qn, kn, seq)
    ctx = l0.gqa(qr, kr, v.reshape(S, KVH, HD))
    o = lproj(ctx.reshape(S, NQ), "o", L)
    x1 = x + o
    x1n = rms(x1, ALL_LN[L]["ffn_norm"])
    g = lproj(x1n, "g", L); up = lproj(x1n, "u", L)
    hu = (g / (1 + np.exp(-g))) * up
    m = lproj(hu, "d", L)
    if L == 0:
        Uq, _ = Lo[0]["q"]
        dump("kvprobe_L0_dq_ref.bin", q[10].reshape(-1))      # DQ un-roped q
        dump("kvprobe_L0_dk_ref.bin", k[10].reshape(-1))      # DK un-roped k
        dump("kvprobe_L0_dv_ref.bin", v[10].reshape(-1))      # DV v
        dump("kvprobe_L0_do_ref.bin", o[10])                  # DO o-proj
        dump("kvprobe_L0_x1_ref.bin", x1[10])                 # X1 residual (x+o)
        dump("kvprobe_L0_x1n_ref.bin", x1n[10])               # X1N ffn_norm out
        dump("kvprobe_L0_gr_ref.bin", g[10])                  # gate
        dump("kvprobe_L0_up_ref.bin", up[10])                 # up
        dump("kvprobe_L0_hu_ref.bin", hu[10])                 # silu(g)*up
        Uk, _ = Lo[0]["k"]; Uv, _ = Lo[0]["v"]; Uo, _ = Lo[0]["o"]
        Ug, _ = Lo[0]["g"]; Uu, _ = Lo[0]["u"]; Ud, _ = Lo[0]["d"]
        dump("kvprobe_L0_sk_ref.bin", xn[10] @ Uk)
        dump("kvprobe_L0_sv_ref.bin", xn[10] @ Uv)
        dump("kvprobe_L0_so_ref.bin", ctx[10].reshape(-1) @ Uo)
        dump("kvprobe_L0_sg_ref.bin", x1n[10] @ Ug)
        dump("kvprobe_L0_su_ref.bin", x1n[10] @ Uu)
        dump("kvprobe_L0_sd_ref.bin", hu[10] @ Ud)
    if L == 17:
        Uq, _ = Lo[17]["q"]
        dump("kvprobe_L17_xn_ref.bin", xn[10])
        dump("kvprobe_L17_dqr_ref.bin", qr[10].reshape(-1))
        dump("kvprobe_L17_dkr_ref.bin", kr[10].reshape(-1))
        dump("kvprobe_L17_ctx_ref.bin", ctx[10].reshape(-1))
        dump("kvprobe_L17_m2_ref.bin", m[10])
        dump("kvprobe_L17_sq_ref.bin", xn[10] @ Uq)
    x = x1 + m
print("kvprobe_ref done")
