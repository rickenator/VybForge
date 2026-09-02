#!/usr/bin/env python3
"""Numpy reference for the FULL-MANIFEST milestone: per-token RESPONSE forward conditioned on the
429-token capabilities manifest (fullmanifest_ids.bin), producing response hiddens (84) + masked CE —
the scaled GPU's forward target. O(resp*ctx), feasible at S=513 in numpy. Uses layer0_ref rope/gqa
(split-half rope, causal gqa)."""
import os, importlib.util, numpy as np
repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out = os.path.join(repo, "native", "out")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.path.join(repo, "artifacts", "vybos-configurator-lora"))
ctx_ids = np.fromfile(os.path.join(out, "fullmanifest_ids.bin"), "<i8")
resp_ids = np.fromfile(os.path.join(out, "kvresp_ids.bin"), "<i8")
NCTX, RESP = int(ctx_ids.shape[0]), int(resp_ids.shape[0])
D, H, KVH, HD, FF = 2560, 32, 8, 128, 9728
NQ, NKV, R, alpha_r = H * HD, KVH * HD, 2, 2.0
print(f"full-manifest: NCTX={NCTX} RESP={RESP} S={NCTX+RESP}")
l0s = importlib.util.spec_from_file_location("l0", os.path.join(repo, "native/gguf/layer0_ref.py"))
l0 = importlib.util.module_from_spec(l0s); l0s.loader.exec_module(l0)
tens = l0.parse_tsv()
WK = {"q": "attn_q", "k": "attn_k", "v": "attn_v", "o": "attn_output", "g": "ffn_gate", "u": "ffn_up", "d": "ffn_down"}
proj_shapes = {"q": (D, NQ), "k": (D, NKV), "v": (D, NKV), "o": (NQ, D), "g": (D, FF), "u": (D, FF), "d": (FF, D)}
ALL_W = [{nm: l0.read_weight(tens[f"blk.{L}.{WK[nm]}.weight"], tuple(tens[f"blk.{L}.{WK[nm]}.weight"]["shape"])) for nm in WK} for L in range(36)]
ALL_LN = [{n: l0.read_weight(tens[f"blk.{L}.{n}.weight"], tuple(tens[f"blk.{L}.{n}.weight"]["shape"])) for n in ["attn_norm", "attn_q_norm", "attn_k_norm", "ffn_norm"]} for L in range(36)]
def rms(v, w): return v * (1.0 / np.sqrt((v ** 2).mean(-1, keepdims=True) + 1e-6)) * w
def load_Lo(L):
    return {nm: (np.fromfile(os.path.join(out, f"m2e_l{L}_U{nm}.bin"), "<f8").reshape(proj_shapes[nm][0], R),
                 np.fromfile(os.path.join(out, f"m2e_l{L}_V{nm}.bin"), "<f8").reshape(R, proj_shapes[nm][1])) for nm in proj_shapes}
Lo = {L: load_Lo(L) for L in range(36)}
def lproj(a, nm, L):
    U, V = Lo[L][nm]; return a @ ALL_W[L][nm] + alpha_r * ((a @ U) @ V)
te = tens["token_embd.weight"]
with open("/home/rick/Models/qwen3/Qwen3-4B-Q4_K_M.gguf", "rb") as fh:
    fh.seek(te["off"]); emb = l0.dequant_q6k(fh.read(te["numel"] * 210 // 256)).reshape(te["shape"][1], D)

# ---------------- context K/V cache (once) ----------------
ctx_emb = emb[ctx_ids].astype("<f8")
ctx_K = np.zeros((36, NCTX, KVH, HD)); ctx_V = np.zeros((36, NCTX, KVH, HD))
CX = ctx_emb
for L in range(36):
    xn = rms(CX, ALL_LN[L]["attn_norm"])
    q = lproj(xn, "q", L); k = lproj(xn, "k", L); v = lproj(xn, "v", L)
    qn = rms(q.reshape(NCTX, H, HD), ALL_LN[L]["attn_q_norm"])
    kn = rms(k.reshape(NCTX, KVH, HD), ALL_LN[L]["attn_k_norm"])
    qr, kr = l0.rope(qn, kn, np.arange(NCTX))
    ctx_K[L] = kr; ctx_V[L] = v.reshape(NCTX, KVH, HD)
    ctx = l0.gqa(qr, kr, v.reshape(NCTX, KVH, HD))
    o = lproj(ctx.reshape(NCTX, NQ), "o", L)
    x1 = CX + o
    x1n = rms(x1, ALL_LN[L]["ffn_norm"])
    g = lproj(x1n, "g", L); up = lproj(x1n, "u", L)
    hu = (g / (1 + np.exp(-g))) * up
    m = lproj(hu, "d", L)
    CX = x1 + m
print("context cache built; ctx final hidden norm^2 =", float((CX ** 2).sum()))

# ---------------- per-token response forward ----------------
resp_emb = emb[resp_ids].astype("<f8")
RH = resp_emb.copy()
resp_K = np.zeros((36, RESP, KVH, HD)); resp_V = np.zeros((36, RESP, KVH, HD))
for L in range(36):
    xn = rms(RH, ALL_LN[L]["attn_norm"])
    q = lproj(xn, "q", L); k = lproj(xn, "k", L); v = lproj(xn, "v", L)
    qn = rms(q.reshape(RESP, H, HD), ALL_LN[L]["attn_q_norm"])
    kn = rms(k.reshape(RESP, KVH, HD), ALL_LN[L]["attn_k_norm"])
    atn = np.empty((RESP, H, HD))
    for r in range(RESP):
        pos = NCTX + r
        qr_i, kr_i = l0.rope(qn[r:r + 1], kn[r:r + 1], np.array([pos]))
        resp_K[L, r] = kr_i[0]; resp_V[L, r] = v.reshape(RESP, KVH, HD)[r]
        Kc = np.concatenate([ctx_K[L], resp_K[L, :r]], 0)     # [NCTX+r, KVH, HD]
        Vc = np.concatenate([ctx_V[L], resp_V[L, :r]], 0)
        atn[r] = l0.gqa(qr_i, Kc, Vc)[0].reshape(H, HD)
    o = lproj(atn.reshape(RESP, NQ), "o", L)
    x1 = RH + o
    x1n = rms(x1, ALL_LN[L]["ffn_norm"])
    g = lproj(x1n, "g", L); up = lproj(x1n, "u", L)
    hu = (g / (1 + np.exp(-g))) * up
    m = lproj(hu, "d", L)
    RH = x1 + m
resp_hidden = RH.astype("<f8")
resp_hidden.tofile(os.path.join(out, "kvresp429_hidden_gpu_target.bin"))
print("resp_hidden", resp_hidden.shape)

# ---------------- head: output_norm -> tied lm_head -> masked CE ----------------
on_t = tens["output_norm.weight"]; ON = l0.read_weight(on_t, tuple(on_t["shape"]))
h = rms(resp_hidden, ON)
logits = h @ emb.T
lse = logits.max(-1, keepdims=True); e = np.exp(logits - lse); soft = e / e.sum(-1, keepdims=True)
labels = np.roll(resp_ids, -1); labels[-1] = -1
mask = labels >= 0; NV = int(mask.sum())
loss = float(-np.log(soft[np.arange(RESP), labels][mask] + 1e-30).mean())
print("KVRESP429_FWD_CE =", f"{loss:.6f}", "NV=", NV)
np.savetxt(os.path.join(out, "kvresp429_fwd_ce.txt"), [loss], fmt="%.10g")
print("done")
