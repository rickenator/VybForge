#!/usr/bin/env python3
"""AUTHORITATIVE reference for the KV-aware response forward (kvrespfwd.vyb).

Runs the full [9-token goal context | 84-token response] sequence (S=93) through all 36 layers
with the seed LoRA at 0-based positions 0..92 (matching the kvctx context build -> POS=0), and
dumps the RESPONSE region hidden rows (positions 9..92 = layer-36 output for the response tokens)
to native/out/kvresp_hidden_ref.bin [84, D]. kvrespfwd's HALL (response hiddens) must match this.
Disambiguates the rope-POS question: the reference is built at the natural 0-based positions, so a
correct kvrespfwd (context 0..8 + response 9..92) reproduces it exactly.
"""
import os, importlib.util, numpy as np, json
repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out = os.path.join(repo, "native", "out")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.path.join(repo, "artifacts", "vybos-configurator-lora"))
for line in open(os.path.join(repo, "data/vybos-configurator-train.jsonl")):
    d = json.loads(line); msgs = d["messages"]
    if "Hyprland desktop workstation" in " ".join(m["content"] for m in msgs if m["role"] == "user"):
        usr = [m["content"] for m in msgs if m["role"] == "user"][0]
        asst = [m["content"] for m in msgs if m["role"] == "assistant"][-1]
        break
ctx_ids = np.array(tok.encode(usr, add_special_tokens=False), dtype=np.int64)     # 9
resp_ids = np.array(tok.encode(asst, add_special_tokens=False), dtype=np.int64)   # 84
S = len(ctx_ids) + len(resp_ids)   # 93
NCTX = len(ctx_ids)

D, H, KVH, HD, FF = 2560, 32, 8, 128, 9728
NQ, NKV, R, alpha_r, EPS = H * HD, KVH * HD, 2, 2.0, 1e-6
l0spec = importlib.util.spec_from_file_location("l0", os.path.join(repo, "native/gguf/layer0_ref.py"))
l0 = importlib.util.module_from_spec(l0spec); l0spec.loader.exec_module(l0)
tens = l0.parse_tsv()
WK = {"q": "attn_q", "k": "attn_k", "v": "attn_v", "o": "attn_output", "g": "ffn_gate", "u": "ffn_up", "d": "ffn_down"}
proj_shapes = {"q": (D, NQ), "k": (D, NKV), "v": (D, NKV), "o": (NQ, D), "g": (D, FF), "u": (D, FF), "d": (FF, D)}
ALL_W = [{nn: l0.read_weight(tens[f"blk.{L}.{nn}.weight"], tuple(tens[f"blk.{L}.{nn}.weight"]["shape"]))
          for nn in ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]} for L in range(36)]
ALL_LN = [{n: l0.read_weight(tens[f"blk.{L}.{n}.weight"], tuple(tens[f"blk.{L}.{n}.weight"]["shape"]))
           for n in ["attn_norm", "attn_q_norm", "attn_k_norm", "ffn_norm"]} for L in range(36)]
def rms(v, w):
    return v * (1.0 / np.sqrt((v ** 2).mean(-1, keepdims=True) + EPS)) * w
def load_Lo(L):
    return {nm: (np.fromfile(os.path.join(out, f"m2e_l{L}_U{nm}.bin"), "<f8").reshape(proj_shapes[nm][0], R),
                 np.fromfile(os.path.join(out, f"m2e_l{L}_V{nm}.bin"), "<f8").reshape(R, proj_shapes[nm][1])) for nm in proj_shapes}
Lo = {L: load_Lo(L) for L in range(36)}
def lproj(a, nm, L):
    U, V = Lo[L][nm]; return a @ ALL_W[L][WK[nm]] + alpha_r * ((a @ U) @ V)
te = tens["token_embd.weight"]
with open("/home/rick/Models/qwen3/Qwen3-4B-Q4_K_M.gguf", "rb") as fh:
    fh.seek(te["off"]); emb = l0.dequant_q6k(fh.read(te["numel"] * 210 // 256)).reshape(te["shape"][1], D)
INVF = np.fromfile(os.path.join(out, "layer0_invfreq.bin"), "<f8")
seq = np.arange(S)                       # 0-based positions 0..92
x = emb[np.concatenate([ctx_ids, resp_ids])].astype("<f8")
saved = {}
for L in range(36):
    xn = rms(x, ALL_LN[L]["attn_norm"])
    q = lproj(xn, "q", L); k = lproj(xn, "k", L); v = lproj(xn, "v", L)
    qn = rms(q.reshape(S, H, HD), ALL_LN[L]["attn_q_norm"])
    kn = rms(k.reshape(S, KVH, HD), ALL_LN[L]["attn_k_norm"])
    qr, kr = l0.rope(qn, kn, seq)
    ctx = l0.gqa(qr, kr, v.reshape(S, KVH, HD))
    if L == 0:
        qr[NCTX:S].reshape(-1).astype("<f8").tofile(os.path.join(out, "kvresp_L0_q_ref.bin"))
        ctx[NCTX:S].reshape(-1).astype("<f8").tofile(os.path.join(out, "kvresp_L0_ctx_ref.bin"))
        xn[NCTX:S].reshape(-1).astype("<f8").tofile(os.path.join(out, "kvresp_L0_xn_ref.bin"))
        qn[NCTX:S].reshape(-1).astype("<f8").tofile(os.path.join(out, "kvresp_L0_dqn_ref.bin"))
    o = lproj(ctx.reshape(S, NQ), "o", L)
    x1 = x + o
    x1n = rms(x1, ALL_LN[L]["ffn_norm"])
    g = lproj(x1n, "g", L); up = lproj(x1n, "u", L)
    m = lproj((g / (1 + np.exp(-g))) * up, "d", L)
    x = x1 + m
    if L == 0: saved["l0"] = x[NCTX:S].copy()
    if L in (0, 8, 17, 26): saved[f"l{L+1}"] = x[NCTX:S].copy()
# response hidden = layer-36 output rows NCTX..S-1
resp_hidden = x[NCTX:S].astype("<f8")     # [84, D]
resp_hidden.tofile(os.path.join(out, "kvresp_hidden_ref.bin"))
np.savetxt(os.path.join(out, "kvresp_hidden_ref_head.txt"), resp_hidden.reshape(-1)[:256], fmt="%.9g")
np.asarray(emb[resp_ids[:3]], dtype=np.float64).reshape(-1).tofile(os.path.join(out, "kvresp_emb_ref.bin"))
np.asarray(ALL_LN[0]["attn_norm"], dtype=np.float64).reshape(-1).tofile(os.path.join(out, "kvresp_L0_n1_ref.bin"))
for k in ["l1", "l9", "l18", "l27"]:
    if k in saved:
        saved[k].astype("<f8").tofile(os.path.join(out, f"kvresp_L{k}_ref.bin"))
print("kvresp_ref done S=", S, "NCTX=", NCTX, "resp_hidden", resp_hidden.shape, "norm=", float(np.linalg.norm(resp_hidden.reshape(-1)) ** 2))
