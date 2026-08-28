#!/usr/bin/env python3
"""numpy greedy autoregressive decode of real Qwen3-4B (verification-only).

Same per-layer machinery as prefill_ref.py, but loops generation: after each
full-36-layer forward on the current token sequence x[S,D], takes the tied-lm_head
argmax as the next token id, appends that token's token_embd row, and repeats.
Saves the generated token-id sequence (native/out/decode_ids_ref.txt) — the GPU
decode driver must reproduce it exactly.

Cost is ~90s+ per step (re-dequants all 36 layers per step); keep steps small.
"""
import os, importlib.util, numpy as np

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = "/home/rick/Models/qwen3/Qwen3-4B-Q4_K_M.gguf"
spec = importlib.util.spec_from_file_location("l0", os.path.join(repo, "native/gguf/layer0_ref.py"))
l0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l0)
D, H, KVH, HD, FF = 2560, 32, 8, 128, 9728
tens = l0.parse_tsv()

SEED = [0, 1]          # seed token ids (token 0,1 = same as prefill)
GEN = 6                # tokens to generate

def forward_layer(x, W):
    xn = l0.rmsnorm(x, W["attn_norm"])
    q = (xn @ W["attn_q"]).reshape(x.shape[0], H, HD)
    k = (xn @ W["attn_k"]).reshape(x.shape[0], KVH, HD)
    v = (xn @ W["attn_v"]).reshape(x.shape[0], KVH, HD)
    q = l0.rmsnorm(q, W["attn_q_norm"])
    k = l0.rmsnorm(k, W["attn_k_norm"])
    pos = np.arange(x.shape[0])
    q, k = l0.rope(q, k, pos)
    o = l0.gqa(q, k, v)
    layer = x + (o @ W["attn_output"])
    h = l0.rmsnorm(layer, W["ffn_norm"])
    gate = h @ W["ffn_gate"]
    up = h @ W["ffn_up"]
    s = gate / (1.0 + np.exp(-gate))
    out = layer + ((s * up) @ W["ffn_down"])
    return out

def layer_weights(layer):
    pre = f"blk.{layer}."
    W = {}
    for n in ["attn_k", "attn_norm", "attn_output", "attn_q", "attn_v",
              "ffn_down", "ffn_gate", "ffn_norm", "ffn_up", "attn_q_norm", "attn_k_norm"]:
        shape = tuple(tens[pre + n + ".weight"]["shape"])
        W[n] = l0.read_weight(tens[pre + n + ".weight"], shape)
    return W

def emb_row(token_id):
    te = tens["token_embd.weight"]
    nblk = D // 256
    with open(MODEL, "rb") as fh:
        fh.seek(te["off"] + token_id * nblk * 210)
        return l0.dequant_q6k(fh.read(nblk * 210)).reshape(-1)

def logits_for(hidden):
    on_shape = tuple(tens["output_norm.weight"]["shape"])
    h = l0.rmsnorm(hidden, l0.read_weight(tens["output_norm.weight"], on_shape))
    te = tens["token_embd.weight"]
    with open(MODEL, "rb") as fh:
        fh.seek(te["off"])
        emb = l0.dequant_q6k(fh.read(te["numel"] * 210 // 256)).reshape(te["shape"][1], D)
    return h @ emb.T   # [S, V]

cache = None   # (layer -> W) to avoid re-dequant node each step if RAM allows; keep simple
ids = list(SEED)
for step in range(GEN):
    # CORRECT greedy recompute-full-prefix: re-embed the WHOLE current token
    # sequence each step (not carry the previous hidden forward — that was a bug
    # that fed stale hidden tensors back through the stack and invalidated the ref).
    x = np.stack([emb_row(t) for t in ids])
    for layer in range(36):
        x = forward_layer(x, layer_weights(layer))
    lg = logits_for(x)
    if step == 0:
        on_shape = tuple(tens["output_norm.weight"]["shape"])
        hn = l0.rmsnorm(x[-1], l0.read_weight(tens["output_norm.weight"], on_shape))
        print("REF_DIAG DHrow0=" + " ".join(f"{v:.6g}" for v in hn[:8]) + " | logits0=" + " ".join(f"{v:.6g}" for v in lg[-1][:8]), flush=True)
    nxt = int(lg[-1].argmax())              # next token (from last position)
    ids.append(nxt)
    print(f"  step {step}: generated token id {nxt}", flush=True)

np.savetxt(os.path.join(repo, "native/out/decode_ids_ref.txt"), np.array(ids), fmt="%d")
print("DECODE_IDS_REF:", ids)
print("saved native/out/decode_ids_ref.txt")
