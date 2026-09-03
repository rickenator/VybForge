#!/usr/bin/env python3
"""Full-36-layer Qwen3-4B prefill reference (verification-only) for S=2.

Reuses layer0_ref.py's vectorized dequant + pinned layer math, looping blk.0..35
so only ONE layer's weights are resident at a time. Produces the numeric contract
the GPU model driver must match:
  native/out/prefill_hidden_ref.txt  -- final layer-36 hidden [S,D] flat
  native/out/prefill_top1_ref.txt    -- per-token argmax logit id (lm_head tied to token_embd)
  native/out/prefill_hidden.bin      -- final hidden f64 LE (GPU input would reuse layer0_input.bin)
"""
import os, importlib.util, numpy as np

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = "/home/rick/Models/qwen3/Qwen3-4B-Q4_K_M.gguf"
TSV = os.path.join(repo, "native/out/qwen3_4b_tensors.tsv")

spec = importlib.util.spec_from_file_location("l0", os.path.join(repo, "native/gguf/layer0_ref.py"))
l0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l0)

D, H, KVH, HD, FF = 2560, 32, 8, 128, 9728
S = 2


def forward_layer(x, W, LN):
    """One layer-0..35 forward on x[S,D]. W = layer weight dict, LN = norm dict
    (separate to keep indexing simple: attn_norm etc are per-layer too)."""
    xn = l0.rmsnorm(x, W["attn_norm"])
    q = (xn @ W["attn_q"]).reshape(S, H, HD)
    k = (xn @ W["attn_k"]).reshape(S, KVH, HD)
    v = (xn @ W["attn_v"]).reshape(S, KVH, HD)
    q = l0.rmsnorm(q, W["attn_q_norm"])
    k = l0.rmsnorm(k, W["attn_k_norm"])
    pos = np.arange(S)
    q, k = l0.rope(q, k, pos)
    o = l0.gqa(q, k, v)
    layer = x + (o @ W["attn_output"])
    h = l0.rmsnorm(layer, W["ffn_norm"])
    gate = h @ W["ffn_gate"]
    up = h @ W["ffn_up"]
    silu = gate / (1.0 + np.exp(-gate))
    out = layer + ((silu * up) @ W["ffn_down"])
    return out


def layer_weight(tens, name, shape):
    return l0.read_weight(tens[name], shape)


def main():
    tens = l0.parse_tsv()
    # input = token_embd rows 0,1 (same as layer-0 contract)
    te = tens["token_embd.weight"]
    nblk_emb = D // 256
    x = np.empty((S, D), np.float64)
    for s in range(S):
        with open(MODEL, "rb") as fh:
            fh.seek(te["off"] + s * nblk_emb * 210)
            x[s] = l0.dequant_q6k(fh.read(nblk_emb * 210)).reshape(-1)

    for layer in range(36):
        pre = f"blk.{layer}."
        W = {}
        for n in ["attn_k", "attn_norm", "attn_output", "attn_q", "attn_v",
                  "ffn_down", "ffn_gate", "ffn_norm", "ffn_up",
                  "attn_q_norm", "attn_k_norm"]:
            shape = tuple(tens[pre + n + ".weight"]["shape"])
            W[n] = layer_weight(tens, pre + n + ".weight", shape)
        x = forward_layer(x, W, None)
        if layer in (0, 1, 35):
            print(f"  layer {layer}: hidden range [{x.min():.6g},{x.max():.6g}]", flush=True)
        if layer == 1:
            np.savetxt(os.path.join(repo, "native/out/prefill_hidden_l2_ref.txt"), x.reshape(-1), fmt="%.17g")

    # final: output_norm then logits = h @ token_embd.T (tied lm_head)
    on_shape = tuple(tens["output_norm.weight"]["shape"])
    h = l0.rmsnorm(x, l0.read_weight(tens["output_norm.weight"], on_shape))
    print("dequantizing token_embd (tied lm_head, Q6_K)...", flush=True)
    te_nb = te["numel"] * 210 // 256
    with open(MODEL, "rb") as fh:
        fh.seek(te["off"])
        emb = l0.dequant_q6k(fh.read(te_nb))
    emb = emb.reshape(te["shape"][1], D)   # [vocab, D]
    logits = h @ emb.T                      # [S, vocab]
    top1 = logits.argmax(axis=1)
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    topk = logits.argsort(axis=-1)[:, ::-1][:, :5]

    np.savetxt(os.path.join(repo, "native/out/prefill_hidden_ref.txt"), x.reshape(-1), fmt="%.17g")
    x.tofile(os.path.join(repo, "native/out/prefill_hidden.bin"))
    np.savetxt(os.path.join(repo, "native/out/prefill_top1_ref.txt"), top1, fmt="%d")
    np.savetxt(os.path.join(repo, "native/out/prefill_top5_ref.txt"), topk.reshape(-1), fmt="%d")
    print(f"final hidden range [{x.min():.6g},{x.max():.6g}]")
    print("per-token top5 ids:")
    for s in range(S):
        print(f"  tok{s}: {topk[s].tolist()}  (top1={top1[s]}, p={probs[s, top1[s]]:.6f})")


if __name__ == "__main__":
    import time
    t0 = time.time()
    main()
    print(f"elapsed {time.time()-t0:.1f}s")
