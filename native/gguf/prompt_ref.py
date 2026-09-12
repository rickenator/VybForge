#!/usr/bin/env python3
"""Full-36-layer Qwen3-4B prefill reference over an ARBITRARY prompt (verify-only).

Generalizes prefill_ref.py (which is hardcoded to ids [0,1]) to embed an
arbitrary token-id sequence from the real token_embd and run the full forward,
producing the per-position lm_head top-1. The GPU model_driver (VYB_PROMPT path)
must match these top-1 ids token-for-token — the Checkpoint-B gate for
arbitrary-prompt llm.chat.

Prompts: ids come from the stdlib/vllm CPU tokenizer (llm_encode_probe.vyb).
Reads native/out/prompt_ids.txt (one comma-separated line).
Writes:
  native/out/prompt_top1_ref.txt -- per-position argmax logit id
  native/out/prompt_hidden_ref.txt -- final layer-36 hidden [S,D] flat
  native/out/prompt_hidden.bin   -- final hidden f64 LE
"""
import os, importlib.util, numpy as np

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = "/home/rick/Models/qwen3/Qwen3-4B-Q4_K_M.gguf"
TSV = os.path.join(repo, "native/out/qwen3_4b_tensors.tsv")

spec = importlib.util.spec_from_file_location("l0", os.path.join(repo, "native/gguf/layer0_ref.py"))
l0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l0)

D, H, KVH, HD, FF = 2560, 32, 8, 128, 9728


def forward_layer(x, W):
    xn = l0.rmsnorm(x, W["attn_norm"])
    S = x.shape[0]
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


def main():
    ids = [int(x) for x in open(os.path.join(repo, "native/out/prompt_ids.txt")).read().strip().split(",")]
    S = len(ids)
    print(f"prompt ids = {ids}, S = {S}", flush=True)

    tens = l0.parse_tsv()
    te = tens["token_embd.weight"]
    nblk_emb = D // 256
    x = np.empty((S, D), np.float64)
    for s, tid in enumerate(ids):
        with open(MODEL, "rb") as fh:
            fh.seek(te["off"] + tid * nblk_emb * 210)
            x[s] = l0.dequant_q6k(fh.read(nblk_emb * 210)).reshape(-1)

    for layer in range(36):
        pre = f"blk.{layer}."
        W = {}
        for n in ["attn_k", "attn_norm", "attn_output", "attn_q", "attn_v",
                  "ffn_down", "ffn_gate", "ffn_norm", "ffn_up",
                  "attn_q_norm", "attn_k_norm"]:
            shape = tuple(tens[pre + n + ".weight"]["shape"])
            W[n] = l0.read_weight(tens[pre + n + ".weight"], shape)
        x = forward_layer(x, W)
        if layer in (0, 35):
            print(f"  layer {layer}: hidden range [{x.min():.6g},{x.max():.6g}]", flush=True)

    on_shape = tuple(tens["output_norm.weight"]["shape"])
    h = l0.rmsnorm(x, l0.read_weight(tens["output_norm.weight"], on_shape))
    print("dequantizing token_embd (tied lm_head, Q6_K)...", flush=True)
    te_nb = te["numel"] * 210 // 256
    with open(MODEL, "rb") as fh:
        fh.seek(te["off"])
        emb = l0.dequant_q6k(fh.read(te_nb))
    emb = emb.reshape(te["shape"][1], D)
    logits = h @ emb.T
    top1 = logits.argmax(axis=1)
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    topk = logits.argsort(axis=-1)[:, ::-1][:, :5]

    np.savetxt(os.path.join(repo, "native/out/prompt_top1_ref.txt"), top1, fmt="%d")
    np.savetxt(os.path.join(repo, "native/out/prompt_hidden_ref.txt"), x.reshape(-1), fmt="%.17g")
    x.tofile(os.path.join(repo, "native/out/prompt_hidden.bin"))
    print(f"final hidden range [{x.min():.6g},{x.max():.6g}]")
    print("per-position top1:", top1.tolist())
    print("per-position top5:")
    for s in range(S):
        print(f"  tok{s}: {topk[s].tolist()}  (top1={top1[s]}, p={probs[s, top1[s]]:.6f})")


if __name__ == "__main__":
    import time
    t0 = time.time()
    main()
    print(f"elapsed {time.time()-t0:.1f}s")
