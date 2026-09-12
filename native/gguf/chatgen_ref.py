#!/usr/bin/env python3
"""Checkpoint C gold: greedy autoregressive Qwen3-4B decode over a REAL prompt.

Generalizes decode_ref.py (hardcoded SEED=[0,1]) to an arbitrary prompt: reads
native/out/prompt_ids.txt (ids from llm_encode_probe / stdlib vllm tokenizer),
embeds those token rows, then runs GEN greedy recompute-full-prefix steps (the
CORRECT approach — re-embed the WHOLE current sequence each step, not carry a
stale hidden). Writes the full generated id sequence to
native/out/chatgen_ids_ref.txt — the GPU autoregressive chat driver must match it
token-for-token. The last N ids (after the prompt) are the fluent response.
"""
import os, importlib.util, numpy as np

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = "/home/rick/Models/qwen3/Qwen3-4B-Q4_K_M.gguf"
TSV = os.path.join(repo, "native/out/qwen3_4b_tensors.tsv")

spec = importlib.util.spec_from_file_location("l0", os.path.join(repo, "native/gguf/layer0_ref.py"))
l0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l0)

D, H, KVH, HD, FF = 2560, 32, 8, 128, 9728
GEN = 3   # response tokens to generate after the prompt


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
    print(f"prompt ids = {ids}  GEN = {GEN} (full seq len {len(ids)+GEN})", flush=True)
    tens = l0.parse_tsv()

    # pre-load all 36 layers' weights once (RAM permitting ~ 36*~1.5GB?? keep per-layer)
    # Decode_ref reloads per-layer per-step; keep simple: load on demand each step.

    def emb_row(tid):
        te = tens["token_embd.weight"]
        nblk = D // 256
        with open(MODEL, "rb") as fh:
            fh.seek(te["off"] + tid * nblk * 210)
            return l0.dequant_q6k(fh.read(nblk * 210)).reshape(-1)

    def logits_for(hidden):
        on_shape = tuple(tens["output_norm.weight"]["shape"])
        h = l0.rmsnorm(hidden, l0.read_weight(tens["output_norm.weight"], on_shape))
        te = tens["token_embd.weight"]
        with open(MODEL, "rb") as fh:
            fh.seek(te["off"])
            emb = l0.dequant_q6k(fh.read(te["numel"] * 210 // 256)).reshape(te["shape"][1], D)
        return h @ emb.T

    def layer_weights(layer):
        W = {}
        for n in ["attn_k", "attn_norm", "attn_output", "attn_q", "attn_v",
                  "ffn_down", "ffn_gate", "ffn_norm", "ffn_up",
                  "attn_q_norm", "attn_k_norm"]:
            pre = f"blk.{layer}.{n}.weight"
            shape = tuple(tens[pre]["shape"])
            W[n] = l0.read_weight(tens[pre], shape)
        return W

    out_ids = list(ids)
    for step in range(GEN):
        x = np.stack([emb_row(t) for t in out_ids])
        for layer in range(36):
            x = forward_layer(x, layer_weights(layer))
        lg = logits_for(x)
        nxt = int(lg[-1].argmax())
        out_ids.append(nxt)
        print(f"  step {step}: generated token id {nxt}", flush=True)

    np.savetxt(os.path.join(repo, "native/out/chatgen_ids_ref.txt"), np.array(out_ids), fmt="%d")
    print("CHATGEN_IDS_REF:", out_ids)
    print(f"response tokens (after prompt) = {out_ids[len(ids):]}")
    print("saved native/out/chatgen_ids_ref.txt")


if __name__ == "__main__":
    import time
    t0 = time.time()
    main()
    print(f"elapsed {time.time()-t0:.1f}s")
