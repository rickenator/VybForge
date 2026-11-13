"""P3/P4 gate: DSA_INDEXER_VERIFY + DSA_ATTN_VERIFY — the checkpoint's real DSA
indexer + masked MLA attention at S>1, numpy (dsa_ref) vs HuggingFace glm5_next.

Why this exists: `verify_mla.py`'s DSA_SELECT_VERIFY gates `mla_ref.indexer_select`
against exllamav3's `QSAIndexer.select_indices_ref` with *synthetic* q/pooled arrays.
That reference models a rope q/k + mean-pooled-blocks indexer whose tensors do not
exist in this checkpoint (layer 3-44 carry `indexer.wq_b`, `indexer.wk`,
`indexer.k_norm.{weight,bias}`, `indexer.weights_proj`,
`indexer.index_kpool_compress_{ape,gate}` — the learned-gate pooling design). So
selection *geometry* was gated and the *weights path* was not; S=1 gates can not see
the difference (every token is visible, any pooling rule selects the same tokens).

This gate runs a real DSA layer (default 3) on a real token sequence: layers 0-2 are
chained in numpy to produce the layer-3 input streams, HF then runs the layer alone
with layer-3's tensors and the numpy port reproduces its indexer output, its
attention output, and every projection on the way.

Usage: env -u PYTHONPATH ~/venvs/exl3/bin/python native/glm53/verify_dsa.py [layer]
"""

import json
import sys

import numpy as np
import torch
from safetensors import safe_open
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

sys.path.insert(0, "/home/rick/Projects/VybForge/native/glm53")
import verify_chain3 as V  # noqa: E402
import dsa_ref  # noqa: E402

H = 4096
CKPT = V.CKPT
IDX = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
PROMPT = ("The capital of France is Paris, and the largest city in Germany is Berlin, "
          "which was divided for decades.")   # ~25 real tokens: 6 complete pools + tail


def read(name, a=None, b=None):
    with safe_open(f"{CKPT}/{IDX[name]}", framework="pt", device="cpu") as f:
        if a is None:
            t = f.get_tensor(name)
        else:
            t = f.get_slice(name)[a:b]
    return t.float().numpy() if t.dtype == torch.bfloat16 else t.numpy()


def read_layer3_non_expert(layer):
    pre = f"model.language_model.layers.{layer}."
    out = {}
    for k in IDX:
        if not k.startswith(pre):
            continue
        short = k[len(pre):]
        if ".experts." in short:
            continue
        out[short] = read(k).astype(np.float64)
    return out


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


def main(layer=3):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{CKPT}/tokenizer.json")
    ids = tok.encode(PROMPT).ids
    s = len(ids)
    print(f"prompt {PROMPT!r}: {s} tokens {ids}")

    x = np.stack([read("model.language_model.embed_tokens.weight", i, i + 1)[0].astype(np.float64)
                  for i in ids])

    # ---- chain the KDA layers before `layer` in numpy (CHAIN3_VERIFY-proven math) ----
    streams = np.repeat(x[:, None, :], 4, axis=1)
    for i in range(layer):
        w = V.to_gdn_weights(V.read_layer(i))
        streams, _, _ = V.numpy_layer_from_streams(streams, w)
    print(f"chained layers 0-{layer - 1}: streams {streams.shape}")

    t = read_layer3_non_expert(layer)
    pre = f"model.language_model.layers.{layer}."
    ipre = "self_attn.indexer."
    idx_w = {"wq_b": t[ipre + "wq_b.weight"], "wk": t[ipre + "wk.weight"],
             "k_norm.weight": t[ipre + "k_norm.weight"], "k_norm.bias": t[ipre + "k_norm.bias"],
             "weights_proj": t[ipre + "weights_proj.weight"],
             "compress_ape": t[ipre + "index_kpool_compress_ape"],
             "compress_gate": t[ipre + "index_kpool_compress_gate"]}
    # attention projections by base name, HF (out,in) as stored
    w = {k[len("self_attn."):-len(".weight")]: v for k, v in t.items()
         if k.startswith("self_attn.") and k.endswith(".weight") and ".indexer." not in k}

    # ---- numpy side: hc mix -> input_layernorm -> DSA attention ----
    post, comb, collapsed, _ = dsa_ref_hc_mix(streams, t)
    h_a = dsa_ref.rms_norm(collapsed, t["input_layernorm.weight"])
    y_attn, topk_mine, inter = dsa_ref.layer_attention(h_a, w, idx_w)

    # ---- HF side (bf16), attention-only: the MLP is built dense so 14 GB of expert
    #      tensors are never allocated (the MLP output is not compared here) ----
    cfg = Glm5NextConfig.from_dict(json.load(open(V.CFG)))
    tc = cfg.text_config
    tc.num_hidden_layers = 1
    tc.layer_types = ["deepseek_sparse_attention"]
    tc.mlp_layer_types = ["dense"]
    tc.indexer_types = ["full"]
    if hasattr(tc, "num_nextn_predict_layers"):
        tc.num_nextn_predict_layers = 0
    lac = tc.linear_attn_config
    if isinstance(lac, dict):
        lac["kda_layers"], lac["full_attn_layers"] = [], [0]
    else:
        lac.kda_layers, lac.full_attn_layers = [], [0]
    torch.set_default_dtype(torch.bfloat16)
    model = Glm5NextTextModel(tc)
    model.eval()

    alias = {"attn_hc.fn": "hc_attn_fn", "attn_hc.base": "hc_attn_base", "attn_hc.scale": "hc_attn_scale",
             "ffn_hc.fn": "hc_ffn_fn", "ffn_hc.base": "hc_ffn_base", "ffn_hc.scale": "hc_ffn_scale"}
    sd = {}
    for name, p in {**dict(model.named_parameters()), **dict(model.named_buffers())}.items():
        short = alias.get(name.replace("layers.0.", ""), name.replace("layers.0.", ""))
        a = t.get(short)
        if a is None:
            continue
        src = a if a.shape == tuple(p.shape) else a.T
        if src.shape != tuple(p.shape):
            continue
        sd[name] = torch.from_numpy(np.ascontiguousarray(src.astype(np.float32))).to(torch.bfloat16)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"HF load: {len(sd)} tensors; unexpected {list(unexpected)}; "
          f"missing {[m for m in missing if 'mlp' not in m]}")

    hf_layer = model.layers[0]
    g = {}

    def hook(tag):
        def fn(mm, inp, out):
            tt = out[0] if isinstance(out, (tuple, list)) else out
            g[tag] = tt.detach().float().cpu().numpy()
        return fn

    hf_layer.input_layernorm.register_forward_hook(hook("inln"))
    hf_layer.self_attn.q_a_proj.register_forward_hook(hook("qap"))
    hf_layer.self_attn.q_b_proj.register_forward_hook(hook("qb"))
    hf_layer.self_attn.kv_a_layernorm.register_forward_hook(hook("kva_ln"))
    hf_layer.self_attn.indexer.register_forward_hook(hook("topk"))
    hf_layer.self_attn.register_forward_hook(hook("attn"))
    with torch.no_grad():
        hf_layer(hidden_states=torch.from_numpy(streams[None].astype(np.float32)).to(torch.bfloat16),
                 attention_mask=torch.ones(1, s, dtype=torch.bool),
                 position_ids=torch.arange(s)[None], use_cache=False)

    # ---- compare ----
    print(f"input_layernorm:  rel {rel(h_a, g['inln'][0]):.3e}")
    print(f"q_a latent:       rel {rel(inter['q_resid'], g['qap'][0]):.3e}")
    print(f"q_b:              rel {rel(inter['qb'], g['qb'][0]):.3e}" if "qb" in inter
          else "q_b:              (not captured)")
    print(f"kv_a_layernorm:   rel {rel(inter['ckv'], g['kva_ln'][0]):.3e}")

    hf_topk = np.asarray(g["topk"], dtype=np.int64)[0]
    print(f"topk shapes: mine {topk_mine.shape} hf {hf_topk.shape}")
    set_ok = True
    for q in range(s):
        a = set(int(v) for v in topk_mine[q] if v >= 0)
        b = set(int(v) for v in hf_topk[q] if v >= 0)
        same = a == b
        set_ok &= same
        print(f"  q{q}: sel {sorted(a)} vs selected {sorted(b)}  set-agree={same}")
    order_same = bool(np.array_equal(topk_mine, hf_topk[:, :topk_mine.shape[1]]))
    n_diff = int((topk_mine != hf_topk[:, :topk_mine.shape[1]]).sum())
    print(f"topk exact (flag order incl.): {order_same}  ({n_diff} differing entries of {topk_mine.size})")
    if not order_same:
        bad = [q for q in range(s) if not np.array_equal(topk_mine[q], hf_topk[q, :topk_mine.shape[1]])]
        for q in bad[:3]:
            print(f"  order q{q}:")
            print(f"    mine {[(c, int(v)) for c, v in enumerate(topk_mine[q]) if v >= 0]}")
            print(f"    hf   {[(c, int(v)) for c, v in enumerate(hf_topk[q]) if v >= 0]}")

    hf_attn = np.asarray(g["attn"], dtype=np.float64)[0]
    d_attn = rel(y_attn, hf_attn)
    print(f"attention out: |hf| {np.abs(hf_attn).max():.4f}  rel {d_attn:.3e}")

    ok = set_ok and d_attn < 1e-2
    print("DSA_INDEXER_VERIFY:", "OK" if set_ok else "FAIL")
    print("DSA_ATTN_VERIFY:", "OK" if d_attn < 1e-2 else "FAIL")
    return 0 if ok else 1


def dsa_ref_hc_mix(streams, t):
    """hc mix on the checkpoint's hc_attn tensors (hc_ref owns the math)."""
    import hc_ref
    post, comb, collapsed, stream_out = hc_ref.mix(
        streams, t["hc_attn_fn"], t["hc_attn_base"], t["hc_attn_scale"], 1e-5)
    return post, comb, collapsed, stream_out


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 3))
