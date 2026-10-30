"""Gold for the layer-3 DSA attention: layers 0-2 chained in numpy (KDA), then layer 3's
attention site (hc mix -> input_layernorm -> DSA attention -> o_proj) on real weights.

The layer-3 attention input `h_a` is written to native/out/gold_l3_ha.txt as `(v*2^40) as
Int` integers (Vyb cannot reinterpret bytes as f64), so the Vyb driver can run the DSA
path on the identical input while the layer chaining itself is still numpy-side.

Writes:
  native/out/gold_l3.json         digests (attn, h_a, and per-stage)
  native/out/gold_l3_ha.txt       h_a [22, 4096] as scaled integers
  native/out/gold_l3_attn.npy     gold attention output [22, 4096] (fp64)
  native/out/gold_l3_chain.npy    streams after layer 2 (for the chaining gate later)

Usage: env -u PYTHONPATH ~/venvs/exl3/bin/python native/glm53/gold_l3_attn.py
"""

import json
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, "/home/rick/Projects/VybForge/native/glm53")
import hc_ref  # noqa: E402
import layer0_ref  # noqa: E402
import gdn_ref  # noqa: E402
import dsa_ref  # noqa: E402

CKPT = "/mnt/apex/rick/vybforge/glm53-checkpoint"
OUT = "native/out"
H = 4096
HEADS, HDIM = 64, 128
KDIM = HDIM
INTER = 12288
EPS = 1e-5
S40 = 1099511627776.0
PROMPT = ("The capital of France is Paris, and the largest city in Germany is Berlin, "
          "which was divided for decades.")


def read(name):
    idx = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
    with safe_open(f"{CKPT}/{idx[name]}", framework="pt", device="cpu") as f:
        t = f.get_tensor(name)
    return t.float().numpy().astype(np.float64)


def weights(layer, kda=True):
    pre = f"model.language_model.layers.{layer}."
    T = lambda n: read(pre + n).T  # noqa: E731
    w = {}
    if kda:
        w["q"] = T("self_attn.q_proj.weight")
        w["k"] = T("self_attn.k_proj.weight")
        w["v"] = T("self_attn.v_proj.weight")
        w["o"] = T("self_attn.o_proj.weight")
        w["conv_q"] = read(pre + "self_attn.q_conv1d.weight")[:, 0, :]
        w["conv_k"] = read(pre + "self_attn.k_conv1d.weight")[:, 0, :]
        w["conv_v"] = read(pre + "self_attn.v_conv1d.weight")[:, 0, :]
        w["b"] = T("self_attn.b_proj.weight")
        w["f_a"] = T("self_attn.f_a_proj.weight")
        w["f_b"] = T("self_attn.f_b_proj.weight")
        w["g_a"] = T("self_attn.g_a_proj.weight")
        w["g_b"] = T("self_attn.g_b_proj.weight")
        w["dt_bias"] = read(pre + "self_attn.dt_bias")
        w["A_log"] = read(pre + "self_attn.A_log")
        w["o_norm"] = read(pre + "self_attn.o_norm.weight")
        w["mlp_gate"] = T("mlp.gate_proj.weight")
        w["mlp_up"] = T("mlp.up_proj.weight")
        w["mlp_down"] = T("mlp.down_proj.weight")
    w["in_ln"] = read(pre + "input_layernorm.weight")
    w["post_ln"] = read(pre + "post_attention_layernorm.weight")
    for site, key in (("attn", "hc_attn"), ("ffn", "hc_ffn")):
        w[f"{site}_fn"] = read(pre + f"{key}_fn")
        w[f"{site}_base"] = read(pre + f"{key}_base")
        w[f"{site}_scale"] = read(pre + f"{key}_scale")
    return w


def kda_layer(streams, w):
    post, comb, collapsed, _ = hc_ref.mix(streams, w["attn_fn"], w["attn_base"], w["attn_scale"], EPS)
    h = layer0_ref.rms_weight(collapsed, w["in_ln"])
    y = layer0_ref.kda(h, w)
    streams1 = hc_ref.apply_(streams, y, post, comb)
    post2, comb2, collapsed2, _ = hc_ref.mix(streams1, w["ffn_fn"], w["ffn_base"], w["ffn_scale"], EPS)
    h2 = layer0_ref.rms_weight(collapsed2, w["post_ln"])
    y2 = layer0_ref.mlp_dense(h2, w)
    return hc_ref.apply_(streams1, y2, post2, comb2), y, y2


def dig(v):
    f = np.ascontiguousarray(v, dtype=np.float64).ravel()
    out = {"n": len(f), "sum": int(f.sum() * S40), "abs": int(np.abs(f).sum() * S40),
           "sq": int((f * f).sum() * S40), "max": int(np.abs(f).max() * S40)}
    for k in range(8):
        out[f"v{k}"] = int(f[k] * S40) if k < len(f) else 0
    return out


def dsa_weights(layer):
    pre = f"model.language_model.layers.{layer}."
    ipre = pre + "self_attn.indexer."
    # dsa_ref takes the checkpoint's own (out,in) layout for every projection
    R_ = lambda n: read(pre + n)  # noqa: E731
    w = {"q_a_proj": R_("self_attn.q_a_proj.weight"),
         "q_a_layernorm": read(pre + "self_attn.q_a_layernorm.weight"),
         "q_b_proj": R_("self_attn.q_b_proj.weight"),
         "kv_a_proj_with_mqa": R_("self_attn.kv_a_proj_with_mqa.weight"),
         "kv_a_layernorm": read(pre + "self_attn.kv_a_layernorm.weight"),
         "kv_b_proj": read(pre + "self_attn.kv_b_proj.weight"),
         "o_proj": R_("self_attn.o_proj.weight")}
    iw = {"wq_b": read(ipre + "wq_b.weight"),
          "wk": read(ipre + "wk.weight"),
          "k_norm.weight": read(ipre + "k_norm.weight"),
          "k_norm.bias": read(ipre + "k_norm.bias"),
          "weights_proj": read(ipre + "weights_proj.weight"),
          "compress_ape": read(ipre + "index_kpool_compress_ape"),
          "compress_gate": read(ipre + "index_kpool_compress_gate")}
    return w, iw


def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{CKPT}/tokenizer.json")
    ids = tok.encode(PROMPT).ids
    s = len(ids)
    emb = {}
    x = np.stack([read_row(i) for i in ids])
    streams = np.repeat(x[:, None, :], 4, axis=1)
    for lay in (0, 1, 2):
        w = weights(lay, kda=True)
        streams, _, _ = kda_layer(streams, w)
    np.save(f"{OUT}/gold_l3_chain.npy", streams)
    print(f"chained layers 0-2: streams {streams.shape}")

    # layer 3 attention site
    w3 = weights(3, kda=False)
    post, comb, collapsed, _ = hc_ref.mix(streams, w3["attn_fn"], w3["attn_base"], w3["attn_scale"], EPS)
    h_a = layer0_ref.rms_weight(collapsed, w3["in_ln"])
    wa, iw = dsa_weights(3)
    y_attn, topk, inter = dsa_ref.layer_attention(h_a, wa, iw)
    streams1 = hc_ref.apply_(streams, y_attn, post, comb)

    np.save(f"{OUT}/gold_l3_attn.npy", y_attn)
    np.save(f"{OUT}/gold_l3_o_pre.npy", inter["o_pre"])
    np.save(f"{OUT}/gold_l3_ckv.npy", inter["ckv"])
    np.save(f"{OUT}/gold_l3_qb.npy", inter["qb"])
    np.save(f"{OUT}/gold_l3_qres.npy", inter["q_resid"])
    np.save(f"{OUT}/gold_l3_streams1.npy", streams1)
    np.save(f"{OUT}/gold_l3_ha.npy", h_a)
    with open(f"{OUT}/gold_l3_ha.txt", "w") as f:
        for row in h_a:
            f.write(" ".join(str(int(v * S40)) for v in row) + "\n")

    sel = [[int(v) for v in topk[q] if v >= 0] for q in range(s)]
    meta = {"ids": ids, "tokens": s,
            "stages": {"ha": dig(h_a), "attn": dig(y_attn), "streams1": dig(streams1),
                       "chain": dig(streams)},
            "topk": sel}
    json.dump(meta, open(f"{OUT}/gold_l3.json", "w"), indent=1)
    print(f"GOLD_L3_OK tokens={s}")
    for k, v in meta["stages"].items():
        print(f"  {k:9s} n={v['n']:7d} max={v['max'] / S40:.6e} sum={v['sum'] / S40:.6e}")
    print(f"  topk: q0 sel={sel[0]}  q{s - 1} n_sel={len(sel[s - 1])}")
    return 0


_EMB = {}


def read_row(i):
    if "w" not in _EMB:
        idx = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
        name = "model.language_model.embed_tokens.weight"
        with safe_open(f"{CKPT}/{idx[name]}", framework="pt", device="cpu") as f:
            _EMB["w"] = f.get_tensor(name).float().numpy()
    return _EMB["w"][i].astype(np.float64)


if __name__ == "__main__":
    sys.exit(main())
