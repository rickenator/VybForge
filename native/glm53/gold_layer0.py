"""Gold for the layer-0 Vyb forward: the full GLM-5.3 layer 0 in numpy fp64 on the
REAL checkpoint weights, for the same tokens the Vyb driver runs.

Everything here comes from native/glm53/*.py references that are already gated
(hc_ref HC_REF_VERIFY, gdn_ref GDN_RULE_VERIFY, layer0_ref LAYER0_VERIFY against HF).
This script only rewires them to read the apex checkpoint instead of the .bin fixtures.

Writes native/out/gold_l0.json (checksums + stage tables) and native/out/gold_l0_*.npy
(full arrays) for the Vyb side to diff.

Usage: env -u PYTHONPATH ~/venvs/exl3/bin/python native/glm53/gold_layer0.py
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

CKPT = "/mnt/apex/rick/vybforge/glm53-checkpoint"
OUT = "native/out"
H = 4096
HEADS, HDIM = 64, 128
KDIM = HDIM
INTER = 12288
EPS = 1e-5
PROMPT = ("The capital of France is Paris, and the largest city in Germany is Berlin, "
          "which was divided for decades.")


def read(name):
    idx = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
    with safe_open(f"{CKPT}/{idx[name]}", framework="pt", device="cpu") as f:
        t = f.get_tensor(name)
    return t.float().numpy().astype(np.float64)


def weights(layer=0):
    pre = f"model.language_model.layers.{layer}."
    T = lambda n: read(pre + n).T  # noqa: E731  checkpoint is (out,in)
    w = {}
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


def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{CKPT}/tokenizer.json")
    ids = tok.encode(PROMPT).ids
    s = len(ids)
    x = np.stack([_embed_row(i) for i in ids])
    w = weights(0)

    streams = np.repeat(x[:, None, :], 4, axis=1)
    # the hc-mix gemm result itself (mv = rms(streams) @ fn^T), for the Vyb side's stage diff
    flat = hc_ref.rms_norm_unweighted(
        streams.reshape(s, 4 * H).astype(np.float64), EPS)
    mv_attn = flat @ w["attn_fn"].astype(np.float64).T
    post, comb, collapsed, _ = hc_ref.mix(streams, w["attn_fn"], w["attn_base"], w["attn_scale"], EPS)
    h = layer0_ref.rms_weight(collapsed, w["in_ln"])
    # KDA intermediates, in the order the Vyb driver computes them
    qpre = h @ w["q"]
    kpre = h @ w["k"]
    vpre = h @ w["v"]
    qc = gdn_ref.short_conv_silu(qpre[None], w["conv_q"])[0]
    kc = gdn_ref.short_conv_silu(kpre[None], w["conv_k"])[0]
    vc = gdn_ref.short_conv_silu(vpre[None], w["conv_v"])[0]
    qn = (gdn_ref.l2norm(qc.reshape(s, HEADS, KDIM)) * (KDIM ** -0.5)).reshape(s, HEADS * KDIM)
    kn = gdn_ref.l2norm(kc.reshape(s, HEADS, KDIM)).reshape(s, HEADS * KDIM)
    bet = 1.0 / (1.0 + np.exp(-(h @ w["b"])))
    fa_ = h @ w["f_a"]
    fraw = fa_ @ w["f_b"]
    gdec = gdn_ref.kda_log_decay(fraw, w["dt_bias"], w["A_log"])
    ga_ = h @ w["g_a"]
    zraw = ga_ @ w["g_b"]
    y_attn = layer0_ref.kda(h, w)
    streams1 = hc_ref.apply_(streams, y_attn, post, comb)
    post2, comb2, collapsed2, _ = hc_ref.mix(streams1, w["ffn_fn"], w["ffn_base"], w["ffn_scale"], EPS)
    h2 = layer0_ref.rms_weight(collapsed2, w["post_ln"])
    y_mlp = layer0_ref.mlp_dense(h2, w)
    streams2 = hc_ref.apply_(streams1, y_mlp, post2, comb2)

    import os
    os.makedirs(OUT, exist_ok=True)
    np.save(f"{OUT}/gold_l0_x.npy", x)
    np.save(f"{OUT}/gold_l0_h.npy", h)
    np.save(f"{OUT}/gold_l0_attn.npy", y_attn)
    np.save(f"{OUT}/gold_l0_h2.npy", h2)
    np.save(f"{OUT}/gold_l0_mlp.npy", y_mlp)
    np.save(f"{OUT}/gold_l0_streams2.npy", streams2)
    np.save(f"{OUT}/gold_l0_mv_attn.npy", mv_attn)
    np.save(f"{OUT}/gold_l0_qpre.npy", qpre)
    np.save(f"{OUT}/gold_l0_qc.npy", qc)
    np.save(f"{OUT}/gold_l0_qn.npy", qn)
    np.save(f"{OUT}/gold_l0_kn.npy", kn)
    np.save(f"{OUT}/gold_l0_beta.npy", bet)
    np.save(f"{OUT}/gold_l0_fraw.npy", fraw)
    np.save(f"{OUT}/gold_l0_gdec.npy", gdec)
    np.save(f"{OUT}/gold_l0_zraw.npy", zraw)
    np.save(f"{OUT}/gold_l0_collapsed.npy", collapsed)
    np.save(f"{OUT}/gold_l0_post_attn.npy", post)
    np.save(f"{OUT}/gold_l0_comb_attn.npy", comb)

    def digest(a):
        f = np.ascontiguousarray(a, dtype=np.float64).ravel()
        return {"sum": float(f.sum()), "abs": float(np.abs(f).sum()),
                "sq": float((f * f).sum()), "max": float(np.abs(f).max()),
                "n": int(f.size),
                "head": [float(v) for v in f[:4]]}

    meta = {"ids": ids, "tokens": s, "stages": {
        "x": digest(x), "h": digest(h), "attn": digest(y_attn), "h2": digest(h2),
        "mlp": digest(y_mlp), "streams2": digest(streams2)}}
    json.dump(meta, open(f"{OUT}/gold_l0.json", "w"), indent=1)
    print(f"GOLD_L0_OK tokens={s} ids={ids}")
    for k, v in meta["stages"].items():
        print(f"  {k:9s} n={v['n']:7d} max={v['max']:.6e} sum={v['sum']:.6e} sq={v['sq']:.6e}")
    return 0


_EMB = {}


def _embed_row(i):
    """One row of embed_tokens.weight (1.27 GB tensor, read by the slice)."""
    if "w" not in _EMB:
        idx = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
        name = "model.language_model.embed_tokens.weight"
        with safe_open(f"{CKPT}/{idx[name]}", framework="pt", device="cpu") as f:
            _EMB["w"] = f.get_tensor(name).float().numpy()
    return _EMB["w"][i].astype(np.float64)


if __name__ == "__main__":
    sys.exit(main())
