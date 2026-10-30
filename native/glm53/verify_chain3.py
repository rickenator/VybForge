"""P4 Gate A: CHAIN3_VERIFY — layers 0,1,2 (KDA + dense MLP) composed in numpy,
chained, vs the HF glm5_next decoder layers on the same chained inputs.

Layer 0's input is the expanded embedding; layers 1+ take the previous layer's
4-stream output directly. Both sides see identical weights (real bf16 checkpoint
tensors read from the apex copy) and identical inputs at every step, so any
drift beyond tolerance means the composition is wrong, not the weights.

Usage: env -u PYTHONPATH ~/venvs/exl3/bin/python native/glm53/verify_chain3.py
"""

import json
import sys

import numpy as np
import torch
from safetensors import safe_open
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

sys.path.insert(0, "/home/rick/Projects/VybForge/native/glm53")
import layer0_ref  # noqa: E402
import hc_ref  # noqa: E402

CKPT = "/mnt/apex/rick/vybforge/glm53-checkpoint"
CFG = CKPT + "/config.json"
H = 4096
HEADS, HDIM = 64, 128
INTER = 12288
RMS_EPS = 1e-5
LAYERS = [0, 1, 2]
S = 8


def read_layer(i):
    """Layer i's 26 tensors from the apex shards -> numpy f64 dict (layer0_ref layout)."""
    idx = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
    pre = f"model.language_model.layers.{i}."
    out = {}
    for k, shard in idx.items():
        if not k.startswith(pre):
            continue
        name = k[len(pre):]
        with safe_open(f"{CKPT}/{shard}", framework="pt", device="cpu") as f:
            t = f.get_tensor(k)
        if t.dtype == torch.bfloat16:
            a = t.float().numpy()          # torch converts bf16 -> f32 exactly
        else:
            a = t.numpy()
        out[name] = a.astype(np.float64)
    assert len(out) == 26, f"layer {i}: {len(out)} tensors, want 26"
    return out


def to_gdn_weights(t):
    """checkpoint names -> layer0_ref 'w' layout (transposed to (in,out))."""
    w = {}
    T = lambda n: t[n].T  # noqa: E731  HF stores (out,in)
    w["q"], w["k"], w["v"] = T("self_attn.q_proj.weight"), T("self_attn.k_proj.weight"), T("self_attn.v_proj.weight")
    w["o"] = T("self_attn.o_proj.weight")
    w["conv_q"] = t["self_attn.q_conv1d.weight"][:, 0, :]
    w["conv_k"] = t["self_attn.k_conv1d.weight"][:, 0, :]
    w["conv_v"] = t["self_attn.v_conv1d.weight"][:, 0, :]
    w["b"] = T("self_attn.b_proj.weight")
    w["f_a"], w["f_b"] = T("self_attn.f_a_proj.weight"), T("self_attn.f_b_proj.weight")
    w["g_a"], w["g_b"] = T("self_attn.g_a_proj.weight"), T("self_attn.g_b_proj.weight")
    w["dt_bias"], w["A_log"] = t["self_attn.dt_bias"], t["self_attn.A_log"]
    w["o_norm"] = t["self_attn.o_norm.weight"]
    w["mlp_gate"], w["mlp_up"], w["mlp_down"] = T("mlp.gate_proj.weight"), T("mlp.up_proj.weight"), T("mlp.down_proj.weight")
    w["in_ln"], w["post_ln"] = t["input_layernorm.weight"], t["post_attention_layernorm.weight"]
    for site, key in (("attn", "hc_attn"), ("ffn", "hc_ffn")):
        w[f"{site}_fn"] = t[f"{key}_fn"]
        w[f"{site}_base"] = t[f"{key}_base"]
        w[f"{site}_scale"] = t[f"{key}_scale"]
    return w


def numpy_layer_from_streams(streams, w):
    """streams [S,4,H] -> streams out. Same math as layer0_ref, minus the embed expand."""
    post, comb, collapsed, _ = hc_ref.mix(streams, w["attn_fn"], w["attn_base"], w["attn_scale"], RMS_EPS)
    h = collapsed / np.sqrt((collapsed * collapsed).mean(-1, keepdims=True) + RMS_EPS) * w["in_ln"]
    y = layer0_ref.kda(h, w)
    streams = hc_ref.apply_(streams, y, post, comb)
    post, comb, collapsed, _ = hc_ref.mix(streams, w["ffn_fn"], w["ffn_base"], w["ffn_scale"], RMS_EPS)
    h = collapsed / np.sqrt((collapsed * collapsed).mean(-1, keepdims=True) + RMS_EPS) * w["post_ln"]
    y = layer0_ref.mlp_dense(h, w)
    return hc_ref.apply_(streams, y, post, comb), y, y


def hf_layer_module(i):
    """1-layer HF model with layer i's weights -> (model, hook target)."""
    d = json.load(open(CFG))
    cfg = Glm5NextConfig.from_dict(d)
    tc = cfg.text_config
    lac = tc.linear_attn_config
    kda_set = set(lac["kda_layers"] if isinstance(lac, dict) else lac.kda_layers)
    is_kda = i in kda_set
    tc.num_hidden_layers = 1
    tc.layer_types = ["linear_attention" if is_kda else "deepseek_sparse_attention"]
    tc.mlp_layer_types = ["dense" if i < getattr(tc, "first_k_dense_replace", 3) else "sparse"]
    tc.indexer_types = ["full"]
    if hasattr(tc, "num_nextn_predict_layers"):
        tc.num_nextn_predict_layers = 0
    if isinstance(lac, dict):
        lac["kda_layers"] = [0]
        lac["full_attn_layers"] = []
    else:
        lac.kda_layers = [0]
        lac.full_attn_layers = []
    model = Glm5NextTextModel(tc).to(torch.float32)
    model.eval()

    t = read_layer(i)
    sd = {}
    put = lambda name, arr, tr=False: sd.__setitem__(  # noqa: E731
        name, torch.from_numpy(np.ascontiguousarray(arr.T if tr else arr).astype(np.float32)))
    # explicit, readable mapping (checkpoint (out,in) -> HF (out,in); my numpy side wants (in,out))
    put("layers.0.self_attn.q_proj.weight", t["self_attn.q_proj.weight"])
    put("layers.0.self_attn.k_proj.weight", t["self_attn.k_proj.weight"])
    put("layers.0.self_attn.v_proj.weight", t["self_attn.v_proj.weight"])
    put("layers.0.self_attn.o_proj.weight", t["self_attn.o_proj.weight"])
    put("layers.0.self_attn.conv1d.weight",
        np.concatenate([t["self_attn.q_conv1d.weight"], t["self_attn.k_conv1d.weight"],
                        t["self_attn.v_conv1d.weight"]], axis=0))
    put("layers.0.self_attn.forget_gate.dt_bias", t["self_attn.dt_bias"])
    put("layers.0.self_attn.forget_gate.A_log", t["self_attn.A_log"])
    put("layers.0.self_attn.forget_gate.f_a_proj.weight", t["self_attn.f_a_proj.weight"])
    put("layers.0.self_attn.forget_gate.f_b_proj.weight", t["self_attn.f_b_proj.weight"])
    put("layers.0.self_attn.b_proj.weight", t["self_attn.b_proj.weight"])
    put("layers.0.self_attn.g_a_proj.weight", t["self_attn.g_a_proj.weight"])
    put("layers.0.self_attn.g_b_proj.weight", t["self_attn.g_b_proj.weight"])
    put("layers.0.self_attn.o_norm.weight", t["self_attn.o_norm.weight"])
    put("layers.0.mlp.gate_proj.weight", t["mlp.gate_proj.weight"])
    put("layers.0.mlp.up_proj.weight", t["mlp.up_proj.weight"])
    put("layers.0.mlp.down_proj.weight", t["mlp.down_proj.weight"])
    put("layers.0.input_layernorm.weight", t["input_layernorm.weight"])
    put("layers.0.post_attention_layernorm.weight", t["post_attention_layernorm.weight"])
    for site, key in (("attn", "hc_attn"), ("ffn", "hc_ffn")):
        put(f"layers.0.{site}_hc.fn", t[f"{key}_fn"])
        put(f"layers.0.{site}_hc.base", t[f"{key}_base"])
        put(f"layers.0.{site}_hc.scale", t[f"{key}_scale"])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [m for m in missing if m not in ("embed_tokens.weight", "norm.weight")]
    assert not real_missing and not unexpected, (real_missing, unexpected)
    return model


def main():
    rs = np.random.RandomState(0)
    x = (rs.randn(S, H) * 0.02).astype(np.float64)      # the embedding, collapsed

    grabbed = {}

    def hook(mod, inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        grabbed["out"] = t.detach().float().cpu().numpy()

    worst = 0.0
    for i in LAYERS:
        t = read_layer(i)
        w = to_gdn_weights(t)
        model = hf_layer_module(i)
        layer = model.model.layers[0] if hasattr(model, "model") else model.layers[0]
        layer.register_forward_hook(hook)

        if i == 0:
            streams_in = np.repeat(x[:, None, :], 4, axis=1)          # ExpandStreams
        else:
            streams_in = prev_streams

        with torch.no_grad():
            layer(hidden_states=torch.from_numpy(streams_in[None].astype(np.float32)))
        hf_out = grabbed["out"][0].astype(np.float64)                 # [S,4,H]

        my_out, y_attn, y_mlp = numpy_layer_from_streams(streams_in, w)
        d = np.abs(my_out - hf_out).max() / max(np.abs(hf_out).max(), 1e-12)
        print(f"layer {i} ({'KDA' if i in (0,1,2) else '?'}): streams rel {d:.3e}  "
              f"|out| {np.abs(hf_out).max():.4f}")
        worst = max(worst, d)
        prev_streams = my_out
        del model, t, w

    ok = worst < 2e-3
    print(f"worst layer rel {worst:.3e}")
    print("CHAIN3_VERIFY:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
