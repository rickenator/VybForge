"""GLM-5.3 layer 0 (KDA + dense MLP + two hyper-connection sites) — numpy, real weights.

Layer 0 is entirely bf16 in the checkpoint (no EXL3), 579 MB. Weights come from
native/glm53/layer0/ (extracted with ~/.hermes/cache/scratch/extract_hc.py).

Composition (matches the HF glm5_next text layer, whose parameter names are listed
in verify_layer0.py):

    streams(4) = hc_attn.mix(x)                      -> post_a, comb_a, collapsed
    h  = input_layernorm(collapsed)
    y  = o_proj( gated_rmsnorm_o( kda(h), z(h) ) )
    streams = post_a ⊗ y + comb_aᵀ streams
    streams2 = hc_ffn.mix(streams)                   -> post_f, comb_f, collapsed2
    h2 = post_attention_layernorm(collapsed2)
    y2 = down_proj( silu(gate(h2)) * up(h2) )
    out = post_f ⊗ y2 + comb_fᵀ streams
"""

import json
import os

import numpy as np

import gdn_ref
import hc_ref

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "layer0")
H = 4096
HEADS, HDIM = 64, 128
INTER = 12288
RMS_EPS = 1e-5


def load_bf16(name, shape):
    """bf16 -> f32 (uint16 << 16 reinterpreted)."""
    a = np.fromfile(f"{FIX}/model_language_model_layers_0_{name}.bin", dtype="<u2")
    a = (a.astype(np.uint32) << 16).view(np.float32).astype(np.float32)
    return a.reshape(shape)


def load_f32(name, shape):
    return np.fromfile(f"{FIX}/model_language_model_layers_0_{name}.bin",
                       dtype="<f4").reshape(shape)


def load_weights():
    w = {}
    w["q"] = load_bf16("self_attn_q_proj_weight", (HEADS * HDIM, H)).astype(np.float64).T
    w["k"] = load_bf16("self_attn_k_proj_weight", (HEADS * HDIM, H)).astype(np.float64).T
    w["v"] = load_bf16("self_attn_v_proj_weight", (HEADS * HDIM, H)).astype(np.float64).T
    w["o"] = load_bf16("self_attn_o_proj_weight", (H, HEADS * HDIM)).astype(np.float64).T
    w["conv_q"] = load_bf16("self_attn_q_conv1d_weight", (HEADS * HDIM, 1, 4))[:, 0, :].astype(np.float64)
    w["conv_k"] = load_bf16("self_attn_k_conv1d_weight", (HEADS * HDIM, 1, 4))[:, 0, :].astype(np.float64)
    w["conv_v"] = load_bf16("self_attn_v_conv1d_weight", (HEADS * HDIM, 1, 4))[:, 0, :].astype(np.float64)
    w["b"] = load_bf16("self_attn_b_proj_weight", (HEADS, H)).astype(np.float64).T      # [H, H_heads]
    w["f_a"] = load_bf16("self_attn_f_a_proj_weight", (128, H)).astype(np.float64).T   # [H,128]
    w["f_b"] = load_bf16("self_attn_f_b_proj_weight", (HEADS * HDIM, 128)).astype(np.float64).T
    w["g_a"] = load_bf16("self_attn_g_a_proj_weight", (128, H)).astype(np.float64).T
    w["g_b"] = load_bf16("self_attn_g_b_proj_weight", (HEADS * HDIM, 128)).astype(np.float64).T
    w["dt_bias"] = load_f32("self_attn_dt_bias", (HEADS * HDIM,)).astype(np.float64)
    w["A_log"] = load_f32("self_attn_A_log", (HEADS,)).astype(np.float64)
    w["o_norm"] = load_bf16("self_attn_o_norm_weight", (HDIM,)).astype(np.float64)
    w["mlp_gate"] = load_bf16("mlp_gate_proj_weight", (INTER, H)).astype(np.float64).T
    w["mlp_up"] = load_bf16("mlp_up_proj_weight", (INTER, H)).astype(np.float64).T
    w["mlp_down"] = load_bf16("mlp_down_proj_weight", (H, INTER)).astype(np.float64).T
    w["in_ln"] = load_bf16("input_layernorm_weight", (H,)).astype(np.float64)
    w["post_ln"] = load_bf16("post_attention_layernorm_weight", (H,)).astype(np.float64)
    for site, key in (("attn", "hc_attn"), ("ffn", "hc_ffn")):
        w[f"{site}_fn"] = load_bf16(f"{key}_fn", (24, 4 * H)).astype(np.float64)
        w[f"{site}_base"] = load_f32(f"{key}_base", (24,)).astype(np.float64)
        w[f"{site}_scale"] = load_f32(f"{key}_scale", (3,)).astype(np.float64)
    return w


def rms_weight(x, weight, eps=RMS_EPS):
    x = np.asarray(x, dtype=np.float64)
    n = x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
    return n * weight


def kda(out_proj_input, w, beta_scale=1.0):
    """q/k/v projections -> conv+silu -> gates -> delta rule -> gated norm."""
    h = out_proj_input                       # [s, H]
    s = h.shape[0]
    q = gdn_ref.short_conv_silu((h @ w["q"])[None], w["conv_q"])[0].reshape(s, HEADS, HDIM)
    k = gdn_ref.short_conv_silu((h @ w["k"])[None], w["conv_k"])[0].reshape(s, HEADS, HDIM)
    v = gdn_ref.short_conv_silu((h @ w["v"])[None], w["conv_v"])[0].reshape(s, HEADS, HDIM)

    beta = (1.0 / (1.0 + np.exp(-(h @ w["b"])))) * beta_scale        # [s, HEADS]
    f = (h @ w["f_a"]) @ w["f_b"]                                     # [s, HEADS*HDIM]
    g = gdn_ref.kda_log_decay(f, w["dt_bias"], w["A_log"])
    z = (h @ w["g_a"]) @ w["g_b"]                                     # [s, HEADS*HDIM]

    o, _ = gdn_ref.kda_recurrent(q[None], k[None], v[None], g[None], beta[None])
    o = o[0].reshape(s, HEADS * HDIM)
    zz = z.reshape(s, HEADS, HDIM)
    o = o.reshape(s, HEADS, HDIM)
    o = gdn_ref.gated_rmsnorm(o, zz, w["o_norm"], eps=RMS_EPS, activation="sigmoid")
    return o.reshape(s, HEADS * HDIM) @ w["o"]


def mlp_dense(h, w):
    g = h @ w["mlp_gate"]
    u = h @ w["mlp_up"]
    act = (g * (1.0 / (1.0 + np.exp(-g)))) * u
    return act @ w["mlp_down"]


def layer_forward_attn(x, w):
    """x [s,H] (the collapsed input) -> (streams [s,4,H], attn out [s,H])."""
    streams = np.repeat(x[:, None, :], 4, axis=1)               # ExpandStreams: 4 copies
    post, comb, collapsed, _ = hc_ref.mix(streams, w["attn_fn"], w["attn_base"],
                                          w["attn_scale"], RMS_EPS)
    h = rms_weight(collapsed, w["in_ln"])
    y = kda(h, w)
    streams = hc_ref.apply_(streams, y, post, comb)
    return streams, y


def layer_forward_ffn(streams, w):
    post, comb, collapsed, _ = hc_ref.mix(streams, w["ffn_fn"], w["ffn_base"],
                                          w["ffn_scale"], RMS_EPS)
    h = rms_weight(collapsed, w["post_ln"])
    y = mlp_dense(h, w)
    return hc_ref.apply_(streams, y, post, comb), y


def layer_forward(x, w=None):
    if w is None:
        w = load_weights()
    streams, y_attn = layer_forward_attn(x, w)
    streams, y_mlp = layer_forward_ffn(streams, w)
    return streams, y_attn, y_mlp


if __name__ == "__main__":
    rs = np.random.RandomState(0)
    x = (rs.randn(8, H) * 0.02)
    st, ya, ym = layer_forward(x)
    print(f"streams {st.shape} |attn out| {np.abs(ya).max():.4f} "
          f"|mlp out| {np.abs(ym).max():.4f} finite {np.isfinite(st).all()}")
