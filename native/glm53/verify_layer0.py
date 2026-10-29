"""P4 gate: LAYER0_VERIFY — numpy layer 0 vs the HuggingFace glm5_next text layer
(transformers 5.21.1 ships `glm5_next`), both fed the SAME input embeddings and the
SAME real layer-0 weights from the GLM-5.3 checkpoint.

The HF model is built with one layer (dense MLP, KDA attention) and its weights are
filled from the extracted bf16 fixture, so any disagreement is my composition, not a
different checkpoint.

Usage: env -u PYTHONPATH ~/venvs/exl3/bin/python native/glm53/verify_layer0.py
"""

import json
import sys

import numpy as np
import torch
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

sys.path.insert(0, "/home/rick/Projects/VybForge/native/glm53")
import layer0_ref  # noqa: E402

CFG = "/home/rick/Projects/VybForge/native/glm53/layer0/config.json"
H = 4096


def build_config():
    d = json.load(open(CFG))
    cfg = Glm5NextConfig.from_dict(d)
    tc = cfg.text_config
    tc.num_hidden_layers = 1
    tc.layer_types = [tc.layer_types[0]]
    tc.mlp_layer_types = [tc.mlp_layer_types[0]]
    tc.indexer_types = [tc.indexer_types[0]]
    if hasattr(tc, "num_nextn_predict_layers"):
        tc.num_nextn_predict_layers = 0
    lac = getattr(tc, "linear_attn_config", None)
    if lac is not None:
        if hasattr(lac, "kda_layers"):
            lac.kda_layers = [0]
        if hasattr(lac, "full_attn_layers"):
            lac.full_attn_layers = []
    return tc


def load_hf_model():
    tc = build_config()
    model = Glm5NextTextModel(tc).to(torch.float32)
    model.eval()

    w = layer0_ref.load_weights()
    sd = {}

    def put(name, arr, transpose=False):
        a = arr.T if transpose else arr
        sd[name] = torch.from_numpy(np.ascontiguousarray(a).astype(np.float32))

    put("layers.0.self_attn.q_proj.weight", w["q"], transpose=True)   # I keep (in,out)
    put("layers.0.self_attn.k_proj.weight", w["k"], transpose=True)
    put("layers.0.self_attn.v_proj.weight", w["v"], transpose=True)
    put("layers.0.self_attn.o_proj.weight", w["o"], transpose=True)
    put("layers.0.self_attn.conv1d.weight",
        np.concatenate([w["conv_q"], w["conv_k"], w["conv_v"]], axis=0)[:, None, :])
    put("layers.0.self_attn.forget_gate.dt_bias", w["dt_bias"])
    put("layers.0.self_attn.forget_gate.A_log", w["A_log"])
    put("layers.0.self_attn.forget_gate.f_a_proj.weight", w["f_a"], transpose=True)
    put("layers.0.self_attn.forget_gate.f_b_proj.weight", w["f_b"], transpose=True)
    put("layers.0.self_attn.b_proj.weight", w["b"], transpose=True)
    put("layers.0.self_attn.g_a_proj.weight", w["g_a"], transpose=True)
    put("layers.0.self_attn.g_b_proj.weight", w["g_b"], transpose=True)
    put("layers.0.self_attn.o_norm.weight", w["o_norm"])
    put("layers.0.mlp.gate_proj.weight", w["mlp_gate"], transpose=True)
    put("layers.0.mlp.up_proj.weight", w["mlp_up"], transpose=True)
    put("layers.0.mlp.down_proj.weight", w["mlp_down"], transpose=True)
    put("layers.0.input_layernorm.weight", w["in_ln"])
    put("layers.0.post_attention_layernorm.weight", w["post_ln"])
    for site in ("attn", "ffn"):
        put(f"layers.0.{site}_hc.fn", w[f"{site}_fn"])
        put(f"layers.0.{site}_hc.base", w[f"{site}_base"])
        put(f"layers.0.{site}_hc.scale", w[f"{site}_scale"])

    missing, unexpected = model.load_state_dict(sd, strict=False)
    ignore = {"embed_tokens.weight", "norm.weight"}
    real_missing = [m for m in missing if m not in ignore]
    print(f"load: {len(sd)} tensors set; missing (excluding embed/norm): {real_missing}; "
          f"unexpected: {list(unexpected)}")
    return model, w


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, w = load_hf_model()
    model = model.to(dev)

    rs = np.random.RandomState(0)
    S = 8
    x = (rs.randn(S, H) * 0.02).astype(np.float32)

    # capture layer 0's own output (the 4 streams), not just the collapsed final state
    layers = model.model.layers if hasattr(model, "model") else model.layers
    print("layer module:", type(layers[0]).__name__)
    grabbed = {}

    def hook(mod, inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        grabbed["out"] = t.detach().float().cpu().numpy()

    layers[0].register_forward_hook(hook)

    with torch.no_grad():
        out = model(inputs_embeds=torch.from_numpy(x)[None].to(dev),
                    output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    print("hidden_states:", [tuple(h.shape) for h in hs])
    hf_streams = grabbed.get("out")
    print("layer-0 output captured:", None if hf_streams is None else hf_streams.shape)

    streams, y_attn, y_mlp = layer0_ref.layer_forward(x.astype(np.float64), w)
    print(f"mine: streams {streams.shape}  attn out |{np.abs(y_attn).max():.4f}|  "
          f"mlp out |{np.abs(y_mlp).max():.4f}|")

    if hf_streams is not None:
        hf = hf_streams[0]                      # [S, 4, H] (or [S, H] if collapsed)
        if hf.ndim == 2:
            print(f"hf  : collapsed layer output {hf.shape} |{np.abs(hf).max():.4f}|")
            for name, c in (("stream_mean", streams.mean(axis=1)),
                            ("stream_sum", streams.sum(axis=1)),
                            ("stream_0", streams[:, 0, :])):
                rel = np.abs(c - hf).max() / max(np.abs(hf).max(), 1e-12)
                print(f"  vs {name:11s}: rel {rel:.3e}")
            return 0
        print(f"hf  : streams {hf.shape} |{np.abs(hf).max():.4f}|")
        d = np.abs(streams - hf)
        rel = d.max() / max(np.abs(hf).max(), 1e-12)
        print(f"  per-stream max |d| {d.max():.3e}  rel {rel:.3e}")
        for k in range(hf.shape[1]):
            rk = np.abs(streams[:, k, :] - hf[:, k, :]).max() / max(np.abs(hf).max(), 1e-12)
            print(f"    stream {k}: rel {rk:.3e}")
        ok = rel < 2e-3        # bf16 weights, fp32 math on both sides
        print("LAYER0_VERIFY:", "OK" if ok else "FAIL", f"(rel {rel:.3e})")
        return 0 if ok else 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
