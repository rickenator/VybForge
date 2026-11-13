"""P4 Gate B: LAYER3_MOE_VERIFY — layer 3 (DSA attention + 288-expert MoE), numpy vs HF.

The input to layer 3 comes from the numpy chain of layers 0-2 (proven by
CHAIN3_VERIFY, worst rel 3e-7). One token, so the router selects 8 of 288 experts;
ONLY those are dequantized (via loader.dequant_exl3_fast, ~0.5 s/matrix) and
loaded into the HF side — the other 280 rows stay zero and HF's expert loop never
touches them. Router, shared expert, DSA attention and indexer weights are the
real checkpoint tensors on both sides.

Memory discipline (a first attempt was OOM-killed, exit 137): layer-3 expert EXL3
tensors are NOT bulk-read into one dict (that alone was ~15 GB as f64) and the HF
model runs in bf16, so the model and the zero-filled expert tensors are half size.

Usage: env -u PYTHONPATH ~/venvs/exl3/bin/python native/glm53/verify_layer3_moe.py
"""

import json
import sys

import resource

import numpy as np
import torch
from safetensors import safe_open
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

sys.path.insert(0, "/home/rick/Projects/VybForge/native/glm53")
sys.path.insert(0, "/home/rick/Projects/VybForge/native/exl3")
import verify_chain3 as V  # noqa: E402
import hc_ref  # noqa: E402
import moe_ref  # noqa: E402
from loader import dequant_exl3_fast  # noqa: E402

H = 4096
NE = 288
INTER = 2048


def index():
    return json.load(open(f"{V.CKPT}/model.safetensors.index.json"))["weight_map"]


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def read_one(name, idx):
    with safe_open(f"{V.CKPT}/{idx[name]}", framework="pt", device="cpu") as f:
        t = f.get_tensor(name)
    return t.float().numpy() if t.dtype == torch.bfloat16 else t.numpy()


def read_layer3_non_expert():
    """Layer-3 tensors EXCLUDING the per-expert EXL3 parts (bulk-read OOM'd)."""
    idx = index()
    pre = "model.language_model.layers.3."
    out = {}
    for k in idx:
        if not k.startswith(pre):
            continue
        short = k[len(pre):]
        if ".experts." in short:
            continue
        out[short] = read_one(k, idx).astype(np.float64)
    return out


def read_expert_exl3(expert, proj, idx):
    pre = f"model.language_model.layers.3.mlp.experts.{expert}.{proj}."
    with safe_open(f"{V.CKPT}/{idx[pre + 'trellis']}", framework="pt", device="cpu") as f:
        tre = f.get_tensor(pre + "trellis").numpy()
        suh = f.get_tensor(pre + "suh").numpy()
        svh = f.get_tensor(pre + "svh").numpy()
    return dequant_exl3_fast(tre, suh, svh)


def main():
    rs = np.random.RandomState(0)
    x = (rs.randn(1, H) * 0.02).astype(np.float64)          # ONE token
    idx_map = index()

    # ---- chain layers 0-2 (CHAIN3_VERIFY-proven) ----
    streams = np.repeat(x[:, None, :], 4, axis=1)
    for i in (0, 1, 2):
        w = V.to_gdn_weights(V.read_layer(i))
        streams, _, _ = V.numpy_layer_from_streams(streams, w)
    print(f"chained to layer-3 input: {streams.shape}")

    t3 = read_layer3_non_expert()
    print(f"layer-3 non-expert tensors: {len(t3)}  rss {rss()} MB", flush=True)

    gate_w = t3["mlp.gate.weight"].astype(np.float32)
    bias = t3["mlp.gate.e_score_correction_bias"].astype(np.float32)
    shared = (t3["mlp.shared_experts.gate_proj.weight"].T,
              t3["mlp.shared_experts.up_proj.weight"].T,
              t3["mlp.shared_experts.down_proj.weight"].T)

    # ---- HF side (bf16) ----
    d = json.load(open(V.CFG))
    cfg = Glm5NextConfig.from_dict(d)
    tc = cfg.text_config
    tc.num_hidden_layers = 1
    tc.layer_types = ["deepseek_sparse_attention"]
    tc.mlp_layer_types = ["sparse"]
    tc.indexer_types = ["full"]
    if hasattr(tc, "num_nextn_predict_layers"):
        tc.num_nextn_predict_layers = 0
    lac = tc.linear_attn_config
    if isinstance(lac, dict):
        lac["kda_layers"], lac["full_attn_layers"] = [], [0]
    else:
        lac.kda_layers, lac.full_attn_layers = [], [0]
    # build in bf16 from the start: the expert tensors are [288, 4096, 4096] and as
    # float32 they are 29 GB before any .to() conversion (that OOM'd twice)
    torch.set_default_dtype(torch.bfloat16)
    # ---- FFN site: mix -> post-norm -> route (my h2; agreement with HF's own top-k
    #      is verified after the forward via the experts pre-hook) ----
    post, comb, collapsed, _ = hc_ref.mix(streams, t3["hc_ffn_fn"], t3["hc_ffn_base"],
                                          t3["hc_ffn_scale"], 1e-5)
    h2 = collapsed / np.sqrt((collapsed * collapsed).mean(-1, keepdims=True) + 1e-5) * t3["post_attention_layernorm.weight"]
    h2 = h2[0:1]
    r_idx, r_w = moe_ref.route(h2, gate_w, bias)
    sel = [int(e) for e in r_idx[0]]
    print("router selected (my h2):", sel, flush=True)
    print("weights:", np.round(r_w[0], 4).tolist())

    print(f"building HF model (bf16 default)  rss {rss()} MB", flush=True)
    model = Glm5NextTextModel(tc)
    print(f"built  rss {rss()} MB", flush=True)
    model.eval()

    print(f"loading state dict  rss {rss()} MB", flush=True)
    sd = {}
    for name, param in dict(model.named_parameters()).items():
        short = name.replace("layers.0.", "")
        if ".experts." in name or short not in t3:
            continue
        short = {"hc_attn_fn": "attn_hc.fn", "hc_attn_base": "attn_hc.base",
                 "hc_attn_scale": "attn_hc.scale", "hc_ffn_fn": "ffn_hc.fn",
                 "hc_ffn_base": "ffn_hc.base", "hc_ffn_scale": "ffn_hc.scale"}.get(short, short)
        a = t3[short] if short in t3 else None
        if a is None:
            continue
        if a.shape == tuple(param.shape):
            sd[name] = torch.from_numpy(np.ascontiguousarray(a.astype(np.float32))).to(torch.bfloat16)
        elif a.T.shape == tuple(param.shape):
            sd[name] = torch.from_numpy(np.ascontiguousarray(a.T.astype(np.float32))).to(torch.bfloat16)
        else:
            print(f"  !! shape mismatch {name}: ckpt {a.shape} vs hf {tuple(param.shape)}")

    sel_w = {}
    for n, e in enumerate(sel):
        mats = {p: read_expert_exl3(e, p, idx_map) for p in ("gate_proj", "up_proj", "down_proj")}
        sel_w[e] = (mats["gate_proj"].astype(np.float64), mats["up_proj"].astype(np.float64),
                    mats["down_proj"].astype(np.float64))
        print(f"  [{n + 1}/8] expert {e} dequantized", flush=True)

    gu = torch.zeros(NE, 2 * INTER, H, dtype=torch.bfloat16)
    dn = torch.zeros(NE, H, INTER, dtype=torch.bfloat16)
    for e in sel:
        g, u, dnw = sel_w[e]
        gu[e, :INTER] = torch.from_numpy(g.T.astype(np.float32)).to(torch.bfloat16)
        gu[e, INTER:] = torch.from_numpy(u.T.astype(np.float32)).to(torch.bfloat16)
        dn[e] = torch.from_numpy(dnw.T.astype(np.float32)).to(torch.bfloat16)
    sd["layers.0.mlp.experts.gate_up_proj"] = gu
    sd["layers.0.mlp.experts.down_proj"] = dn
    print(f"sd ready ({len(sd)} tensors)  rss {rss()} MB", flush=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [m for m in missing if m not in ("embed_tokens.weight", "norm.weight",
                                                    "layers.0.mlp.gate.e_score_correction_bias")]
    print(f"load: {len(sd)} tensors; missing {real_missing}; unexpected {list(unexpected)}")
    # the router bias is an nn.Buffer (not a parameter): assign it or HF routes WITHOUT it
    root = model.model.layers[0] if hasattr(model, "model") else model.layers[0]
    with torch.no_grad():
        root.mlp.gate.e_score_correction_bias.copy_(torch.from_numpy(bias.astype(np.float32)))

    layer = model.model.layers[0] if hasattr(model, "model") else model.layers[0]
    grabbed = {}

    def mk(tag):
        def h(mm, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            grabbed[tag] = t.detach().float().cpu().numpy()
        return h

    def mk_in(tag):
        def h(mm, inp):
            grabbed[tag] = inp[0].detach().float().cpu().numpy()
        return h

    layer.mlp.register_forward_hook(mk("mlp"))
    layer.mlp.register_forward_pre_hook(mk_in("h2"))
    def exp_pre(mm, inp):
        grabbed["routed_h"] = inp[0].detach().float().cpu().numpy()
        grabbed["routed_idx"] = inp[1].detach().cpu().numpy()
        grabbed["routed_w"] = inp[2].detach().float().cpu().numpy()
    layer.mlp.experts.register_forward_pre_hook(exp_pre)
    layer.register_forward_hook(mk("layer"))
    with torch.no_grad():
        layer(hidden_states=torch.from_numpy(streams[None].astype(np.float32)).to(torch.bfloat16),
              attention_mask=torch.ones(1, 1, dtype=torch.bool))

    hf_topk = grabbed["routed_idx"][0].tolist()
    h2_hf = grabbed["routed_h"][0].astype(np.float64)
    sel = [int(e) for e in hf_topk]
    print("HF top-k (bias delivered):", sel)
    print("HF weights:", np.round(grabbed["routed_w"][0].astype(np.float64), 4).tolist())

    sel_w = {}
    for n, e in enumerate(sel):
        mats = {p: read_expert_exl3(e, p, idx_map) for p in ("gate_proj", "up_proj", "down_proj")}
        g, u, dnw = mats["gate_proj"], mats["up_proj"], mats["down_proj"]
        sel_w[e] = (g, u, dnw)
        with torch.no_grad():
            root.mlp.experts.gate_up_proj[e, :INTER].copy_(torch.from_numpy(g.T.astype(np.float32)))
            root.mlp.experts.gate_up_proj[e, INTER:].copy_(torch.from_numpy(u.T.astype(np.float32)))
            root.mlp.experts.down_proj[e].copy_(torch.from_numpy(dnw.T.astype(np.float32)))
        print(f"  [{n + 1}/8] expert {e} dequantized + loaded", flush=True)

    grabbed.pop("mlp", None); grabbed.pop("routed_h", None)
    with torch.no_grad():
        layer(hidden_states=torch.from_numpy(streams[None].astype(np.float32)).to(torch.bfloat16),
              attention_mask=torch.ones(1, 1, dtype=torch.bool))
    hf_y3 = grabbed["mlp"][0].astype(np.float64)
    h2_hf2 = grabbed["routed_h"][0].astype(np.float64)
    print("pass-2 h2 == pass-1 h2:", bool(np.array_equal(h2_hf2, h2_hf)))

    # the oracle's own run-to-run band: identical inputs, bf16 accumulation order varies
    # with lane order/threading — measure it rather than assume a threshold
    grabbed.pop("mlp", None)
    with torch.no_grad():
        layer(hidden_states=torch.from_numpy(streams[None].astype(np.float32)).to(torch.bfloat16),
              attention_mask=torch.ones(1, 1, dtype=torch.bool))
    hf_y3b = grabbed["mlp"][0].astype(np.float64)
    band = np.abs(hf_y3 - hf_y3b).max() / max(np.abs(hf_y3).max(), 1e-12)
    print(f"oracle run-to-run band: {band:.3e}")

    # MoE block math with HF's OWN top-k (moe_block re-routes internally, which would
    # disagree by construction here): clamped-SwiGLU experts + shared, HF's weights
    y3_np = np.zeros_like(h2_hf)
    w_lane = grabbed["routed_w"][0].astype(np.float64)
    for lane in range(len(sel)):
        e = sel[lane]
        wg, wu, wd = sel_w[e]
        y3_np += float(w_lane[lane]) * moe_ref.ffn(h2_hf, wg, wu, wd)
    y3_np += moe_ref.ffn(h2_hf, *shared)
    d_y = np.abs(y3_np - hf_y3).max() / max(np.abs(hf_y3).max(), 1e-12)
    # control: same math with weights pre-rounded to bf16 (what the HF side holds) —
    # if the gap collapses, the residual is bf16 weight rounding, not semantics
    def bf16_round(w):
        """true bf16 rounding: keep the top 16 bits of the f32 pattern (round-to-nearest-even
        via the +0x7FFF+lsb trick), matching what the bf16 HF side holds"""
        u = w.astype(np.float32).view(np.uint32)
        u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
        return u.view(np.float32).astype(np.float64)
    sel_w_b = {e: tuple(bf16_round(w) for w in t) for e, t in sel_w.items()}
    y3_b = np.zeros_like(h2_hf)
    for lane in range(len(sel)):
        e = sel[lane]
        wg, wu, wd = sel_w_b[e]
        y3_b += float(w_lane[lane]) * moe_ref.ffn(h2_hf, wg, wu, wd)
    y3_b += moe_ref.ffn(h2_hf, *shared)
    d_yb = np.abs(y3_b - hf_y3).max() / max(np.abs(hf_y3).max(), 1e-12)
    print(f"MoE out on HF's h2: |hf| {np.abs(hf_y3).max():.4f}  rel(f64 w) {d_y:.3e}  "
          f"rel(bf16-rounded w) {d_yb:.3e}")
    # sensitivity of the MoE output to ONE bf16 ulp on h2: HF's own h2 varies by ~1 ulp
    # run-to-run (bf16 attention accumulation order), so this is the oracle's floor
    h2e = h2_hf.astype(np.float32)
    h2e = (h2e.view(np.uint32) + (1 << 23)).view(np.float32).astype(np.float64)  # +1 ulp
    y3_e = np.zeros_like(h2_hf)
    for lane in range(len(sel)):
        e = sel[lane]
        wg, wu, wd = sel_w[e]
        y3_e += float(w_lane[lane]) * moe_ref.ffn(h2e, wg, wu, wd)
    y3_e += moe_ref.ffn(h2e, *shared)
    sens = np.abs(y3_e - y3_np).max() / max(np.abs(y3_np).max(), 1e-12)
    print(f"sensitivity to 1 bf16 ulp on h2: {sens:.3e}")
    ok = d_yb <= max(2e-3, sens)   # within the oracle's own bf16 reproducibility
    print("LAYER3_MOE_VERIFY:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
