"""P4 Gate C: LAYER3_FULL_VERIFY — the complete layer 3 (DSA attention + hc + MoE + hc)
composed in numpy vs HF, on the chained input from layers 0-2.

Sublayer checks: (1) DSA attention at 1 token (absorbed MLA: q_b from the q_a latent,
kv_b split per head into W_UK/W_UV, softmax weight 1 at S=1), (2) MoE routing agreement
on the post-attention h2, (3) full-layer output streams. Routing is chaotically
sensitive (Gate B: 335%/ulp) — if the h2 differs at bf16-ulp level the selections can
diverge; that is reported as the documented boundary class rather than a failure of
the math (the MoE block itself is gated in LAYER3_MOE_VERIFY on identical h2).

Usage: env -u PYTHONPATH ~/venvs/exl3/bin/python native/glm53/verify_layer3_attn.py
"""

import json
import sys

import numpy as np
import torch
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

sys.path.insert(0, "/home/rick/Projects/VybForge/native/glm53")
sys.path.insert(0, "/home/rick/Projects/VybForge/native/exl3")
import verify_chain3 as V  # noqa: E402
import verify_layer3_moe as B  # noqa: E402
import hc_ref  # noqa: E402
import moe_ref  # noqa: E402

H = 4096
NE = 288
INTER = 2048
HEADS, NOPE, VDIM = 64, 256, 256


def rms(x, weight, eps):
    x = np.asarray(x, dtype=np.float64)
    n = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    return n * np.asarray(weight, dtype=np.float64)


def main():
    rs = np.random.RandomState(0)
    x = (rs.randn(1, H) * 0.02).astype(np.float64)      # ONE token
    idx_map = B.index()

    streams = np.repeat(x[:, None, :], 4, axis=1)
    for i in (0, 1, 2):
        w = V.to_gdn_weights(V.read_layer(i))
        streams, _, _ = V.numpy_layer_from_streams(streams, w)
    print(f"chained to layer-3 input: {streams.shape}")

    t3 = B.read_layer3_non_expert()

    # ---- attention site: hc mix -> input_layernorm -> DSA attention (absorbed, S=1) ----
    post_a, comb_a, collapsed_a, _ = hc_ref.mix(streams, t3["hc_attn_fn"], t3["hc_attn_base"],
                                                t3["hc_attn_scale"], 1e-5)
    h_a = rms(collapsed_a, t3["input_layernorm.weight"], 1e-5)
    q_resid = h_a @ t3["self_attn.q_a_proj.weight"].T
    q = (rms(q_resid, t3["self_attn.q_a_layernorm.weight"], 1e-5)
         @ t3["self_attn.q_b_proj.weight"].T).reshape(HEADS, NOPE)
    ckv = rms((h_a @ t3["self_attn.kv_a_proj_with_mqa.weight"].T)[:512],
              t3["self_attn.kv_a_layernorm.weight"], 1e-5)
    kv_b = t3["self_attn.kv_b_proj.weight"].reshape(HEADS, 2 * VDIM, 512)
    uk, uv = kv_b[:, :NOPE, :], kv_b[:, NOPE:, :]          # [64, 256, 512] each
    # S=1: softmax weight is exactly 1, so the attention output is the value path
    o = np.concatenate([ckv @ uv[h].T for h in range(HEADS)], axis=1).reshape(-1)  # [16384]
    y_attn = (o @ t3["self_attn.o_proj.weight"].T)[None, :]            # [1, 4096]
    kvb_mine = np.concatenate([np.concatenate([ckv @ uk[h].T, ckv @ uv[h].T])
                               for h in range(HEADS)])[None]
    STAGES = {"qa_ln": rms(q_resid, t3["self_attn.q_a_layernorm.weight"], 1e-5),
              "kva_ln": ckv[None], "qb": q.reshape(1, -1),
              "kvb": kvb_mine, "opre": y_attn}
    streams_mid = hc_ref.apply_(streams, y_attn, post_a, comb_a)

    # ---- ffn site: hc mix -> post-norm -> route -> MoE (my h2, my routing) ----
    post_f, comb_f, collapsed_f, _ = hc_ref.mix(streams_mid, t3["hc_ffn_fn"], t3["hc_ffn_base"],
                                                t3["hc_ffn_scale"], 1e-5)
    h2 = rms(collapsed_f, t3["post_attention_layernorm.weight"], 1e-5)[0:1]
    r_idx, r_w = moe_ref.route(h2, t3["mlp.gate.weight"].astype(np.float32),
                               t3["mlp.gate.e_score_correction_bias"].astype(np.float32))
    sel = [int(e) for e in r_idx[0]]
    print("my routing (on my h2):", sel)

    sel_w = {}
    for n, e in enumerate(sel):
        mats = {p: B.read_expert_exl3(e, p, idx_map) for p in ("gate_proj", "up_proj", "down_proj")}
        sel_w[e] = (mats["gate_proj"], mats["up_proj"], mats["down_proj"])
    y3 = np.zeros_like(h2)
    for lane in range(len(sel)):
        e = sel[lane]
        wg, wu, wd = sel_w[e]
        y3 += float(r_w[0][lane]) * moe_ref.ffn(h2, wg, wu, wd)
    shared = (t3["mlp.shared_experts.gate_proj.weight"].T,
              t3["mlp.shared_experts.up_proj.weight"].T,
              t3["mlp.shared_experts.down_proj.weight"].T)
    y3 += moe_ref.ffn(h2, *shared)
    streams_out = hc_ref.apply_(streams_mid, y3, post_f, comb_f)

    # ---- HF side ----
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
    torch.set_default_dtype(torch.bfloat16)
    model = Glm5NextTextModel(tc)
    model.eval()

    # hc fn/base/scale are BUFFERS, not parameters — named_parameters() misses them,
    # and an unloaded fn means the hc mix runs on random init (the 19% divergence)
    hf_tensors = dict(model.named_parameters())
    hf_tensors.update(dict(model.named_buffers()))
    sd = {}
    for name, param in hf_tensors.items():
        short = name.replace("layers.0.", "")
        short = {"attn_hc.fn": "hc_attn_fn", "attn_hc.base": "hc_attn_base",
                 "attn_hc.scale": "hc_attn_scale", "ffn_hc.fn": "hc_ffn_fn",
                 "ffn_hc.base": "hc_ffn_base", "ffn_hc.scale": "hc_ffn_scale"}.get(short, short)
        if ".experts." in name or short not in t3:
            continue
        a = t3[short]
        if a.shape == tuple(param.shape):
            sd[name] = torch.from_numpy(np.ascontiguousarray(a.astype(np.float32))).to(torch.bfloat16)
        elif a.T.shape == tuple(param.shape):
            sd[name] = torch.from_numpy(np.ascontiguousarray(a.T.astype(np.float32))).to(torch.bfloat16)
        else:
            print(f"  !! shape mismatch {name}")

    gu = torch.zeros(NE, 2 * INTER, H, dtype=torch.bfloat16)
    dn = torch.zeros(NE, H, INTER, dtype=torch.bfloat16)
    for e in sel:                                    # my routing, so the MoE is comparable
        wg, wu, wd = sel_w[e]
        gu[e, :INTER] = torch.from_numpy(wg.T.astype(np.float32)).to(torch.bfloat16)
        gu[e, INTER:] = torch.from_numpy(wu.T.astype(np.float32)).to(torch.bfloat16)
        dn[e] = torch.from_numpy(wd.T.astype(np.float32)).to(torch.bfloat16)
    sd["layers.0.mlp.experts.gate_up_proj"] = gu
    sd["layers.0.mlp.experts.down_proj"] = dn
    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [m for m in missing if m not in ("embed_tokens.weight", "norm.weight",
                                                    "layers.0.mlp.gate.e_score_correction_bias")]
    print(f"load: {len(sd)} tensors; missing {real_missing}; unexpected {list(unexpected)}")
    root = model.model.layers[0] if hasattr(model, "model") else model.layers[0]
    with torch.no_grad():
        root.mlp.gate.e_score_correction_bias.copy_(
            torch.from_numpy(t3["mlp.gate.e_score_correction_bias"].astype(np.float32)))

    layer = root
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

    layer.self_attn.register_forward_hook(mk("attn"))
    # stage-by-stage: find the first divergent sub-step (forward hooks: post-stage values)
    layer.input_layernorm.register_forward_hook(mk("inln"))
    def attn_in(mm, args, kwargs):
        hs = args[0] if args else kwargs.get("hidden_states")
        grabbed["attn_in"] = hs.detach().float().cpu().numpy()
        grabbed["fired"] = grabbed.get("fired", []) + ["attn_in"]
    layer.self_attn.register_forward_pre_hook(attn_in, with_kwargs=True)
    def hc_hook(mm, inp, out):
        post_h, comb_h, coll_h = out
        grabbed["hc_post"] = post_h.detach().float().cpu().numpy()
        grabbed["hc_comb"] = comb_h.detach().float().cpu().numpy()
        grabbed["hc_coll"] = coll_h.detach().float().cpu().numpy()
    layer.attn_hc.register_forward_hook(hc_hook)
    def qap_hook(mm, inp, out):
        grabbed["qap"] = out.detach().float().cpu().numpy()
        grabbed["fired"] = grabbed.get("fired", []) + ["qap"]
    layer.self_attn.q_a_proj.register_forward_hook(qap_hook)
    for mod_name, tag in (("q_a_layernorm", "qa_ln"), ("kv_a_layernorm", "kva_ln"),
                          ("q_b_proj", "qb"), ("kv_b_proj", "kvb"),
                          ("o_proj", "opre")):
        getattr(layer.self_attn, mod_name).register_forward_hook(mk(tag))
    layer.mlp.register_forward_hook(mk("mlp"))
    layer.mlp.register_forward_pre_hook(mk_in("h2"))
    layer.mlp.experts.register_forward_pre_hook(
        lambda mm, inp: grabbed.__setitem__("routed_idx", inp[1].detach().cpu().numpy()))
    layer.register_forward_hook(mk("layer"))

    with torch.no_grad():
        layer(hidden_states=torch.from_numpy(streams[None].astype(np.float32)).to(torch.bfloat16),
              attention_mask=torch.ones(1, 1, dtype=torch.bool))

    # ---- comparisons ----
    hf_attn = grabbed["attn"][0].astype(np.float64)
    d_attn = np.abs(y_attn - hf_attn).max() / max(np.abs(hf_attn).max(), 1e-12)
    print(f"attention out: |hf| {np.abs(hf_attn).max():.4f}  rel {d_attn:.3e}")
    for tag, mine in STAGES.items():
        if tag in grabbed:
            hf_v = grabbed[tag].astype(np.float64)
            r = np.abs(mine - hf_v).max() / max(np.abs(hf_v).max(), 1e-12) if mine.shape == hf_v.shape                 else float("nan")
            print(f"  stage {tag:6s}: hf shape {hf_v.shape}  rel {r:.3e}")
    print("hooks fired:", grabbed.get("fired", []))
    for hname, ckname in (("fn", "hc_attn_fn"), ("base", "hc_attn_base"), ("scale", "hc_attn_scale")):
        hp = getattr(root.attn_hc, hname).detach().float().cpu().numpy()
        ck = t3[ckname].astype(np.float32)
        same_shape = hp.shape == ck.shape
        d = np.abs(hp - ck).max() / max(np.abs(ck).max(), 1e-12) if same_shape else float("nan")
        print(f"  hc load {hname:5s}: hf {hp.shape} ckpt {ck.shape} same_shape={same_shape} rel {d:.3e}")
    if "hc_coll" in grabbed:
        hc_coll = grabbed["hc_coll"].astype(np.float64).reshape(1, -1)
        print(f"  hc collapsed vs mine: rel {np.abs(collapsed_a - hc_coll).max() / max(np.abs(hc_coll).max(), 1e-12):.3e}")
        print(f"  hc post vs mine:      rel {np.abs(post_a - grabbed['hc_post'].astype(np.float64).reshape(1,-1)).max() / max(np.abs(post_a).max(), 1e-12):.3e}")
        print(f"  hc comb vs mine:      rel {np.abs(comb_a.reshape(1, -1) - grabbed['hc_comb'].astype(np.float64).reshape(1, -1)).max() / max(np.abs(comb_a).max(), 1e-12):.3e}")
    if "attn_in" in grabbed:
        hf_ain = grabbed["attn_in"].astype(np.float64).reshape(1, -1)
        h_a = rms(collapsed_a, t3["input_layernorm.weight"], 1e-5)
        print(f"  attn_in vs my h_a: rel {np.abs(h_a - hf_ain).max() / max(np.abs(hf_ain).max(), 1e-12):.3e}")
    if "inln" in grabbed:
        hf_inln = grabbed["inln"].astype(np.float64).reshape(1, -1)
        print(f"  inln vs my h_a:    rel {np.abs(h_a - hf_inln).max() / max(np.abs(hf_inln).max(), 1e-12):.3e}")
    if "qap" in grabbed:
        hf_qap = grabbed["qap"].astype(np.float64).reshape(1, -1)
        print(f"  qap vs my q_resid: rel {np.abs(q_resid - hf_qap).max() / max(np.abs(hf_qap).max(), 1e-12):.3e}")

    hf_h2 = grabbed["h2"][0].astype(np.float64)
    d_h2 = np.abs(h2 - hf_h2).max() / max(np.abs(hf_h2).max(), 1e-12)
    print(f"h2 (mine vs HF, after attention): rel {d_h2:.3e}")

    hf_sel = sorted(int(e) for e in np.asarray(grabbed["routed_idx"]).reshape(-1)[:8])
    agree = hf_sel == sorted(sel)   # SET agreement: lane order varies with accumulation
    print(f"routing: mine {sorted(sel)}  HF {hf_sel}  set-agree={agree}")

    hf_y3 = grabbed["mlp"][0].astype(np.float64)
    d_y3 = np.abs(y3 - hf_y3).max() / max(np.abs(hf_y3).max(), 1e-12)
    print(f"mlp out: |hf| {np.abs(hf_y3).max():.4f}  rel {d_y3:.3e}")

    hf_out = grabbed["layer"][0].astype(np.float64)
    d_full = np.abs(streams_out - hf_out).max() / max(np.abs(hf_out).max(), 1e-12)
    print(f"full layer streams: |hf| {np.abs(hf_out).max():.4f}  rel {d_full:.3e}")

    # bf16-noise tolerances: the HF side computes in bf16, so 3e-3..1e-2 residuals are
    # its own rounding, verified by the stage table above (hc/attn_in/qap all ~3-6e-3)
    ok = (d_attn < 1e-2 and d_h2 < 1e-2 and agree and d_y3 < 1e-2 and d_full < 1.5e-2)
    print("LAYER3_FULL_VERIFY:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
