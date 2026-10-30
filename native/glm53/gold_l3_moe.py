"""Gold for layer 3's MoE ffn site: router (sigmoid + noaux_tc bias + top-8 + renorm*2.5),
the clamped shared expert, and the routed experts (EXL3-dequantized through the P1/P2
oracle path), then the hc apply that closes the layer.

Input h2 (the pre-ffn layernorm output) comes from the numpy chain of layers 0-2 + layer
3's attention, so the Vyb side can run the same router/shared-expert code on identical
inputs. `gold_l3_moe_h2.txt` holds h2 as `(v*2^40) as Int` integers (Vyb cannot
reinterpret bytes as f64).

Usage: env -u PYTHONPATH ~/venvs/exl3/bin/python native/glm53/gold_l3_moe.py
"""

import json
import sys

import numpy as np

sys.path.insert(0, "/home/rick/Projects/VybForge/native/glm53")
import hc_ref  # noqa: E402
import layer0_ref  # noqa: E402
import moe_ref  # noqa: E402
import verify_layer3_moe as B  # noqa: E402

CKPT = "/mnt/apex/rick/vybforge/glm53-checkpoint"
OUT = "native/out"
H = 4096
S40 = 1099511627776.0
EPS = 1e-5


def read(name):
    import torch
    from safetensors import safe_open
    idx = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
    with safe_open(f"{CKPT}/{idx[name]}", framework="pt", device="cpu") as f:
        t = f.get_tensor(name)
    return t.float().numpy().astype(np.float64)


def dig(v):
    f = np.ascontiguousarray(v, dtype=np.float64).ravel()
    return {"n": len(f), "sum": int(f.sum() * S40), "abs": int(np.abs(f).sum() * S40),
            "sq": int((f * f).sum() * S40), "max": int(np.abs(f).max() * S40),
            **{f"v{k}": int(f[k] * S40) if k < len(f) else 0 for k in range(8)}}


def main():
    h2 = np.load(f"{OUT}/gold_l3_moe_h2.npy") if False else None
    h2 = load_h2()
    with open(f"{OUT}/gold_l3_moe_h2.txt", "w") as f:
        for row in h2:
            f.write(" ".join(str(int(v * S40)) for v in row) + "\n")

    pre = "model.language_model.layers.3."
    gate_w = read(pre + "mlp.gate.weight").astype(np.float32)
    bias = read(pre + "mlp.gate.e_score_correction_bias").astype(np.float32)
    shared = (read(pre + "mlp.shared_experts.gate_proj.weight").T,
              read(pre + "mlp.shared_experts.up_proj.weight").T,
              read(pre + "mlp.shared_experts.down_proj.weight").T)

    r_idx, r_w = moe_ref.route(h2, gate_w, bias)          # [R,8], [R,8]
    sel = [[int(e) for e in r_idx[t]] for t in range(h2.shape[0])]

    # per-token routed experts: dequantize the union once (this is the oracle path)
    union = sorted({e for row in sel for e in row})
    print(f"route: {len(union)} distinct experts over {h2.shape[0]} tokens")
    ew = {}
    for e in union:
        ew[e] = {p: B.read_expert_exl3(e, p, B.index()).astype(np.float64)
                 for p in ("gate_proj", "up_proj", "down_proj")}

    y = np.zeros_like(h2)
    for t in range(h2.shape[0]):
        for lane in range(8):
            e = sel[t][lane]
            w = float(r_w[t][lane])
            y[t] += w * moe_ref.ffn(h2[t:t + 1], ew[e]["gate_proj"], ew[e]["up_proj"],
                                    ew[e]["down_proj"])[0]
    y_shared = moe_ref.ffn(h2, *shared)
    y += y_shared
    np.save(f"{OUT}/gold_l3_shared.npy", y_shared)

    np.save(f"{OUT}/gold_l3_mlp.npy", y)
    np.save(f"{OUT}/gold_l3_route.npy", np.array(sel))
    np.save(f"{OUT}/gold_l3_rw.npy", r_w)
    meta = {"tokens": int(h2.shape[0]),
            "stages": {"h2": dig(h2), "shared": dig(y_shared), "mlp": dig(y)},
            "route": sel, "rw": [[float(x) for x in row] for row in r_w],
            "distinct_experts": union}
    json.dump(meta, open(f"{OUT}/gold_l3_moe.json", "w"), indent=1)
    print(f"GOLD_L3_MOE_OK tokens={h2.shape[0]} mlp max={np.abs(y).max():.6e}")
    print(f"  route[0]={sel[0]}  route[21]={sel[21]}")
    return 0


def load_h2():
    """h2 from the layer-3 chain (attention output -> hc apply -> ffn hc mix -> post-norm)."""
    import gold_l3_attn as G3
    pre = f"model.language_model.layers.3."
    T = lambda n: G3.read(pre + n).T  # noqa: E731
    # recompute the chain: layers 0-2 then layer 3's attention site + hc apply + ffn hc mix
    ids = json.load(open(f"{OUT}/gold_l3.json"))["ids"]
    x = np.stack([G3.read_row(i) for i in ids])
    streams = np.repeat(x[:, None, :], 4, axis=1)
    for lay in (0, 1, 2):
        streams, _, _ = G3.kda_layer(streams, G3.weights(lay, kda=True))
    w3 = G3.weights(3, kda=False)
    post, comb, collapsed, _ = hc_ref.mix(streams, w3["attn_fn"], w3["attn_base"],
                                          w3["attn_scale"], EPS)
    h_a = layer0_ref.rms_weight(collapsed, w3["in_ln"])
    wa, iw = G3.dsa_weights(3)
    import dsa_ref
    y_attn, _, _ = dsa_ref.layer_attention(h_a, wa, iw)
    streams1 = hc_ref.apply_(streams, y_attn, post, comb)

    # ffn site: hc mix -> post-attention layernorm
    post2, comb2, collapsed2, _ = hc_ref.mix(streams1, w3["ffn_fn"], w3["ffn_base"],
                                             w3["ffn_scale"], EPS)
    h2 = layer0_ref.rms_weight(collapsed2, w3["post_ln"])
    np.save(f"{OUT}/gold_l3_streams1.npy", streams1)
    np.save(f"{OUT}/gold_l3_post2.npy", post2)
    np.save(f"{OUT}/gold_l3_comb2.npy", comb2)
    np.save(f"{OUT}/gold_l3_moe_h2.npy", h2)
    return h2


if __name__ == "__main__":
    sys.exit(main())
