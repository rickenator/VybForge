"""Gold for the layer-3 ATTENTION-site hc mix/apply (native/host/glm_l3_hc_check.vyb).

Recomputes, from the dumped streams0 text and the checkpoint's hc_attn_* tensors:
  mv, collapsed, postA, combA, h_a, streams1
and prints the same digests the Vyb check prints, so the two can be diffed directly.
Also writes gold_l3_attn.txt (the attention output as scaled integers) for the check.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/rick/Projects/VybForge/native/glm53")
import gold_l3_attn as G3  # noqa: E402
import hc_ref  # noqa: E402
import layer0_ref  # noqa: E402

OUT = "native/out"
S40 = 1099511627776.0
EPS = 1e-5
pre = "model.language_model.layers.3."


def dig(name, v):
    f = np.ascontiguousarray(v, dtype=np.float64).ravel()
    parts = [f"n={len(f)}", f"sum={int(f.sum() * S40)}", f"abs={int(np.abs(f).sum() * S40)}",
             f"sq={int((f * f).sum() * S40)}", f"max={int(np.abs(f).max() * S40)}"]
    for k in range(8):
        parts.append(f"v{k}={int(f[k] * S40) if k < len(f) else 0}")
    return name + " " + " ".join(parts)


def main():
    y_attn = np.load(f"{OUT}/gold_l3_attn.npy").astype(np.float64)
    with open(f"{OUT}/gold_l3_attn.txt", "w") as f:
        for row in y_attn:
            f.write(" ".join(str(int(v * S40)) for v in row) + " ")

    s0 = np.array([int(x) for x in open(f"{OUT}/gold_l3_streams0.txt").read().split()],
                  dtype=np.float64) / S40
    streams0 = s0.reshape(22, 4, 4096)

    fn = G3.read(pre + "hc_attn_fn").astype(np.float64)        # [24, 16384]
    base = G3.read(pre + "hc_attn_base").astype(np.float64)    # [24]
    scale = G3.read(pre + "hc_attn_scale").astype(np.float64)  # [3]
    in_ln = G3.read(pre + "input_layernorm.weight").astype(np.float64)

    flat = hc_ref.rms_norm_unweighted(streams0.reshape(22, 4 * 4096), EPS)
    mv = flat @ fn.T
    with open(f"{OUT}/l3hc_mv_gold.txt", "w") as f:
        f.write(" ".join(str(int(v * S40)) for v in mv.ravel()) + " ")
    post, comb, collapsed, _ = hc_ref.mix(streams0, fn, base, scale, EPS)
    h_a = layer0_ref.rms_weight(collapsed, in_ln)
    streams1 = hc_ref.apply_(streams0, y_attn, post, comb)

    out = [dig("mv", mv), dig("collapsed", collapsed), dig("postA", post), dig("combA", comb),
           dig("h_a", h_a), dig("streams1", streams1)]
    open(f"{OUT}/gold_l3hc.txt", "w").write("\n".join(out) + "\n")
    for line in out:
        print(line.split()[0], "abs=", line.split()[3])
    s1 = np.load(f"{OUT}/gold_l3_streams1.npy")
    print("sanity streams1 vs saved:", float(np.abs(streams1 - s1).max()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
