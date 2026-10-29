"""P3 gate: GDN_RULE_VERIFY — numpy KDA delta rule vs exllamav3's own labelled
reference for this variant (`gated_delta_rule.py::torch_recurrent_kda`, "mirroring
the GLM5.3/Kimi reference"), at the real head shape (64 heads x 128) and real log-decay
values from the checkpoint config's gate (lower bound -5.0).

Usage: env -u PYTHONPATH ~/venvs/exl3/bin/python native/glm53/verify_gdn.py
"""

import sys

import numpy as np
import torch

sys.path.insert(0, "/home/rick/Projects/VybForge/native/glm53")
import gdn_ref  # noqa: E402
from exllamav3.modules.gated_delta_net_fn.gated_delta_rule import torch_recurrent_kda  # noqa: E402

H, DK, DV = 64, 128, 128
S = 12
B = 1


def main():
    rs = np.random.RandomState(7)
    q = (rs.randn(B, S, H, DK) * 0.5).astype(np.float16)
    k = (rs.randn(B, S, H, DK) * 0.5).astype(np.float16)
    v = (rs.randn(B, S, H, DV) * 0.5).astype(np.float16)

    # log-decay through the real gate: f ~ N(0,1), dt_bias ~ 0.1, a_log ~ 0.5,
    # lower bound -5 -> g in (-5, 0), i.e. per-step decay exp(g) in (0.0067, 1)
    f = (rs.randn(B, S, H * DK) * 1.0).astype(np.float32)
    dt_bias = (rs.randn(H * DK) * 0.1).astype(np.float32)
    a_log = (rs.randn(H) * 0.5).astype(np.float32)
    g = gdn_ref.kda_log_decay(f, dt_bias, a_log)
    print(f"log-decay g: min {g.min():.4f} max {g.max():.4f} "
          f"(decay per step {np.exp(g.min()):.4f}..{np.exp(g.max()):.4f})")

    beta = (1.0 / (1.0 + np.exp(-rs.randn(B, S, H)))).astype(np.float32)   # sigmoid, ~[0,1]

    out_np, st_np = gdn_ref.kda_recurrent(q, k, v, g, beta)

    out_t = torch_recurrent_kda(
        torch.from_numpy(q.astype(np.float32)),
        torch.from_numpy(k.astype(np.float32)),
        torch.from_numpy(v.astype(np.float32)),
        torch.from_numpy(g.astype(np.float32)),
        torch.from_numpy(beta.astype(np.float32)),
        torch.zeros(B, H, DK, DV, dtype=torch.float32),
    ).float().numpy()

    # the shipped reference returns bf16, so compare at that resolution: how many
    # elements round to the identical bf16 value?
    out_bf16 = torch.from_numpy(out_np.astype(np.float32)).to(torch.bfloat16)
    same = (out_bf16 == torch.from_numpy(out_t)).float().mean().item()
    ref = out_t.astype(np.float64)
    scale = float(np.abs(ref).max())
    d = np.abs(out_np - ref).max() / scale
    print(f"out: scale {scale:.4f}  rel {d:.3e}  bf16-exact {same * 100:.3f}% of "
          f"{out_np.size:,} elements")
    print(f"state: |S| max {np.abs(st_np).max():.4f}, nonzero {int((st_np != 0).sum())}")

    # ---- short conv + silu, against the shipped torch fallback ----
    from exllamav3.modules.gated_delta_net_fn.conv1d import causal_conv1d_update_function_torch

    C = 192          # a slice of the q|k|v channels
    SQ = 9
    xs = (rs.randn(1, C, SQ) * 0.5).astype(np.float32)          # [b, dim, seq] for torch
    wconv = (rs.randn(C, 1, 4) * 0.3).astype(np.float32)
    state = torch.zeros(1, C, 4, dtype=torch.float32)
    conv_t = causal_conv1d_update_function_torch(
        torch.from_numpy(xs), state, torch.from_numpy(wconv[:, 0, :]), None).numpy()
    conv_np = gdn_ref.short_conv_silu(xs.transpose(0, 2, 1),
                                      wconv[:, 0, :]).transpose(0, 2, 1)
    d_conv = np.abs(conv_np - conv_t).max() / max(np.abs(conv_t).max(), 1e-12)
    print(f"conv+silu: rel {d_conv:.3e}  (scale {np.abs(conv_t).max():.4f})")

    # ---- gated RMSNorm (KDA sigmoid form), against the shipped torch fallback ----
    from exllamav3.modules.gated_rmsnorm import GatedRMSNorm

    xn = (rs.randn(1, SQ, H, DV) * 2.0).astype(np.float32)
    zz = (rs.randn(1, SQ, H, DV) * 2.0).astype(np.float32)
    wn = (rs.randn(DV) * 0.1 + 1.0).astype(np.float32)
    gn = object.__new__(GatedRMSNorm)
    gn.rms_norm_eps = 1e-5
    gn.constant_bias = 0.0
    gn.groups = 1
    gn.gate_first = False
    gn.gate_activation = "sigmoid"
    gn.out_dtype = torch.float32
    gn.weight = torch.from_numpy(wn)
    with torch.no_grad():
        norm_t = gn.forward(torch.from_numpy(xn), {}, gate=torch.from_numpy(zz)).numpy()
    norm_np = gdn_ref.gated_rmsnorm(xn, zz, wn, eps=1e-5, activation="sigmoid")
    d_norm = np.abs(norm_np - norm_t).max() / max(np.abs(norm_t).max(), 1e-12)
    print(f"gated norm (sigmoid, after weight): rel {d_norm:.3e} "
          f"(scale {np.abs(norm_t).max():.4f})")

    # per-step sanity: decay must shrink the state when g is very negative
    g_zero = np.zeros_like(g)
    out_nogate, _ = gdn_ref.kda_recurrent(q, k, v, g_zero, beta)
    print(f"no-gate variant out scale {np.abs(out_nogate).max():.4f} "
          f"(vs gated {np.abs(out_np).max():.4f})")

    ok = (d < 5e-3 and same > 0.999 and d_conv < 1e-6 and d_norm < 1e-6
          and np.isfinite(out_np).all())
    print("GDN_RULE_VERIFY:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
