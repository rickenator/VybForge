"""P3 gate 1: HC_REF_VERIFY — numpy mHC reference vs exllamav3's own torch path
(`HyperConnection._mix_unfused` / `apply_`), on the REAL layer-3 hc_attn and
hc_ffn weights from the GLM-5.3 EXL3 checkpoint.

The module's own __init__/load() is bypassed (it needs a full Config/stc); the
object is given fn/base/scale directly and its unweighted RMSNorm is replaced by
the same definition, so the mix/apply MATH under test is the shipped source.

Usage: env -u PYTHONPATH ~/venvs/exl3/bin/python native/glm53/verify_hc.py
"""

import sys

import numpy as np
import torch

sys.path.insert(0, "/home/rick/Projects/VybForge/native/glm53")
import hc_ref  # noqa: E402
from exllamav3.modules.hyperconnections import HyperConnection  # noqa: E402

FIX = "/home/rick/Projects/VybForge/native/glm53/fixture"
H, HIDDEN = 4, 4096
RMS_EPS = 1e-6
SITES = ["hc_attn", "hc_ffn"]


def load(site):
    pre = f"model_language_model_layers_3_{site}"
    fn = np.fromfile(f"{FIX}/{pre}_fn.bin", dtype="<u2").reshape(24, H * HIDDEN)
    fn = (fn.astype(np.uint32) << 16).view(np.float32).astype(np.float32)   # bf16 -> f32
    base = np.fromfile(f"{FIX}/{pre}_base.bin", dtype="<f4")
    scale = np.fromfile(f"{FIX}/{pre}_scale.bin", dtype="<f4")
    return fn, base, scale


class NormShim:
    """Unweighted RMSNorm (unweighted=True -> no weight, constant bias 0 / scale 1)."""

    def forward(self, x, params):
        return x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + RMS_EPS)


def make_module(fn, base, scale, dev):
    m = object.__new__(HyperConnection)
    m.hc_mult = H
    m.hidden_size = HIDDEN
    m.sinkhorn_iters = hc_ref.SINKHORN_ITERS
    m.hc_eps = hc_ref.Hc_EPS
    m.rms_eps = RMS_EPS
    m.fn = torch.from_numpy(fn.astype(np.float32)).to(dev)
    m.fn_h = None
    m.base = torch.from_numpy(base.astype(np.float32)).to(dev)
    m.scale = torch.from_numpy(scale.astype(np.float32)).to(dev)
    m.norm = NormShim()
    return m


def to_dev(a, dev, contiguous=True):
    t = torch.from_numpy(a).reshape(1, *a.shape).to(dev)
    if not contiguous:
        # non-contiguous but logically identical: forces the torch fallback path
        t = t.permute(0, 1, 3, 2).permute(0, 1, 3, 2)
    return t


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"oracle device: {dev}")
    rs = np.random.RandomState(11)
    R = 3
    streams = (rs.randn(R, H, HIDDEN) * 0.5).astype(np.float32)
    y = (rs.randn(R, HIDDEN) * 0.3).astype(np.float32)

    worst = {}
    for site in SITES:
        fn, base, scale = load(site)
        post_np, comb_np, coll_np, pre_np = hc_ref.mix(streams, fn, base, scale, RMS_EPS)

        m = make_module(fn, base, scale, dev)
        xs = to_dev(streams, dev, contiguous=(dev.type == "cuda"))
        ys = torch.from_numpy(y).to(dev)
        with torch.no_grad():
            post_t, comb_t, coll_t = m._mix_unfused(xs, {})
            post_t = post_t.reshape(R, H).cpu().numpy()
            comb_t = comb_t.reshape(R, H, H).cpu().numpy()
            coll_t = coll_t.reshape(R, HIDDEN).cpu().numpy()

            x_np = hc_ref.apply_(streams, y, post_np, comb_np)
            x_out = m.apply_(xs,
                             ys,
                             torch.from_numpy(post_np).float().to(dev).reshape(1, R, H),
                             torch.from_numpy(comb_np).float().to(dev).reshape(1, R, H, H), {})
            x_t = x_out.reshape(R, H, HIDDEN).cpu().numpy()

        d_post = np.abs(post_np - post_t).max()
        d_comb = np.abs(comb_np - comb_t).max()
        # the fused ext path returns `collapsed` as fp16 (both block consumers cast it
        # immediately), so compare against the fp16-rounded reference
        coll_h = coll_np.astype(np.float16).astype(np.float64)
        d_coll16 = np.abs(coll_h - coll_t).max() / max(np.abs(coll_t).max(), 1e-12)
        d_coll32 = np.abs(coll_np - coll_t).max() / max(np.abs(coll_t).max(), 1e-12)
        n_bits = int((coll_h != coll_t.astype(np.float64)).sum())
        d_apply = np.abs(x_np - x_t).max() / max(np.abs(x_t).max(), 1e-12)

        rowsum = comb_np.sum(axis=-1)
        colsum = comb_np.sum(axis=-2)
        print(f"{site}: post |d| {d_post:.2e}  comb |d| {d_comb:.2e}  "
              f"collapsed rel {d_coll16:.2e} (fp16-rounded; fp32 {d_coll32:.2e}; "
              f"{n_bits}/{coll_t.size} elements differ)  apply rel {d_apply:.2e}")
        print(f"   comb rowsum [{rowsum.min():.5f} {rowsum.max():.5f}] "
              f"colsum [{colsum.min():.5f} {colsum.max():.5f}]  "
              f"pre [{pre_np.min():.4f} {pre_np.max():.4f}] "
              f"post [{post_np.min():.4f} {post_np.max():.4f}]")
        worst[site] = max(d_post, d_comb, d_apply)

    # post/comb/apply are fp32 exact-ish; collapsed is stored as fp16 by the fused path
    ok = all(v < 2e-5 for v in worst.values())
    print("HC_REF_VERIFY:", "OK" if ok else "FAIL", {k: f"{v:.2e}" for k, v in worst.items()})
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
