"""GLM-5.3 multi-token hyper-connections (mHC), numpy reference.

Semantics transcribed from exllamav3/modules/hyperconnections.py `_mix_unfused` and
`apply_` (the torch fallback path, which is the readable definition of the fused
ext.hc_mix kernel). Per sublayer site (attn and ffn) the layer carries
`{key}_fn` [ (2+H)*H , H*hidden ] bf16, `{key}_base` [ (2+H)*H ] f32 and
`{key}_scale` [3] f32 (three scalars: pre/post/comb).

  flat = rms_norm_unweighted(streams.reshape(R, H*D))   # mHC streams are fp32
  mix  = flat @ fnᵀ                                     # -> pre(H), post(H), comb(H*H)
  pre  = sigmoid(pre_w*pre_s + pre_b) + hc_eps
  post = 2 * sigmoid(post_w*post_s + post_b)
  comb = comb_w*comb_s + comb_b ; softmax(-1) + hc_eps
  comb = comb / (colsum + hc_eps) ; then (sinkhorn_iters-1) x {rowsum, colsum} normalise
  collapsed = Σ_h pre_h * stream_h                     # what the sublayer consumes
  x ← post ⊗ y + combᵀ x                               # residual update (apply_)
"""

import numpy as np

H = 4                 # hc_mult
HIDDEN = 4096
Hc_EPS = 0.000001     # hc_eps (config hc_eps)
SINKHORN_ITERS = 20      # config: hc_sinkhorn_iters = 20


def rms_norm_unweighted(x, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def mix(streams, fn, base, scale, rms_eps, hc_eps=Hc_EPS, sinkhorn_iters=SINKHORN_ITERS):
    """streams (R, H, D) fp32 -> (post (R,H), comb (R,H,H), collapsed (R,D))."""
    R, hc, D = streams.shape
    assert hc == H
    flat = rms_norm_unweighted(streams.reshape(R, hc * D).astype(np.float64), rms_eps)
    mv = flat @ fn.astype(np.float64).T                  # (R, 24)
    pre_w = mv[:, :hc]
    post_w = mv[:, hc:2 * hc]
    comb_w = mv[:, 2 * hc:].reshape(R, hc, hc)
    pre_b = base[:hc]
    post_b = base[hc:2 * hc]
    comb_b = base[2 * hc:].reshape(hc, hc)
    pre_s, post_s, comb_s = [float(s) for s in scale]

    pre = 1.0 / (1.0 + np.exp(-(pre_w * pre_s + pre_b))) + hc_eps
    post = 2.0 * (1.0 / (1.0 + np.exp(-(post_w * post_s + post_b))))
    comb = comb_w * comb_s + comb_b
    e = np.exp(comb - comb.max(axis=-1, keepdims=True))
    comb = e / e.sum(axis=-1, keepdims=True)
    comb = comb + hc_eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + hc_eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + hc_eps)

    collapsed = (pre[:, :, None] * streams.astype(np.float64)).sum(axis=1)
    return post, comb, collapsed, pre


def apply_(x, y, post, comb):
    """x (R,H,D) streams, y (R,D) sublayer output -> post ⊗ y + combᵀ x."""
    return post[:, :, None] * y.astype(np.float64)[:, None, :] + \
        np.einsum("rjh,rjd->rhd", comb.astype(np.float64), x.astype(np.float64))
