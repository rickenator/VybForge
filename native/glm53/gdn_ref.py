"""GLM-5.3 linear-attention (KDA) reference, numpy.

Variant pinned from the checkpoint config (`linear_attn_config`):
  num_heads 64, head_dim 128 (both k and v), short_conv_kernel_size 4,
  gate_lower_bound -5.0, and every linear layer is KDA (kda_layers covers all 34).

Math transcribed from exllamav3 `modules/gated_delta_net.py` (KDA branch) and
`modules/gated_delta_net_fn/gated_delta_rule.py::torch_recurrent_kda` (the module's
own labelled reference for this variant), plus `modules/gated_rmsnorm.py`.

Per layer, with x [b, s, 4096]:

  qkv   = qkv_proj(x)                       # q|k|v, 3 x 64 heads x 128
  q,k,v = split(qkv); short causal depthwise conv (k=4) + silu
  z     = g_b_proj(g_a_proj(x))             # low-rank SIGNMOID output gate [b,s,64,128]
  beta  = sigmoid(b_proj(x)) * beta_scale
  f     = f_b_proj(f_a_proj(x))             # low-rank forget gate [b,s,64*128]
  g     = -5.0 * sigmoid((f + dt_bias) * exp(a_log))     # "safe gate" log-decay
  q̂ = l2norm(q) * dk^-0.5 ; k̂ = l2norm(k)            # eps 1e-6
  state (h, 128, 128): per step i
      state *= exp(g_i)                     # per-k-channel decay
      kv    = (state * k̂_i).sum(-2)
      state += k̂_i ⊗ ((v_i - kv) * beta_i)
      o_i   = (state * q̂_i).sum(-2)
  o → gated RMSNorm(o, z) → o_proj → hyper-connection apply
"""

import numpy as np

HEADS = 64
KDIM = 128
VDIM = 128
CONV_K = 4
GATE_LOWER_BOUND = -5.0
L2_EPS = 1e-6


def l2norm(x, eps=L2_EPS):
    return x / np.sqrt((x * x).sum(axis=-1, keepdims=True) + eps)


def kda_log_decay(f, dt_bias, a_log, lower_bound=GATE_LOWER_BOUND):
    """f [.., HEADS*KDIM] -> log decay [.., HEADS, KDIM] (per-k-channel, negative)."""
    decay = np.exp(np.asarray(a_log, dtype=np.float64)).reshape(*([1] * (f.ndim - 1)), HEADS, 1)
    gf = (np.asarray(f, dtype=np.float64).reshape(*f.shape[:-1], HEADS, KDIM)
          + np.asarray(dt_bias, dtype=np.float64).reshape(HEADS, KDIM))
    return (1.0 / (1.0 + np.exp(-(gf * decay)))) * lower_bound


def short_conv_silu(x, weight, k=CONV_K):
    """Causal depthwise conv over the token axis, then silu.

    x [b, s, C], weight [C, k] with weight[:, -1] applying to the current token
    (verified against the module in verify_gdn.py).
    """
    b, s, c = x.shape
    xp = np.concatenate([np.zeros((b, k - 1, c), x.dtype), x], axis=1)
    out = np.zeros_like(x, dtype=np.float64)
    for j in range(k):
        out += xp[:, j:j + s, :] * weight[None, None, :, j]
    return out * (1.0 / (1.0 + np.exp(-out)))


def kda_recurrent(q, k, v, g, beta, state=None):
    """q/k/v [b,s,h,d], g [b,s,h,dk] log-decay, beta [b,s,h]. -> (out [b,s,h,dv], state)."""
    b, s, h, dk = k.shape
    dv = v.shape[-1]
    q = l2norm(np.asarray(q, dtype=np.float64)) * (dk ** -0.5)
    k = l2norm(np.asarray(k, dtype=np.float64))
    v = np.asarray(v, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)

    st = np.zeros((b, h, dk, dv), dtype=np.float64) if state is None else state.astype(np.float64)
    out = np.empty((b, s, h, dv), dtype=np.float64)
    for i in range(s):
        st *= np.exp(g[:, i])[:, :, :, None]                 # (b,h,dk,1) per-channel decay
        kv = (st * k[:, i][:, :, :, None]).sum(axis=-2)      # (b,h,dv)
        delta = (v[:, i] - kv) * beta[:, i][:, :, None]
        st += k[:, i][:, :, :, None] * delta[:, :, None, :]
        out[:, i] = (st * q[:, i][:, :, :, None]).sum(axis=-2)
    return out, st


def gated_rmsnorm(x, gate, weight, eps=1e-5, activation="sigmoid"):
    """GLM's gated norm. KDA layers use activation='sigmoid' and apply it AFTER the
    weighted norm (`gated_rmsnorm.py`: "sigmoid (KDA) ... gate applied after the
    weighted norm, per the KDA reference"); Mamba2-style sites use silu before."""
    x = np.asarray(x, dtype=np.float64)
    gt = np.asarray(gate, dtype=np.float64)
    if activation == "sigmoid":
        n = x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
        if weight is not None:
            n = n * np.asarray(weight, dtype=np.float64)
        return n * (1.0 / (1.0 + np.exp(-gt)))
    n = x * (gt / (1.0 + np.exp(-gt)))                  # gate first (silu), then norm
    n = n / np.sqrt((n * n).mean(axis=-1, keepdims=True) + eps)
    if weight is not None:
        n = n * np.asarray(weight, dtype=np.float64)
    return n
