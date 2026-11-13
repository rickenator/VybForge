"""GLM-5.3 MoE block reference (P2 in doc/ROADMAP-GLM53-ON-VYB.md).

Semantics pinned from exllamav3 (the engine that serves this checkpoint):
  modules/block_sparse_mlp_routing.py:_routing_ds3_torch  (the shipped reference
  the CUDA router kernel is tested against) and modules/mlp.py:gated_act_slopes
  for the swiglu_limit clamp.

Routing (scoring_func=sigmoid, topk_method=noaux_tc, n_group=topk_group=1):
  logits  = y @ gate.weight^T                      # [S, 288], fp32
  scores  = sigmoid(logits)
  choice  = scores + e_score_correction_bias       # bias used for SELECTION only
  idx     = top8(choice)                           # no group step when n_group=1
  w       = scores[idx] / (sum + 1e-20) * routed_scaling_factor   # RAW scores
Expert FFN (per selected expert, clamped SwiGLU, limit = swiglu_limit):
  g = gate_proj(x); u = up_proj(x)
  g = min(g, LIMIT); u = clamp(u, -LIMIT, LIMIT)
  e = (silu(g) * u) @ down_proj
  y_out = sum_i w_i * e_i + shared_expert(x)        # shared expert uses the same clamp
"""

import os

import numpy as np

HIDDEN = 4096
INTER = 2048
N_EXPERTS = 288
TOPK = 8
SCALE = 2.5
SWIGLU_LIMIT = 10.0

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fixture")


def _load(name, dtype, shape):
    p = os.path.join(FIXTURE, name)
    a = np.fromfile(p, dtype=dtype)
    return a.reshape(shape)


def load_router():
    """Returns (gate W as f32 [N_EXPERTS, HIDDEN], bias f32 [N_EXPERTS])."""
    w = _load("layers_3_mlp_gate_weight.bin", "<u2", (N_EXPERTS, HIDDEN))
    bias = _load("layers_3_mlp_gate_e_score_correction_bias.bin", "<f4", (N_EXPERTS,))
    return bf16_to_f32(w).astype(np.float32), bias.astype(np.float32)


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    """bf16 = top 16 bits of f32."""
    return (u16.astype(np.uint32) << 16).view(np.float32)


def load_shared():
    """Checkpoint stores [out, in] (HuggingFace convention); return (in, out)."""
    g = _load("layers_3_mlp_shared_experts_gate_proj_weight.bin", "<u2", (INTER, HIDDEN))
    u = _load("layers_3_mlp_shared_experts_up_proj_weight.bin", "<u2", (INTER, HIDDEN))
    d = _load("layers_3_mlp_shared_experts_down_proj_weight.bin", "<u2", (HIDDEN, INTER))
    return (bf16_to_f32(g).astype(np.float64).T,
            bf16_to_f32(u).astype(np.float64).T,
            bf16_to_f32(d).astype(np.float64).T)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def route(y, gate_w, bias, topk: int = TOPK):
    """y: [S, HIDDEN] float32/64; gate_w f32 [N_EXPERTS, HIDDEN].
    -> (idx [S,topk], weights [S,topk])."""
    logits = y.astype(np.float64) @ gate_w.astype(np.float64).T
    scores = sigmoid(logits)
    choice = scores + bias.astype(np.float64)[None, :]
    idx = np.argsort(-choice, axis=-1)[:, :topk]
    w = np.take_along_axis(scores, idx, axis=-1)
    w = w / (w.sum(axis=-1, keepdims=True) + 1e-20)
    w = w * SCALE
    return idx, w


def swiglu(g, u, limit: float = SWIGLU_LIMIT):
    g = np.minimum(g, limit)
    u = np.clip(u, -limit, limit)
    return (g * sigmoid(g)) * u


def ffn(x, wg, wu, wd):
    """x: [S, HIDDEN]; wg/wu [HIDDEN, INTER]; wd [INTER, HIDDEN] (in,out layout)."""
    return swiglu(x @ wg, x @ wu) @ wd


def moe_block(y, gate_w, bias, expert_w, shared_w, topk: int = TOPK):
    """y: [S, HIDDEN]. expert_w: dict index -> (wg, wu, wd) in (in,out) layout.

    The expert FFN is computed once per DISTINCT weight set (cached by the tuple's
    identity): in the current fixture every lane maps to expert 0's weights, and in
    general a lane's FFN depends only on the weights it is given.
    """
    idx, w = route(y, gate_w, bias, topk)
    S = y.shape[0]
    out = np.zeros((S, HIDDEN), dtype=np.float64)
    for s in range(S):
        cache = {}
        for lane in range(topk):
            e = int(idx[s, lane])
            wg, wu, wd = expert_w[e]
            key = (id(wg), id(wu), id(wd))
            if key not in cache:
                cache[key] = ffn(y[s:s + 1], wg, wu, wd)[0]
            out[s] += w[s, lane] * cache[key]
    out += ffn(y, *shared_w)
    return out, idx, w
