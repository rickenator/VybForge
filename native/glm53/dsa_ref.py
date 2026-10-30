"""GLM-5.3 DSA indexer + masked MLA attention — numpy, HF `glm5_next` semantics.

This is the CORRECTED indexer reference. `mla_ref.py` models exllamav3's
`QSAIndexer` (rope on a learned q/k + *mean* pooling of complete blocks); the
checkpoint this port has to run does not contain those tensors. Layer 3-44's
indexer weights are, verbatim from the shards:

    self_attn.indexer.wq_b.weight                 [32*128, 1536]   (from the q_a latent)
    self_attn.indexer.wk.weight                   [128, 4096]      (from hidden)
    self_attn.indexer.k_norm.{weight,bias}        [128]            LayerNorm, eps 1e-6
    self_attn.indexer.weights_proj.weight         [32, 4096]       (per-head score weight)
    self_attn.indexer.index_kpool_compress_ape    [4, 128]         per-slot pooling bias
    self_attn.indexer.index_kpool_compress_gate   [128, 4096]      learned pooling gate

that is, **learned gate-weighted pooling with a per-slot bias** and **no rope**, which
is what `transformers.models.glm5_next.modeling_glm5_next.Glm5NextTextIndexer` computes:

    q   = wq_b(q_resid)                          [S, 32, 128]
    k   = k_norm(wk(hidden))                     [S, 128]
    gsc = compress_gate(hidden)                  [S, 128]     (pooling logits per token)
    pool slots: complete kpool=4 groups of raw tokens, starting at the first valid token
    logits = gsc[slot] + ape[slot_i]             softmax over the 4 slots -> probs
    pool_k = Σ_slot probs * k[slot]              (invalid slots -inf -> probs 0)
    score  = relu(q · pool_k * (1/sqrt(128)))    [32, P]
    w      = weights_proj(hidden) * 32^-0.5      [32]
    idx_score = w · score                        [P] per query
    candidates = pools whose LAST token is visible (causal + valid) and fully valid
    select topk min(index_topk//4, P) pools -> expand to their raw token ids, -1 if invalid
    + append the visible incomplete tail (visible_count % 4 tokens after the last full pool)
    + pad with -1 to index_topk + 3 columns

`mla_ref.indexer_select` is retained only because `verify_mla.py` gates it against
exllamav3; it must NOT be used to run this checkpoint.

Usage (gate): `verify_dsa.py`.
"""

import numpy as np

HEADS, NOPE, VDIM, KV_LORA = 64, 256, 256, 512
INDEX_HEADS, INDEX_DIM = 32, 128
INDEX_KPOOL = 4
INDEX_TOPK = 2048
LN_EPS = 1e-6


def rms_norm(x, weight, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * np.asarray(weight, dtype=np.float64)


def layer_norm(x, weight, bias, eps=LN_EPS):
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * np.asarray(weight, dtype=np.float64) + np.asarray(bias, dtype=np.float64)


def indexer_topk(hidden, q_resid, iw, attention_mask=None, kpool=INDEX_KPOOL,
                 index_topk=INDEX_TOPK):
    """HF Glm5NextTextIndexer.forward (no cache, batch 1) -> int32 [S, index_topk+kpool-1].

    iw: dict with wq_b, wk, k_norm.weight, k_norm.bias, weights_proj,
        compress_ape [kpool,128], compress_gate [128,H]  (all (out,in) as stored).
    """
    hidden = np.asarray(hidden, dtype=np.float64)
    q_resid = np.asarray(q_resid, dtype=np.float64)
    s = hidden.shape[0]
    valid = np.ones(s, dtype=bool) if attention_mask is None else np.asarray(attention_mask, bool)

    q = (q_resid @ np.asarray(iw["wq_b"]).T).reshape(s, INDEX_HEADS, INDEX_DIM)
    k = layer_norm(hidden @ np.asarray(iw["wk"]).T, iw["k_norm.weight"], iw["k_norm.bias"])
    gsc = hidden @ np.asarray(iw["compress_gate"]).T                       # [S,128]

    # ---- pools of complete kpool groups, starting at the first valid token ----
    first_key = int(np.argmax(valid)) if valid.any() else s
    n_pools = (s + kpool - 1) // kpool
    offsets = np.arange(n_pools * kpool).reshape(n_pools, kpool)
    pool_idx = first_key + offsets                                          # [P,kpool]
    safe = np.clip(pool_idx, 0, s - 1)
    gk = k[safe]                                                           # keys per slot
    gg = gsc[safe]
    gval = valid[safe] & (pool_idx < s)                                     # [P,kpool]
    pool_valid = gval.all(-1)
    pool_idx_out = np.where(gval, pool_idx, -1)

    ape = np.asarray(iw["compress_ape"], dtype=np.float64)                  # [kpool,128]
    logits = gg + ape[None]
    logits = np.where(gval[..., None], logits, -np.inf)
    m = logits.max(-1, keepdims=True)
    e = np.exp(logits - np.where(np.isfinite(m), m, 0.0))
    e = np.where(np.isfinite(logits), e, 0.0)
    den = e.sum(-1, keepdims=True)
    probs = np.divide(e, den, out=np.zeros_like(e), where=den > 0)          # nan_to_num(softmax)
    pool_k = (probs * gk).sum(1)                                            # [P,128]

    # HF drops pools that are not fully valid (`keep = pool_valid.any(0)`), which also
    # narrows select_k — this decides the topk LAYOUT (pool block width vs tail offset)
    keep = pool_valid
    pool_k, pool_idx_out, pool_valid = pool_k[keep], pool_idx_out[keep], pool_valid[keep]
    if pool_k.shape[0] == 0:
        pool_k = np.zeros((0, INDEX_DIM))

    scores = np.maximum((q @ pool_k.T) * (INDEX_DIM ** -0.5), 0.0)          # [S,P]
    n_pools = pool_k.shape[0]
    w = (hidden @ np.asarray(iw["weights_proj"]).T) * (INDEX_HEADS ** -0.5)  # [S,32]
    index_scores = np.einsum("sh,shp->sp", w, scores)                       # [S,P]

    # ---- visibility: causal AND valid, tested at the pool's LAST token ----
    kv_pos = np.arange(s)
    causal = kv_pos[None, :] <= kv_pos[:, None]                             # [S(query),S(key)]
    visible = causal & valid[None, :]                                       # [S,S]
    pool_end = np.clip(pool_idx_out[..., -1], 0, s - 1)
    pool_visible = visible[:, pool_end]                                     # [S,P]
    candidates = pool_visible & pool_valid[None]

    index_scores = np.where(candidates, index_scores, -np.inf)
    select_k = min(index_topk // kpool, n_pools)
    order = np.argsort(-index_scores, axis=-1, kind="stable")[:, :select_k]  # [S,K]
    sel_valid = np.take_along_axis(candidates, order, axis=-1)
    sel_pools = np.take(pool_idx_out, order, axis=0)                        # [S,K,kpool]
    tok = sel_pools.reshape(s, -1)                                          # [S,K*kpool]
    ok = np.repeat(sel_valid, kpool, axis=-1)
    tok = np.where(ok, tok, -1)

    # ---- append the visible incomplete tail ----
    max_tail = kpool - 1
    width = index_topk + kpool - 1
    if max_tail > 0:
        vis_count = visible.sum(-1)                                         # [S]
        tail_count = vis_count % kpool
        tail_start = first_key + vis_count - tail_count
        tail_off = np.arange(max_tail)
        tail_idx = tail_start[:, None] + tail_off[None, :]
        tail_ok = (tail_off[None, :] < tail_count[:, None]) & (tail_idx < s)
        tail_vis = np.take_along_axis(visible, np.clip(tail_idx, 0, s - 1), axis=-1)
        tail_idx = np.where(tail_ok & tail_vis, tail_idx, -1)
        tok = np.concatenate([tok, tail_idx], axis=-1)

    out = np.full((s, width), -1, dtype=np.int32)
    n = min(width, tok.shape[-1])
    out[:, :n] = tok[:, :n]
    out = np.where(valid[:, None], out, -1)
    return out


def masked_attention(q, ckv, w_uk, w_uv, topk, scale=None):
    """MLA attention restricted to the indexer's selected tokens.

    q [S,64,256]; ckv [S,512]; w_uk/w_uv [64,512,256]; topk [S,K] (-1 = unused).
    """
    q = np.asarray(q, dtype=np.float64)
    ckv = np.asarray(ckv, dtype=np.float64)
    s = ckv.shape[0]
    sc = (NOPE ** -0.5) if scale is None else scale
    # w_* are [H, out, in] as stored (kv_b_proj is [H*(nope+v), kv_lora])
    k = np.einsum("sj,hij->hsi", ckv, np.asarray(w_uk, dtype=np.float64))   # [H,S,256]
    v = np.einsum("sj,hij->hsi", ckv, np.asarray(w_uv, dtype=np.float64))
    qh = q.transpose(1, 0, 2)                                              # [H,S,256]
    scores = np.einsum("hqd,hkd->hqk", qh, k) * sc                          # [H,S,S]
    keep = np.zeros((s, s), dtype=bool)
    sel = np.asarray(topk, dtype=np.int64)
    valid = sel >= 0
    rows = np.broadcast_to(np.arange(s)[:, None], sel.shape)
    keep[rows[valid], sel[valid]] = True
    scores = np.where(keep[None], scores, -np.inf)
    m = scores.max(-1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.exp(scores - m)
    den = e.sum(-1, keepdims=True)
    p = np.divide(e, den, out=np.zeros_like(e), where=den > 0)
    o = np.einsum("hqk,hkd->hqd", p, v).transpose(1, 0, 2)                  # [S,64,256]
    return o.reshape(s, HEADS * VDIM)


def layer_attention(h_a, w, iw, attention_mask=None):
    """h_a [S,H] (already input_layernorm'd) -> attention out [S,H], plus intermediates."""
    s = h_a.shape[0]
    q_resid = h_a @ np.asarray(w["q_a_proj"]).T
    q = (rms_norm(q_resid, w["q_a_layernorm"]) @ np.asarray(w["q_b_proj"]).T).reshape(s, HEADS, NOPE)
    ckv = rms_norm((h_a @ np.asarray(w["kv_a_proj_with_mqa"]).T)[:, :KV_LORA], w["kv_a_layernorm"])
    kv_b = np.asarray(w["kv_b_proj"]).reshape(HEADS, NOPE + VDIM, KV_LORA)
    uk, uv = kv_b[:, :NOPE, :], kv_b[:, NOPE:, :]
    topk = indexer_topk(h_a, q_resid, iw, attention_mask=attention_mask)
    o = masked_attention(q, ckv, uk, uv, topk)
    return ((o @ np.asarray(w["o_proj"]).T), topk,
            {"q_resid": q_resid, "qb": q.reshape(s, -1), "ckv": ckv, "o_pre": o})
