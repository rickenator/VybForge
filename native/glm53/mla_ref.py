"""GLM-5.3 MLA + DSA indexer reference, numpy.

Config (checkpoint): q_lora_rank 1536, kv_lora_rank 512, num_q_heads 64,
qk_nope_head_dim 256, qk_rope_head_dim **0** (no rope in the main attention at all),
v_head_dim 256. Indexer: index_n_heads 32, index_head_dim 128, index_topk 2048,
index_kpool 4 (compress_ratio), kpool_compress True, always_select_tail True, and
every layer is a "full" indexer (no shared selection).

MLA (dense equivalent of the absorbed form used at decode):

    q    = q_b( q_a_layernorm( q_a(x) ) )              # [.., 64, 256]
    ckv  = kv_a_layernorm( kv_a_proj(x)[..., :512] )   # cached latent
    k[h] = ckv @ W_UK[h]      (512 -> 256)             # W_UK from kv_b_proj
    v[h] = ckv @ W_UV[h]      (512 -> 256)
    o[h] = softmax(q[h]·k[h] / sqrt(256) + causal) @ v[h]
    out  = o_proj(o.reshape(.., 64*256))

DSA indexer (selection only — it never changes the attention math, it masks tokens):

    qk     = index_qk_proj(x) -> q [32*128], raw_k [128]   (raw_k cached unnormed/unroped)
    q      = q_layernorm(q) then rope
    pooled = mean over complete blocks of cr=4 raw keys, k_layernorm, rope at block start
    score  = relu(q · pooled).sum(heads) * (1/sqrt(128))
    blocks visible to a query = arange(nb) < (pos+1)//cr, top-k blocks, each expanded to its
    cr tokens, plus the tail (incomplete) block's tokens, dropped past pos, padded with -1.
"""

import numpy as np

HEADS = 64
NOPE = 256
VDIM = 256
KV_LORA = 512
Q_LORA = 1536
INDEX_HEADS = 32
INDEX_DIM = 128
INDEX_KPOOL = 4
INDEX_TOPK = 2048


def rms_norm(x, weight=None, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    n = x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
    return n if weight is None else n * np.asarray(weight, dtype=np.float64)


def mla_attention(q, ckv, w_uk, w_uv, scale=None, causal=True):
    """q [b,s,H,NOPE]; ckv [b,s,KV_LORA]; w_uk/w_uv [H, KV_LORA, NOPE/VDIM]."""
    q = np.asarray(q, dtype=np.float64)
    ckv = np.asarray(ckv, dtype=np.float64)
    b, s, h, d = q.shape
    sc = (d ** -0.5) if scale is None else scale
    k = np.einsum("bsj,hjd->bhsd", ckv, np.asarray(w_uk, dtype=np.float64))   # [b,H,s,d]
    v = np.einsum("bsj,hjd->bhsd", ckv, np.asarray(w_uv, dtype=np.float64))   # [b,H,s,dv]
    scores = np.einsum("bhsd,bhtd->bhst", q.transpose(0, 2, 1, 3), k) * sc
    if causal:
        tri = np.triu(np.ones((s, s), dtype=bool), 1)
        scores = np.where(tri[None, None], -np.inf, scores)
    m = scores.max(axis=-1, keepdims=True)
    e = np.exp(scores - m)
    p = e / e.sum(axis=-1, keepdims=True)
    o = np.einsum("bhst,bhtd->bhsd", p, v)                                   # [b,H,s,dv]
    return o.transpose(0, 2, 1, 3).reshape(b, s, h * v.shape[-1])


def mla_attention_absorbed(q, ckv, w_uk, w_uv, scale=None, causal=True):
    """Same result via the absorbed query (q·W_UK first), which is what decode runs."""
    q = np.asarray(q, dtype=np.float64)
    ckv = np.asarray(ckv, dtype=np.float64)
    b, s, h, d = q.shape
    sc = (d ** -0.5) if scale is None else scale
    q_lat = np.einsum("bhsd,hjd->bhsj", q.transpose(0, 2, 1, 3),
                      np.asarray(w_uk, dtype=np.float64))                    # [b,H,s,KV_LORA]
    scores = np.einsum("bhsj,btj->bhst", q_lat, ckv) * sc
    if causal:
        tri = np.triu(np.ones((s, s), dtype=bool), 1)
        scores = np.where(tri[None, None], -np.inf, scores)
    m = scores.max(axis=-1, keepdims=True)
    e = np.exp(scores - m)
    p = e / e.sum(axis=-1, keepdims=True)
    o_lat = np.einsum("bhst,btj->bhsj", p, ckv)                              # [b,H,s,KV_LORA]
    o = np.einsum("bhsj,hjd->bhsd", o_lat, np.asarray(w_uv, dtype=np.float64))
    return o.transpose(0, 2, 1, 3).reshape(b, s, h * o.shape[-1])


def indexer_pool(raw_k, cr=INDEX_KPOOL):
    """raw_k [b,T,dk] -> block means [b, T//cr, dk] (complete blocks only)."""
    raw_k = np.asarray(raw_k, dtype=np.float64)
    b, total, dk = raw_k.shape
    nb = total // cr
    if nb == 0:
        return np.zeros((b, 0, dk), dtype=np.float64)
    return raw_k[:, :nb * cr].reshape(b, nb, cr, dk).mean(axis=2)


def indexer_block_scores(q, pooled, scale=INDEX_DIM ** -0.5):
    """q [b,s,H,dk], pooled [b,nb,dk] -> [b,s,nb] (relu then sum over heads, scaled)."""
    s_ = np.einsum("bshd,bnd->bshn", np.asarray(q, dtype=np.float64),
                   np.asarray(pooled, dtype=np.float64))
    return np.maximum(s_, 0.0).sum(axis=2) * scale


def indexer_select(q_idx, pooled, past_len, block_topk, cr=INDEX_KPOOL,
                   pad_mult=32, scale=INDEX_DIM ** -0.5):
    """Mirror of QSAIndexer.select_indices_ref: -> int32 [bsz*seq, k_pad], -1 = unused."""
    q_idx = np.asarray(q_idx, dtype=np.float64)
    bsz, seq = q_idx.shape[:2]
    nb = pooled.shape[1]
    k_pad = -(-(block_topk * cr + cr - 1) // pad_mult) * pad_mult
    out = np.full((bsz, seq, k_pad), -1, dtype=np.int32)

    pos = past_len + np.arange(seq)
    nbq = (pos + 1) // cr
    scores = indexer_block_scores(q_idx, pooled, scale)                       # [b,seq,nb]
    scores = np.where(np.arange(nb)[None, None, :] >= nbq[None, :, None], -np.inf, scores)
    ksel = min(block_topk, nb)
    order = np.argsort(-scores, axis=-1, kind="stable")[..., :ksel]           # topk order
    vals = np.take_along_axis(scores, order, axis=-1)
    sel_ok = vals > -np.inf
    sel_tok = (order * cr)[..., None] + np.arange(cr)                         # [b,seq,ksel,cr]
    sel_tok = sel_tok.reshape(bsz, seq, ksel * cr)
    sel_ok = np.repeat(sel_ok, cr, axis=-1)
    tail = (nbq[:, None] * cr) + np.arange(cr)[None, :]                       # [seq,cr]
    tail = np.broadcast_to(tail[None], (bsz, seq, cr))
    tok = np.concatenate([sel_tok, tail], axis=2)
    ok = np.concatenate([sel_ok, np.ones_like(tail, dtype=bool)], axis=2)
    ok &= tok <= pos[None, :, None]
    L = tok.shape[2]
    out[:, :, :L] = np.where(ok, tok, -1)
    return out.reshape(bsz * seq, k_pad)
