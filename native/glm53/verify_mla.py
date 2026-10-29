"""P3 gate: MLA + DSA indexer.

1. DSA_SELECT_VERIFY — numpy indexer selection vs exllamav3's own
   `QSAIndexer.select_indices_ref` (the shipped torch reference "for tests") at the
   real indexer shape (32 heads x 128, kpool 4, block_topk = index_topk/kpool).
2. MLA_ABSORB_VERIFY — the absorbed-query MLA (what decode runs) vs the explicit
   per-head k = ckv·W_UK / v = ckv·W_UV form, at real dims (64 heads, 512 latent,
   256 nope).

Usage: env -u PYTHONPATH ~/venvs/exl3/bin/python native/glm53/verify_mla.py
"""

import sys

import numpy as np
import torch

sys.path.insert(0, "/home/rick/Projects/VybForge/native/glm53")
import mla_ref  # noqa: E402
from exllamav3.modules.qsa_indexer import QSAIndexer  # noqa: E402

H, DIM = 32, 128
CR = 4
BLOCK_TOPK = 2048 // CR      # index_topk / kpool
QHEADS, NOPE, VDIM, KV_LORA = 64, 256, 256, 512


def gate_select():
    rs = np.random.RandomState(3)
    bsz, seq = 2, 24
    nb = 16                       # 64 tokens of history
    q_idx = (rs.randn(bsz, seq, H, DIM)).astype(np.float32)
    pooled = (rs.randn(bsz, nb, DIM)).astype(np.float32)
    past_len = 40                 # queries start at absolute position 40

    ref = QSAIndexer.select_indices_ref(
        _stub_indexer(), torch.from_numpy(q_idx), torch.from_numpy(pooled),
        past_len, batch_stride=0).numpy()
    mine = mla_ref.indexer_select(q_idx, pooled, past_len, BLOCK_TOPK, CR)
    same = (ref == mine).all()
    n_diff = int((ref != mine).sum())
    n_sel = int((ref >= 0).sum())
    print(f"select: {n_sel} tokens selected of {ref.size}; shapes ref {ref.shape} "
          f"mine {mine.shape}; identical {same} ({n_diff} differing entries)")
    return same


def _stub_indexer():
    """Only these attributes are touched by select_indices_ref / block_scores."""
    idx = object.__new__(QSAIndexer)
    idx.compress_ratio = CR
    idx.block_topk = BLOCK_TOPK
    idx.scale = float(DIM ** -0.5)
    return idx


def gate_mla():
    rs = np.random.RandomState(5)
    b, s = 1, 6
    q = (rs.randn(b, s, QHEADS, NOPE) * 0.3).astype(np.float32)
    ckv = (rs.randn(b, s, KV_LORA) * 0.5).astype(np.float32)
    w_uk = (rs.randn(QHEADS, KV_LORA, NOPE) * 0.02).astype(np.float32)
    w_uv = (rs.randn(QHEADS, KV_LORA, VDIM) * 0.02).astype(np.float32)

    a = mla_ref.mla_attention(q, ckv, w_uk, w_uv)
    b2 = mla_ref.mla_attention_absorbed(q, ckv, w_uk, w_uv)
    rel = np.abs(a - b2).max() / max(np.abs(a).max(), 1e-12)
    print(f"mla: shape {a.shape}  absorbed vs explicit rel {rel:.3e}  "
          f"|out| max {np.abs(a).max():.4f}")
    return rel < 1e-10


def main():
    ok_sel = gate_select()
    ok_mla = gate_mla()
    print("DSA_SELECT_VERIFY:", "OK" if ok_sel else "FAIL")
    print("MLA_ABSORB_VERIFY:", "OK" if ok_mla else "FAIL")
    return 0 if (ok_sel and ok_mla) else 1


if __name__ == "__main__":
    sys.exit(main())
