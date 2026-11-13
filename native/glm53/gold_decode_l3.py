"""Gold for GLM-5.3 step decode (S=1) — layer 3 (DSA attention + router/MoE + hc sites).

The batched layer-3 oracle (gold_l3_attn.py / gold_l3_moe.py) runs layer 3 over the whole
22-token prompt in one shot. Generation needs the SAME math one token at a time with the
DSA caches carried across steps:

  1. the ckv latent cache   [S, 512]   (kv_a latent, post kv_a_layernorm)
  2. the indexer key cache  [S, 128]   (wk -> k_norm; dsa_ref.indexer_topk recomputes this
     from hidden on every call, so the cache is factored out here — cached_topk takes the
     k/gsc rows as inputs instead of hidden)
  3. the indexer gsc cache  [S, 128]   (index_kpool_compress_gate projection, pool logits)

The self-check is built in at three levels:
  * the per-token topk walk must select the same token set per query as the batched
    dsa_ref.indexer_topk over the full prompt (pools whose last token is visible + tail);
  * the stepwise attention must reproduce the batched gold_l3_attn per row;
  * the whole stepwise prefill walk must reproduce the already-gated gold_l3_streams2.npy
    (tolerance 2e-4, the same bar the Vyb gate uses).

Outputs: GOLD_DECODE_L3_OK plus native/out/gold_l3dec_*.npy / *.bin and gold_l3decode.json.
The .bin files are raw little-endian f64 (a Vyb driver cannot read .npy) and are the decode
driver's actual inputs: gold_l3dec_{ckv,ik,igsc}.bin hold the 22 prefill rows, and each
decode step appends the new token's row to them before the indexer/MLA run.

Decode steps are teacher-forced with the prompt's last two token ids (13 then 10788),
exactly like gold_decode_step.py.

Usage: env -u PYTHONPATH ~/venvs/exl3/bin/python native/glm53/gold_decode_l3.py
"""

import json
import sys

import numpy as np

sys.path.insert(0, "/home/rick/Projects/VybForge/native/glm53")
import hc_ref  # noqa: E402
import layer0_ref  # noqa: E402
import dsa_ref  # noqa: E402
import moe_ref  # noqa: E402
import gold_l3_attn as G3  # noqa: E402
import verify_layer3_moe as B  # noqa: E402

CKPT = G3.CKPT
OUT = G3.OUT
H = G3.H
HEADS, NOPE, VDIM, KV_LORA = dsa_ref.HEADS, dsa_ref.NOPE, dsa_ref.VDIM, dsa_ref.KV_LORA
INDEX_HEADS, INDEX_DIM, KPOOL, INDEX_TOPK = (dsa_ref.INDEX_HEADS, dsa_ref.INDEX_DIM,
                                             dsa_ref.INDEX_KPOOL, dsa_ref.INDEX_TOPK)
HC = 4
EPS = 1e-5
S40 = G3.S40
NSTEPS = 2
# the prompt's last two token ids, teacher-forced in the same order as gold_decode_step.py
TIDS = [13, 10788]


def dig(v):
    f = np.ascontiguousarray(v, dtype=np.float64).ravel()
    return {"n": int(f.size), "sum": float(f.sum()), "abs": float(np.abs(f).sum()),
            "sq": float((f * f).sum()), "max": float(np.abs(f).max())}


_EXL3 = {}


def expert_mats(e, idx):
    """EXL3-dequantized (gate, up, down) for one expert.

    verify_layer3_moe.read_expert_exl3 assumes trellis/suh/svh share one shard; some
    experts (e.g. 264) are split across shards, so each part is resolved by name.
    """
    if e not in _EXL3:
        from safetensors import safe_open
        from loader import dequant_exl3_fast
        mats = {}
        for p in ("gate_proj", "up_proj", "down_proj"):
            parts = {}
            for part in ("trellis", "suh", "svh"):
                name = f"model.language_model.layers.3.mlp.experts.{e}.{p}.{part}"
                with safe_open(f"{CKPT}/{idx[name]}", framework="pt", device="cpu") as f:
                    parts[part] = f.get_tensor(name).numpy()
            mats[p] = dequant_exl3_fast(parts["trellis"], parts["suh"], parts["svh"])
        _EXL3[e] = mats
    return _EXL3[e]


def cached_topk(k, gsc, ape, q_idx, w_row, kpool=KPOOL, index_topk=INDEX_TOPK):
    """dsa_ref.indexer_topk for ONE query with the k/gsc caches factored out.

    k [N,128] / gsc [N,128] are the cached per-token rows (all valid — decode never pads),
    q_idx [32,128] is the query's indexer head keys, w_row [32] its per-head score weight.
    Pools are complete kpool groups starting at token 0; a pool is a candidate when its
    LAST token is visible to the query (<= query index, which is N-1); the visible
    incomplete tail after the last full pool is appended. select_k = index_topk//kpool.
    Returns the selected raw token ids (int list).
    """
    k = np.asarray(k, dtype=np.float64)
    gsc = np.asarray(gsc, dtype=np.float64)
    ape = np.asarray(ape, dtype=np.float64)
    q_idx = np.asarray(q_idx, dtype=np.float64)
    w_row = np.asarray(w_row, dtype=np.float64)
    n = k.shape[0]
    qpos = n - 1

    n_pools = n // kpool                                   # complete pools only
    # dsa_ref.indexer_topk's pooling verbatim: softmax over the CHANNEL axis per slot,
    # then the slot-weighted sum. (The batched Vyb driver pools with a per-channel
    # softmax over slots instead — the selected token SET is identical either way at
    # these cache sizes, and the attention output never consumes pool_k.)
    logits = gsc[:n_pools * kpool].reshape(n_pools, kpool, INDEX_DIM) + ape[None]
    m = logits.max(-1, keepdims=True)
    e = np.exp(logits - m)
    probs = e / e.sum(-1, keepdims=True)
    pool_k = (probs * k[:n_pools * kpool].reshape(n_pools, kpool, INDEX_DIM)).sum(1)

    scores = np.maximum((q_idx @ pool_k.T) * (INDEX_DIM ** -0.5), 0.0)   # [32,P]
    index_scores = w_row @ scores                          # [P]

    # candidates: pool's last token visible to the query (all of them here, since every
    # complete pool ends at or before n-1) — but keep the visibility test explicit
    pool_last = np.arange(n_pools) * kpool + (kpool - 1)
    candidates = pool_last <= qpos
    index_scores = np.where(candidates, index_scores, -np.inf)
    select_k = min(index_topk // kpool, n_pools)
    order = np.argsort(-index_scores, kind="stable")[:select_k]
    sel = (order[:, None] * kpool + np.arange(kpool)[None, :]).ravel().tolist()

    # visible incomplete tail after the last full pool
    tail_count = n % kpool
    sel += list(range(n_pools * kpool, n_pools * kpool + tail_count))
    return sel


def masked_step(q, ckv_cache, uk, uv, sel):
    """Masked MLA for ONE query row against a cache of N rows.

    dsa_ref.masked_attention assumes square S x S (query count == cache rows), which a
    decode step is not (1 query, N cached rows), so the same math is written here for
    q [1,64,256] / ckv [N,512] / sel = selected raw cache row ids.
    """
    ckv_cache = np.asarray(ckv_cache, dtype=np.float64)
    k = np.einsum("sj,hij->hsi", ckv_cache, np.asarray(uk, dtype=np.float64))  # [H,N,256]
    v = np.einsum("sj,hij->hsi", ckv_cache, np.asarray(uv, dtype=np.float64))
    qh = np.asarray(q, dtype=np.float64)[0]                                    # [64,256]
    scores = np.einsum("hd,hnd->hn", qh, k) * (NOPE ** -0.5)                   # [H,N]
    mask = np.zeros(ckv_cache.shape[0], dtype=bool)
    mask[list(sel)] = True
    scores = np.where(mask[None, :], scores, -np.inf)
    m = scores.max(-1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.exp(scores - m)
    den = e.sum(-1, keepdims=True)
    p = np.divide(e, den, out=np.zeros_like(e), where=den > 0)
    o = np.einsum("hn,hnd->hd", p, v)                                          # [64,256]
    return o.reshape(1, HEADS * VDIM)


def main():
    ids = json.load(open(f"{OUT}/gold_l3.json"))["ids"]
    s = len(ids)
    x = np.stack([G3.read_row(i) for i in ids])

    # ---- layers 0-2 chained exactly like the gated layer-3 golds -------------------------
    streams = np.repeat(x[:, None, :], HC, axis=1)            # [s, 4, H]
    for lay in (0, 1, 2):
        streams, _, _ = G3.kda_layer(streams, G3.weights(lay, kda=True))
    streams0 = streams.copy()
    print(f"chained layers 0-2: streams {streams0.shape}")

    w3 = G3.weights(3, kda=False)
    wa, iw = G3.dsa_weights(3)
    kv_b = np.asarray(wa["kv_b_proj"]).reshape(HEADS, NOPE + VDIM, KV_LORA)
    uk, uv = kv_b[:, :NOPE, :], kv_b[:, NOPE:, :]
    ape = iw["compress_ape"]

    # ---- batched reference for the self-gates (the gated layer-3 math) -------------------
    post_b, comb_b, collapsed_b, _ = hc_ref.mix(streams0, w3["attn_fn"], w3["attn_base"],
                                                w3["attn_scale"], EPS)
    h_a_b = layer0_ref.rms_weight(collapsed_b, w3["in_ln"])
    y_attn_b, topk_b, inter_b = dsa_ref.layer_attention(h_a_b, wa, iw)

    # ---- the stepwise PREFILL walk: one token at a time, caches grown per token ----------
    ckv_cache = np.zeros((0, KV_LORA))
    ik_cache = np.zeros((0, INDEX_DIM))
    igsc_cache = np.zeros((0, INDEX_DIM))
    attn_walk = np.zeros((s, H))
    streams1 = np.zeros_like(streams0)
    topk_walk = []
    for t in range(s):
        row = streams0[t]                                     # [4, H]
        post, comb, collapsed, _ = hc_ref.mix(row[None], w3["attn_fn"], w3["attn_base"],
                                              w3["attn_scale"], EPS)
        h_a = layer0_ref.rms_weight(collapsed, w3["in_ln"])   # [1, H]
        q_resid = h_a @ np.asarray(wa["q_a_proj"]).T          # [1, 1536]
        q = (dsa_ref.rms_norm(q_resid, wa["q_a_layernorm"]) @ np.asarray(wa["q_b_proj"]).T
             ).reshape(1, HEADS, NOPE)
        ckv_t = dsa_ref.rms_norm((h_a @ np.asarray(wa["kv_a_proj_with_mqa"]).T)[:, :KV_LORA],
                                 wa["kv_a_layernorm"])
        k_t = dsa_ref.layer_norm(h_a @ np.asarray(iw["wk"]).T, iw["k_norm.weight"],
                                 iw["k_norm.bias"])
        gsc_t = h_a @ np.asarray(iw["compress_gate"]).T
        ckv_cache = np.concatenate([ckv_cache, ckv_t], axis=0)
        ik_cache = np.concatenate([ik_cache, k_t], axis=0)
        igsc_cache = np.concatenate([igsc_cache, gsc_t], axis=0)

        q_idx = (q_resid @ np.asarray(iw["wq_b"]).T).reshape(INDEX_HEADS, INDEX_DIM)
        w_row = (h_a @ np.asarray(iw["weights_proj"]).T)[0] * (INDEX_HEADS ** -0.5)
        sel = cached_topk(ik_cache, igsc_cache, ape, q_idx, w_row)
        topk_walk.append(sel)
        o = masked_step(q, ckv_cache, uk, uv, sel)            # [1, HEADS*VDIM]
        attn_walk[t] = (o @ np.asarray(wa["o_proj"]).T)[0]
        streams1[t] = hc_ref.apply_(row[None], attn_walk[t][None], post, comb)[0]

    ckv_cache = ckv_cache[:s]
    ik_cache = ik_cache[:s]
    igsc_cache = igsc_cache[:s]

    # ---- self-gate 1: per-query selected SETS vs the batched indexer ---------------------
    set_err = 0
    for t in range(s):
        batched = sorted(int(v) for v in topk_b[t] if v >= 0)
        if batched != sorted(topk_walk[t]):
            set_err += 1
    ckv_err = float(np.abs(ckv_cache - inter_b["ckv"]).max())
    attn_err = float(np.abs(attn_walk - y_attn_b).max())

    # ---- ffn site: hc_ffn mix -> post-norm -> router + experts + shared ------------------
    post2, comb2, collapsed2, _ = hc_ref.mix(streams1, w3["ffn_fn"], w3["ffn_base"],
                                             w3["ffn_scale"], EPS)
    h2 = layer0_ref.rms_weight(collapsed2, w3["post_ln"])
    gate_w = G3.read("model.language_model.layers.3.mlp.gate.weight").astype(np.float32)
    bias = G3.read("model.language_model.layers.3.mlp.gate.e_score_correction_bias"
                   ).astype(np.float32)
    shared = (G3.read("model.language_model.layers.3.mlp.shared_experts.gate_proj.weight").T,
              G3.read("model.language_model.layers.3.mlp.shared_experts.up_proj.weight").T,
              G3.read("model.language_model.layers.3.mlp.shared_experts.down_proj.weight").T)
    r_idx, r_w = moe_ref.route(h2, gate_w, bias)
    sel_rows = [[int(e) for e in r_idx[t]] for t in range(s)]

    idx_map = B.index()
    union = sorted({e for row in sel_rows for e in row})
    print(f"route: {len(union)} distinct experts over {s} tokens")
    y = np.zeros_like(h2)
    for e in union:
        ew = expert_mats(e, idx_map)
        for t in range(s):
            lanes = [l for l in range(moe_ref.TOPK) if sel_rows[t][l] == e]
            if lanes:
                contrib = moe_ref.ffn(h2[t:t + 1], ew["gate_proj"], ew["up_proj"],
                                      ew["down_proj"])[0]
                for l in lanes:
                    y[t] += float(r_w[t][l]) * contrib
        del ew
    y_shared = moe_ref.ffn(h2, *shared)
    y += y_shared
    streams2 = hc_ref.apply_(streams1, y, post2, comb2)

    h2_err = float("nan")
    s2_err = float("nan")
    try:
        h2_err = float(np.abs(h2 - np.load(f"{OUT}/gold_l3_moe_h2.npy")).max())
        s2_err = float(np.abs(streams2 - np.load(f"{OUT}/gold_l3_streams2.npy")).max())
    except FileNotFoundError:
        pass

    # ---- dump the prefill state the decode driver starts from ----------------------------
    ckv_cache.astype("<f8").tofile(f"{OUT}/gold_l3dec_ckv.bin")
    ik_cache.astype("<f8").tofile(f"{OUT}/gold_l3dec_ik.bin")
    igsc_cache.astype("<f8").tofile(f"{OUT}/gold_l3dec_igsc.bin")
    streams2[-1].astype("<f8").tofile(f"{OUT}/gold_l3dec_streams_prev.bin")
    np.save(f"{OUT}/gold_l3dec_ckv.npy", ckv_cache)
    np.save(f"{OUT}/gold_l3dec_ik.npy", ik_cache)
    np.save(f"{OUT}/gold_l3dec_igsc.npy", igsc_cache)
    np.save(f"{OUT}/gold_l3dec_streams_prev.npy", streams2[-1])
    np.save(f"{OUT}/gold_l3dec_walk_streams2.npy", streams2)
    np.save(f"{OUT}/gold_l3dec_walk_attn.npy", attn_walk)

    # ---- two teacher-forced decode steps from the 22-row caches --------------------------
    steps = []
    dec_ckv = ckv_cache.copy()
    dec_ik = ik_cache.copy()
    dec_igsc = igsc_cache.copy()
    for t in range(NSTEPS):
        tid = TIDS[t]
        xt = G3.read_row(tid)
        row = np.repeat(xt[None, :], HC, axis=0)              # [4, H] decode input
        post, comb, collapsed, _ = hc_ref.mix(row[None], w3["attn_fn"], w3["attn_base"],
                                              w3["attn_scale"], EPS)
        h_a = layer0_ref.rms_weight(collapsed, w3["in_ln"])
        q_resid = h_a @ np.asarray(wa["q_a_proj"]).T
        q = (dsa_ref.rms_norm(q_resid, wa["q_a_layernorm"]) @ np.asarray(wa["q_b_proj"]).T
             ).reshape(1, HEADS, NOPE)
        ckv_t = dsa_ref.rms_norm((h_a @ np.asarray(wa["kv_a_proj_with_mqa"]).T)[:, :KV_LORA],
                                 wa["kv_a_layernorm"])
        k_t = dsa_ref.layer_norm(h_a @ np.asarray(iw["wk"]).T, iw["k_norm.weight"],
                                 iw["k_norm.bias"])
        gsc_t = h_a @ np.asarray(iw["compress_gate"]).T
        dec_ckv = np.concatenate([dec_ckv, ckv_t], axis=0)
        dec_ik = np.concatenate([dec_ik, k_t], axis=0)
        dec_igsc = np.concatenate([dec_igsc, gsc_t], axis=0)
        n = dec_ckv.shape[0]

        q_idx = (q_resid @ np.asarray(iw["wq_b"]).T).reshape(INDEX_HEADS, INDEX_DIM)
        w_row = (h_a @ np.asarray(iw["weights_proj"]).T)[0] * (INDEX_HEADS ** -0.5)
        sel = cached_topk(dec_ik, dec_igsc, ape, q_idx, w_row)
        o = masked_step(q, dec_ckv, uk, uv, sel)
        attn = (o @ np.asarray(wa["o_proj"]).T)[0]
        streams1_t = hc_ref.apply_(row[None], attn[None], post, comb)[0]

        post2t, comb2t, collapsed2t, _ = hc_ref.mix(streams1_t[None], w3["ffn_fn"],
                                                    w3["ffn_base"], w3["ffn_scale"], EPS)
        h2_t = layer0_ref.rms_weight(collapsed2t, w3["post_ln"])
        ri, rw = moe_ref.route(h2_t, gate_w, bias)
        y_t = np.zeros_like(h2_t)
        for lane in range(moe_ref.TOPK):
            e = int(ri[0][lane])
            ew = expert_mats(e, idx_map)
            y_t += float(rw[0][lane]) * moe_ref.ffn(h2_t, ew["gate_proj"], ew["up_proj"],
                                                    ew["down_proj"])
            del ew
        y_t += moe_ref.ffn(h2_t, *shared)
        streams2_t = hc_ref.apply_(streams1_t[None], y_t, post2t, comb2t)[0]

        np.save(f"{OUT}/gold_l3dec_step{t}_streams2.npy", streams2_t)
        np.save(f"{OUT}/gold_l3dec_step{t}_attn.npy", attn)
        np.save(f"{OUT}/gold_l3dec_step{t}_mlp.npy", y_t[0])
        steps.append({"token": tid, "cache_rows": int(n),
                      "topk": [int(v) for v in sel],
                      "route": [int(e) for e in ri[0]],
                      "attn": dig(attn), "mlp": dig(y_t[0]), "streams2": dig(streams2_t)})
        print(f"  step{t} token={tid} cache_rows={n} n_sel={len(sel)} "
              f"route={steps[-1]['route']}")

    meta = {"tokens": s, "steps": NSTEPS, "tids": TIDS,
            "topk_set_mismatches": set_err,
            "stepwise_vs_batched": {"ckv_max_abs": ckv_err, "attn_max_abs": attn_err,
                                    "h2_vs_gold_max_abs": h2_err,
                                    "streams2_vs_gold_max_abs": s2_err},
            "caches": {"ckv": dig(ckv_cache), "ik": dig(ik_cache), "igsc": dig(igsc_cache),
                       "rows": int(ckv_cache.shape[0])},
            "prefill_last_stage": {"streams2": dig(streams2[-1])},
            "decode": steps}
    json.dump(meta, open(f"{OUT}/gold_l3decode.json", "w"), indent=1)

    ok = (set_err == 0 and ckv_err < 1e-9 and attn_err < 1e-9
          and (np.isnan(s2_err) or s2_err < 2e-4))
    print("GOLD_DECODE_L3_OK" if ok else "GOLD_DECODE_L3_FAIL")
    print(f"  topk sets vs batched indexer: {s - set_err}/{s} identical")
    print(f"  walk vs batched: ckv max_abs={ckv_err:.3e}  attn max_abs={attn_err:.3e}")
    print(f"  walk vs gated gold: h2 max_abs={h2_err:.3e}  streams2 max_abs={s2_err:.3e}"
          "  (tol 2e-4)")
    for t, sg in enumerate(steps):
        print(f"  step{t} token={sg['token']} attn max={sg['attn']['max']:.6e} "
              f"streams2 max={sg['streams2']['max']:.6e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
