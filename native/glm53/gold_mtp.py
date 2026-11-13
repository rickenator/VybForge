"""Gold for the GLM-5.3-Flash MTP head (layer 45) — numpy fp64 oracle for DECODE_MTP_MATCH.

Reads the chained decode driver's dumps from native/out/decchain_vyb.txt:
    mtptrunk{r}   16384 values = the layer-44 streams2 [4, 4096] for MTP row r
                  (rows 0..19 are the phase-1 prefill tokens, rows 20..20+NSTEPS-1 the
                  teacher-forced decode tokens 20, 21)
    mtphin{r}     4096 values = the driver's eh_proj output for row r (gate A)
    mtpx{r}       4096 values = the driver's post-layer-45 hidden for row r (gate B)
    mtplogits{t}  154880 values = the driver's final-step lm_head logits (gate C)

and recomputes everything downstream of the trunk with real checkpoint weights:

  1. trunk state = mean over the 4 layer-44 streams -> model.language_model.norm
     (plain RMSNorm, eps 1e-5) -> layers.45.hnorm (plain RMSNorm).
     NOTE: the handoff pinned hnorm/enorm with constant_bias=1.0, but the shipped
     exllamav3 GLM-5.3 code (architecture/glm5_next_mtp.py) explicitly passes
     constant_bias=0.0 to Qwen3_5MTPInputLayer, and HF's generic MtpLayer uses plain
     RMSNorm as well — so the scale is x_norm * w with NO +1. That convention is used
     here and in the driver.
  2. e = layers.45.enorm(embed(next_token)), same plain scale.
  3. concat [enorm(emb), hnorm(h)] -> eh_proj (BF16 [4096, 8192] out-in) = the MTP input.
     (Concat order pinned from the shipped impl: embedding first.)
  4. layer 45 as a plain residual block (no hyper-connections): input_layernorm ->
     DSA attention (no rope, own ckv/indexer-k/indexer-gsc cache primed by running the
     MTP rows 0..N-1 in order — rows carry causal visibility) -> residual ->
     post_attention_layernorm -> 288-expert MoE (dots router, sigmoid + noaux_tc bias,
     top-8, renorm * 2.5) -> residual.
  5. shared_head.norm (plain RMSNorm) -> lm_head (untied, BF16 [154880, 4096]) ->
     argmax = draft token.

Routed experts are dequantized with loader.dequant_exl3_fast (fp16 butterflies), the
same convention the gated L3 oracles use, so the honest tolerance vs the Vyb driver is
the 2e-4 class on post-MoE arrays (the EXL3 fp16 noise), tighter on everything else.

Outputs: native/out/gold_mtp_hin.npy [N,4096], gold_mtp_x.npy [N,4096],
gold_mtp_hidden.npy [4096] (last row), gold_mtp_logits.npy [154880],
gold_mtp.json (top-1 + digests). The verifier (verify_decode_mtp.py) compares the
driver's dumps element-wise and prints DECODE_MTP_MATCH.

Usage: env -u PYTHONPATH ~/venvs/exl3/bin/python native/glm53/gold_mtp.py
"""

import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, "/home/rick/Projects/VybForge/native/glm53")
import dsa_ref  # noqa: E402
import moe_ref  # noqa: E402
import gold_l3_attn as G3  # noqa: E402
from loader import dequant_exl3_fast  # noqa: E402

CKPT = G3.CKPT
OUT = G3.OUT
H = 4096
HEADS, NOPE, VDIM, KV_LORA = dsa_ref.HEADS, dsa_ref.NOPE, dsa_ref.VDIM, dsa_ref.KV_LORA
INDEX_HEADS, INDEX_DIM, KPOOL, INDEX_TOPK = (dsa_ref.INDEX_HEADS, dsa_ref.INDEX_DIM,
                                             dsa_ref.INDEX_KPOOL, dsa_ref.INDEX_TOPK)
EPS = 1e-5
VOCAB = 154880
S40 = G3.S40
PRE45 = "model.language_model.layers.45."


def dig(v):
    f = np.ascontiguousarray(v, dtype=np.float64).ravel()
    return {"n": int(f.size), "sum": float(f.sum()), "abs": float(np.abs(f).sum()),
            "sq": float((f * f).sum()), "max": float(np.abs(f).max())}


def read_dumps(path, prefix, n_expected=None):
    """Parse `prefix<k> <ints>` dumps lines -> {k: np.float64 array / S40}."""
    rx = re.compile(r"^" + re.escape(prefix) + r"(\d+) (.*)$")
    out = {}
    for line in open(path):
        m = rx.match(line.strip())
        if m:
            out[int(m.group(1))] = np.array([int(x) for x in m.group(2).split()],
                                            dtype=np.float64) / S40
    return out


def read(name):
    return G3.read(name)


_EXL3 = {}


def expert_mats(e, idx):
    """EXL3-dequantized (gate, up, down) for one layer-45 expert, cached."""
    if e not in _EXL3:
        from safetensors import safe_open
        mats = {}
        for p in ("gate_proj", "up_proj", "down_proj"):
            parts = {}
            for part in ("trellis", "suh", "svh"):
                name = f"{PRE45}mlp.experts.{e}.{p}.{part}"
                with safe_open(f"{CKPT}/{idx[name]}", framework="pt", device="cpu") as f:
                    parts[part] = f.get_tensor(name).numpy()
            mats[p] = dequant_exl3_fast(parts["trellis"], parts["suh"], parts["svh"])
        _EXL3[e] = mats
        print(f"    deq expert {e}", flush=True)
    return _EXL3[e]


def cached_topk(k, gsc, ape, q_idx, w_row, n_valid, kpool=KPOOL, index_topk=INDEX_TOPK):
    """Indexer pool + top-k for ONE query over an n_valid-row cache.

    Same math as gold_decode_l3.cached_topk / dsa_ref.indexer_topk; the driver's pool
    loop (per-channel softmax over the 4 slots) is mirrored exactly.
    """
    k = np.asarray(k, dtype=np.float64)[:n_valid]
    gsc = np.asarray(gsc, dtype=np.float64)[:n_valid]
    n = n_valid
    n_pools = n // kpool
    logits = gsc[:n_pools * kpool].reshape(n_pools, kpool, INDEX_DIM) + ape[None]
    m = logits.max(-1, keepdims=True)
    e = np.exp(logits - m)
    probs = e / e.sum(-1, keepdims=True)
    pool_k = (probs * k[:n_pools * kpool].reshape(n_pools, kpool, INDEX_DIM)).sum(1)
    scores = np.maximum((q_idx @ pool_k.T) * (INDEX_DIM ** -0.5), 0.0)
    index_scores = w_row @ scores
    pool_last = np.arange(n_pools) * kpool + (kpool - 1)
    candidates = pool_last <= n - 1
    index_scores = np.where(candidates, index_scores, -np.inf)
    select_k = min(index_topk // kpool, n_pools)
    order = np.argsort(-index_scores, kind="stable")[:select_k]
    sel = (order[:, None] * kpool + np.arange(kpool)[None, :]).ravel().tolist()
    tail = n % kpool
    sel += list(range(n_pools * kpool, n_pools * kpool + tail))
    return sel


def main():
    nsteps = int(os.environ.get("VYBFORGE_NSTEPS", "2"))
    dec_path = f"{OUT}/decchain_vyb.txt"

    trunk = read_dumps(dec_path, "mtptrunk")
    nrows = 20 + nsteps
    missing = [r for r in range(nrows) if r not in trunk]
    if missing:
        print(f"GOLD_MTP_FAIL: mtptrunk dumps missing rows {missing}")
        return 1
    print(f"read {nrows} mtptrunk rows from {dec_path}")

    # ---- weights ------------------------------------------------------------
    trunk_norm_w = read("model.language_model.norm.weight")
    hnorm_w = read(PRE45 + "hnorm.weight")
    enorm_w = read(PRE45 + "enorm.weight")
    eh = read(PRE45 + "eh_proj.weight")                      # [4096, 8192] out-in
    in_ln_w = read(PRE45 + "input_layernorm.weight")
    post_ln_w = read(PRE45 + "post_attention_layernorm.weight")
    shn_w = read(PRE45 + "shared_head.norm.weight")
    gate_w = read(PRE45 + "mlp.gate.weight")
    gate_bias = read(PRE45 + "mlp.gate.e_score_correction_bias")
    shared = (read(PRE45 + "mlp.shared_experts.gate_proj.weight").T,
              read(PRE45 + "mlp.shared_experts.up_proj.weight").T,
              read(PRE45 + "mlp.shared_experts.down_proj.weight").T)
    lm_head = read("lm_head.weight")                         # [154880, 4096] out-in
    wa, iw = G3.dsa_weights(45)
    kv_b = np.asarray(wa["kv_b_proj"]).reshape(HEADS, NOPE + VDIM, KV_LORA)
    uk, uv = kv_b[:, :NOPE, :], kv_b[:, NOPE:, :]
    ape = iw["compress_ape"]
    idx_map = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]

    # embedding rows for every MTP row's token (prompt ids 0..19 + decode 20..)
    from tokenizers import Tokenizer
    ids = json.load(open(f"{OUT}/gold_l3.json"))["ids"]
    emb_all = G3.read_row  # function
    dec_tids = ids[20:20 + nsteps]
    row_tids = ids[:20] + dec_tids
    print(f"row tokens: {row_tids}")

    def mtp_input(tvec, tid):
        """Steps 1-3: dumped trunk state (the mean over the 4 layer-44 streams, [H])
        -> eh_proj output [H]."""
        tvec = np.asarray(tvec, dtype=np.float64).ravel()
        h = dsa_ref.rms_norm(tvec, trunk_norm_w, EPS)
        h = dsa_ref.rms_norm(h, hnorm_w, EPS)
        e = dsa_ref.rms_norm(emb_all(tid), enorm_w, EPS)
        return (np.concatenate([e, h]) @ eh.T)

    # ---- gate A: input composition vs the driver's mtphin --------------------
    hin_g = np.stack([mtp_input(trunk[r], row_tids[r]) for r in range(nrows)])
    drv_hin = read_dumps(dec_path, "mtphin")
    gate_a = 0.0
    for r in range(nrows):
        if r in drv_hin:
            gate_a = max(gate_a, float(np.abs(hin_g[r] - drv_hin[r]).max()))
    print(f"gate A (eh_proj input) worst max_abs {gate_a:.3e}")

    # ---- layer 45 over all rows, own cache primed in order --------------------
    ckv_cache = np.zeros((0, KV_LORA))
    ik_cache = np.zeros((0, INDEX_DIM))
    gsc_cache = np.zeros((0, INDEX_DIM))
    x_out = np.zeros((nrows, H))
    route_sel = []
    for r in range(nrows):
        hin = hin_g[r:r + 1]
        h_a = dsa_ref.rms_norm(hin, in_ln_w, EPS)                       # [1, H]
        q_resid = h_a @ np.asarray(wa["q_a_proj"]).T                    # [1, 1536]
        q = (dsa_ref.rms_norm(q_resid, wa["q_a_layernorm"]) @ np.asarray(wa["q_b_proj"]).T
             ).reshape(1, HEADS, NOPE)
        ckv_t = dsa_ref.rms_norm((h_a @ np.asarray(wa["kv_a_proj_with_mqa"]).T)[:, :KV_LORA],
                                 wa["kv_a_layernorm"])
        k_t = dsa_ref.layer_norm(h_a @ np.asarray(iw["wk"]).T,
                                 iw["k_norm.weight"], iw["k_norm.bias"])
        gsc_t = h_a @ np.asarray(iw["compress_gate"]).T
        ckv_cache = np.concatenate([ckv_cache, ckv_t], axis=0)
        ik_cache = np.concatenate([ik_cache, k_t], axis=0)
        gsc_cache = np.concatenate([gsc_cache, gsc_t], axis=0)
        n = ckv_cache.shape[0]
        q_idx = (q_resid @ np.asarray(iw["wq_b"]).T).reshape(INDEX_HEADS, INDEX_DIM)
        w_row = (h_a @ np.asarray(iw["weights_proj"]).T)[0] * (INDEX_HEADS ** -0.5)
        sel = cached_topk(ik_cache, gsc_cache, ape, q_idx, w_row, n)
        # masked MLA over the selected rows (gold_decode_l3.masked_step math)
        ksel = np.einsum("sj,hij->hsi", ckv_cache, uk)                  # [H,N,NOPE]
        vsel = np.einsum("sj,hij->hsi", ckv_cache, uv)
        qh = q[0]
        scores = np.einsum("hd,hnd->hn", qh, ksel) * (NOPE ** -0.5)
        mask = np.zeros(n, dtype=bool)
        mask[list(sel)] = True
        scores = np.where(mask[None, :], scores, -np.inf)
        m = scores.max(-1, keepdims=True)
        m = np.where(np.isfinite(m), m, 0.0)
        ee = np.exp(scores - m)
        den = ee.sum(-1, keepdims=True)
        p = np.divide(ee, den, out=np.zeros_like(ee), where=den > 0)
        o = np.einsum("hn,hnd->hd", p, vsel)
        attn = (o.reshape(1, HEADS * VDIM) @ np.asarray(wa["o_proj"]).T)
        x1 = hin + attn                                                  # plain residual
        h2 = dsa_ref.rms_norm(x1, post_ln_w, EPS)
        r_idx, r_w = moe_ref.route(h2, gate_w, gate_bias)
        route_sel.append([int(e) for e in r_idx[0]])
        y = np.zeros_like(h2)
        for lane in range(moe_ref.TOPK):
            e = int(r_idx[0][lane])
            ew = expert_mats(e, idx_map)
            y += float(r_w[0][lane]) * moe_ref.ffn(h2, ew["gate_proj"], ew["up_proj"],
                                                   ew["down_proj"])
            del ew
        y += moe_ref.ffn(h2, *shared)
        x_out[r] = (x1 + y)[0]
        print(f"  row {r} token={row_tids[r]} cache_rows={n} n_sel={len(sel)} "
              f"route={route_sel[-1]}", flush=True)

    # ---- final head on the LAST row ------------------------------------------
    fin = dsa_ref.rms_norm(x_out[-1], shn_w, EPS)
    logits = fin @ lm_head.T
    top1 = int(np.argmax(logits))
    top5 = np.argsort(-logits)[:5].tolist()

    np.save(f"{OUT}/gold_mtp_hin.npy", hin_g)
    np.save(f"{OUT}/gold_mtp_x.npy", x_out)
    np.save(f"{OUT}/gold_mtp_hidden.npy", x_out[-1])
    np.save(f"{OUT}/gold_mtp_logits.npy", logits)
    meta = {"nrows": nrows, "nsteps": nsteps, "row_tids": row_tids,
            "top1": top1, "top5": top5,
            "hin_vs_driver_max_abs": gate_a,
            "routes": route_sel,
            "digests": {"hin": dig(hin_g), "x": dig(x_out), "logits": dig(logits)}}
    json.dump(meta, open(f"{OUT}/gold_mtp.json", "w"), indent=1)
    print(f"GOLD_MTP_OK rows={nrows} top1={top1} top5={top5}")
    print(f"  gate A (eh_proj input) worst max_abs {gate_a:.3e}")
    print(f"  final hidden max {np.abs(x_out[-1]).max():.4e}  "
          f"logits max {np.abs(logits).max():.4f} margin "
          f"{np.sort(logits)[-1] - np.sort(logits)[-2]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
