#!/usr/bin/env python3
"""FROZEN-CONTEXT KV TRAINING oracle — the authoritative GPU target for kvresp_train.vyb.

The KV-cache context-prefix trainer (KVCACHE-CONTEXT.md) caches the frozen 9-token goal context's
per-layer roped-K/V once, then per training step runs a per-token RESPONSE forward attending the
combined cache (== the validated full-sequence S=93 forward for response rows), a masked CE on the
response positions only, a FROZEN-CONTEXT backward, and AdamW on the response LoRA.

KEY EQUIVALENCE (what makes this cheap and authoritative): the response forward is VERIFIED equal to
the full-sequence forward (KVRESPFWD_VERIFY OK), and the frozen-context backward equals the
full-sequence backward with the CONTEXT-REGION (rows 0..NCTX-1) attention gradients ZEROED:
response queries are the only backprop'd queries; context K/V are read but their grads are DROPPED
(context frozen), and context rows never update LoRA. So this oracle = the committed full-sequence
oracle (train_corpus_ce_ref.py, FD-verified + descending) extended with the context boundary mask.

Outputs (native/out, the .bin the GPU loads):
  kvresp_train_loss_ref.txt       — per-step masked CE loss (GPU must match + descend)
  kvresp_train_DUQ_L{L}_ref.bin   — step-1 dU_q (flattened [D*R]) at L in {0,17,35} (GPU gradient gate)
  kvresp_train_DVQ_L{L}_ref.bin   — step-1 dV_q
  kvresp_train_UQ_L0_stepN_ref.txt — final L0 Uq (first 128) parity gate after the loop
"""
import os, importlib.util, numpy as np

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out = os.path.join(repo, "native", "out")
MODEL = "/home/rick/Models/qwen3/Qwen3-4B-Q4_K_M.gguf"
spec = importlib.util.spec_from_file_location("l0", os.path.join(repo, "native/gguf/layer0_ref.py"))
l0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l0)

D, H, KVH, HD, FF = 2560, 32, 8, 128, 9728
NQ, NKV = H * HD, KVH * HD
R, alpha_r = 2, 2.0
VOCAB = 151936
EPS = 1e-6


def _load_record():
    import json
    msgs = None
    with open(os.path.join(repo, "data/vybos-configurator-train.jsonl")) as fh:
        for line in fh:
            d = json.loads(line)
            u = " ".join(m["content"] for m in d["messages"] if m["role"] == "user")
            if "Hyprland desktop workstation" in u:
                msgs = d["messages"]; break
    if msgs is None:
        raise SystemExit("goal-desktop record not found")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(repo, "artifacts", "vybos-configurator-lora"))
    usr = [m["content"] for m in msgs if m["role"] == "user"][0]
    asst = [m["content"] for m in msgs if m["role"] == "assistant"][-1]
    return (np.array(tok.encode(usr, add_special_tokens=False), dtype=np.int64),
            np.array(tok.encode(asst, add_special_tokens=False), dtype=np.int64))


_ctx, _resp = _load_record()
NCTX = int(_ctx.shape[0])
S = NCTX + int(_resp.shape[0])
INPUT_IDS = np.concatenate([_ctx, _resp])
LABELS = np.full(S, -1, dtype=np.int64)
for k in range(len(_resp) - 1):
    LABELS[NCTX + k] = _resp[k + 1]
NV = int((LABELS >= 0).sum())
print(f"frozen-context KV: S={S} ctx(resp)={NCTX} valid labels={NV}")
seq = np.arange(S)
tens = l0.parse_tsv()
proj_shapes = {"q": (D, NQ), "k": (D, NKV), "v": (D, NKV), "o": (NQ, D), "g": (D, FF), "u": (D, FF), "d": (FF, D)}
WK = {"q": "attn_q", "k": "attn_k", "v": "attn_v", "o": "attn_output", "g": "ffn_gate", "u": "ffn_up", "d": "ffn_down"}


def rw(L, n):
    return l0.read_weight(tens[f"blk.{L}.{n}.weight"], tuple(tens[f"blk.{L}.{n}.weight"]["shape"]))


ALL_W = [{n: rw(L, n) for n in ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]} for L in range(36)]
ALL_LN = [{n: rw(L, n) for n in ["attn_norm", "attn_q_norm", "attn_k_norm", "ffn_norm"]} for L in range(36)]


def load_Lo(L):
    return {nm: (np.fromfile(os.path.join(out, f"m2e_l{L}_U{nm}.bin"), "<f8").reshape(proj_shapes[nm][0], R),
                 np.fromfile(os.path.join(out, f"m2e_l{L}_V{nm}.bin"), "<f8").reshape(R, proj_shapes[nm][1])) for nm in proj_shapes}


Lo = {L: load_Lo(L) for L in range(36)}
mU = {L: {nm: np.zeros_like(U) for nm, (U, V) in Lo[L].items()} for L in range(36)}
vU = {L: {nm: np.zeros_like(U) for nm, (U, V) in Lo[L].items()} for L in range(36)}
mV = {L: {nm: np.zeros_like(V) for nm, (U, V) in Lo[L].items()} for L in range(36)}
vV = {L: {nm: np.zeros_like(V) for nm, (U, V) in Lo[L].items()} for L in range(36)}


def rms(v, w):
    return v * (1.0 / np.sqrt((v ** 2).mean(-1, keepdims=True) + EPS)) * w


INVF = np.fromfile(os.path.join(out, "layer0_invfreq.bin"), "<f8")
freqs = lambda s_: s_ * INVF[:HD // 2]


def lproj(a, nm, L):
    U, V = Lo[L][nm]; s = a @ U
    return a @ ALL_W[L][WK[nm]] + alpha_r * (s @ V), s


def rmsnorm_layer(v_, gy, wn, dim):
    inv = 1.0 / np.sqrt((v_ ** 2).mean() + 1e-6); T = (gy * wn * v_).sum()
    return (gy * wn * inv - v_ * (inv ** 3 / dim) * T)


def rope_adj(gr):
    S_, XH, HD_ = gr.shape; go = np.zeros_like(gr)
    for s_ in range(S_):
        for xh in range(XH):
            for i in range(HD_ // 2):
                f_ = freqs(s_)[i]; c, sn = np.cos(f_), np.sin(f_)
                di, dj = gr[s_, xh, i], gr[s_, xh, i + HD_ // 2]
                go[s_, xh, i] = c * di + sn * dj; go[s_, xh, i + HD_ // 2] = -sn * di + c * dj
    return go


INPUT_IDS.astype(np.int64).tofile(os.path.join(out, "kvresp_train_input_ids.bin"))
LABELS.astype(np.int64).tofile(os.path.join(out, "kvresp_train_labels.bin"))

te = tens["token_embd.weight"]
with open(MODEL, "rb") as fh:
    fh.seek(te["off"]); emb = l0.dequant_q6k(fh.read(te["numel"] * 210 // 256)).reshape(te["shape"][1], D)
on_t = tens["output_norm.weight"]; ON = l0.read_weight(on_t, tuple(on_t["shape"]))


def forward_layers(x):
    cache = {}; cur = x
    for L in range(36):
        W = ALL_W[L]; lnn = ALL_LN[L]; c = {}
        xn = rms(cur, lnn["attn_norm"]); c["xn"] = xn
        q, sq = lproj(xn, "q", L); c["q"] = q; c["s_q"] = sq
        k, sk = lproj(xn, "k", L); c["k"] = k; c["s_k"] = sk
        v, sv = lproj(xn, "v", L); c["v"] = v; c["s_v"] = sv
        qn = rms(q.reshape(S, H, HD), lnn["attn_q_norm"]); c["qn"] = qn
        kn = rms(k.reshape(S, KVH, HD), lnn["attn_k_norm"]); c["kn"] = kn
        qr, kr = l0.rope(qn, kn, seq); c["qr"] = qr; c["kr"] = kr
        ctx = l0.gqa(qr, kr, v.reshape(S, KVH, HD)); c["ctx"] = ctx
        o, so = lproj(ctx.reshape(S, NQ), "o", L); c["o"] = o; c["s_o"] = so
        x1 = cur + o; c["x1"] = x1
        x1n = rms(x1, lnn["ffn_norm"]); c["x1n"] = x1n
        g, sg = lproj(x1n, "g", L); c["gate"] = g; c["s_g"] = sg
        up, su = lproj(x1n, "u", L); c["up"] = up; c["s_u"] = su
        hu = (g / (1 + np.exp(-g))) * up; c["hu"] = hu
        m, sd = lproj(hu, "d", L); c["m"] = m; c["s_d"] = sd
        xo = x1 + m; c["xo"] = xo
        cache[L] = c; cur = xo
    cc = {}; cur = x
    for L in range(36):
        cc[L] = (cache[L], cur); cur = cache[L]["xo"]
    return cc, cur


CTXMASK = np.zeros(S, dtype=bool); CTXMASK[:NCTX] = True   # rows 0..NCTX-1 = frozen context


def backward_layer(L, dG, c, xin):
    W = ALL_W[L]; LoL = Lo[L]; lnn = ALL_LN[L]; gu = {}

    def lo_bwd(a, nm, dout):
        U, V = LoL[nm]; s = a @ U; gv = alpha_r * dout
        ds = gv @ V.T; dU = a.T @ ds; dV = s.T @ gv
        return dout @ W[WK[nm]].T + (ds @ U.T), dU, dV

    dm = dG.copy(); dx1 = dG.copy()
    dHu, gu["dU_d"], gu["dV_d"] = lo_bwd(c["hu"], "d", dm)
    sg = 1.0 / (1.0 + np.exp(-c["gate"])); silu_g = c["gate"] * sg; silup = sg + c["gate"] * sg * (1.0 - sg)
    dgate = dHu * c["up"] * silup; dup = dHu * silu_g
    d_g, gu["dU_g"], gu["dV_g"] = lo_bwd(c["x1n"], "g", dgate)
    d_u, gu["dU_u"], gu["dV_u"] = lo_bwd(c["x1n"], "u", dup)
    dX1n = d_g + d_u
    dX1l = np.zeros((S, D))
    for s_ in range(S): dX1l[s_] = rmsnorm_layer(c["x1"][s_], dX1n[s_], lnn["ffn_norm"], D)
    dx1 = dx1 + dX1l
    dctx, gu["dU_o"], gu["dV_o"] = lo_bwd(c["ctx"], "o", dx1)
    Dat = dctx.reshape(S, H, HD)
    qr = c["qr"].reshape(S, H, HD); kr = c["kr"].reshape(S, KVH, HD); vv = c["v"].reshape(S, KVH, HD)
    scale = 1.0 / np.sqrt(HD)
    gq = np.zeros_like(qr); gk = np.zeros_like(kr); gv_ = np.zeros_like(vv)
    # --------- FROZEN CONTEXT: only RESPONSE rows (s_ >= NCTX) act as queries -------------
    for s_ in range(S):
        if CTXMASK[s_]:
            continue
        for h in range(H):
            kvh = h // (H // KVH); qs = qr[s_, h]
            sc = np.array([scale * np.dot(qs, kr[tt, kvh]) for tt in range(s_ + 1)])
            mx = sc.max(); ax = np.exp(sc - mx); ssum = ax.sum(); w = ax / ssum
            l = np.array([np.dot(Dat[s_, h], vv[tt, kvh]) for tt in range(s_ + 1)])
            mean = (w * l).sum(); dsc = ax * (l - mean) / ssum
            gq[s_, h] = scale * sum(dsc[tt] * kr[tt, kvh] for tt in range(s_ + 1))
            for tt in range(s_ + 1):
                # accumulate k/v grads into row tt; CONTEXT rows (tt<NCtx) are dropped (frozen)
                if not CTXMASK[tt]:
                    gk[tt, kvh] += scale * dsc[tt] * qs
                    gv_[tt, kvh] += w[tt] * Dat[s_, h]
    dQn = rope_adj(gq); dKn = rope_adj(gk)
    dq_ = np.zeros((S, H, HD)); dk_ = np.zeros((S, KVH, HD))
    for s_ in range(S):
        if CTXMASK[s_]:
            continue
        for h in range(H): dq_[s_, h] = rmsnorm_layer(c["q"].reshape(S, H, HD)[s_, h], dQn[s_, h], lnn["attn_q_norm"], HD)
        for h in range(KVH): dk_[s_, h] = rmsnorm_layer(c["k"].reshape(S, KVH, HD)[s_, h], dKn[s_, h], lnn["attn_k_norm"], HD)
    dxn, gu["dU_q"], gu["dV_q"] = lo_bwd(c["xn"], "q", dq_.reshape(S, NQ))
    dxk, gu["dU_k"], gu["dV_k"] = lo_bwd(c["xn"], "k", dk_.reshape(S, NKV))
    dxv, gu["dU_v"], gu["dV_v"] = lo_bwd(c["xn"], "v", gv_.reshape(S, NKV))
    dXN = dxn + dxk + dxv
    dx = np.zeros((S, D))
    for s_ in range(S):
        if CTXMASK[s_]:
            continue
        dx[s_] = rmsnorm_layer(xin[s_], dXN[s_], lnn["attn_norm"], D)
    dx = dx + dx1   # residual-identity
    return gu, dx


def ce_loss_and_headgrad(xo):
    h = rms(xo, ON)
    logits = h @ emb.T
    lse = logits.max(axis=-1, keepdims=True)
    e = np.exp(logits - lse); soft = e / e.sum(axis=-1, keepdims=True)
    mask = LABELS >= 0; NV_ = int(mask.sum())
    loss = float(-np.log(soft[np.arange(S), LABELS][mask] + 1e-30).mean())
    onehot = np.zeros_like(soft); onehot[np.arange(S), LABELS] = 1.0
    dlog = np.where(mask[:, None], (soft - onehot) / NV_, 0.0)
    dh = dlog @ emb
    dxo = np.zeros((S, D))
    for s_ in range(S):
        if CTXMASK[s_]:
            continue
        dxo[s_] = rmsnorm_layer(xo[s_], dh[s_], ON, D)
    return float(loss), dxo, logits, soft


NSTP = 4; LR = 0.00005; B1 = 0.9; B2 = 0.999; EPP = 1e-8; WDD = 0.0


def adamw(P, g, m, v, st):
    m[:] = B1 * m + (1 - B1) * g; v[:] = B2 * v + (1 - B2) * g * g
    P[:] -= LR * (m / (1 - B1 ** st)) / (np.sqrt(v / (1 - B2 ** st)) + EPP)


x0 = emb[INPUT_IDS].astype("<f8")
losses = []
for st in range(1, NSTP + 1):
    cc, xo = forward_layers(x0)
    loss, dxo, logits, soft = ce_loss_and_headgrad(xo)
    losses.append(loss)
    dG = dxo.copy()
    for L in range(35, -1, -1):
        c, xin = cc[L]
        gu, dG = backward_layer(L, dG.copy(), c, xin)
        if st == 1 and L in (0, 17, 35):
            np.asarray(gu["dU_q"], dtype=np.float64).reshape(-1).tofile(os.path.join(out, f"kvresp_train_DUQ_L{L}_ref.bin"))
            np.asarray(gu["dV_q"], dtype=np.float64).reshape(-1).tofile(os.path.join(out, f"kvresp_train_DVQ_L{L}_ref.bin"))
        for nm in proj_shapes:
            adamw(Lo[L][nm][0], gu["dU_" + nm], mU[L][nm], vU[L][nm], st)
            adamw(Lo[L][nm][1], gu["dV_" + nm], mV[L][nm], vV[L][nm], st)
    print(f"step {st} CE loss {loss:.6f}", flush=True)
np.savetxt(os.path.join(out, "kvresp_train_loss_ref.txt"), np.array(losses), fmt="%.10g")
np.savetxt(os.path.join(out, "kvresp_train_UQ_L0_stepN_ref.txt"), Lo[0]["q"][0].reshape(-1)[:128], fmt="%.9g")
print("KVRESP_TRAIN_ORACLE_DONE steps", NSTP, "loss0", losses[0], "lossN", losses[-1], "descend", losses[-1] < losses[0])
