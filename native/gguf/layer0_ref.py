#!/usr/bin/env python3
"""
Layer-0 reference: dequant the real blk.0 weights from the Qwen3-4B GGUF and
compute one Qwen3 layer-0 forward in numpy (verification ONLY; runtime stays Vyb).

This produces the NUMERIC CONTRACT the GPU kernels must match:
  native/out/layer0_input.txt  -- input x [S,D] as flat f64 text (GPU loads same)
  native/out/layer0_ref.txt    -- layer-0 output [S,D] flat f64 text

Pinned semantics (transformers modeling_qwen3, real GGUF metadata):
  D=2560, H=32, KVH=8, HD=128, FF=9728, rms_norm_eps=1e-6, rope.freq_base=5e6.
  input_layernorm(D) -> q/k/v proj -> q_norm/k_norm RMSNorm(HD) -> RoPE(split-half,
  base 5e6) -> causal GQA (scaling=HD^-0.5, fp32 softmax) -> o_proj -> +residual
  -> post_attention_layernorm(D) -> SiLU(gate)*up -> down -> +residual.
  Weights GGUF [in,out] row-major; type 12=Q4_K, 14=Q6_K, 0=F32.
"""
import os, struct
import numpy as np

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = "/home/rick/Models/qwen3/Qwen3-4B-Q4_K_M.gguf"
TSV = os.path.join(repo, "native/out/qwen3_4b_tensors.tsv")
OUT_IN = os.path.join(repo, "native/out/layer0_input.txt")
OUT_REF = os.path.join(repo, "native/out/layer0_ref.txt")
OUT_IN_B = os.path.join(repo, "native/out/layer0_input.bin")     # f64 LE for Vyb
OUT_INVF_B = os.path.join(repo, "native/out/layer0_invfreq.bin") # 64 f64 LE for Vyb

D, H, KVH, HD, FF = 2560, 32, 8, 128, 9728
EPS = 1e-6
BASE = 5e6
SCALING = HD ** -0.5
S = 2          # 2 tokens -> positions 0 (identity) and 1 (exercises rope)


def parse_tsv():
    tens = {}
    for line in open(TSV):
        toks = line.rstrip("\n").split("\t")
        name, shape = toks[0], toks[1]
        ttype = int(toks[2])
        off = int([t for t in toks if t.startswith("off=")][0][4:])
        dims = [int(x) for x in shape.split("x")]
        tens[name] = dict(shape=dims, type=ttype, off=off,
                          numel=int(np.prod(dims, dtype=np.int64)))
    return tens


def dequant_q4k(raw):
    """type 12 block_q4_K: 144 B -> 256 f64. Vectorized over blocks."""
    nb = len(raw) // 144
    b = np.frombuffer(raw, np.uint8).reshape(nb, 144)
    d = b[:, 0:2].copy().view("<f2").astype(np.float64)[:, 0]
    dmin = b[:, 2:4].copy().view("<f2").astype(np.float64)[:, 0]
    scales = b[:, 4:16].astype(np.int64)
    qs = b[:, 16:144].astype(np.int64)
    out = np.empty((nb, 256), np.float64)
    for j in range(0, 256, 64):
        t = j // 64
        for gl in (0, 1):
            is_ = 2 * t + gl
            if is_ < 4:
                sc = scales[:, is_] & 63
                mn = scales[:, is_ + 4] & 63
            else:
                sc = (scales[:, is_ + 4] & 0xF) | ((scales[:, is_ - 4] >> 6) << 4)
                mn = (scales[:, is_ + 4] >> 4) | ((scales[:, is_] >> 6) << 4)
            d1 = d * sc
            m1 = dmin * mn
            q = qs[:, t * 32:(t + 1) * 32]
            nib = (q & 0xF) if gl == 0 else (q >> 4)
            idx = j + gl * 32 + np.arange(32)
            out[:, idx] = d1[:, None] * nib - m1[:, None]
    return out.reshape(-1)


def dequant_q6k(raw):
    """type 14 block_q6_K: 210 B -> 256 f64. Vectorized over blocks."""
    nb = len(raw) // 210
    b = np.frombuffer(raw, np.uint8).reshape(nb, 210)
    d = b[:, 208:210].copy().view("<f2").astype(np.float64)[:, 0]
    ql = b[:, 0:128].astype(np.int64)
    qh = b[:, 128:192].astype(np.int64)
    sc = b[:, 192:208].astype(np.int64)
    sc[sc >= 128] -= 256
    out = np.empty((nb, 256), np.float64)
    for o in (0, 128):
        g = o // 128
        ql2 = ql[:, g * 64:g * 64 + 64]
        qh2 = qh[:, g * 32:g * 32 + 32]
        scb = g * 8
        for l in range(32):
            isv = l // 16
            q1 = ((ql2[:, l] & 0xF) | (((qh2[:, l] >> 0) & 3) << 4)) - 32
            q2 = ((ql2[:, 32 + l] & 0xF) | (((qh2[:, l] >> 2) & 3) << 4)) - 32
            q3 = ((ql2[:, l] >> 4) | (((qh2[:, l] >> 4) & 3) << 4)) - 32
            q4 = ((ql2[:, 32 + l] >> 4) | (((qh2[:, l] >> 6) & 3) << 4)) - 32
            out[:, o + l] = d * sc[:, scb + isv + 0] * q1
            out[:, o + 32 + l] = d * sc[:, scb + isv + 2] * q2
            out[:, o + 64 + l] = d * sc[:, scb + isv + 4] * q3
            out[:, o + 96 + l] = d * sc[:, scb + isv + 6] * q4
    return out.reshape(-1)


def read_weight(t, shape):
    """dequant one tensor, returned reshaped [in,out] (GGUF row-major)."""
    packed = {"numel": t["numel"]}
    if t["type"] == 12:
        nbytes = t["numel"] * 144 // 256
        with open(MODEL, "rb") as fh:
            fh.seek(t["off"]); raw = fh.read(nbytes)
        assert len(raw) == nbytes
        w = dequant_q4k(raw)
    elif t["type"] == 14:
        nbytes = t["numel"] * 210 // 256
        with open(MODEL, "rb") as fh:
            fh.seek(t["off"]); raw = fh.read(nbytes)
        assert len(raw) == nbytes
        w = dequant_q6k(raw)
    elif t["type"] == 0:
        with open(MODEL, "rb") as fh:
            fh.seek(t["off"]); raw = fh.read(t["numel"] * 4)
        assert len(raw) == t["numel"] * 4
        w = np.frombuffer(raw, "<f4").astype(np.float64)
    else:
        raise RuntimeError(f"unhandled type {t['type']}")
    assert w.size == t["numel"], (w.size, t["numel"])
    return w.reshape(shape)


def rmsnorm(x, wgt):
    """x:[...,dim] over last dim; wgt:[dim]. x*wgt/sqrt(mean(x^2)+eps)."""
    var = (x * x).mean(-1, keepdims=True)
    return x * wgt / np.sqrt(var + EPS)


def rope(q, k, pos):
    """split-half rotary (transformers Qwen3), base 5e6. q[S,H,HD], k[S,KVH,HD].
    emb=cat(freqs,freqs); cos/sin=[S,1,HD]; out = x*cos + rot(x)*sin, rot=cat(-xR,xL)."""
    i = np.arange(HD // 2)                       # 0..63
    invf = BASE ** (-2.0 * i / HD)               # [HD/2]
    freqs = pos[:, None] * invf[None, :]         # [S,HD/2]
    emb = np.concatenate([freqs, freqs], axis=1) # [S,HD]
    cos = np.cos(emb)[:, None, :]                # [S,1,HD]
    sin = np.sin(emb)[:, None, :]
    half = HD // 2
    ql, qr = q[..., :half], q[..., half:]
    qrot = np.concatenate([-qr, ql], axis=-1)
    kl, kr = k[..., :half], k[..., half:]
    krot = np.concatenate([-kr, kl], axis=-1)
    return q * cos + qrot * sin, k * cos + krot * sin


def gqa(q, k, v):
    """q[S,H,HD], k/v[S,KVH,HD] -> causal attn out[S,H,HD]. qh->kv=qh//(H/KVH)."""
    n_rep = H // KVH
    S_ = q.shape[0]
    attn = np.empty((H, S_, HD), np.float64)
    for qh in range(H):
        kv = qh // n_rep
        scores = (q[:, qh, :] @ k[:, kv, :].T) * SCALING   # [S,S]
        for s2 in range(S_):
            for s1 in range(s2):
                scores[s1, s2] = -np.inf                    # causal mask
        e = np.exp(scores - scores.max(axis=-1, keepdims=True))
        e /= e.sum(axis=-1, keepdims=True)
        attn[qh] = e @ v[:, kv, :]                          # [S,HD]
    o = attn.transpose(1, 0, 2)                            # [S,H,HD]
    return o.reshape(S_, H * HD)


def main():
    tens = parse_tsv()
    blk = {n: tens[n] for n in tens if n.startswith("blk.0.")}
    names = sorted(blk)

    print("dequantizing blk.0 weights (11 tensors)...")
    W = {}
    for n in names:
        shape = tuple(blk[n]["shape"])         # 1-D for norms, else [in,out]
        W[n] = read_weight(blk[n], shape)
        print(f"  {n}: {blk[n]['type']} {blk[n]['shape']} numel={blk[n]['numel']}")

    # input: 2 real token rows of token_embd (Q6_K), positions 0 and 1.
    te = tens["token_embd.weight"]
    print("dequantizing 2 token_embd rows (positions 0,1)...")
    nblk_emb = D // 256          # 10 Q6_K blocks per 2560-wide token row
    x = np.empty((S, D), np.float64)
    for s in range(S):
        with open(MODEL, "rb") as fh:
            fh.seek(te["off"] + s * nblk_emb * 210)
            raw = fh.read(nblk_emb * 210)
        x[s] = dequant_q6k(raw).reshape(-1)
    pos = np.arange(S)

    # ---- forward ----
    xn = rmsnorm(x, W["blk.0.attn_norm.weight"])
    q = (xn @ W["blk.0.attn_q.weight"]).reshape(S, H, HD)
    k = (xn @ W["blk.0.attn_k.weight"]).reshape(S, KVH, HD)
    v = (xn @ W["blk.0.attn_v.weight"]).reshape(S, KVH, HD)
    q = rmsnorm(q, W["blk.0.attn_q_norm.weight"])
    k = rmsnorm(k, W["blk.0.attn_k_norm.weight"])
    q, k = rope(q, k, pos)
    np.savetxt(os.path.join(repo, "native/out/layer0_qr.txt"), q.reshape(-1), fmt="%.17g")
    np.savetxt(os.path.join(repo, "native/out/layer0_kr.txt"), k.reshape(-1), fmt="%.17g")
    o = gqa(q, k, v)
    layer = x + (o @ W["blk.0.attn_output.weight"])          # [S,D]

    h = rmsnorm(layer, W["blk.0.ffn_norm.weight"])
    gate = h @ W["blk.0.ffn_gate.weight"]                    # [S,FF]
    up = h @ W["blk.0.ffn_up.weight"]
    silu = gate / (1.0 + np.exp(-gate))
    ff_out = (silu * up) @ W["blk.0.ffn_down.weight"]
    out = layer + ff_out

    # ---- save contract ----
    np.savetxt(OUT_IN, x.reshape(-1), fmt="%.17g")
    np.savetxt(OUT_REF, out.reshape(-1), fmt="%.17g")
    invf = BASE ** (-2.0 * np.arange(HD // 2) / HD)            # [64] f64
    x.tofile(OUT_IN_B)                                          # f64 LE
    invf.astype(np.float64).tofile(OUT_INVF_B)                  # f64 LE
    print(f"wrote {OUT_IN} ({x.size} f64), {OUT_REF} ({out.size} f64),"
          f" {OUT_IN_B}, {OUT_INVF_B} ({invf.size} f64)")
    print(f"input  x   range [{x.min():.6g},{x.max():.6g}]")
    print(f"layer-0 out range [{out.min():.6g},{out.max():.6g}]")


if __name__ == "__main__":
    import time
    t0 = time.time()
    main()
    print(f"elapsed {time.time()-t0:.1f}s")
