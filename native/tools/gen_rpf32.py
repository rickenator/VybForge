#!/usr/bin/env python3
"""gen_rpf32.py -- f32-compute chunk kernels for the Ridge batched prefill (VYB_PFB).

GB10 runs f64 at a small fraction of the f32 rate, so the f64 chunk kernels (delta_seq, attn_pf,
norm_gated_b, l2n_bs, silu32) dominated pp512 once the GEMMs were on tensor cores.  These keep the
same buffers and parameter blocks (f64 in, f64/f32 out) but compute in f32 with the native
ex2/rsqrt/sqrt/max intrinsics.  Compile: vyb --kernel --gpu sm_80 --kernel-shared 1024 rpf32.vyb

  delta_seq32  gated delta rule over the chunk, f32 state in registers; params = delta_seq
  attn_pf2     causal GQA attention, warp per (query, kv group), online softmax; params = attn_pf
               grid ceil(S/8)*KVH x 256; needs HD == 256 and H/KVH == GQ (generated)
  ng_b32       norm_gated_b, warp per row, writes f32 (fuses the cast32); params X Z W Y(f32) NR D eps
  l2n_b32      l2n_bs, warp per head vector; params = l2n_bs + NR@64; grid ceil(NR/8) x 256
  silu_b32     silu(g)*u -> f32 (silu32 signature)
"""
import sys

L2E = 1.4426950408889634

def F(v):
    return f"({v!r} as Float32)"

def delta_seq32():
    o = ["delta_seq32(p<Int>) -> {"]
    I = lambda s, n=1: o.append("    " * n + s)
    for nm, off in [("Q", 0), ("K", 8), ("V", 16), ("G", 24), ("BE", 32), ("ST", 40), ("OUT", 48), ("NT", 56),
                    ("SSd", 64), ("HK", 72), ("HV", 80), ("QKS", 96), ("GKS", 104)]:
        I(f"{nm}<Int> = ld_i64(p + {off})")
    I("sc<Float32> = (ld_f64(p + 88) as Float32)")
    I(f"l2e<Float32> = {F(L2E)}")
    for s in ["tid<Int> = tid_x()", "b<Int> = blk_x()", "h<Int> = b / 4", "j<Int> = (b % 4) * 32 + tid / 4",
              "qt<Int> = tid % 4", "hk<Int> = h % HK",
              # thread (j, qt) owns columns 4e+qt (interleaved: conflict-free shared reads)
              "sb<Int> = ST + (SSd * SSd * h + j * SSd + qt) * 8"]:
        I(s)
    for e in range(32):
        I(f"s{e}<Float32> = (ld_f64(sb + {32 * e}) as Float32)")
    I("kn<Float> = ld_f64(K + (hk * SSd + tid) * 8)")
    I("qn<Float> = ld_f64(Q + (hk * SSd + tid) * 8)")
    I("t<Int> = 0")
    I("while (t < NT) {")
    I("st_shared_f32(tid * 4, (kn as Float32))", 2)
    I("st_shared_f32(512 + tid * 4, (qn as Float32))", 2)
    I("gl<Float> = ld_f64(G + (t * HV + h) * 8)", 2)
    I("be<Float> = ld_f64(BE + (t * HV + h) * 8)", 2)
    I("vv<Float> = ld_f64(V + (t * QKS + h * SSd + j) * 8)", 2)
    I("kernel_barrier()", 2)
    I("if (t + 1 < NT) {", 2)
    I("kn = ld_f64(K + ((t + 1) * GKS + hk * SSd + tid) * 8)", 3)
    I("qn = ld_f64(Q + ((t + 1) * GKS + hk * SSd + tid) * 8)", 3)
    I("}", 2)
    I(f"acc<Float32> = {F(0.0)}", 2)
    I(f"acq<Float32> = {F(0.0)}", 2)
    I(f"kq<Float32> = {F(0.0)}", 2)
    for e in range(32):
        I(f"k{e}<Float32> = ld_shared_f32(({e} * 4 + qt) * 4)", 2)
        I(f"q{e}<Float32> = ld_shared_f32(512 + ({e} * 4 + qt) * 4)", 2)
        I(f"acc = fma_f32(s{e}, k{e}, acc)", 2)
        I(f"acq = fma_f32(s{e}, q{e}, acq)", 2)
        I(f"kq = fma_f32(k{e}, q{e}, kq)", 2)
    for v in ["acc", "acq", "kq"]:
        I(f"{v} = {v} + shfl_xor_f32({v}, 1)", 2)
        I(f"{v} = {v} + shfl_xor_f32({v}, 2)", 2)
    I("gg<Float32> = ex2_f32((gl as Float32) * l2e)", 2)
    I("del<Float32> = ((vv as Float32) - gg * acc) * (be as Float32)", 2)
    I("if (qt == 0) { st_f64(OUT + (t * HV * SSd + h * SSd + j) * 8, (sc * (gg * acq + del * kq)) as Float) }", 2)
    for e in range(32):
        I(f"s{e} = fma_f32(gg, s{e}, del * k{e})", 2)
    I("kernel_barrier()", 2)
    I("t = t + 1", 2)
    I("}")
    for e in range(32):
        I(f"st_f64(sb + {32 * e}, (s{e} as Float))")
    o.append("}")
    return "\n".join(o)

def attn_pf2(GQ=6, HD=256):
    M = HD // 32
    o = ["attn_pf2(p<Int>) -> {"]
    I = lambda s, n=1: o.append("    " * n + s)
    for nm, off in [("Q", 0), ("KC", 8), ("VC", 16), ("CTX", 24), ("H", 32), ("KVH", 40), ("HDp", 48), ("P0", 56), ("S", 64)]:
        I(f"{nm}<Int> = ld_i64(p + {off})")
    I(f"scl<Float32> = ((1.0 / vsqrt(HDp as Float, 40)) * {L2E!r}) as Float32")
    I("b<Int> = blk_x()")
    I("g<Int> = b % KVH")
    I("tid<Int> = tid_x()")
    I("lane<Int> = tid % 32")
    I("s<Int> = (b / KVH) * 8 + tid / 32")
    I("if (s < S) {")
    I("T<Int> = P0 + s + 1", 2)
    I("NKV<Int> = KVH * HDp", 2)
    I("kb<Int> = KC + (g * HDp + lane) * 8", 2)
    I("vb<Int> = VC + (g * HDp + lane) * 8", 2)
    for i in range(GQ):
        I(f"qb{i}<Int> = Q + (s * H * HDp + (g * {GQ} + {i}) * HDp + lane) * 8", 2)
        for m in range(M):
            I(f"q{i}_{m}<Float32> = (ld_f64(qb{i} + {256 * m}) as Float32) * scl", 2)
            I(f"a{i}_{m}<Float32> = {F(0.0)}", 2)
        I(f"mx{i}<Float32> = {F(-1e30)}", 2)
        I(f"ls{i}<Float32> = {F(0.0)}", 2)
    I("t<Int> = 0", 2)
    I("while (t < T) {", 2)
    I("ko<Int> = t * NKV * 8", 3)
    for m in range(M):
        I(f"k{m}<Float32> = (ld_f64(kb + ko + {256 * m}) as Float32)", 3)
    for m in range(M):
        I(f"v{m}<Float32> = (ld_f64(vb + ko + {256 * m}) as Float32)", 3)
    for i in range(GQ):
        I(f"d{i}<Float32> = q{i}_0 * k0", 3)
        for m in range(1, M):
            I(f"d{i} = fma_f32(q{i}_{m}, k{m}, d{i})", 3)
    for sh in [16, 8, 4, 2, 1]:
        for i in range(GQ):
            I(f"d{i} = d{i} + shfl_xor_f32(d{i}, {sh})", 3)
    for i in range(GQ):
        I(f"mn{i}<Float32> = max_f32(mx{i}, d{i})", 3)
        I(f"c{i}<Float32> = ex2_f32(mx{i} - mn{i})", 3)
        I(f"pe{i}<Float32> = ex2_f32(d{i} - mn{i})", 3)
        I(f"mx{i} = mn{i}", 3)
        I(f"ls{i} = fma_f32(ls{i}, c{i}, pe{i})", 3)
        for m in range(M):
            I(f"a{i}_{m} = fma_f32(a{i}_{m}, c{i}, pe{i} * v{m})", 3)
    I("t = t + 1", 3)
    I("}", 2)
    for i in range(GQ):
        I(f"inv{i}<Float32> = {F(1.0)} / ls{i}", 2)
        I(f"ob{i}<Int> = CTX + (s * H * HDp + (g * {GQ} + {i}) * HDp + lane) * 8", 2)
        for m in range(M):
            I(f"st_f64(ob{i} + {256 * m}, (a{i}_{m} * inv{i}) as Float)", 2)
    I("}")
    o.append("}")
    return "\n".join(o)

HAND = r'''
// norm_gated_b in f32, one warp per row of D = 128 (4 elements per lane), f32 output (fuses cast32).
// Params X@0 Z@8 W@16 (f64) Y@24 (f32) NR@32 rows D@40 eps@48. Grid ceil(NR/8) x 256.
ng_b32(p<Int>) -> {
    X<Int> = ld_i64(p + 0)
    Z<Int> = ld_i64(p + 8)
    W<Int> = ld_i64(p + 16)
    Y<Int> = ld_i64(p + 24)
    NR<Int> = ld_i64(p + 32)
    D<Int> = ld_i64(p + 40)
    eps<Float32> = (ld_f64(p + 48) as Float32)
    tid<Int> = tid_x()
    lane<Int> = tid % 32
    r<Int> = blk_x() * 8 + tid / 32
    l2e<Float32> = (1.4426950408889634 as Float32)
    one<Float32> = (1.0 as Float32)
    if (r < NR) {
        xb<Int> = X + (r * D + lane) * 8
        zb<Int> = Z + (r * D + lane) * 8
        v0<Float32> = (ld_f64(xb) as Float32)
        v1<Float32> = (ld_f64(xb + 256) as Float32)
        v2<Float32> = (ld_f64(xb + 512) as Float32)
        v3<Float32> = (ld_f64(xb + 768) as Float32)
        ss<Float32> = v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3
        ss = ss + shfl_xor_f32(ss, 16)
        ss = ss + shfl_xor_f32(ss, 8)
        ss = ss + shfl_xor_f32(ss, 4)
        ss = ss + shfl_xor_f32(ss, 2)
        ss = ss + shfl_xor_f32(ss, 1)
        ri<Float32> = one / sqrt_f32(ss / (D as Float32) + eps)
        yb<Int> = Y + (r * D + lane) * 4
        z0<Float32> = (ld_f64(zb) as Float32)
        z1<Float32> = (ld_f64(zb + 256) as Float32)
        z2<Float32> = (ld_f64(zb + 512) as Float32)
        z3<Float32> = (ld_f64(zb + 768) as Float32)
        st_f32n(yb, v0 * ri * (ld_f64(W + lane * 8) as Float32) * (z0 / (one + ex2_f32(z0 * (0.0 - l2e)))))
        st_f32n(yb + 128, v1 * ri * (ld_f64(W + (lane + 32) * 8) as Float32) * (z1 / (one + ex2_f32(z1 * (0.0 - l2e)))))
        st_f32n(yb + 256, v2 * ri * (ld_f64(W + (lane + 64) * 8) as Float32) * (z2 / (one + ex2_f32(z2 * (0.0 - l2e)))))
        st_f32n(yb + 384, v3 * ri * (ld_f64(W + (lane + 96) * 8) as Float32) * (z3 / (one + ex2_f32(z3 * (0.0 - l2e)))))
    }
}

// l2n_bs in f32, one warp per q/k head vector of D = 128. Params = l2n_bs + NR@64 (= S*2*HK).
// Grid ceil(NR/8) x 256.
l2n_b32(p<Int>) -> {
    X<Int> = ld_i64(p + 0)
    QN<Int> = ld_i64(p + 8)
    KN<Int> = ld_i64(p + 16)
    HK<Int> = ld_i64(p + 24)
    D<Int> = ld_i64(p + 32)
    QKS<Int> = ld_i64(p + 40)
    GK<Int> = ld_i64(p + 48)
    eps<Float32> = (ld_f64(p + 56) as Float32)
    NR<Int> = ld_i64(p + 64)
    tid<Int> = tid_x()
    lane<Int> = tid % 32
    b<Int> = blk_x() * 8 + tid / 32
    if (b < NR) {
        s<Int> = b / (2 * HK)
        r<Int> = b % (2 * HK)
        src<Int> = X + (s * QKS + r * D + lane) * 8
        dst<Int> = QN + (s * GK + r * D + lane) * 8
        if (r >= HK) { dst = KN + (s * GK + (r - HK) * D + lane) * 8 }
        v0<Float32> = (ld_f64(src) as Float32)
        v1<Float32> = (ld_f64(src + 256) as Float32)
        v2<Float32> = (ld_f64(src + 512) as Float32)
        v3<Float32> = (ld_f64(src + 768) as Float32)
        ss<Float32> = v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3
        ss = ss + shfl_xor_f32(ss, 16)
        ss = ss + shfl_xor_f32(ss, 8)
        ss = ss + shfl_xor_f32(ss, 4)
        ss = ss + shfl_xor_f32(ss, 2)
        ss = ss + shfl_xor_f32(ss, 1)
        sq<Float32> = max_f32(sqrt_f32(ss), eps)
        iv<Float32> = (1.0 as Float32) / sq
        st_f64(dst, (v0 * iv) as Float)
        st_f64(dst + 256, (v1 * iv) as Float)
        st_f64(dst + 512, (v2 * iv) as Float)
        st_f64(dst + 768, (v3 * iv) as Float)
    }
}

// silu32 in f32: h = silu(g) * u (g, u f64; h f32).
silu_b32(g<Int>, u<Int>, h<Int>, n<Int>) -> {
    gid<Int> = blk_x() * dim_x() + tid_x()
    if (gid < n) {
        gv<Float32> = (ld_f64(g + gid * 8) as Float32)
        uv<Float32> = (ld_f64(u + gid * 8) as Float32)
        st_f32n(h + gid * 4, gv * uv / ((1.0 as Float32) + ex2_f32(gv * (-1.4426950408889634 as Float32))))
    }
}
'''

if __name__ == "__main__":
    out = ["// GENERATED by native/tools/gen_rpf32.py -- f32-compute chunk kernels for the Ridge batched prefill.",
           "// " + __doc__.strip().replace("\n", "\n// "), "import vmath::{vsqrt}", "",
           delta_seq32(), "", attn_pf2(), "", HAND]
    print("\n".join(out))
