#!/usr/bin/env python3
"""Generate native/kernels/rpf_gemm.vyb: tiled tensor-core GEMMs that read packed GGUF weights in
place (batched prefill).  Y[m, n] (f64, row stride LDY) = RS[m, n] + sum_k X[m, k] * W[n, k]
(LDY < 0: Y is f32, row stride -LDY, RS ignored.)
X: f32 [S][K] (16-byte aligned rows), W: packed rows of type T (Q4_K/Q5_K/Q6_K/Q8_0/IQ2_S/IQ3_S).
Tile 64 (--bm128: 128) tokens x 128 outputs x 64 k, 256 threads = 8 warps (2 along M x 4 along N), each warp
32x32 via mma.m16n8k16 (f16 in, f32 accumulate).  Shared: A [64][72] halves @0, B [128][72] @9216.
Params: W@0 X@8 Y@16 S@24 N@32 K@40 LDY@48 RS@56 G@64 NMT@72.  Grid = NMT * ceil(N/128), block 256.
Compile: vyb --kernel --gpu sm_80 --kernel-shared 27648 (ursela/gpu-features compiler).
--bm128 --vdeq --gsh (rpf_gemm128, the default IQ prefill tiles): --kernel-shared 45056 (A+B tiles + 8 KB IQ grid)."""
import re
import sys

def deq_q4k(o, q5=False):
    bs = 176 if q5 else 144
    o.append(f"bb<Int> = rb + sbi * {bs}")
    o.append("dd<Float32> = ld_f16(bb)")
    o.append("dmn<Float32> = ld_f16(bb + 2)")
    o.append("scv<Int> = 0")
    o.append("mnv<Int> = 0")
    o.append("if (sub < 4) { scv = (ld_i8(bb + 4 + sub) as Int) & 63 ; mnv = (ld_i8(bb + 8 + sub) as Int) & 63 } else {")
    o.append("    scv = ((ld_i8(bb + 8 + sub) as Int) & 15) | ((((ld_i8(bb + sub) as Int) & 255) >> 6) << 4)")
    o.append("    mnv = (((ld_i8(bb + 8 + sub) as Int) & 255) >> 4) | ((((ld_i8(bb + 4 + sub) as Int) & 255) >> 6) << 4) }")
    o.append("ds<Float32> = dd * (scv as Float32)")
    o.append("dm<Float32> = dmn * (mnv as Float32)")
    qso = 48 if q5 else 16
    o.append(f"qa<Int> = bb + {qso} + 32 * (sub / 2)")
    o.append("shq<UInt32> = (4 * (sub % 2)) as UInt32")
    if q5:
        o.append("shh<UInt32> = sub as UInt32")
    for i in range(8):
        o.append(f"u{i}<UInt32> = ld_v4_u32(qa + {16 * (i // 4)}, {i % 4}) >> shq")
        if q5:
            o.append(f"h{i}<UInt32> = ld_v4_u32(bb + {16 + 16 * (i // 4)}, {i % 4}) >> shh")
    for i in range(8):
        for b in range(4):
            j = 4 * i + b
            q = f"((u{i} >> {8 * b}) & 15)"
            if q5:
                q = f"({q} | (((h{i} >> {8 * b}) & 1) << 4))"
            o.append(f"dw{j}<Float32> = ds * ({q} as Float32) - dm")

def deq_q6k(o):
    o.append("bb<Int> = rb + sbi * 210")
    o.append("nh<Int> = sub / 4")
    o.append("q4<Int> = sub % 4")
    o.append("qla<Int> = bb + 64 * nh + 32 * (q4 % 2)")
    o.append("qls<Int> = 4 * (q4 / 2)")
    o.append("qha<Int> = bb + 128 + 32 * nh")
    o.append("qhs<Int> = 2 * q4")
    o.append("dd<Float32> = ld_f16(bb + 208)")
    o.append("d0<Float32> = dd * ((ld_i8(bb + 192 + 8 * nh + 2 * q4) as Int) as Float32)")
    o.append("d1<Float32> = dd * ((ld_i8(bb + 193 + 8 * nh + 2 * q4) as Int) as Float32)")
    for l in range(32):
        q = f"(((((ld_i8(qla + {l}) as Int) & 255) >> qls) & 15) | (((((ld_i8(qha + {l}) as Int) & 255) >> qhs) & 3) << 4))"
        o.append(f"dw{l}<Float32> = d{l // 16} * (({q} - 32) as Float32)")

def deq_q80(o):
    o.append("bb<Int> = W + nrow * (K / 32) * 34 + (kofs / 32) * 34")
    o.append("dd<Float32> = ld_f16(bb)")
    for l in range(32):
        o.append(f"dw{l}<Float32> = dd * ((ld_i8(bb + {2 + l}) as Int) as Float32)")

def deq_iq2s(o):
    o.append("bb<Int> = rb + sbi * 82")
    o.append("dd<Float32> = ld_f16(bb)")
    o.append("scb<Int> = (ld_i8(bb + 74 + sub) as Int) & 255")
    o.append("dl0<Float32> = dd * (((scb & 15) as Float32) + (0.5 as Float32)) * (0.25 as Float32)")
    o.append("dl1<Float32> = dd * (((scb >> 4) as Float32) + (0.5 as Float32)) * (0.25 as Float32)")
    o.append("qhb<Int> = (ld_i8(bb + 66 + sub) as Int) & 255")
    for l2 in range(4):
        o.append(f"ix{l2}<Int> = ((ld_i8(bb + 2 + 4 * sub + {l2}) as Int) & 255) | ((qhb << {8 - 2 * l2}) & 768)")
        o.append(f"ga{l2}<UInt32> = ld_v2_u32(G + ix{l2} * 8, 0)")
        o.append(f"gb{l2}<UInt32> = ld_v2_u32(G + ix{l2} * 8, 1)")
        o.append(f"sg{l2}<UInt32> = (((ld_i8(bb + 34 + 4 * sub + {l2}) as Int) & 255) as UInt32)")
        for j in range(8):
            wd = f"ga{l2}" if j < 4 else f"gb{l2}"
            o.append(f"dw{8 * l2 + j}<Float32> = dl{l2 // 2} * (((({wd} >> {8 * (j % 4)}) & 255) as Float32) * ((1.0 as Float32) - (2.0 as Float32) * (((sg{l2} >> {j}) & 1) as Float32)))")

def deq_iq3s(o):
    o.append("bb<Int> = rb + sbi * 110")
    o.append("dd<Float32> = ld_f16(bb)")
    o.append("nib<Int> = (((ld_i8(bb + 106 + sub / 2) as Int) & 255) >> (4 * (sub % 2))) & 15")
    o.append("db<Float32> = dd * ((1 + 2 * nib) as Float32)")
    o.append("qhb<Int> = (ld_i8(bb + 66 + sub) as Int) & 255")
    for l3 in range(4):
        o.append(f"qo{l3}<Int> = bb + 2 + 8 * sub + {2 * l3}")
        o.append(f"ia{l3}<Int> = ((ld_i8(qo{l3}) as Int) & 255) | ((qhb << {8 - 2 * l3}) & 256)")
        o.append(f"ib{l3}<Int> = ((ld_i8(qo{l3} + 1) as Int) & 255) | ((qhb << {7 - 2 * l3}) & 256)")
        o.append(f"ga{l3}<UInt32> = ld_u32(G + ia{l3} * 4)")
        o.append(f"gb{l3}<UInt32> = ld_u32(G + ib{l3} * 4)")
        o.append(f"sg{l3}<UInt32> = (((ld_i8(bb + 74 + 4 * sub + {l3}) as Int) & 255) as UInt32)")
        for j in range(8):
            wd = f"ga{l3}" if j < 4 else f"gb{l3}"
            o.append(f"dw{8 * l3 + j}<Float32> = db * (((({wd} >> {8 * (j % 4)}) & 255) as Float32) * ((1.0 as Float32) - (2.0 as Float32) * (((sg{l3} >> {j}) & 1) as Float32)))")

def _fw(o, sc, wd, sg, base):
    # 4 weights from grid word wd, sign bits sg >> base (same float formula as deq_iq2s/deq_iq3s)
    for j in range(4):
        o.append(f"dw{base + j}<Float32> = {sc} * (((({wd} >> {8 * j}) & 255) as Float32) * ((1.0 as Float32) - (2.0 as Float32) * ((({sg} >> {base % 8 + j}) & 1) as Float32)))")

def deq_iq2s_v(o):
    # --vdeq: same math as deq_iq2s (bit-identical) with the per-sub-block qs / sign bytes fetched as
    # 2 x u16 each (blocks are 82 B, so 2-byte aligned) instead of 8 byte loads
    o.append("bb<Int> = rb + sbi * 82")
    o.append("dd<Float32> = ld_f16(bb)")
    o.append("scb<Int> = (ld_i8(bb + 74 + sub) as Int) & 255")
    o.append("dl0<Float32> = dd * (((scb & 15) as Float32) + (0.5 as Float32)) * (0.25 as Float32)")
    o.append("dl1<Float32> = dd * (((scb >> 4) as Float32) + (0.5 as Float32)) * (0.25 as Float32)")
    o.append("qhb<Int> = (ld_i8(bb + 66 + sub) as Int) & 255")
    o.append("qw0<Int> = ld_u16(bb + 2 + 4 * sub) as Int")
    o.append("qw1<Int> = ld_u16(bb + 4 + 4 * sub) as Int")
    o.append("sw0<UInt32> = (ld_u16(bb + 34 + 4 * sub) as Int) as UInt32")
    o.append("sw1<UInt32> = (ld_u16(bb + 36 + 4 * sub) as Int) as UInt32")
    for l2 in range(4):
        qw = f"qw{l2 // 2}"
        o.append(f"ix{l2}<Int> = (({qw} >> {8 * (l2 % 2)}) & 255) | ((qhb << {8 - 2 * l2}) & 768)")
        o.append(f"ga{l2}<UInt32> = ld_v2_u32(G + ix{l2} * 8, 0)")
        o.append(f"gb{l2}<UInt32> = ld_v2_u32(G + ix{l2} * 8, 1)")
        o.append(f"sg{l2}<UInt32> = sw{l2 // 2} >> {8 * (l2 % 2)}")
        _fw(o, f"dl{l2 // 2}", f"ga{l2}", f"sg{l2}", 8 * l2)
        _fw(o, f"dl{l2 // 2}", f"gb{l2}", f"sg{l2}", 8 * l2 + 4)

def deq_iq3s_v(o):
    o.append("bb<Int> = rb + sbi * 110")
    o.append("dd<Float32> = ld_f16(bb)")
    o.append("nib<Int> = (((ld_i8(bb + 106 + sub / 2) as Int) & 255) >> (4 * (sub % 2))) & 15")
    o.append("db<Float32> = dd * ((1 + 2 * nib) as Float32)")
    o.append("qhb<Int> = (ld_i8(bb + 66 + sub) as Int) & 255")
    o.append("sw0<UInt32> = (ld_u16(bb + 74 + 4 * sub) as Int) as UInt32")
    o.append("sw1<UInt32> = (ld_u16(bb + 76 + 4 * sub) as Int) as UInt32")
    for l3 in range(4):
        o.append(f"qp{l3}<Int> = ld_u16(bb + 2 + 8 * sub + {2 * l3}) as Int")
        o.append(f"ia{l3}<Int> = (qp{l3} & 255) | ((qhb << {8 - 2 * l3}) & 256)")
        o.append(f"ib{l3}<Int> = ((qp{l3} >> 8) & 255) | ((qhb << {7 - 2 * l3}) & 256)")
        o.append(f"ga{l3}<UInt32> = ld_u32(G + ia{l3} * 4)")
        o.append(f"gb{l3}<UInt32> = ld_u32(G + ib{l3} * 4)")
        o.append(f"sg{l3}<UInt32> = sw{l3 // 2} >> {8 * (l3 % 2)}")
        _fw(o, "db", f"ga{l3}", f"sg{l3}", 8 * l3)
        _fw(o, "db", f"gb{l3}", f"sg{l3}", 8 * l3 + 4)

TYPES = {
    "q4k": (lambda o: deq_q4k(o, False), "(K / 256) * 144"),
    "q5k": (lambda o: deq_q4k(o, True), "(K / 256) * 176"),
    "q6k": (deq_q6k, "(K / 256) * 210"),
    "q80": (deq_q80, "(K / 32) * 34"),
    "iq2s": (deq_iq2s, "(K / 256) * 82"),
    "iq3s": (deq_iq3s, "(K / 256) * 110"),
}

VDEQ = {"iq2s": None, "iq3s": None}  # filled below (--vdeq: vsub4 sign-folded IQ dequant)


# --cpa (with --xh, needs vyb-wt cp_async_*): cp.async f16 A tile; bit-identical but slower (578 vs 620 tok/s pp512;
# single-buffered, no static-shared room for a 2nd 128-row A tile). Not used by default.
def gen(name, deq, rowbytes, BM=64, PF=False, GSH=False, XH=False, CPA=False):
    MI = BM // 32
    BOFF = BM * 144
    o = []
    I = lambda s, n=1: o.append("    " * n + s)
    o.append(f"gemm_{name}(p<Int>) -> {{")
    for nm, off in [("W", 0), ("X", 8), ("Y", 16), ("S", 24), ("N", 32), ("K", 40), ("LDY", 48), ("RS", 56), ("G", 64), ("NMT", 72)]:
        I(f"{nm}<Int> = ld_i64(p + {off})")
    for s in ["tid<Int> = tid_x()", "lane<Int> = tid % 32", "wp<Int> = tid / 32", "wm<Int> = wp / 4", "wn<Int> = wp % 4",
              "g<Int> = lane / 4", "t4<Int> = lane % 4", "bid<Int> = blk_x()", f"m0<Int> = (bid % NMT) * {BM}",
              "n0<Int> = (bid / NMT) * 128", "ar<Int> = tid / 4", "akq<Int> = (tid % 4) * 16", "arow<Int> = m0 + ar",
              "nl<Int> = tid / 2", "hf<Int> = tid % 2", "nrow<Int> = n0 + nl", f"rb<Int> = W + nrow * {rowbytes}",
              "aso<Int> = ar * 144 + akq * 2", f"bso<Int> = {BOFF} + nl * 144 + hf * 64",
              f"fa<Int> = (wm * {BM // 2} + g) * 144 + t4 * 4", f"fb<Int> = {BOFF} + (wn * 32 + g) * 144 + t4 * 4"]:
        I(s)
    GB = {"iq2s": 8192, "iq3s": 2048}.get(name, 0) if GSH else 0
    GOFF = BOFF + 128 * 144
    if GB:
        # --gsh: copy the IQ grid table (GB bytes) into shared memory after the A/B tiles and read it there
        for c in range(0, GB // 16, 256):
            I(f"if (tid + {c} < {GB // 16}) {{")
            I(f"st_shared_v4_u32({GOFF} + (tid + {c}) * 16, ld_v4_u32(G + (tid + {c}) * 16, 0), ld_v4_u32(G + (tid + {c}) * 16, 1), ld_v4_u32(G + (tid + {c}) * 16, 2), ld_v4_u32(G + (tid + {c}) * 16, 3))", 2)
            I("}")
        I("kernel_barrier()")
        deq0 = deq
        def deq(d):
            t = []
            deq0(t)
            t = [re.sub(r"ld_v2_u32\(G \+ (\w+) \* 8, 0\)", lambda m: f"ld_shared_u32({GOFF} + {m.group(1)} * 8)", x) for x in t]
            t = [re.sub(r"ld_v2_u32\(G \+ (\w+) \* 8, 1\)", lambda m: f"ld_shared_u32({GOFF + 4} + {m.group(1)} * 8)", x) for x in t]
            t = [re.sub(r"ld_u32\(G \+ (\w+) \* 4\)", lambda m: f"ld_shared_u32({GOFF} + {m.group(1)} * 4)", x) for x in t]
            d.extend(t)
    for mi in range(MI):
        for ni in range(4):
            for e in range(4):
                I(f"c{mi}{ni}{e}<Float32> = 0.0")
    def tile_load(KE, n, decl):
        # global loads + dequant of the k-tile at KE into registers xu*/wq* (declared at function scope)
        for pa in range(BM // 64):
            for i in range(8):
                I(f"xu{i}_{pa} = 0 as UInt32", n)
            I(f"if (arow + {64 * pa} < S) {{", n)
            I(f"xa{pa}<Int> = X + ((arow + {64 * pa}) * K + {KE} + akq) * 4", n + 1)
            for v in range(4):
                for e in range(4):
                    I(f"xf{4 * v + e}_{pa}<Float32> = ld_v4_f32(xa{pa} + {16 * v}, {e})", n + 1)
            for i in range(8):
                I(f"xu{i}_{pa} = h2_pack(xf{2 * i}_{pa}, xf{2 * i + 1}_{pa})", n + 1)
            I("}", n)
        for i in range(16):
            I(f"wq{i} = 0 as UInt32", n)
        I("if (nrow < N) {", n)
        I(f"kofs<Int> = {KE} + 32 * hf", n + 1)
        I("sbi<Int> = kofs / 256", n + 1)
        I("sub<Int> = (kofs % 256) / 32", n + 1)
        d = []
        deq(d)
        for s_ in d:
            I(s_, n + 1)
        for i in range(16):
            I(f"wq{i} = h2_pack(dw{2 * i}, dw{2 * i + 1})", n + 1)
        I("}", n)
    def tile_store(n):
        for pa in range(BM // 64):
            I(f"st_shared_v4_u32(aso + {64 * 144 * pa}, xu0_{pa}, xu1_{pa}, xu2_{pa}, xu3_{pa})", n)
            I(f"st_shared_v4_u32(aso + {64 * 144 * pa + 16}, xu4_{pa}, xu5_{pa}, xu6_{pa}, xu7_{pa})", n)
        for v in range(4):
            I(f"st_shared_v4_u32(bso + {16 * v}, wq{4 * v}, wq{4 * v + 1}, wq{4 * v + 2}, wq{4 * v + 3})", n)
    if PF:
        # register prefetch: the next tile's global loads + dequant are issued before this tile's MMAs
        for pa in range(BM // 64):
            for i in range(8):
                I(f"xu{i}_{pa}<UInt32> = 0 as UInt32")
        for i in range(16):
            I(f"wq{i}<UInt32> = 0 as UInt32")
        I("if (K > 0) {")
        tile_load("0", 2, False)
        I("}")
        tile_store(1)
        I("kernel_barrier()")
        I("k0<Int> = 0")
        I("while (k0 < K) {")
        I("if (k0 + 64 < K) {", 2)
        tile_load("k0 + 64", 3, False)
        I("}", 2)
    else:
        if CPA:
            # --cpa (with --xh): the f16 A tile is copied global->shared with cp.async (no registers, no st.shared);
            # rows >= S are zeroed once (never overwritten), the next k-tile's copy is issued right after the
            # post-MMA barrier so it overlaps the next B dequant. Same bytes in shared -> bit-identical.
            for pa in range(BM // 64):
                I(f"if (arow + {64 * pa} >= S) {{")
                z = "0 as UInt32"
                I(f"st_shared_v4_u32(aso + {64 * 144 * pa}, {z}, {z}, {z}, {z})", 2)
                I(f"st_shared_v4_u32(aso + {64 * 144 * pa + 16}, {z}, {z}, {z}, {z})", 2)
                I("} else {")
                I(f"xa{pa}<Int> = X + ((arow + {64 * pa}) * K + akq) * 2", 2)
                I(f"cp_async_16(aso + {64 * 144 * pa}, xa{pa})", 2)
                I(f"cp_async_16(aso + {64 * 144 * pa + 16}, xa{pa} + 16)", 2)
                I("}")
            I("cp_async_commit()")
        I("k0<Int> = 0")
        I("while (k0 < K) {")
    if CPA:
        pass
    elif not PF and XH:
        # --xh: X is f16 [S][K] (converted once by cvt_h): two 16-byte loads straight into the A fragment words
        for pa in range(BM // 64):
            for i in range(8):
                I(f"xu{i}_{pa}<UInt32> = 0 as UInt32", 2)
            I(f"if (arow + {64 * pa} < S) {{", 2)
            I(f"xa{pa}<Int> = X + ((arow + {64 * pa}) * K + k0 + akq) * 2", 3)
            for i in range(8):
                I(f"xu{i}_{pa} = ld_v4_u32(xa{pa} + {16 * (i // 4)}, {i % 4})", 3)
            I("}", 2)
    elif not PF:
        for pa in range(BM // 64):
            for i in range(8):
                I(f"xu{i}_{pa}<UInt32> = 0 as UInt32", 2)
            I(f"if (arow + {64 * pa} < S) {{", 2)
            I(f"xa{pa}<Int> = X + ((arow + {64 * pa}) * K + k0 + akq) * 4", 3)
            for v in range(4):
                for e in range(4):
                    I(f"xf{4 * v + e}_{pa}<Float32> = ld_v4_f32(xa{pa} + {16 * v}, {e})", 3)
            for i in range(8):
                I(f"xu{i}_{pa} = h2_pack(xf{2 * i}_{pa}, xf{2 * i + 1}_{pa})", 3)
            I("}", 2)
    if not PF:
        for pa in range(BM // 64 if not CPA else 0):
            I(f"st_shared_v4_u32(aso + {64 * 144 * pa}, xu0_{pa}, xu1_{pa}, xu2_{pa}, xu3_{pa})", 2)
            I(f"st_shared_v4_u32(aso + {64 * 144 * pa + 16}, xu4_{pa}, xu5_{pa}, xu6_{pa}, xu7_{pa})", 2)
        for i in range(16):
            I(f"wq{i}<UInt32> = 0 as UInt32", 2)
        I("if (nrow < N) {", 2)
        I("kofs<Int> = k0 + 32 * hf", 3)
        I("sbi<Int> = kofs / 256", 3)
        I("sub<Int> = (kofs % 256) / 32", 3)
        d = []
        deq(d)
        for s in d:
            I(s, 3)
        for i in range(16):
            I(f"wq{i} = h2_pack(dw{2 * i}, dw{2 * i + 1})", 3)
        I("}", 2)
        for v in range(4):
            I(f"st_shared_v4_u32(bso + {16 * v}, wq{4 * v}, wq{4 * v + 1}, wq{4 * v + 2}, wq{4 * v + 3})", 2)
    if not PF:
        if CPA:
            I("cp_async_wait(0)", 2)
        I("kernel_barrier()", 2)
    for kk in range(4):
        for mi in range(MI):
            base = f"fa + {mi * 16 * 144 + kk * 32}"
            I(f"a{mi}0_{kk}<UInt32> = ld_shared_u32({base})", 2)
            I(f"a{mi}1_{kk}<UInt32> = ld_shared_u32({base} + {8 * 144})", 2)
            I(f"a{mi}2_{kk}<UInt32> = ld_shared_u32({base} + 16)", 2)
            I(f"a{mi}3_{kk}<UInt32> = ld_shared_u32({base} + {8 * 144 + 16})", 2)
        for ni in range(4):
            base = f"fb + {ni * 8 * 144 + kk * 32}"
            I(f"b{ni}0_{kk}<UInt32> = ld_shared_u32({base})", 2)
            I(f"b{ni}1_{kk}<UInt32> = ld_shared_u32({base} + 16)", 2)
        for mi in range(MI):
            for ni in range(4):
                c = [f"c{mi}{ni}{e}" for e in range(4)]
                args = f"a{mi}0_{kk}, a{mi}1_{kk}, a{mi}2_{kk}, a{mi}3_{kk}, b{ni}0_{kk}, b{ni}1_{kk}, {', '.join(c)}"
                for e in range(4):
                    I(f"n{mi}{ni}{e}_{kk}<Float32> = mma_16816({args}, {e})", 2)
                for e in range(4):
                    I(f"c{mi}{ni}{e} = n{mi}{ni}{e}_{kk}", 2)
    I("kernel_barrier()", 2)
    if CPA:
        I("if (k0 + 64 < K) {", 2)
        for pa in range(BM // 64):
            I(f"if (arow + {64 * pa} < S) {{", 3)
            I(f"xn{pa}<Int> = X + ((arow + {64 * pa}) * K + k0 + 64 + akq) * 2", 4)
            I(f"cp_async_16(aso + {64 * 144 * pa}, xn{pa})", 4)
            I(f"cp_async_16(aso + {64 * 144 * pa + 16}, xn{pa} + 16)", 4)
            I("}", 3)
        I("cp_async_commit()", 3)
        I("}", 2)
    if PF:
        I("if (k0 + 64 < K) {", 2)
        tile_store(3)
        I("}", 2)
        I("kernel_barrier()", 2)
    I("k0 = k0 + 64", 2)
    I("}")
    for mi in range(MI):
        for hh in range(2):
            I(f"r{mi}{hh}<Int> = m0 + wm * {BM // 2} + {mi * 16 + hh * 8} + g")
            I(f"if (r{mi}{hh} < S) {{")
            for ni in range(4):
                for e in range(2):
                    col = f"n0 + wn * 32 + {ni * 8 + e} + t4 * 2"
                    I(f"cl{mi}{hh}{ni}{e}<Int> = {col}", 2)
                    I(f"if (cl{mi}{hh}{ni}{e} < N) {{", 2)
                    # LDY < 0: Y is f32 with row stride -LDY (no RS); exact when the consumer casts f64 -> f32 anyway
                    I("if (LDY < 0) {", 3)
                    I(f"st_f32n(Y + (r{mi}{hh} * (0 - LDY) + cl{mi}{hh}{ni}{e}) * 4, c{mi}{ni}{2 * hh + e})", 4)
                    I("} else {", 3)
                    I(f"yo{mi}{hh}{ni}{e}<Int> = (r{mi}{hh} * LDY + cl{mi}{hh}{ni}{e}) * 8", 4)
                    I(f"yv{mi}{hh}{ni}{e}<Float> = c{mi}{ni}{2 * hh + e} as Float", 4)
                    I(f"if (RS != 0) {{ yv{mi}{hh}{ni}{e} = yv{mi}{hh}{ni}{e} + ld_f64(RS + yo{mi}{hh}{ni}{e}) }}", 4)
                    I(f"st_f64(Y + yo{mi}{hh}{ni}{e}, yv{mi}{hh}{ni}{e})", 4)
                    I("}", 3)
                    I("}", 2)
            I("}")
    o.append("}")
    return "\n".join(o)

def gen_delta_seq():
    """delta_seq: the gated delta rule over a whole chunk of tokens, state resident in registers.
    Grid HV*4 blocks x 128 threads: block (h, quarter) owns 32 state rows j; thread (jl, qt) holds
    S[j][32qt .. 32qt+31] (f64).  Per token: k/q of head h%HK to shared, row dots by 4-lane
    shuffles, out[j] = sc*(g*S.q + del*k.q), S' = g*S + del*k (the delta_step_b maths).
    Params: Q@0 K@8 (row stride GKS) V@16 (row stride QKS) G@24 BE@32 ([T][HV]) ST@40 ([HV][SS][SS])
    OUT@48 ([T][HV*SS]) NT@56 SS@64 HK@72 HV@80 sc@88 QKS@96 GKS@104.  SS must be 128."""
    o = ["delta_seq(p<Int>) -> {"]
    I = lambda s, n=1: o.append("    " * n + s)
    for nm, off in [("Q", 0), ("K", 8), ("V", 16), ("G", 24), ("BE", 32), ("ST", 40), ("OUT", 48), ("NT", 56),
                    ("SSd", 64), ("HK", 72), ("HV", 80), ("QKS", 96), ("GKS", 104)]:
        I(f"{nm}<Int> = ld_i64(p + {off})")
    I("sc<Float> = ld_f64(p + 88)")
    for s in ["tid<Int> = tid_x()", "b<Int> = blk_x()", "h<Int> = b / 4", "j<Int> = (b % 4) * 32 + tid / 4",
              "qt<Int> = tid % 4", "hk<Int> = h % HK", "ib<Int> = qt * 32", "sb<Int> = ST + (SSd * SSd * h + j * SSd + ib) * 8"]:
        I(s)
    for e in range(32):
        I(f"s{e}<Float> = ld_f64(sb + {8 * e})")
    I("t<Int> = 0")
    I("while (t < NT) {")
    I("st_shared_f64(tid, ld_f64(K + (t * GKS + hk * SSd + tid) * 8))", 2)
    I("st_shared_f64(128 + tid, ld_f64(Q + (t * GKS + hk * SSd + tid) * 8))", 2)
    I("kernel_barrier()", 2)
    I("acc<Float> = 0.0", 2)
    I("acq<Float> = 0.0", 2)
    I("kq<Float> = 0.0", 2)
    for e in range(32):
        I(f"k{e}<Float> = ld_shared_f64(ib + {e})", 2)
        I(f"q{e}<Float> = ld_shared_f64(128 + ib + {e})", 2)
        I(f"acc = acc + s{e} * k{e}", 2)
        I(f"acq = acq + s{e} * q{e}", 2)
        I(f"kq = kq + k{e} * q{e}", 2)
    for v in ["acc", "acq", "kq"]:
        I(f"{v} = {v} + shfl_xor_f64({v}, 1)", 2)
        I(f"{v} = {v} + shfl_xor_f64({v}, 2)", 2)
    I("gg<Float> = vexp(ld_f64(G + (t * HV + h) * 8))", 2)
    I("be<Float> = ld_f64(BE + (t * HV + h) * 8)", 2)
    I("del<Float> = (ld_f64(V + (t * QKS + h * SSd + j) * 8) - gg * acc) * be", 2)
    I("if (qt == 0) { st_f64(OUT + (t * HV * SSd + h * SSd + j) * 8, sc * (gg * acq + del * kq)) }", 2)
    for e in range(32):
        I(f"s{e} = gg * s{e} + del * k{e}", 2)
    I("kernel_barrier()", 2)
    I("t = t + 1", 2)
    I("}")
    for e in range(32):
        I(f"st_f64(sb + {8 * e}, s{e})")
    o.append("}")
    return "\n".join(o)


VDEQ["iq2s"] = deq_iq2s_v
VDEQ["iq3s"] = deq_iq3s_v


if __name__ == "__main__":
    BM = 128 if "--bm128" in sys.argv else 64
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(TYPES)
    out = ["# GENERATED by native/tools/gen_rpf.py -- tensor-core GEMMs over packed GGUF weights (Ridge batched prefill).",
           "# " + __doc__.strip().replace("\n", "\n# "), "import vmath::{vexp}", ""]
    for n in names:
        dq, rbx = TYPES[n]
        if "--vdeq" in sys.argv and n in VDEQ:
            dq = VDEQ[n]
        out.append(gen(n, dq, rbx, BM=BM, PF="--pf" in sys.argv, GSH="--gsh" in sys.argv, XH="--xh" in sys.argv,
                       CPA="--cpa" in sys.argv and "--xh" in sys.argv and "--pf" not in sys.argv))
        out.append("")
    if "--xh" in sys.argv:
        out += ["// cvt_h: f32 -> f16 copy of a GEMM activation (once per producer, not once per N tile)",
                "cvt_h(src<Int>, dst<Int>, n<Int>, z<Int>) -> {",
                "    gid<Int> = blk_x() * dim_x() + tid_x()",
                "    if (gid < n) { st_f16(dst + gid * 2, ld_f32(src + gid * 4)) }", "}", ""]
    if "--ds" in sys.argv or not names or not [a for a in sys.argv[1:] if not a.startswith("-")]:
        out.append(gen_delta_seq())
        out.append("")
    print("\n".join(out))
