# Generates native/kernels/rgemv2.vyb: R-rows-per-warp variants of the IQ2_S / IQ3_S fused GEMVs
# (x is loaded once per super-block and reused across R rows), plus block-parallel GDN kernels.
import sys
R = int(sys.argv[1]) if len(sys.argv) > 1 else 2
out = []
w = out.append
w(f"""// GENERATED (R={R} rows per warp). Ridge resident-decode kernels, second generation:
//  * gemv_iq2s_r / gemv_iq3s_r: the rgemv.vyb lane map, but each warp owns {R} consecutive rows and
//    loads x once per super-block for all of them (the 1-row kernels were LSU-bound on the x loads:
//    8 f32 loads per 8 weights at 2.56-3.4 bits/weight). Launch ceil(N/{16*R}) blocks x 512.
//  * l2norm_b / norm_gated_b: one block per row, shared-memory tree reduction (the gdn.vyb versions
//    recompute the whole row sum in every thread).
//  * delta_step_c: delta_step_b split 4-ways along the state row and 32 rows per block, so the
//    {{48 x 128 x 128}} f64 state update runs on 192 blocks instead of 48. In-place safe (each state
//    element is read then written by the same thread). Launch Hv*S/32 blocks x 128.
// Same parameter blocks as rgemv.vyb / gdn.vyb. Derived from llama.cpp (MIT, see THIRD-PARTY.md).

import vmath::{{vexp, vsqrt}}
""")

def hdr(name):
    w(f"""{name}(p<Int>) -> {{
    W<Int> = ld_i64(p + 0)
    X<Int> = ld_i64(p + 8)
    Y<Int> = ld_i64(p + 16)
    N<Int> = ld_i64(p + 32)
    K<Int> = ld_i64(p + 40)
    YE<Int> = ld_i64(p + 48)
    RS<Int> = ld_i64(p + 120)
    G<Int> = ld_i64(p + 128)
    tid<Int> = tid_x()
    l<Int> = tid % 32
    n0<Int> = blk_x() * {16*R} + (tid / 32) * {R}
    nsb<Int> = K / 256
    c1<Float32> = 1.0 as Float32
    c2<Float32> = 2.0 as Float32
    ch<Float32> = 0.5 as Float32
    cq<Float32> = 0.25 as Float32""")
    for r in range(R):
        w(f"    acc{r}<Float32> = 0.0")

def xloads(ind):
    w(f"{ind}xb<Int> = xr + sb * 1024")
    for e in range(8):
        w(f"{ind}x{e}<Float32> = ld_f32(xb + {4*e})")

def dot(ind, r, scale):
    for e in range(8):
        sh = "" if e % 4 == 0 else f" >> {8*(e%4)}"
        gsrc = f"g0_{r}" if e < 4 else f"g1_{r}"
        gv = f"(({gsrc}{sh}) & 255)"
        sgb = f"(sg_{r} & 1)" if e == 0 else f"((sg_{r} >> {e}) & 1)"
        w(f"{ind}w{e}_{r}<Float32> = ({gv} as Float32) * (c1 - c2 * ({sgb} as Float32))")
    w(f"{ind}pt{r}<Float32> = x0 * w0_{r}")
    for e in range(1, 8):
        w(f"{ind}pt{r} = pt{r} + x{e} * w{e}_{r}")
    w(f"{ind}acc{r} = acc{r} + {scale} * pt{r}")

def tail():
    for r in range(R):
        decl = "st2<Int> = 16" if r == 0 else "st2 = 16"
        w(f"""    st_shared_f64(tid, acc{r} as Float)
    kernel_barrier()
    {decl}
    while (st2 > 0) {{
        if (l < st2) {{ st_shared_f64(tid, ld_shared_f64(tid) + ld_shared_f64(tid + st2)) }}
        kernel_barrier()
        st2 = st2 / 2
    }}
    if (l == 0) {{
        if (n0 + {r} < N) {{
            if (YE == 4) {{ st_f32(Y + (n0 + {r}) * 4, ld_shared_f64(tid) as Float32) }} else {{
                yv{r}<Float> = ld_shared_f64(tid)
                if (RS != 0) {{ yv{r} = ld_f64(RS + (n0 + {r}) * 8) + yv{r} }}
                st_f64(Y + (n0 + {r}) * 8, yv{r})
            }}
        }}
    }}
    kernel_barrier()""")
    w("}\n")

# ---- IQ2_S ----
hdr("gemv_iq2s_r")
w(f"""    if (n0 < N) {{
        ib<Int> = l / 4
        l2<Int> = l % 4
        shs<Int> = (l2 / 2) * 4
        shq<Int> = 8 - 2 * l2
        xr<Int> = X + l * 32
        rbs<Int> = nsb * 82
        sb<Int> = 0
        while (sb < nsb) {{""")
xloads("            ")
for r in range(R):
    ind = "            "
    w(f"{ind}bb{r}<Int> = W + (n0 + {r}) * rbs + sb * 82")
    if r > 0:
        w(f"{ind}if (n0 + {r} >= N) {{ bb{r} = bb0 }}")
    w(f"{ind}d{r}<Float32> = ld_f16(bb{r})")
    w(f"{ind}scb{r}<Int> = (ld_i8(bb{r} + 74 + ib) as Int) & 255")
    w(f"{ind}dl{r}<Float32> = d{r} * ((((scb{r} >> shs) & 15) as Float32) + ch) * cq")
    w(f"{ind}idx{r}<Int> = ((ld_i8(bb{r} + 2 + l) as Int) & 255) | ((((ld_i8(bb{r} + 66 + ib) as Int) & 255) << shq) & 768)")
    w(f"{ind}g0_{r}<UInt32> = ld_i32(G + idx{r} * 8) as UInt32")
    w(f"{ind}g1_{r}<UInt32> = ld_i32(G + idx{r} * 8 + 4) as UInt32")
    w(f"{ind}sg_{r}<UInt32> = (((ld_i8(bb{r} + 34 + l) as Int) & 255) as UInt32)")
    dot(ind, r, f"dl{r}")
w("""            sb = sb + 1
        }
    }""")
tail()

# ---- IQ3_S ----
hdr("gemv_iq3s_r")
w(f"""    if (n0 < N) {{
        g<Int> = l / 8
        hf<Int> = (l % 8) / 4
        l3<Int> = l % 4
        shs<Int> = hf * 4
        sh1<Int> = 8 - 2 * l3
        sh2<Int> = 7 - 2 * l3
        xr<Int> = X + l * 32
        rbs<Int> = nsb * 110
        sb<Int> = 0
        while (sb < nsb) {{""")
xloads("            ")
for r in range(R):
    ind = "            "
    w(f"{ind}bb{r}<Int> = W + (n0 + {r}) * rbs + sb * 110")
    if r > 0:
        w(f"{ind}if (n0 + {r} >= N) {{ bb{r} = bb0 }}")
    w(f"{ind}d{r}<Float32> = ld_f16(bb{r})")
    w(f"{ind}scb{r}<Int> = (ld_i8(bb{r} + 106 + g) as Int) & 255")
    w(f"{ind}db{r}<Float32> = d{r} * (c1 + c2 * (((scb{r} >> shs) & 15) as Float32))")
    w(f"{ind}qhb{r}<Int> = (ld_i8(bb{r} + 66 + 2 * g + hf) as Int) & 255")
    w(f"{ind}qso{r}<Int> = bb{r} + 2 + 16 * g + 8 * hf + 2 * l3")
    w(f"{ind}i1_{r}<Int> = ((ld_i8(qso{r}) as Int) & 255) | ((qhb{r} << sh1) & 256)")
    w(f"{ind}i2_{r}<Int> = ((ld_i8(qso{r} + 1) as Int) & 255) | ((qhb{r} << sh2) & 256)")
    w(f"{ind}g0_{r}<UInt32> = ld_i32(G + i1_{r} * 4) as UInt32")
    w(f"{ind}g1_{r}<UInt32> = ld_i32(G + i2_{r} * 4) as UInt32")
    w(f"{ind}sg_{r}<UInt32> = (((ld_i8(bb{r} + 74 + 8 * g + 4 * hf + l3) as Int) & 255) as UInt32)")
    dot(ind, r, f"db{r}")
w("""            sb = sb + 1
        }
    }""")
tail()

w("""// l2norm over q (rows1 rows) and k (rows1 rows) in one launch: +0 X1 +8 Y1 +16 X2 +24 Y2
// +32 rows1 +40 D +48 eps. Grid 2*rows1 blocks x D threads (D a power of two <= 512).
l2norm_b(p<Int>) -> {
    X1<Int> = ld_i64(p + 0)
    Y1<Int> = ld_i64(p + 8)
    X2<Int> = ld_i64(p + 16)
    Y2<Int> = ld_i64(p + 24)
    rows1<Int> = ld_i64(p + 32)
    D<Int> = ld_i64(p + 40)
    eps<Float> = ld_f64(p + 48)
    r<Int> = blk_x()
    t<Int> = tid_x()
    X<Int> = X1
    Y<Int> = Y1
    rr<Int> = r
    if (r >= rows1) {
        X = X2
        Y = Y2
        rr = r - rows1
    }
    v<Float> = ld_f64(X + (rr * D + t) * 8)
    st_shared_f64(t, v * v)
    kernel_barrier()
    st2<Int> = D / 2
    while (st2 > 0) {
        if (t < st2) { st_shared_f64(t, ld_shared_f64(t) + ld_shared_f64(t + st2)) }
        kernel_barrier()
        st2 = st2 / 2
    }
    s<Float> = vsqrt(ld_shared_f64(0), 40)
    if (s < eps) { s = eps }
    st_f64(Y + (rr * D + t) * 8, v / s)
}

// norm_gated with one block per row: same params as gdn.vyb norm_gated. Grid rows x D threads.
norm_gated_b(p<Int>) -> {
    X<Int> = ld_i64(p + 0)
    Z<Int> = ld_i64(p + 8)
    W<Int> = ld_i64(p + 16)
    Y<Int> = ld_i64(p + 24)
    D<Int> = ld_i64(p + 40)
    eps<Float> = ld_f64(p + 48)
    r<Int> = blk_x()
    t<Int> = tid_x()
    v<Float> = ld_f64(X + (r * D + t) * 8)
    st_shared_f64(t, v * v)
    kernel_barrier()
    st2<Int> = D / 2
    while (st2 > 0) {
        if (t < st2) { st_shared_f64(t, ld_shared_f64(t) + ld_shared_f64(t + st2)) }
        kernel_barrier()
        st2 = st2 / 2
    }
    rinv<Float> = 1.0 / vsqrt(ld_shared_f64(0) / (D as Float) + eps, 40)
    wv<Float> = ld_f64(W + t * 8)
    zv<Float> = ld_f64(Z + (r * D + t) * 8)
    sg<Float> = 1.0 + vexp(0.0 - zv)
    st_f64(Y + (r * D + t) * 8, v * rinv * wv * (zv / sg))
}

// delta_step_b, 4 threads per state row, 32 rows per block. Same params. Grid Hv*S/32 x 128.
delta_step_c(p<Int>) -> {
    Q<Int> = ld_i64(p + 0)
    Kp<Int> = ld_i64(p + 8)
    V<Int> = ld_i64(p + 16)
    G<Int> = ld_i64(p + 24)
    BE<Int> = ld_i64(p + 32)
    SIN<Int> = ld_i64(p + 40)
    SOUT<Int> = ld_i64(p + 48)
    OUT<Int> = ld_i64(p + 56)
    S<Int> = ld_i64(p + 72)
    Hk<Int> = ld_i64(p + 80)
    sc<Float> = ld_f64(p + 96)
    b<Int> = blk_x()
    bph<Int> = S / 32
    h<Int> = b / bph
    rbase<Int> = (b % bph) * 32
    t<Int> = tid_x()
    jr<Int> = t / 4
    q<Int> = t % 4
    j<Int> = rbase + jr
    hk<Int> = h % Hk
    i0<Int> = t
    while (i0 < S) {
        st_shared_f64(i0, ld_f64(Kp + (i0 + S * hk) * 8))
        st_shared_f64(256 + i0, ld_f64(Q + (i0 + S * hk) * 8))
        i0 = i0 + 128
    }
    kernel_barrier()
    gexp<Float> = vexp(ld_f64(G + h * 8))
    rw<Int> = SIN + (S * S * h + j * S) * 8
    acc<Float> = 0.0
    acq<Float> = 0.0
    m<Int> = q
    while (m < S) {
        sv<Float> = ld_f64(rw + m * 8)
        acc = acc + sv * ld_shared_f64(m)
        acq = acq + sv * ld_shared_f64(256 + m)
        m = m + 4
    }
    st_shared_f64(512 + t, acc)
    st_shared_f64(1024 + t, acq)
    kernel_barrier()
    if (q == 0) {
        a<Float> = ld_shared_f64(512 + t) + ld_shared_f64(513 + t) + ld_shared_f64(514 + t) + ld_shared_f64(515 + t)
        aq<Float> = ld_shared_f64(1024 + t) + ld_shared_f64(1025 + t) + ld_shared_f64(1026 + t) + ld_shared_f64(1027 + t)
        kq<Float> = 0.0
        i<Int> = 0
        for (i in 0..S - 1) { kq = kq + ld_shared_f64(i) * ld_shared_f64(256 + i) }
        be<Float> = ld_f64(BE + h * 8)
        del<Float> = (ld_f64(V + (j + S * h) * 8) - gexp * a) * be
        st_f64(OUT + (j + S * h) * 8, sc * (gexp * aq + del * kq))
        st_shared_f64(1536 + jr, del)
    }
    kernel_barrier()
    del2<Float> = ld_shared_f64(1536 + jr)
    ww<Int> = SOUT + (S * S * h + j * S) * 8
    m = q
    while (m < S) {
        sv3<Float> = ld_f64(rw + m * 8)
        st_f64(ww + m * 8, gexp * sv3 + del2 * ld_shared_f64(m))
        m = m + 4
    }
}
""")
open(sys.argv[2] if len(sys.argv) > 2 else "rgemv2.vyb", "w").write("\n".join(out))
