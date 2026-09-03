#!/usr/bin/env python3
"""Rebuild native/train/kvresp_train_kv.vyb from the committed kvresp_train.vyb (verified, 0d03267)
by applying (A) per-token helper insertion (proj_lor + resp_layer_kv, resp_layer_kv modified for
per-projection LoRA s capture), (B) CK/CV/RSHI allocations, (C) per-token response forward replacing
the batched run_layer forward. Uses direct open() so nothing truncates. Output written atomically.
"""
import os
src_path = "native/train/kvresp_train.vyb"
dst_path = "native/train/kvresp_train_kv.vyb"
src = open(src_path).read()

SLAYOUT = 50292912
# ASLB per-layer byte offsets (field order, f64):
OFF = dict(XI=0, XN=1904640, DQ=3809280, DK=6856704, DV=7618560, DQN=8380416, DKN=11427840,
           DQr=12189696, DKr=15237120, DCtx=15998976, DO=19046400, X1=20951040, X1N=22855680,
           Gr=24760320, Up=31997952, Hu=39235584, M2=46473216, XOcur=48377856,
           sq=50282496, sk=50283984, sv=50285472, so=50286960, sg=50288448, su=50289936, sd=50291424)

# ---------------- (A) helpers before main() ----------------
assert src.count("main()<Int> -> {") == 1
helpers = '''proj_lor(xa<Int>, W<Int>, out<Int>, U<Int>, V<Int>, slo<Int>, shi<Int>,
         M<Int>, N<Int>, K<Int>, R2<Int>, AR<Float>,
         gf<loc<CVoid>>, ads<loc<CVoid>>, GM<Int>, PB<Int>)<Int> -> {
    freedom {
        AF<Float> = 1.0; BF<Float> = 0.0
        cuMemcpyHtoD_v2(GM+0,loc(xa),8); cuMemcpyHtoD_v2(GM+8,loc(W),8); cuMemcpyHtoD_v2(GM+16,loc(out),8); cuMemcpyHtoD_v2(GM+24,loc(M),8); cuMemcpyHtoD_v2(GM+32,loc(N),8); cuMemcpyHtoD_v2(GM+40,loc(K),8); cuMemcpyHtoD_v2(GM+48,loc(AF),8); cuMemcpyHtoD_v2(GM+56,loc(BF),8)
        cuda_launch4i(from<loc<CVoid>>(addr(gf)),(M*N+255)/256,1,1,256,1,1,GM,0,0,0)
        cuMemcpyHtoD_v2(GM+0,loc(xa),8); cuMemcpyHtoD_v2(GM+8,loc(U),8); cuMemcpyHtoD_v2(GM+16,loc(slo),8); cuMemcpyHtoD_v2(GM+24,loc(M),8); cuMemcpyHtoD_v2(GM+32,loc(R2),8); cuMemcpyHtoD_v2(GM+40,loc(K),8); cuMemcpyHtoD_v2(GM+48,loc(AF),8); cuMemcpyHtoD_v2(GM+56,loc(BF),8)
        cuda_launch4i(from<loc<CVoid>>(addr(gf)),(M*R2+255)/256,1,1,256,1,1,GM,0,0,0)
        cuMemcpyHtoD_v2(GM+0,loc(slo),8); cuMemcpyHtoD_v2(GM+8,loc(V),8); cuMemcpyHtoD_v2(GM+16,loc(shi),8); cuMemcpyHtoD_v2(GM+24,loc(M),8); cuMemcpyHtoD_v2(GM+32,loc(N),8); cuMemcpyHtoD_v2(GM+40,loc(R2),8); cuMemcpyHtoD_v2(GM+48,loc(AF),8); cuMemcpyHtoD_v2(GM+56,loc(BF),8)
        cuda_launch4i(from<loc<CVoid>>(addr(gf)),(M*N+255)/256,1,1,256,1,1,GM,0,0,0)
        cuMemcpyHtoD_v2(PB+0,loc(out),8); cuMemcpyHtoD_v2(PB+8,loc(shi),8); cuMemcpyHtoD_v2(PB+16,loc(AR),8); cuMemcpyHtoD_v2(PB+24,loc(M*N),8)
        cuda_launch4i(from<loc<CVoid>>(addr(ads)),(M*N+255)/256,1,1,256,1,1,PB,0,0,0)
        return 0
    }
    return 0
}

// resp_layer_kv: per-token KV forward helper (ported from verified kvrespfwd). One response/context
// token at absolute position P through ONE layer, LoRA on all 7 projections. Writes per-token
// activations into the passed buffer args at row P; stores this position's roped-K (DKr) and V (DV)
// into CK/CV row P; runs attention over the combined cache (rows 0..P). Modified vs kvrespfwd to
// keep a SEPARATE LoRA low-rank A output per projection (sqB..sdB) so the backward gets sq..sd.
resp_layer_kv(P<Int>, Sctx<Int>, D<Int>, H<Int>, KVH<Int>, HD<Int>, FF<Int>, NQ<Int>, NKV<Int>,
              R2<Int>, EPS<Float>, AR<Float>, HB<Int>, XO<Int>, CK<Int>, CV<Int>,
              gf<loc<CVoid>>, ads<loc<CVoid>>, rn<loc<CVoid>>, rp<loc<CVoid>>, att<loc<CVoid>>,
              sl<loc<CVoid>>, rs<loc<CVoid>>, dcp<loc<CVoid>>, GM<Int>, RN<Int>, ROP<Int>, ATT<Int>, PB<Int>,
              N1<Int>, Wq<Int>, Wk<Int>, Wv<Int>, Nq<Int>, Nk<Int>, DF<Int>, Wo<Int>, N2<Int>,
              Wg<Int>, Wu<Int>, Wd<Int>, Uq<Int>, Vq<Int>, Uk<Int>, Vk<Int>, Uv<Int>, Vv<Int>,
              Uo<Int>, Vo<Int>, Ug<Int>, Vg<Int>, Uu<Int>, Vu<Int>, Ud<Int>, Vd<Int>,
              XN<Int>, DQ<Int>, DK<Int>, DV<Int>, DQN<Int>, DKN<Int>, DQr<Int>, DKr<Int>,
              DCtx<Int>, DO<Int>, X1<Int>, X1N<Int>, Gr<Int>, Up<Int>, Hu<Int>, M2<Int>,
              sqB<Int>, skB<Int>, svB<Int>, soB<Int>, sgB<Int>, suB<Int>, sdB<Int>, shi<Int>)<Int> -> {
    Pp<Int> = P; DFf<Int> = DF; Hf<Int> = H; HDf<Int> = HD; KVHf<Int> = KVH; Df2<Int> = D; NQf<Int> = NQ; NKVf<Int> = NKV
    one<Int> = 1
    freedom {
        e2<Int> = 0
        hbP<Int> = HB + P * D * 8; xoP<Int> = XO + P * D * 8
        xnP<Int> = XN + P * D * 8
        dqP<Int> = DQ + P * NQ * 8; dkP<Int> = DK + P * NKV * 8; dvP<Int> = DV + P * NKV * 8
        dqnP<Int> = DQN + P * NQ * 8; dknP<Int> = DKN + P * NKV * 8
        dqrP<Int> = DQr + P * NQ * 8; dkrP<Int> = DKr + P * NKV * 8
        dctxP<Int> = DCtx + P * NQ * 8; doP<Int> = DO + P * D * 8
        x1P<Int> = X1 + P * D * 8; x1nP<Int> = X1N + P * D * 8
        grP<Int> = Gr + P * FF * 8; upP<Int> = Up + P * FF * 8; huP<Int> = Hu + P * FF * 8; m2P<Int> = M2 + P * D * 8
        sqP<Int> = sqB + P * R2 * 8; skP<Int> = skB + P * R2 * 8; svP<Int> = svB + P * R2 * 8
        soP<Int> = soB + P * R2 * 8; sgP<Int> = sgB + P * R2 * 8; suP<Int> = suB + P * R2 * 8; sdP<Int> = sdB + P * R2 * 8
        shiP<Int> = shi + P * FF * 8
        cuMemcpyHtoD_v2(RN + 0, loc(hbP), 8); cuMemcpyHtoD_v2(RN + 8, loc(N1), 8); cuMemcpyHtoD_v2(RN + 16, loc(xnP), 8)
        cuMemcpyHtoD_v2(RN + 24, loc(one), 8); cuMemcpyHtoD_v2(RN + 32, loc(D), 8); cuMemcpyHtoD_v2(RN + 40, loc(EPS), 8)
        cuda_launch4i(from<loc<CVoid>>(addr(rn)), (D + 255) / 256, 1, 1, 256, 1, 1, RN, 0, 0, 0)
        e2 = proj_lor(xnP, Wq, dqP, Uq, Vq, sqP, shiP, one, NQ, D, R2, AR, gf, ads, GM, PB); if (e2 != 0) { return 90 }
        e2 = proj_lor(xnP, Wk, dkP, Uk, Vk, skP, shiP, one, NKV, D, R2, AR, gf, ads, GM, PB); if (e2 != 0) { return 91 }
        e2 = proj_lor(xnP, Wv, dvP, Uv, Vv, svP, shiP, one, NKV, D, R2, AR, gf, ads, GM, PB); if (e2 != 0) { return 92 }
        cuMemcpyHtoD_v2(RN + 0, loc(dqP), 8); cuMemcpyHtoD_v2(RN + 8, loc(Nq), 8); cuMemcpyHtoD_v2(RN + 16, loc(dqnP), 8)
        cuMemcpyHtoD_v2(RN + 24, loc(H), 8); cuMemcpyHtoD_v2(RN + 32, loc(HD), 8); cuMemcpyHtoD_v2(RN + 40, loc(EPS), 8)
        cuda_launch4i(from<loc<CVoid>>(addr(rn)), (NQ + 255) / 256, 1, 1, 256, 1, 1, RN, 0, 0, 0)
        cuMemcpyHtoD_v2(RN + 0, loc(dkP), 8); cuMemcpyHtoD_v2(RN + 8, loc(Nk), 8); cuMemcpyHtoD_v2(RN + 16, loc(dknP), 8)
        cuMemcpyHtoD_v2(RN + 24, loc(KVH), 8); cuMemcpyHtoD_v2(RN + 32, loc(HD), 8); cuMemcpyHtoD_v2(RN + 40, loc(EPS), 8)
        cuda_launch4i(from<loc<CVoid>>(addr(rn)), (NKV + 255) / 256, 1, 1, 256, 1, 1, RN, 0, 0, 0)
        cuMemcpyHtoD_v2(ROP + 80, loc(dqnP), 8); cuMemcpyHtoD_v2(ROP + 88, loc(dknP), 8); cuMemcpyHtoD_v2(ROP + 96, loc(dqrP), 8); cuMemcpyHtoD_v2(ROP + 104, loc(dkrP), 8)
        cuMemcpyHtoD_v2(ROP + 176, loc(DFf), 8); cuMemcpyHtoD_v2(ROP + 184, loc(one), 8); cuMemcpyHtoD_v2(ROP + 200, loc(HDf), 8); cuMemcpyHtoD_v2(ROP + 208, loc(Hf), 8); cuMemcpyHtoD_v2(ROP + 216, loc(KVHf), 8); cuMemcpyHtoD_v2(ROP + 224, loc(Pp), 8)
        cuda_launch4i(from<loc<CVoid>>(addr(rp)), (NQf + NKVf + 255) / 256, 1, 1, 256, 1, 1, ROP, 0, 0, 0)
        cuda_launch4i(from<loc<CVoid>>(addr(dcp)), (NKV + 255) / 256, 1, 1, 256, 1, 1, CK + P * NKV * 8, dkrP, NKV, 0)
        cuda_launch4i(from<loc<CVoid>>(addr(dcp)), (NKV + 255) / 256, 1, 1, 256, 1, 1, CV + P * NKV * 8, dvP, NKV, 0)
        cuMemcpyHtoD_v2(ATT + 80, loc(DQr), 8); cuMemcpyHtoD_v2(ATT + 88, loc(CK), 8); cuMemcpyHtoD_v2(ATT + 96, loc(CV), 8); cuMemcpyHtoD_v2(ATT + 104, loc(DCtx), 8)
        cuMemcpyHtoD_v2(ATT + 184, loc(Sctx), 8); cuMemcpyHtoD_v2(ATT + 200, loc(HD), 8); cuMemcpyHtoD_v2(ATT + 208, loc(H), 8); cuMemcpyHtoD_v2(ATT + 216, loc(KVH), 8)
        cuda_launch4i(from<loc<CVoid>>(addr(att)), (Sctx * H + 255) / 256, 1, 1, 256, 1, 1, ATT, 0, 0, 0)
        e2 = proj_lor(dctxP, Wo, doP, Uo, Vo, soP, shiP, one, D, NQ, R2, AR, gf, ads, GM, PB); if (e2 != 0) { return 93 }
        cuda_launch4i(from<loc<CVoid>>(addr(rs)), (D + 255) / 256, 1, 1, 256, 1, 1, hbP, doP, x1P, D)
        cuMemcpyHtoD_v2(RN + 0, loc(x1P), 8); cuMemcpyHtoD_v2(RN + 8, loc(N2), 8); cuMemcpyHtoD_v2(RN + 16, loc(x1nP), 8)
        cuMemcpyHtoD_v2(RN + 24, loc(one), 8); cuMemcpyHtoD_v2(RN + 32, loc(D), 8); cuMemcpyHtoD_v2(RN + 40, loc(EPS), 8)
        cuda_launch4i(from<loc<CVoid>>(addr(rn)), (D + 255) / 256, 1, 1, 256, 1, 1, RN, 0, 0, 0)
        e2 = proj_lor(x1nP, Wg, grP, Ug, Vg, sgP, shiP, one, FF, D, R2, AR, gf, ads, GM, PB); if (e2 != 0) { return 94 }
        e2 = proj_lor(x1nP, Wu, upP, Uu, Vu, suP, shiP, one, FF, D, R2, AR, gf, ads, GM, PB); if (e2 != 0) { return 95 }
        cuda_launch4i(from<loc<CVoid>>(addr(sl)), (FF + 255) / 256, 1, 1, 256, 1, 1, grP, upP, huP, FF)
        e2 = proj_lor(huP, Wd, m2P, Ud, Vd, sdP, shiP, one, D, FF, R2, AR, gf, ads, GM, PB); if (e2 != 0) { return 96 }
        cuda_launch4i(from<loc<CVoid>>(addr(rs)), (D + 255) / 256, 1, 1, 256, 1, 1, x1P, m2P, xoP, D)
        cuCtxSynchronize()
        return 0
    }
    return 0
}

'''
src = src.replace("main()<Int> -> {", helpers + "main()<Int> -> {", 1)

# ---------------- (B) CK/CV/RSHI allocs + declarations ----------------
old_decl = "        ASLB<Int> = 0"
assert src.count(old_decl) == 1
src = src.replace(old_decl, "        ASLB<Int> = 0; CK<Int> = 0; CV<Int> = 0; RSHI<Int> = 0", 1)

old_alloc = "        if(cuMemAlloc_v2(loc(ASLB),1810544832)!=0){ return 11 }"
assert src.count(old_alloc) == 1
src = src.replace(old_alloc, old_alloc + "\n        # per-token KV caches: combined roped-k/v per layer [36][93][NKV]; RSHI=[93,FF] addscaled scratch\n        if(cuMemAlloc_v2(loc(CK),36*93*NKV*8)!=0){ return 11 }\n        if(cuMemAlloc_v2(loc(CV),36*93*NKV*8)!=0){ return 11 }\n        if(cuMemAlloc_v2(loc(RSHI),93*FF*8)!=0){ return 11 }", 1)

# ---------------- (C) per-token forward replacing the batched forward ----------------
start = "            cuda_launch4i(from<loc<CVoid>>(addr(dcp)),(238080+255)/256,1,1,256,1,1,ASLB+L3*50292912+0,XI,238080,0)"
end = "            tl2<Int> = XI; XI = XOcur; XOcur = tl2"
si = src.index(start)
ei = src.index(end, si) + len(end)

# build offline slices for ASLB slab args
acts = [(OFF["XN"],"XN"),(OFF["DQ"],"DQ"),(OFF["DK"],"DK"),(OFF["DV"],"DV"),(OFF["DQN"],"DQN"),
        (OFF["DKN"],"DKN"),(OFF["DQr"],"DQr"),(OFF["DKr"],"DKr"),(OFF["DCtx"],"DCtx"),(OFF["DO"],"DO"),
        (OFF["X1"],"X1"),(OFF["X1N"],"X1N"),(OFF["Gr"],"Gr"),(OFF["Up"],"Up"),(OFF["Hu"],"Hu"),(OFF["M2"],"M2")]
sargs = ",".join(f"slabA+{o}" for o,_ in acts)
slos = ",".join(f"slabA+{OFF[k]}" for k in ["sq","sk","sv","so","sg","su","sd"])

new_block = f'''                # ---- per-token KV response forward: layer-major, token-minor ----
                # each response token P=9+t attends combined slabK/slabV (context rows 0..8 cached once +
                # response-so-far), writes its per-layer activations DIRECTLY into slabA (==ASLB+L3*SLAYOUT)
                # so the UNCHANGED S=93 frozen backward reads them. LoRA low-rank A outputs land in the
                # per-projection sq..sd slabs of ASLB for the backward.
                RESP<Int> = 84
                slabK<Int> = CK+L3*93*NKV*8; slabV<Int> = CV+L3*93*NKV*8; slabA<Int> = ASLB+L3*{SLAYOUT}
                cuda_launch4i(from<loc<CVoid>>(addr(dcp)),((93-9)*D+255)/256,1,1,256,1,1,slabA+0+9*D*8,XI+9*D*8,(93-9)*D,0)
                t2<Int> = 0
                for (t2 in 0..RESP-1){{
                    e = resp_layer_kv(9+t2, 10+t2, D,H,KVH,HD,FF,NQ,NKV,R,EPS,AR,XI,XOcur,slabK,slabV,
                                      gf,ads,rn,rp,att,sl,rs,dcp,GM,RN,ROP,ATT,PB,
                                      N1,Wq,Wk,Wv,Nq,Nk,DF,Wo,N2,Wg,Wu,Wd,
                                      UqP,VqP,UkP,VkP,UvP,VvP,UoP,VoP,UgP,VgP,UuP,VuP,UdP,VdP,
                                      {sargs},
                                      {slos}, RSHI); if(e!=0){{ return 80 }}
                }}
                cuda_launch4i(from<loc<CVoid>>(addr(dcp)),((93-9)*D+255)/256,1,1,256,1,1,slabA+{OFF['XOcur']}+9*D*8,XOcur+9*D*8,(93-9)*D,0)
                cuCtxSynchronize()
                tl2<Int> = XI; XI = XOcur; XOcur = tl2'''

src = src[:si] + new_block + src[ei:]

open(dst_path, "w").write(src)
print("wrote", dst_path, "lines", src.count(chr(10)))
print("helpers ok:", "resp_layer_kv(P<Int>" in src)
print("per-token fwd ok:", "resp_layer_kv(9+t2" in src)
print("tail ok:", src.rstrip().endswith("}") and "KVRESP_TRAIN_DONE" in src)
