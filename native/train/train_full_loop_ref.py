#!/usr/bin/env python3
"""M2e-3 oracle: FULL 36-layer Qwen3-4B LoRA TRAINING LOOP in numpy, N steps.
Per step: full-stack forward (with LoRA on all 7 proj/layer, from m2e_l{L}_{UV}.bin) ->
MSE grad at output (xo - t) -> full-stack backward chained L35..0 (M2e-2) -> AdamW on ALL 36
layers' U/V (in-place moments). Uses the SAME real cached activations source as M2e-1 (the
forward here recomputes from the input m2e_input.bin so the whole loop is self-contained).
Writes per-step loss = 0.5*mean((xo-t)^2). Matches the GPU train_full_loop.vyb.
IMPORTANT: Adam moments use IN-PLACE updates (mU[:] = ...) or cross-step state won't accumulate.
"""
import os, importlib.util, numpy as np

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = "/home/rick/Models/qwen3/Qwen3-4B-Q4_K_M.gguf"
out = os.path.join(repo, "native", "out")
spec = importlib.util.spec_from_file_location("l0", os.path.join(repo, "native/gguf/layer0_ref.py"))
l0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l0)

D, H, KVH, HD, FF = 2560, 32, 8, 128, 9728
NQ, NKV, S, R, alpha_r = H*HD, KVH*HD, 2, 2, 2.0
scale = 1.0/np.sqrt(HD)
tens = l0.parse_tsv()
proj_shapes = {"q":(D,NQ),"k":(D,NKV),"v":(D,NKV),"o":(NQ,D),"g":(D,FF),"u":(D,FF),"d":(FF,D)}
WK = {"q":"attn_q","k":"attn_k","v":"attn_v","o":"attn_output","g":"ffn_gate","u":"ffn_up","d":"ffn_down"}
INVF = np.fromfile(os.path.join(out, "layer0_invfreq.bin"), "<f8")
freqs = lambda s_: s_ * INVF[:HD//2]
seq = np.arange(S)

def rw(L, n):
    name = f"blk.{L}.{n}.weight"; t = tens[name]
    return l0.read_weight(t, tuple(t["shape"]))
def Wg(L): return {n: rw(L,n) for n in ["attn_q","attn_k","attn_v","attn_output","ffn_gate","ffn_up","ffn_down"]}
def LN(L): return {n: rw(L,n) for n in ["attn_norm","attn_q_norm","attn_k_norm","ffn_norm"]}
def load_Lo(L):
    return {nm:(np.fromfile(os.path.join(out,f"m2e_l{L}_U{nm}.bin"),"<f8").reshape(*proj_shapes[nm][:1]+(R,)),
             np.fromfile(os.path.join(out,f"m2e_l{L}_V{nm}.bin"),"<f8").reshape(R, proj_shapes[nm][1])) for nm in proj_shapes}
def rmsnorm_layer(v_, gy, wn, dim):
    inv=1.0/np.sqrt((v_**2).mean()+1e-6); T=(gy*wn*v_).sum()
    return (gy*wn*inv - v_*(inv**3/dim)*T)
def rms(v_,w): return l0.rmsnorm(v_,w)
def rope_adj(gr):
    S_,XH,HD_=gr.shape; go=np.zeros_like(gr)
    for s_ in range(S_):
        for xh in range(XH):
            for i in range(HD_//2):
                f_=freqs(s_)[i]; c,sn=np.cos(f_),np.sin(f_)
                di,dj=gr[s_,xh,i],gr[s_,xh,i+HD_//2]
                go[s_,xh,i]=c*di+sn*dj; go[s_,xh,i+HD_//2]=-sn*di+c*dj
    return go

# ---------- actual live LoRA (updated by AdamW) ----------
Lo = {L: load_Lo(L) for L in range(36)}
mU={L:{nm:np.zeros_like(U) for nm,(U,V) in Lo[L].items()} for L in range(36)}
vU={L:{nm:np.zeros_like(U) for nm,(U,V) in Lo[L].items()} for L in range(36)}
mV={L:{nm:np.zeros_like(V) for nm,(U,V) in Lo[L].items()} for L in range(36)}
vV={L:{nm:np.zeros_like(V) for nm,(U,V) in Lo[L].items()} for L in range(36)}

def lproj(a, nm, L):
    U,V=Lo[L][nm]; s=a@U; return a@ALL_W[L][WK[nm]]+alpha_r*(s@V), s


# ---------- PRELOAD weights/lora once (do NOT re-read GGUF inside loops) ----------
ALL_W  = [Wg(L) for L in range(36)]
ALL_LN = [LN(L) for L in range(36)]

def forward_layers(x):
    cache = {}
    cur = x
    for L in range(36):
        c = {}
        W=ALL_W[L]; lnn=ALL_LN[L]
        xn = rms(cur, lnn["attn_norm"]); c["xn"]=xn
        q,sq = lproj(xn,"q",L); c["q"]=q; c["s_q"]=sq
        k,sk = lproj(xn,"k",L); c["k"]=k; c["s_k"]=sk
        v,sv = lproj(xn,"v",L); c["v"]=v; c["s_v"]=sv
        qn = rms(q.reshape(S,H,HD), lnn["attn_q_norm"]); c["qn"]=qn
        kn = rms(k.reshape(S,KVH,HD), lnn["attn_k_norm"]); c["kn"]=kn
        qr,kr = l0.rope(qn,kn,seq); c["qr"]=qr; c["kr"]=kr
        ctx = l0.gqa(qr,kr, v.reshape(S,KVH,HD)); c["ctx"]=ctx
        o,so = lproj(ctx.reshape(S,NQ),"o",L); c["o"]=o; c["s_o"]=so
        x1 = cur+o; c["x1"]=x1
        x1n = rms(x1, lnn["ffn_norm"]); c["x1n"]=x1n
        g,sg = lproj(x1n,"g",L); c["gate"]=g; c["s_g"]=sg
        up,su= lproj(x1n,"u",L); c["up"]=up; c["s_u"]=su
        hu = (g/(1+np.exp(-g)))*up; c["hu"]=hu
        m,sd = lproj(hu,"d",L); c["m"]=m; c["s_d"]=sd
        xo = x1+m; c["xo"]=xo
        cache[L]=c; cur=xo
    return cache, cur

def backward_layer(L, dG, c):
    W=ALL_W[L]; LoL=Lo[L]; lnn=ALL_LN[L]
    gu={}
    def lo_bwd(a,nm,dout):
        U,V=LoL[nm]; s=a@U; gv=alpha_r*dout
        ds=gv@V.T; dU=a.T@ds; dV=s.T@gv
        return dout@W[WK[nm]].T + (ds@U.T), dU, dV
    dm=dG.copy(); dx1=dG.copy()
    (g_,u_,hu_)=(c["gate"],c["up"],c["hu"])
    dHu, gu["dU_d"], gu["dV_d"] = lo_bwd(hu_, "d", dm)
    sg=1.0/(1.0+np.exp(-g_)); silu_g=g_*sg; silup=sg+g_*sg*(1.0-sg)
    dgate=dHu*u_*silup; dup=dHu*silu_g
    d_g, gu["dU_g"], gu["dV_g"] = lo_bwd(c["x1n"],"g",dgate)
    d_u, gu["dU_u"], gu["dV_u"] = lo_bwd(c["x1n"],"u",dup)
    dX1n = d_g+d_u
    dX1l=np.zeros((S,D))
    for s_ in range(S): dX1l[s_]=rmsnorm_layer(c["x1"][s_], dX1n[s_], lnn["ffn_norm"], D)
    dx1 = dx1+dX1l
    dctx, gu["dU_o"], gu["dV_o"] = lo_bwd(c["ctx"],"o",dx1)
    Dat=dctx.reshape(S,H,HD)
    qr=c["qr"].reshape(S,H,HD); kr=c["kr"].reshape(S,KVH,HD); vv=c["v"].reshape(S,KVH,HD)
    gq=np.zeros_like(qr); gk=np.zeros_like(kr); gv_=np.zeros_like(vv)
    for s_ in range(S):
        for h in range(H):
            kvh=h//(H//KVH); qs=qr[s_,h]
            sc=np.array([scale*np.dot(qs,kr[t,kvh]) for t in range(s_+1)])
            mx=sc.max(); ax=np.exp(sc-mx); ss=ax.sum(); w=ax/ss
            l=np.array([np.dot(Dat[s_,h],vv[t,kvh]) for t in range(s_+1)])
            mean=(w*l).sum(); dsc=ax*(l-mean)/ss
            gq[s_,h]=scale*sum(dsc[t]*kr[t,kvh] for t in range(s_+1))
            for t in range(s_+1):
                gk[t,kvh]+=scale*dsc[t]*qs; gv_[t,kvh]+=w[t]*Dat[s_,h]
    dQn=rope_adj(gq); dKn=rope_adj(gk)
    dq_=np.zeros((S,H,HD)); dk_=np.zeros((S,KVH,HD))
    for s_ in range(S):
        for h in range(H): dq_[s_,h]=rmsnorm_layer(c["q"].reshape(S,H,HD)[s_,h], dQn[s_,h], lnn["attn_q_norm"], HD)
        for h in range(KVH): dk_[s_,h]=rmsnorm_layer(c["k"].reshape(S,KVH,HD)[s_,h], dKn[s_,h], lnn["attn_k_norm"], HD)
    dxn, gu["dU_q"], gu["dV_q"] = lo_bwd(c["xn"],"q", dq_.reshape(S,NQ))
    dxk, gu["dU_k"], gu["dV_k"] = lo_bwd(c["xn"],"k", dk_.reshape(S,NKV))
    dxv, gu["dU_v"], gu["dV_v"] = lo_bwd(c["xn"],"v", gv_.reshape(S,NKV))
    dXN = dxn+dxk+dxv
    dx=np.zeros((S,D)); x_in = load_x(L)
    for s_ in range(S): dx[s_]=rmsnorm_layer(x_in[s_], dXN[s_], lnn["attn_norm"], D)
    # RESIDUAL IDENTITY FIX: x1 = x + o  =>  dL/dx must ALSO include dx1 (the gradient that lands
    # on x1 = dG + ffn-norm-back, which then flows to x through the identity in x1 = x + o).
    # Without this term each layer drops the residual path and the 36-layer chain under-counts
    # gradients to lower layers (FD says ~7.6 at L0; analytic was ~1e-18).
    dx = dx + dx1  # dx1 held the accumulated (dG + rmsnorm_back(dX1n)) gradient of x1
    return gu, dx

def load_x(L):
    # layer-L input = m2e_input for L=0; else prior layer output. Not cached as .bin per layer
    # here, so fall back: recompute from a stored copy. We cache x per layer during forward.
    return XCACHE[L]

# hyperparams (match M2d + GPU)
NSTP=5; LR=0.00005; B1=0.9; B2=0.999; EPP=1e-8; WDD=0.0

x0 = np.fromfile(os.path.join(out,"m2e_input.bin"),"<f8").reshape(S,D)
tpath=os.path.join(out,"tl_t.bin")
if not os.path.exists(tpath):
    rng_t=np.random.default_rng(71); rng_t.normal(size=(S,D)).astype("<f8").tofile(tpath)
t = np.fromfile(tpath,"<f8").reshape(S,D)

def adamw(P,g,m,v,step):
    m[:]=B1*m+(1-B1)*g; v[:]=B2*v+(1-B2)*g*g
    mh=m/(1-B1**step); vh=v/(1-B2**step)
    P[:]-=LR*(mh/(np.sqrt(vh)+EPP)+WDD*P)

losses=[]
for step in range(1,NSTP+1):
    # forward, caching each layer's INPUT x into XCACHE
    XCACHE = {}
    cache, xo = forward_layers(x0)
    cur = x0
    for L in range(36):
        XCACHE[L] = cur
        cur = cache[L]["xo"]
    dG = xo - t
    loss = 0.5*np.mean(dG**2); losses.append(loss)
    dx = dG
    for L in range(35,-1,-1):
        gu, dx = backward_layer(L, dx, cache[L])
        for nm in proj_shapes:
            adamw(Lo[L][nm][0], gu["dU_"+nm], mU[L][nm], vU[L][nm], step)
            adamw(Lo[L][nm][1], gu["dV_"+nm], mV[L][nm], vV[L][nm], step)
            if step==1 and L==35 and nm=="q":
                np.savetxt(os.path.join(out,"m2e3_dbg_dUq35_ref.txt"), gu["dU_q"].reshape(-1), fmt="%.10g")
                np.savetxt(os.path.join(out,"m2e3_dbg_dVq35_ref.txt"), gu["dV_q"].reshape(-1), fmt="%.10g")
    print(f"step {step} loss {loss:.6g}", flush=True)
np.savetxt(os.path.join(out,"m2e3_loss_ref.txt"), np.array(losses), fmt="%.10g")

# ---- dump final U/V adapters + AdamW moments (for the parity gate: loss alone can hide bad updates) ----
for L in range(36):
    for nm,(U,V) in Lo[L].items():
        U.tofile(os.path.join(out, f"m2e3_L{L}_{nm}_U_final.bin"))
        V.tofile(os.path.join(out, f"m2e3_L{L}_{nm}_V_final.bin"))
        mU[L][nm].tofile(os.path.join(out, f"m2e3_L{L}_{nm}_mU.bin"))
        vU[L][nm].tofile(os.path.join(out, f"m2e3_L{L}_{nm}_vU.bin"))
        mV[L][nm].tofile(os.path.join(out, f"m2e3_L{L}_{nm}_mV.bin"))
        vV[L][nm].tofile(os.path.join(out, f"m2e3_L{L}_{nm}_vV.bin"))

# ---- matching GPU parity selection: first 128 f64 of Uq/Vq + moments for L in {0,17,35} ----
for LL in (0,17,35):
    U,V = Lo[LL]["q"]
    np.savetxt(os.path.join(out, f"m2e3_L{LL}_Uq_U_final_ref.txt"), U.reshape(-1)[:128], fmt="%.10g")
    np.savetxt(os.path.join(out, f"m2e3_L{LL}_Vq_V_final_ref.txt"), V.reshape(-1)[:128], fmt="%.10g")
    np.savetxt(os.path.join(out, f"m2e3_L{LL}_Uq_mU_ref.txt"), mU[LL]["q"].reshape(-1)[:128], fmt="%.10g")
    np.savetxt(os.path.join(out, f"m2e3_L{LL}_Uq_vU_ref.txt"), vU[LL]["q"].reshape(-1)[:128], fmt="%.10g")
    np.savetxt(os.path.join(out, f"m2e3_L{LL}_Vq_mV_ref.txt"), mV[LL]["q"].reshape(-1)[:128], fmt="%.10g")
    np.savetxt(os.path.join(out, f"m2e3_L{LL}_Vq_vV_ref.txt"), vV[LL]["q"].reshape(-1)[:128], fmt="%.10g")
print("M2E3_ORACLE_ADAPTERS_DONE")
print("M2E3_ORACLE_DONE steps",NSTP,"loss0",losses[0],"lossN",losses[-1],"descend",losses[-1]<losses[0])
