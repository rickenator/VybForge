#!/usr/bin/env python3
"""M2e-2 oracle: full 36-layer Qwen3-4B LoRA BACKWARD, chained top-to-bottom (L=35..0).
Consumes the M2e-1 cached activations (.bin) + per-layer weights + LoRA U/V, starting from a
fixed random upstream grad dG on layer-35 output. For every layer produces dU/dV for the 7
LoRA projections (q/k/v/o/gate/up/down) and the dx chained down to the previous layer.
Writes per-layer grads to native/out/m2e_L{L}_dU{nm}_ref.txt (and dV), final dx as
m2e_dx0_ref.txt. LoRA: proj = a@W + alpha_r*((a@U)@V).
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
KEEPL = ["x","xn","q","k","v","qr","kr","ctx","x1","x1n","gate","up","hu","s_q","s_k","s_v","s_o","s_g","s_u","s_d"]
def load_cache(L):
    rng_f = lambda n: np.fromfile(os.path.join(out, f"m2e_L{L}_{n}.bin"), "<f8")
    return {n: rng_f(n) for n in KEEPL}
def load_Lo(L):
    return {nm: (np.fromfile(os.path.join(out, f"m2e_l{L}_U{nm}.bin"),"<f8").reshape(*proj_shapes[nm][:1]+(R,)),
                np.fromfile(os.path.join(out, f"m2e_l{L}_V{nm}.bin"),"<f8").reshape(R, proj_shapes[nm][1])) for nm in proj_shapes}
def rw(L, n):
    name = f"blk.{L}.{n}.weight"
    return l0.read_weight(tens[name], tuple(tens[name]["shape"]))
def LN(L):
    return {n: rw(L, n) for n in ["attn_norm","attn_q_norm","attn_k_norm","ffn_norm"]}
def Wg(L):
    return {n: rw(L, n) for n in ["attn_q","attn_k","attn_v","attn_output","ffn_gate","ffn_up","ffn_down"]}

INVF = np.fromfile(os.path.join(out, "layer0_invfreq.bin"), "<f8")
def freqs(s_): return s_ * INVF[:HD//2]
def rope_adj(gr):
    S_,XH,HD_=gr.shape; go=np.zeros_like(gr)
    for s_ in range(S_):
        for xh in range(XH):
            for i in range(HD_//2):
                f_=freqs(s_)[i]; c,sn=np.cos(f_),np.sin(f_)
                di,dj=gr[s_,xh,i],gr[s_,xh,i+HD_//2]
                go[s_,xh,i]=c*di+sn*dj; go[s_,xh,i+HD_//2]=-sn*di+c*dj
    return go

def rmsnorm_layer(v_, gy, wn, dim):
    inv=1.0/np.sqrt((v_**2).mean()+1e-6); T=(gy*wn*v_).sum()
    return (gy*wn*inv - v_*(inv**3/dim)*T)

def backward_layer(L, dG):
    c = load_cache(L); W = Wg(L); Lo = load_Lo(L); lnn = LN(L)
    gu = {}
    def lo_bwd(a, nm, dout):
        U,V = Lo[nm]; s = a@U; gv=alpha_r*dout
        ds = gv@V.T; dU = a.T@ds; dV = s.T@gv
        return dout@W[WK[nm]].T + (ds@U.T), dU, dV
    # residual xo = x1 + m ; dm=dG ; dx1=dG
    dm = dG.copy(); dx1 = dG.copy()
    A = c  # xo not cached; use x1 as base (we know m = xo-x1; dL/dm = dG)
    c["xn"]=c["xn"].reshape(S,D); c["x1n"]=c["x1n"].reshape(S,D); c["x1"]=c["x1"].reshape(S,D)
    c["ctx"]=c["ctx"].reshape(S,NQ); c["q"]=c["q"].reshape(S,NQ); c["k"]=c["k"].reshape(S,NKV); c["v"]=c["v"].reshape(S,NKV)
    c["gate"]=c["gate"].reshape(S,FF); c["up"]=c["up"].reshape(S,FF); c["hu"]=c["hu"].reshape(S,FF)
    dHu, gu["dU_d"], gu["dV_d"] = lo_bwd(c["hu"], "d", dm)
    sg=1.0/(1.0+np.exp(-c["gate"])); silu_g=c["gate"]*sg; silup=sg+c["gate"]*sg*(1.0-sg)
    dgate = dHu*c["up"]*silup; dup = dHu*silu_g
    d_xn_g, gu["dU_g"], gu["dV_g"] = lo_bwd(c["x1n"], "g", dgate)
    d_xn_u, gu["dU_u"], gu["dV_u"] = lo_bwd(c["x1n"], "u", dup)
    dX1n = d_xn_g + d_xn_u
    dX1l = np.zeros((S,D))
    for s_ in range(S):
        dX1l[s_] = rmsnorm_layer(c["x1"][s_], dX1n[s_], lnn["ffn_norm"], D)
    dx1 = dx1 + dX1l
    dctx, gu["dU_o"], gu["dV_o"] = lo_bwd(c["ctx"], "o", dx1)
    Dat = dctx.reshape(S,H,HD)
    qr=c["qr"].reshape(S,H,HD); kr=c["kr"].reshape(S,KVH,HD); vv=c["v"].reshape(S,KVH,HD)
    gradq=np.zeros_like(qr); gradk=np.zeros_like(kr); gradv=np.zeros_like(vv)
    for s_ in range(S):
        for h in range(H):
            kvh=h//(H//KVH); qs=qr[s_,h]
            sc=np.array([scale*np.dot(qs,kr[t,kvh]) for t in range(s_+1)])
            mx=sc.max(); ax=np.exp(sc-mx); ss=ax.sum(); w=ax/ss
            l=np.array([np.dot(Dat[s_,h],vv[t,kvh]) for t in range(s_+1)])
            mean=(w*l).sum(); dsc=ax*(l-mean)/ss
            gradq[s_,h]=scale*sum(dsc[t]*kr[t,kvh] for t in range(s_+1))
            for t in range(s_+1):
                gradk[t,kvh]+=scale*dsc[t]*qs; gradv[t,kvh]+=w[t]*Dat[s_,h]
    dQn = rope_adj(gradq); dKn = rope_adj(gradk)
    dq=np.zeros((S,H,HD)); dk=np.zeros((S,KVH,HD))
    for s_ in range(S):
        for h in range(H):
            dq[s_,h]=rmsnorm_layer(c["q"].reshape(S,H,HD)[s_,h], dQn[s_,h], lnn["attn_q_norm"], HD)
        for h in range(KVH):
            dk[s_,h]=rmsnorm_layer(c["k"].reshape(S,KVH,HD)[s_,h], dKn[s_,h], lnn["attn_k_norm"], HD)
    dxn, gu["dU_q"], gu["dV_q"] = lo_bwd(c["xn"], "q", dq.reshape(S,NQ))
    dxk, gu["dU_k"], gu["dV_k"] = lo_bwd(c["xn"], "k", dk.reshape(S,NKV))
    dxv, gu["dU_v"], gu["dV_v"] = lo_bwd(c["xn"], "v", gradv.reshape(S,NKV))
    dXN = dxn+dxk+dxv
    dx = np.zeros((S,D))
    x_in = load_cache(L)["x"].reshape(S,D)  # layer input (MUST reshape: flat otherwise)
    for s_ in range(S):
        dx[s_] = rmsnorm_layer(x_in[s_], dXN[s_], lnn["attn_norm"], D)
    # RESIDUAL-IDENTITY FIX: x1 = x + o => dL/dx also carries dx1 (dG + ffn-norm-back),
    # the gradient through the skip connection. Without it the 36-layer chain under-counts
    # lower-layer gradients by orders of magnitude (FD-verified: L0 dU was 1e-18 vs FD 7.62).
    dx = dx + dx1
    if L == 35:
        np.savetxt(os.path.join(out, "m2e_dbg_dq_ref.txt"), dq.reshape(-1), fmt="%.10g")
        np.savetxt(os.path.join(out, "m2e_dbg_dk_ref.txt"), dk.reshape(-1), fmt="%.10g")
        np.savetxt(os.path.join(out, "m2e_dbg_dXN_ref.txt"), dXN.reshape(-1), fmt="%.10g")
        np.savetxt(os.path.join(out, "m2e_dbg_dx35_ref.txt"), dx.reshape(-1), fmt="%.10g")
    return gu, dx

# top upstream grad (fixed seed) on layer-35 output
rng = np.random.default_rng(67)
dG = rng.normal(size=(S,D)).astype("<f8")
dG.tofile(os.path.join(out, "m2e_dG.bin"))
np.savetxt(os.path.join(out, "m2e_dG_ref.txt"), dG.reshape(-1), fmt="%.10g")
dx = dG
for L in range(35, -1, -1):
    gu, dx = backward_layer(L, dx)
    UVmap = {"q":("Uq","Vq"),"k":("Uk","Vk"),"v":("Uv","Vv"),"o":("Uo","Vo"),
             "g":("Ug","Vg"),"u":("Uu","Vu"),"d":("Ud","Vd")}
    for nm,(un,vn) in UVmap.items():
        np.savetxt(os.path.join(out, f"m2e_L{L}_{un}_ref.txt"), gu["dU_"+nm].reshape(-1), fmt="%.10g")
        np.savetxt(os.path.join(out, f"m2e_L{L}_{vn}_ref.txt"), gu["dV_"+nm].reshape(-1), fmt="%.10g")
    np.savetxt(os.path.join(out, f"m2e_L{L}_dx_ref.txt"), dx.reshape(-1), fmt="%.10g")
    print(f"  L{L} backward, |dx|={np.linalg.norm(dx):.6g}", flush=True)
np.savetxt(os.path.join(out, "m2e_dx0_ref.txt"), dx.reshape(-1), fmt="%.10g")
print("M2E2_REF_DONE layers=36 backward, final dx0 norm", np.linalg.norm(dx))
