#!/usr/bin/env python3
"""M2c reference: analytic BACKWARD through ONE real blk.0 forward WITH LoRA, given a
synthetic upstream grad dL/dxo. Produces the LoRA gradients dU/dV for all 7 projections
(q/k/v/o/gate/up/down) and the input gradient dx. LoRA: proj = a@W + alpha_r*((a@U)@V),
s=a@U. Forward identical to train_layer_ref.py. Saves dU/dV + dx as the target (+ dG bin).
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
tens = l0.parse_tsv(); pre = "blk.0."
rw = lambda n: l0.read_weight(tens[pre+n+".weight"], tuple(tens[pre+n+".weight"]["shape"]))
W = {n: rw(n) for n in ["attn_q","attn_k","attn_v","attn_output","ffn_gate","ffn_up","ffn_down"]}
LN = {n: rw(n) for n in ["attn_norm","attn_q_norm","attn_k_norm","ffn_norm"]}
x = np.fromfile(os.path.join(out, "layer0_input.bin"), "<f8").reshape(S, D)
proj_shapes = {"q":(D,NQ),"k":(D,NKV),"v":(D,NKV),"o":(NQ,D),"g":(D,FF),"u":(D,FF),"d":(FF,D)}
Lo = {nm: (np.fromfile(os.path.join(out,f"tl_U{nm}.bin"),"<f8").reshape(*proj_shapes[nm][:1] + (R,)),
           np.fromfile(os.path.join(out,f"tl_V{nm}.bin"),"<f8").reshape(R, proj_shapes[nm][1])) for nm in proj_shapes}

# ---- forward (matches train_layer_ref) ----
xn = l0.rmsnorm(x, LN["attn_norm"])
A = {}
WK = {"q":"attn_q","k":"attn_k","v":"attn_v","o":"attn_output","g":"ffn_gate","u":"ffn_up","d":"ffn_down"}
def lproj(a, nm):  # a@W + alpha_r*(a@U@V); returns (out, s)
    U,V = Lo[nm]; s = a@U; return (a@W[WK[nm]]+alpha_r*(s@V), s)
A["q"],A["s_q"]=lproj(xn,"q"); A["k"],A["s_k"]=lproj(xn,"k"); A["v"],A["s_v"]=lproj(xn,"v")
A["qn"]=l0.rmsnorm(A["q"].reshape(S,H,HD), LN["attn_q_norm"])
A["kn"]=l0.rmsnorm(A["k"].reshape(S,KVH,HD), LN["attn_k_norm"])
A["qr"],A["kr"]=l0.rope(A["qn"], A["kn"], np.arange(S))
A["ctx"]=l0.gqa(A["qr"], A["kr"], A["v"].reshape(S,KVH,HD))
A["o"],A["s_o"]=lproj(A["ctx"].reshape(S,NQ),"o"); A["x1"]=x+A["o"]
A["x1n"]=l0.rmsnorm(A["x1"], LN["ffn_norm"])
A["g"],A["s_g"]=lproj(A["x1n"],"g"); A["u"],A["s_u"]=lproj(A["x1n"],"u")
A["hu"]=(A["g"]/(1+np.exp(-A["g"])))*A["u"]
A["m"],A["s_d"]=lproj(A["hu"],"d"); A["xo"]=A["x1"]+A["m"]

rng = np.random.default_rng(59)
dG = rng.normal(size=(S,D)); dG.astype("<f8").tofile(os.path.join(out,"tl_dG.bin"))

# ---- backward ----
gu = {}
def lo_bwd(a, nm, dout):  # returns dout-to-a (base+lora) and dU,dV
    U,V = Lo[nm]; s = a@U; gv = alpha_r*dout
    ds = gv@V.T; dU = a.T@ds; dV = s.T@gv
    return (dout@W[WK[nm]].T + (ds@U.T)), dU, dV

# residual xo = x1 + m
dm = dG.copy(); dx1 = dG.copy()
# down proj
dHu, gu["dU_d"], gu["dV_d"] = lo_bwd(A["hu"], "d", dm)
# silu: hu = silu(g)*u, silu(g)=g*σ(g), silu'(g)=σ(g)+g*σ(g)*(1-σ(g))
sg = 1.0/(1.0+np.exp(-A["g"]))                 # sigmoid(g)
silu_g = A["g"]*sg
silup = sg + A["g"]*sg*(1.0-sg)                # silu'(g)
dgate = dHu*A["u"]*silup; dup = dHu*silu_g
# gate/up proj
d_xn_g, gu["dU_g"], gu["dV_g"] = lo_bwd(A["x1n"], "g", dgate)
d_xn_u, gu["dU_u"], gu["dV_u"] = lo_bwd(A["x1n"], "u", dup)
dX1n = d_xn_g + d_xn_u
# x1n rmsnorm (x1, ffn_norm)
dx1n = np.zeros_like(dX1n); gu["dN2"]=np.zeros(D)
for s_ in range(S):
    v_=A["x1"][s_]; gy=dX1n[s_]; wn=LN["ffn_norm"]
    inv=1.0/np.sqrt((v_**2).mean()+1e-6); T=(gy*wn*v_).sum()
    dx1n[s_]=gy*wn*inv - v_*(inv**3/D)*T
    gu["dN2"] += gy*v_*inv
dx1 = dx1 + dx1n
do = dx1.copy()
# o proj
dctx, gu["dU_o"], gu["dV_o"] = lo_bwd(A["ctx"].reshape(S,NQ), "o", do)
# attention backward
Dat = dctx.reshape(S,H,HD)
gradq = np.zeros_like(A["qr"]); gradk = np.zeros_like(A["kr"]); gradv = np.zeros_like(A["v"].reshape(S,KVH,HD))
Vv = A["v"].reshape(S,KVH,HD)
for s_ in range(S):
    for h in range(H):
        kvh = h//(H//KVH)
        qs = A["qr"][s_,h]
        sc = np.array([scale*np.dot(qs, A["kr"][t,kvh]) for t in range(s_+1)])
        mx=sc.max(); a=np.exp(sc-mx); ssum=a.sum(); w=a/ssum
        l = np.array([np.dot(Dat[s_,h], Vv[t,kvh]) for t in range(s_+1)])
        mean=(w*l).sum(); dsc=a*(l-mean)/ssum
        gradq[s_,h] = scale*sum(dsc[t]*A["kr"][t,kvh] for t in range(s_+1))
        for t in range(s_+1):
            gradk[t,kvh] += scale*dsc[t]*qs
            gradv[t,kvh] += w[t]*Dat[s_,h]
# rope adjoint (split-half, transpose)
def rope_adj(gr):
    S_,XH,HD_ = gr.shape
    go = np.zeros_like(gr)
    for s_ in range(S_):
        for xh in range(XH):
            for i in range(HD_//2):
                # pair (i, i+half) [first half i, second half i]
                freq_ = freqs(s_)[i]
                c,sn = np.cos(freq_), np.sin(freq_)
                di, dj = gr[s_,xh,i], gr[s_,xh,i+HD_//2]
                go[s_,xh,i]=c*di+sn*dj
                go[s_,xh,i+HD_//2]=-sn*di+c*dj
    return go
seq = np.arange(S)
INVF = np.fromfile(os.path.join(out, "layer0_invfreq.bin"), "<f8")
def freqs(s_):
    half = HD//2
    return s_ * INVF[:half]
dQn = rope_adj(gradq); dKn = rope_adj(gradk)
# q/k rmsnorm backward (use the PRE-NORM input for inv/x-terms)
q_in = A["q"].reshape(S,H,HD); k_in = A["k"].reshape(S,KVH,HD)
dq = np.zeros_like(q_in); dNq=np.zeros(HD)
dkgrad = np.zeros_like(k_in); dNk=np.zeros(HD)
for s_ in range(S):
    for h in range(H):
        xv=q_in[s_,h]; gy=dQn[s_,h]; wn=LN["attn_q_norm"]
        inv=1.0/np.sqrt((xv**2).mean()+1e-6); T=(gy*wn*xv).sum()
        dq[s_,h]=gy*wn*inv - xv*(inv**3/HD)*T
        dNq += gy*xv*inv
    for h in range(KVH):
        xv=k_in[s_,h]; gy=dKn[s_,h]; wn=LN["attn_k_norm"]
        inv=1.0/np.sqrt((xv**2).mean()+1e-6); T=(gy*wn*xv).sum()
        dkgrad[s_,h]=gy*wn*inv - xv*(inv**3/HD)*T
        dNk += gy*xv*inv
gu["dNq"]=dNq; gu["dNk"]=dNk
# q/k/v proj backward
dxn, gu["dU_q"], gu["dV_q"] = lo_bwd(xn,"q", dq.reshape(S,NQ))
dxk, gu["dU_k"], gu["dV_k"] = lo_bwd(xn,"k", dkgrad.reshape(S,NKV))
dxv, gu["dU_v"], gu["dV_v"] = lo_bwd(xn,"v", gradv.reshape(S,NKV))
dXN = dxn + dxk + dxv
# input rmsnorm backward -> dx
dxl = np.zeros_like(x); gu["dN1"]=np.zeros(D)
for s_ in range(S):
    v_=x[s_]; gy=dXN[s_]; wn=LN["attn_norm"]
    inv=1.0/np.sqrt((v_**2).mean()+1e-6); T=(gy*wn*v_).sum()
    dxl[s_]=gy*wn*inv - v_*(inv**3/D)*T
    gu["dN1"] += gy*v_*inv
dx = dxl
for nm in ["q","k","v","o","g","u","d"]:
    np.savetxt(os.path.join(out, f"b_dU{nm}_ref.txt"), gu["dU_"+nm].reshape(-1), fmt="%.10g")
    np.savetxt(os.path.join(out, f"b_dV{nm}_ref.txt"), gu["dV_"+nm].reshape(-1), fmt="%.10g")
np.savetxt(os.path.join(out, "b_dx_ref.txt"), dx.reshape(-1), fmt="%.10g")
np.savetxt(os.path.join(out, "b_dCtx_ref.txt"), dctx.reshape(-1), fmt="%.10g")
np.savetxt(os.path.join(out, "b_dQr_ref.txt"), gradq.reshape(-1), fmt="%.10g")
np.savetxt(os.path.join(out, "b_dQn_ref.txt"), dQn.reshape(-1), fmt="%.10g")
np.savetxt(os.path.join(out, "b_dq_ref.txt"), dq.reshape(-1), fmt="%.10g")
np.savetxt(os.path.join(out, "b_dk_ref.txt"), dkgrad.reshape(-1), fmt="%.10g")
np.savetxt(os.path.join(out, "b_dVu_ref.txt"), gradv.reshape(-1), fmt="%.10g")
print("M2C_REF_DONE")
