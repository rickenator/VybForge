#!/usr/bin/env python3
"""M2d oracle: full teacher-forced REAL blk.0 LoRA TRAINING LOOP in numpy, N steps.
forward (M2b) + MSE grad (xo - t) + verified backward (M2c, all 7 projections) +
AdamW on U/V (in-place moment state). Produces per-step loss = 0.5*mean((xo-t)^2),
matched by the Vyb driver train_layer_train.vyb. Verifies the loop DESCENDS (overfit).

IMPORTANT (project gotcha): stateful cross-step arrays (Adam moments, U/V) must be
updated IN PLACE (U[:] = ...) — rebinding a local never touches the persistent array.
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

# hyperparams matching the Vyb driver
NSTP = 20; LR = 0.00005; B1 = 0.9; B2 = 0.999; EPP = 1e-8; WDD = 0.0

tens = l0.parse_tsv(); pre = "blk.0."
rw = lambda n: l0.read_weight(tens[pre+n+".weight"], tuple(tens[pre+n+".weight"]["shape"]))
W = {n: rw(n) for n in ["attn_q","attn_k","attn_v","attn_output","ffn_gate","ffn_up","ffn_down"]}
LN = {n: rw(n) for n in ["attn_norm","attn_q_norm","attn_k_norm","ffn_norm"]}
x = np.fromfile(os.path.join(out, "layer0_input.bin"), "<f8").reshape(S, D)
# fixed random target (deterministic seed, overfit-to); persists so the Vyb driver reads the same t
tpath = os.path.join(out, "tl_t.bin")
if not os.path.exists(tpath):
    rng_t = np.random.default_rng(71)
    rng_t.normal(size=(S, D)).astype("<f8").tofile(tpath)
t = np.fromfile(tpath, "<f8").reshape(S, D)

proj_shapes = {"q":(D,NQ),"k":(D,NKV),"v":(D,NKV),"o":(NQ,D),"g":(D,FF),"u":(D,FF),"d":(FF,D)}
WK = {"q":"attn_q","k":"attn_k","v":"attn_v","o":"attn_output","g":"ffn_gate","u":"ffn_up","d":"ffn_down"}
Lo = {nm: (np.fromfile(os.path.join(out,f"tl_U{nm}.bin"),"<f8").reshape(*proj_shapes[nm][:1] + (R,)),
           np.fromfile(os.path.join(out,f"tl_V{nm}.bin"),"<f8").reshape(R, proj_shapes[nm][1])) for nm in proj_shapes}
# Adam moments (persistent, updated in place)
mU = {nm: np.zeros_like(U) for nm,(U,V) in Lo.items()}
vU = {nm: np.zeros_like(U) for nm,(U,V) in Lo.items()}
mV = {nm: np.zeros_like(V) for nm,(U,V) in Lo.items()}
vV = {nm: np.zeros_like(V) for nm,(U,V) in Lo.items()}

def lproj(a, nm):
    U,V = Lo[nm]; s = a@U; return (a@W[WK[nm]]+alpha_r*(s@V), s)

def lo_bwd(a, nm, dout):
    U,V = Lo[nm]; s = a@U; gv = alpha_r*dout
    ds = gv@V.T; dU = a.T@ds; dV = s.T@gv
    return (dout@W[WK[nm]].T + (ds@U.T)), dU, dV

def rmsnorm_layer(v_, gy, wn, dim):
    inv = 1.0/np.sqrt((v_**2).mean()+1e-6); T=(gy*wn*v_).sum()
    return (gy*wn*inv - v_*(inv**3/dim)*T), (gy*v_*inv).sum()

def rope_adj(gr, freqs_):
    S_,XH,HD_ = gr.shape; go = np.zeros_like(gr)
    half = HD_//2
    for s_ in range(S_):
        for xh in range(XH):
            for i in range(half):
                f_ = freqs_[i]; c,sn = np.cos(f_), np.sin(f_)
                di, dj = gr[s_,xh,i], gr[s_,xh,i+half]
                go[s_,xh,i]=c*di+sn*dj; go[s_,xh,i+half]=-sn*di+c*dj
    return go

seq = np.arange(S)
INVF = np.fromfile(os.path.join(out, "layer0_invfreq.bin"), "<f8")
freqs = lambda s_: s_ * INVF[:HD//2]

def forward_step():
    A = {}
    xn = l0.rmsnorm(x, LN["attn_norm"]); A["xn"]=xn
    A["q"],A["s_q"]=lproj(xn,"q"); A["k"],A["s_k"]=lproj(xn,"k"); A["v"],A["s_v"]=lproj(xn,"v")
    A["qn"]=l0.rmsnorm(A["q"].reshape(S,H,HD), LN["attn_q_norm"])
    A["kn"]=l0.rmsnorm(A["k"].reshape(S,KVH,HD), LN["attn_k_norm"])
    A["qr"],A["kr"]=l0.rope(A["qn"], A["kn"], seq)
    A["ctx"]=l0.gqa(A["qr"], A["kr"], A["v"].reshape(S,KVH,HD))
    A["o"],A["s_o"]=lproj(A["ctx"].reshape(S,NQ),"o"); A["x1"]=x+A["o"]
    A["x1n"]=l0.rmsnorm(A["x1"], LN["ffn_norm"])
    A["g"],A["s_g"]=lproj(A["x1n"],"g"); A["u"],A["s_u"]=lproj(A["x1n"],"u")
    A["hu"]=(A["g"]/(1+np.exp(-A["g"])))*A["u"]
    A["m"],A["s_d"]=lproj(A["hu"],"d"); A["xo"]=A["x1"]+A["m"]
    return A

def backward_step(A, dG):
    gu = {}
    dm = dG.copy(); dx1 = dG.copy()
    dHu, gu["dU_d"], gu["dV_d"] = lo_bwd(A["hu"], "d", dm)
    sg = 1.0/(1.0+np.exp(-A["g"])); silu_g = A["g"]*sg; silup = sg+A["g"]*sg*(1.0-sg)
    dgate = dHu*A["u"]*silup; dup = dHu*silu_g
    d_xn_g, gu["dU_g"], gu["dV_g"] = lo_bwd(A["x1n"], "g", dgate)
    d_xn_u, gu["dU_u"], gu["dV_u"] = lo_bwd(A["x1n"], "u", dup)
    dX1n = d_xn_g + d_xn_u
    dx1n = np.zeros_like(dX1n); gu["dN2"]=np.zeros(D)
    for s_ in range(S):
        dx1n[s_], gu["dN2"] = rmsnorm_layer(A["x1"][s_], dX1n[s_], LN["ffn_norm"], D)
    dx1 = dx1 + dx1n
    dctx, gu["dU_o"], gu["dV_o"] = lo_bwd(A["ctx"].reshape(S,NQ), "o", dx1.copy())
    Dat = dctx.reshape(S,H,HD)
    gradq=np.zeros_like(A["qr"]); gradk=np.zeros_like(A["kr"]); gradv=np.zeros_like(A["v"].reshape(S,KVH,HD))
    Vv=A["v"].reshape(S,KVH,HD)
    for s_ in range(S):
        for h in range(H):
            kvh=h//(H//KVH); qs=A["qr"][s_,h]
            sc=np.array([scale*np.dot(qs,A["kr"][t,kvh]) for t in range(s_+1)])
            mx=sc.max(); a_=np.exp(sc-mx); ssum=a_.sum(); w=a_/ssum
            l=np.array([np.dot(Dat[s_,h],Vv[t,kvh]) for t in range(s_+1)])
            mean=(w*l).sum(); dsc=a_*(l-mean)/ssum
            gradq[s_,h]=scale*sum(dsc[t]*A["kr"][t,kvh] for t in range(s_+1))
            for t in range(s_+1):
                gradk[t,kvh]+=scale*dsc[t]*qs; gradv[t,kvh]+=w[t]*Dat[s_,h]
    dQn = rope_adj(gradq, freqs(0)); dKn = rope_adj(gradk, freqs(0))
    q_in=A["q"].reshape(S,H,HD); k_in=A["k"].reshape(S,KVH,HD)
    dq=np.zeros_like(q_in); dNq=np.zeros(HD); dkgrad=np.zeros_like(k_in); dNk=np.zeros(HD)
    for s_ in range(S):
        for h in range(H):
            dq[s_,h], _ = rmsnorm_layer(q_in[s_,h], dQn[s_,h], LN["attn_q_norm"], HD)
        for h in range(KVH):
            dkgrad[s_,h], _ = rmsnorm_layer(k_in[s_,h], dKn[s_,h], LN["attn_k_norm"], HD)
    dxn, gu["dU_q"], gu["dV_q"] = lo_bwd(A["xn"],"q", dq.reshape(S,NQ))
    dxk, gu["dU_k"], gu["dV_k"] = lo_bwd(A["xn"],"k", dkgrad.reshape(S,NKV))
    dxv, gu["dU_v"], gu["dV_v"] = lo_bwd(A["xn"],"v", gradv.reshape(S,NKV))
    dXN = dxn+dxk+dxv
    dxl = np.zeros_like(x)
    for s_ in range(S):
        dxl[s_], _ = rmsnorm_layer(x[s_], dXN[s_], LN["attn_norm"], D)
    return gu, dxl

def adamw(param, g, m, v, t_):
    m[:] = B1*m + (1-B1)*g
    v[:] = B2*v + (1-B2)*g*g
    mh = m/(1-B1**t_); vh = v/(1-B2**t_)
    param[:] -= LR*(mh/(np.sqrt(vh)+EPP) + WDD*param)

losses = []
for step in range(1, NSTP+1):
    A = forward_step()
    dG = A["xo"] - t
    loss = 0.5*np.mean((A["xo"]-t)**2)
    losses.append(loss)
    gu, _ = backward_step(A, dG)
    for nm in proj_shapes:
        adamw(Lo[nm][0], gu["dU_"+nm], mU[nm], vU[nm], step)
        adamw(Lo[nm][1], gu["dV_"+nm], mV[nm], vV[nm], step)

np.savetxt(os.path.join(out, "m2d_loss_ref.txt"), np.array(losses), fmt="%.10g")
print("M2D_ORACLE_DONE steps", NSTP, "loss0", losses[0], "lossN", losses[-1], "descend", losses[-1] < losses[0])
