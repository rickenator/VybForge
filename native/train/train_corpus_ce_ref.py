#!/usr/bin/env python3
"""M2e-3-CE reference: teacher-forced NEXT-TOKEN CE on a fixed TINY corpus slice.
Pipeline (all real Qwen3-4B): embed a fixed token slice via token_embd -> 36-layer LoRA
forward -> output_norm -> TIED lm_head logits (h @ token_embd.T) -> target-shifted causal CE
loss -> CE backward (head grads + output_norm + 36-layer chain, with the residual-identity fix)
-> AdamW on all 36 layers' U/V. Writes per-step CE loss + final U q-ref for verification.
Target: the tiny slice's own next tokens (overfit) -- with the fixed backward the loss DESCENDS.
This is the production corpus objective; the MSE-overfit loop was the correctness gate."""
import os, importlib.util, numpy as np

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = "/home/rick/Models/qwen3/Qwen3-4B-Q4_K_M.gguf"
out = os.path.join(repo, "native", "out")
spec = importlib.util.spec_from_file_location("l0", os.path.join(repo, "native/gguf/layer0_ref.py"))
l0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l0)

D, H, KVH, HD, FF = 2560, 32, 8, 128, 9728
NQ, NKV = H*HD, KVH*HD
R, alpha_r = 2, 2.0
VOCAB = 151936
# S derived from the encoded goal-desktop whole-stack record (train rec 6: 'I want a Hyprland
# desktop workstation' -> manifest-v<3 SUMMARY response, 84 tokens). Encode the assistant content
# with the same tokenizer the Vyb decoder uses (transformers artifacts), then S = len(response)-1,
# input = response[0:S], labels = response[1:S+1] (teacher-forced next-token).
def _load_record():
    import json
    rec = None
    with open(os.path.join(repo, "data/vybos-configurator-train.jsonl")) as fh:
        for line in fh:
            d = json.loads(line)
            u = " ".join(m["content"] for m in d["messages"] if m["role"] == "user")
            if "Hyprland desktop workstation" in u:
                rec = [m["content"] for m in d["messages"] if m["role"] == "assistant"][-1]
                break
    if rec is None:
        raise SystemExit("goal-desktop record not found in train corpus")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(repo, "artifacts", "vybos-configurator-lora"))
    return np.array(tok.encode(rec, add_special_tokens=False), dtype=np.int64)

_ids = _load_record()
S = int(_ids.shape[0]) - 1          # 84 tokens -> S=83 labeled positions
INPUT_IDS = _ids[:S]                 # response[0:S]
LABELS = _ids[1:S + 1]               # response[1:S+1]  (next token at each position)
print("whole-stack goal-desktop record: S =", S, " input", INPUT_IDS[:8].tolist(), "...")
seq = np.arange(S)
tens = l0.parse_tsv()
proj_shapes = {"q":(D,NQ),"k":(D,NKV),"v":(D,NKV),"o":(NQ,D),"g":(D,FF),"u":(D,FF),"d":(FF,D)}
WK = {"q":"attn_q","k":"attn_k","v":"attn_v","o":"attn_output","g":"ffn_gate","u":"ffn_up","d":"ffn_down"}

def rw(L, n):
    name = f"blk.{L}.{n}.weight"; t = tens[name]
    return l0.read_weight(t, tuple(t["shape"]))
ALL_W = [{n: rw(L,n) for n in ["attn_q","attn_k","attn_v","attn_output","ffn_gate","ffn_up","ffn_down"]} for L in range(36)]
ALL_LN = [{n: rw(L,n) for n in ["attn_norm","attn_q_norm","attn_k_norm","ffn_norm"]} for L in range(36)]
def load_Lo(L):
    return {nm:(np.fromfile(os.path.join(out,f"m2e_l{L}_U{nm}.bin"),"<f8").reshape(*proj_shapes[nm][:1]+(R,)),
             np.fromfile(os.path.join(out,f"m2e_l{L}_V{nm}.bin"),"<f8").reshape(R,proj_shapes[nm][1])) for nm in proj_shapes}
Lo = {L: load_Lo(L) for L in range(36)}
mU={L:{nm:np.zeros_like(U) for nm,(U,V) in Lo[L].items()} for L in range(36)}
vU={L:{nm:np.zeros_like(U) for nm,(U,V) in Lo[L].items()} for L in range(36)}
mV={L:{nm:np.zeros_like(V) for nm,(U,V) in Lo[L].items()} for L in range(36)}
vV={L:{nm:np.zeros_like(V) for nm,(U,V) in Lo[L].items()} for L in range(36)}
def rms(v,w): return l0.rmsnorm(v,w)
INVF = np.fromfile(os.path.join(out, "layer0_invfreq.bin"), "<f8")
freqs = lambda s_: s_ * INVF[:HD//2]
def lproj(a, nm, L):
    U,V=Lo[L][nm]; s=a@U; return a@ALL_W[L][WK[nm]]+alpha_r*(s@V), s
def rmsnorm_layer(v_, gy, wn, dim):
    inv=1.0/np.sqrt((v_**2).mean()+1e-6); T=(gy*wn*v_).sum()
    return (gy*wn*inv - v_*(inv**3/dim)*T)
def rope_adj(gr):
    S_,XH,HD_=gr.shape; go=np.zeros_like(gr)
    for s_ in range(S_):
        for xh in range(XH):
            for i in range(HD_//2):
                f_=freqs(s_)[i]; c,sn=np.cos(f_),np.sin(f_)
                di,dj=gr[s_,xh,i],gr[s_,xh,i+HD_//2]
                go[s_,xh,i]=c*di+sn*dj; go[s_,xh,i+HD_//2]=-sn*di+c*dj
    return go

# write int64 .bin that the GPU driver (train_full_ce.vyb) loads; regenerate every run so the
# GPU provably trains on the identical slice. (native/out is gitignored.)
INPUT_IDS.astype(np.int64).tofile(os.path.join(out, "m2e3ce_input_ids.bin"))
LABELS.astype(np.int64).tofile(os.path.join(out, "m2e3ce_labels.bin"))

# ---- token_embd dequant once ----
te = tens["token_embd.weight"]
emb = None
with open(MODEL,"rb") as fh:
    fh.seek(te["off"]); emb = l0.dequant_q6k(fh.read(te["numel"]*210//256)).reshape(te["shape"][1], D)
on_t = tens["output_norm.weight"]; ON = l0.read_weight(on_t, tuple(on_t["shape"]))

def embed(ids):
    return emb[ids].astype("<f8")            # [S,D]

def forward_layers(x):
    cache={}; cur=x
    for L in range(36):
        W=ALL_W[L]; lnn=ALL_LN[L]; c={}
        xn=rms(cur,lnn["attn_norm"]); c["xn"]=xn
        q,sq=lproj(xn,"q",L); c["q"]=q; c["s_q"]=sq
        k,sk=lproj(xn,"k",L); c["k"]=k; c["s_k"]=sk
        v,sv=lproj(xn,"v",L); c["v"]=v; c["s_v"]=sv
        qn=rms(q.reshape(S,H,HD),lnn["attn_q_norm"]); c["qn"]=qn
        kn=rms(k.reshape(S,KVH,HD),lnn["attn_k_norm"]); c["kn"]=kn
        qr,kr=l0.rope(qn,kn,seq); c["qr"]=qr; c["kr"]=kr
        ctx=l0.gqa(qr,kr,v.reshape(S,KVH,HD)); c["ctx"]=ctx
        o,so=lproj(ctx.reshape(S,NQ),"o",L); c["o"]=o; c["s_o"]=so
        x1=cur+o; c["x1"]=x1
        x1n=rms(x1,lnn["ffn_norm"]); c["x1n"]=x1n
        g,sg=lproj(x1n,"g",L); c["gate"]=g; c["s_g"]=sg
        up,su=lproj(x1n,"u",L); c["up"]=up; c["s_u"]=su
        hu=(g/(1+np.exp(-g)))*up; c["hu"]=hu
        m,sd=lproj(hu,"d",L); c["m"]=m; c["s_d"]=sd
        xo=x1+m; c["xo"]=xo
        cache[L]=c; cur=xo
    # store each layer's input
    cc={}; cur=x
    for L in range(36):
        cc[L]=(cache[L],cur); cur=cache[L]["xo"]
    return cc, cur

def backward_layer(L, dG, c, xin):
    W=ALL_W[L]; LoL=Lo[L]; lnn=ALL_LN[L]; gu={}
    def lo_bwd(a,nm,dout):
        U,V=LoL[nm]; s=a@U; gv=alpha_r*dout
        ds=gv@V.T; dU=a.T@ds; dV=s.T@gv
        return dout@W[WK[nm]].T + (ds@U.T), dU, dV
    dm=dG.copy(); dx1=dG.copy()
    dHu,gu["dU_d"],gu["dV_d"]=lo_bwd(c["hu"],"d",dm)
    sg=1.0/(1.0+np.exp(-c["gate"])); silu_g=c["gate"]*sg; silup=sg+c["gate"]*sg*(1.0-sg)
    dgate=dHu*c["up"]*silup; dup=dHu*silu_g
    d_g,gu["dU_g"],gu["dV_g"]=lo_bwd(c["x1n"],"g",dgate)
    d_u,gu["dU_u"],gu["dV_u"]=lo_bwd(c["x1n"],"u",dup)
    dX1n=d_g+d_u
    dX1l=np.zeros((S,D))
    for s_ in range(S): dX1l[s_]=rmsnorm_layer(c["x1"][s_], dX1n[s_], lnn["ffn_norm"], D)
    dx1=dx1+dX1l
    dctx,gu["dU_o"],gu["dV_o"]=lo_bwd(c["ctx"],"o",dx1)
    Dat=dctx.reshape(S,H,HD)
    qr=c["qr"].reshape(S,H,HD); kr=c["kr"].reshape(S,KVH,HD); vv=c["v"].reshape(S,KVH,HD)
    scale=1.0/np.sqrt(HD)
    gq=np.zeros_like(qr); gk=np.zeros_like(kr); gv_=np.zeros_like(vv)
    for s_ in range(S):
        for h in range(H):
            kvh=h//(H//KVH); qs=qr[s_,h]
            sc=np.array([scale*np.dot(qs,kr[tt,kvh]) for tt in range(s_+1)])
            mx=sc.max(); ax=np.exp(sc-mx); ssum=ax.sum(); w=ax/ssum
            l=np.array([np.dot(Dat[s_,h],vv[tt,kvh]) for tt in range(s_+1)])
            mean=(w*l).sum(); dsc=ax*(l-mean)/ssum
            gq[s_,h]=scale*sum(dsc[tt]*kr[tt,kvh] for tt in range(s_+1))
            for tt in range(s_+1):
                gk[tt,kvh]+=scale*dsc[tt]*qs; gv_[tt,kvh]+=w[tt]*Dat[s_,h]
    dQn=rope_adj(gq); dKn=rope_adj(gk)
    dq_=np.zeros((S,H,HD)); dk_=np.zeros((S,KVH,HD))
    for s_ in range(S):
        for h in range(H): dq_[s_,h]=rmsnorm_layer(c["q"].reshape(S,H,HD)[s_,h], dQn[s_,h], lnn["attn_q_norm"], HD)
        for h in range(KVH): dk_[s_,h]=rmsnorm_layer(c["k"].reshape(S,KVH,HD)[s_,h], dKn[s_,h], lnn["attn_k_norm"], HD)
    dxn,gu["dU_q"],gu["dV_q"]=lo_bwd(c["xn"],"q",dq_.reshape(S,NQ))
    dxk,gu["dU_k"],gu["dV_k"]=lo_bwd(c["xn"],"k",dk_.reshape(S,NKV))
    dxv,gu["dU_v"],gu["dV_v"]=lo_bwd(c["xn"],"v",gv_.reshape(S,NKV))
    dXN=dxn+dxk+dxv
    dx=np.zeros((S,D))
    for s_ in range(S): dx[s_]=rmsnorm_layer(xin[s_], dXN[s_], lnn["attn_norm"], D)
    dx = dx + dx1   # RESIDUAL-IDENTITY (same fix as MSE loop)
    return gu, dx

def ce_loss_and_headgrad(xo):
    """output_norm -> tied lm_head; returns loss, dL/d(xo) [S,D] (before output_norm back)."""
    h = rms(xo, ON)                     # [S,D]
    logits = h @ emb.T                  # [S,VOCAB]
    lse = logits.max(axis=-1, keepdims=True)
    e = np.exp(logits - lse)
    soft = e / e.sum(axis=-1, keepdims=True)
    # CE, target-shifted, per token
    loss = -np.log(soft[np.arange(S), LABELS]+1e-30).mean()
    # dL/dlogits = (soft - onehot)
    dlog = soft.copy(); dlog[np.arange(S), LABELS] -= 1.0; dlog /= S   # mean reduction
    # back through logits = h @ emb.T -> dL/dh = dlog @ emb
    dh = dlog @ emb                     # [S,D]
    # back through output_norm -> dL/dxo
    dxo = np.zeros((S,D))
    for s_ in range(S): dxo[s_]=rmsnorm_layer(xo[s_], dh[s_], ON, D)
    return float(loss), dxo, logits, soft

NSTP=4; LR=0.00005; B1=0.9; B2=0.999; EPP=1e-8; WDD=0.0
def adamw(P,g,m,v,st):
    m[:]=B1*m+(1-B1)*g; v[:]=B2*v+(1-B2)*g*g
    P[:]-=LR*(m/(1-B1**st))/(np.sqrt(v/(1-B2**st))+EPP)
x0 = embed(INPUT_IDS)
losses=[]
for st in range(1,NSTP+1):
    cc, xo = forward_layers(x0)
    loss, dxo, logits, soft = ce_loss_and_headgrad(xo)
    losses.append(loss)
    dG = dxo.copy()
    for L in range(35,-1,-1):
        c,xin = cc[L]
        gu, dG = backward_layer(L, dG.copy(), c, xin)
        if st==1 and L in (35,17,0):
            np.savetxt(os.path.join(out,f"m2e3ce_dbg_L{L}_dUq_ref.txt"), gu["dU_q"].reshape(-1)[:64], fmt="%.9g")
        for nm in proj_shapes:
            adamw(Lo[L][nm][0], gu["dU_"+nm], mU[L][nm], vU[L][nm], st)
            adamw(Lo[L][nm][1], gu["dV_"+nm], mV[L][nm], vV[L][nm], st)
    print(f"step {st} CE loss {loss:.6f}", flush=True)
np.savetxt(os.path.join(out,"m2e3ce_loss_ref.txt"), np.array(losses), fmt="%.10g")
# dump first-128 of L0 Uq after training for parity gate
np.savetxt(os.path.join(out,"m2e3ce_L0_Uq_U_final_ref.txt"), Lo[0]["q"][0].reshape(-1)[:128], fmt="%.9g")
np.savetxt(os.path.join(out,"m2e3ce_input_ids.txt"), INPUT_IDS, fmt="%d")
np.savetxt(os.path.join(out,"m2e3ce_labels.txt"), LABELS, fmt="%d")
print("M2E3CE_ORACLE_DONE steps",NSTP,"loss0",losses[0],"lossN",losses[-1],"descend",losses[-1]<losses[0])
