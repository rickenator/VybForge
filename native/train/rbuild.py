#!/usr/bin/env python3
"""Build native/train/kvresp_train_r.vyb: R-RESTRICTED per-token KV trainer.

RELIABLE method: extract the COMMITTED (verified) S=93 backward `for (Li2..)` loop body and apply a
MECHANICAL S->R transform, preserving the exact kernel-call sequences. Splice in (a) compact
RESPONSE-ROW activation loads and (b) a COMBINED-cache dattn. Forward + head are copied verbatim.

Sizes (R=84): RB=RESP*D=215040, RNQ=RESP*NQ=344064, RNKV=RESP*NKV=86016, RF=RESP*FF=817152,
RR=RESP*R2=168, RHP=RESP*H=2688, RKP=RESP*KVH=672.
Layout field rows: XI,XN,DO,X1,X1N,M2,XOcur have row=D; DQ,DQN,DQr,DCtx row=NQ; DK,DV,DKN,DKr row=NKV;
Gr,Up,Hu row=FF; sq..sd row=R."""
import re
NCTX, RESP = 9, 84
D, H, KVH, HD, FF, R2 = 2560, 32, 8, 128, 9728, 2
NQ, NKV = H * HD, KVH * HD
S = NCTX + RESP
RB, RF, RNQ, RNKV, RR = RESP * D, RESP * FF, RESP * NQ, RESP * NKV, RESP * R2
RHP, RKP = RESP * H, RESP * KVH
SD, SNQ_, SNKV_, SFF_, SR_ = S * D, S * NQ, S * NKV, S * FF, S * R2
rows = {"XI": D, "XN": D, "DQ": NQ, "DK": NKV, "DV": NKV, "DQN": NQ, "DKN": NKV, "DQr": NQ,
        "DKr": NKV, "DCtx": NQ, "DO": D, "X1": D, "X1N": D, "Gr": FF, "Up": FF, "Hu": FF,
        "M2": D, "XOcur": D}
order = ["XI", "XN", "DQ", "DK", "DV", "DQN", "DKN", "DQr", "DKr", "DCtx", "DO", "X1", "X1N",
         "Gr", "Up", "Hu", "M2", "XOcur", "sq", "sk", "sv", "so", "sg", "su", "sd"]
off = {}; elem = 0
for f in order:
    off[f] = elem * 8
    elem += S * rows.get(f, R2)
SLAYOUT = elem * 8

SRC = "native/train/kvresp_train_kv.vyb"
DST = "native/train/kvresp_train_r.vyb"
src = open(SRC).read()
lines = src.split("\n")
start_line = next(i for i, l in enumerate(lines) if "# ---- BACKWARD" in l)
loop_line = next(i for i in range(start_line, len(lines)) if lines[i].lstrip().startswith("for (Li2 in 0..MAXL-1)"))
def _loop_end():
    depth = 0; j = loop_line
    depth = lines[j].count("{") - lines[j].count("}")
    j += 1
    while j < len(lines):
        t = lines[j].strip()
        if t == "" or t.startswith("//"):
            j += 1; continue
        depth += lines[j].count("{") - lines[j].count("}")
        if depth <= 0:
            return j
        j += 1
    return None
loop_end = _loop_end()
head = "\n".join(lines[:loop_line]).rstrip() + "\n"
tail = "\n".join(lines[loop_end + 1:])
for p in ["losstr", "KVRESP_TRAIN_DONE", "stp_d = stp_d + 1"]:
    if p not in tail:
        raise SystemExit(f"tail missing {p}")

# ---- extract committed loop body ----
body = "\n".join(lines[loop_line + 1:loop_end])

# remove the full-width S-ASLB activation loads (we replace with response-row loads)
body = re.sub(r'\s*cuda_launch4i\(from<loc<CVoid>>\(addr\(dcp\)\),\([^)]*\),1,1,256,1,1,ASLB\+Lb\*50292912\+[0-9]+,[A-Za-z0-9_]+,([0-9]+),0\)', '', body)
# remove the S size-var block (SB/SF/SNQ/SNKV/SR/SHP/SKP) - rebuilt next
body = re.sub(r'\s*SB<Int> = S\*D; SF<Int> = S\*FF; SNQ<Int> = S\*NQ; SNKV<Int> = S\*NKV; SR<Int> = S\*R', '', body)
body = re.sub(r'\s*SHP<Int> = S\*H; SKP<Int> = S\*KVH', '', body)
body = re.sub(r'\s*ONE2<Float> = 1.0; AR2<Float> = 2.0; ZR2<Float> = 0.0', '', body)
# mechanical S->R
body = body.replace(str(SD), str(RB)).replace(str(SNQ_), str(RNQ)).replace(str(SNKV_), str(RNKV)).replace(str(SFF_), str(RF)).replace(str(SR_), str(RR))
body = re.sub(r'\bS\*D\b', 'RB', body)
body = re.sub(r'\bS\*FF\b', 'RF', body)
body = re.sub(r'\bS\*NQ\b', 'RNQ', body)
body = re.sub(r'\bS\*NKV\b', 'RNKV', body)
body = re.sub(r'\bS\*R2\b', 'R2', body)
body = re.sub(r'\bS\*R\b', 'RR', body)
body = re.sub(r'\bS\*H\b', 'RHP', body)
body = re.sub(r'\bS\*KVH\b', 'RKP', body)
body = re.sub(r'loc\(S\),8', 'loc(RESP),8', body)
body = body.replace('loc(SHP),8', 'loc(RHP),8').replace('loc(SKP),8', 'loc(RKP),8')
# grid counts that were (S*H+255) etc -> (RHP+255), (S*H*HD+...) -> handled by S*H->RHP
body = re.sub(r'\(SHP\*HD', '(RHP*HD', body).replace('SKP*HD', 'RKP*HD')

# ---- build the NEW backward ----
W = []
ap = W.append
ap("                # ============ R-RESTRICTED FROZEN BACKWARD (R=%d, combined-cache dattn) ============" % RESP)
ap("                Li2<Int> = 0")
ap("                for (Li2 in 0..MAXL-1){")
ap("                    Lb<Int> = MAXL-1-Li2")
for i, nm in enumerate(["Wq", "Wk", "Wv", "Wo", "Wg", "Wu", "Wd", "N1", "Nq", "Nk", "N2"]):
    ap(f'e=deq_cached(PKD,POF.get(Lb*11+{i}),PNL.get(Lb*11+{i}),PTY.get(Lb*11+{i}),{nm},q4fn,q6fn,efn,PF); if(e!=0){{ return {60+i} }}')
ap("UqP<Int> = LSLB+Lb*917504+0; VqP<Int> = LSLB+Lb*917504+40960; UkP<Int> = LSLB+Lb*917504+106496; VkP<Int> = LSLB+Lb*917504+147456; UvP<Int> = LSLB+Lb*917504+163840; VvP<Int> = LSLB+Lb*917504+204800; UoP<Int> = LSLB+Lb*917504+221184; VoP<Int> = LSLB+Lb*917504+286720; UgP<Int> = LSLB+Lb*917504+327680; VgP<Int> = LSLB+Lb*917504+368640; UuP<Int> = LSLB+Lb*917504+524288; VuP<Int> = LSLB+Lb*917504+565248; UdP<Int> = LSLB+Lb*917504+720896; VdP<Int> = LSLB+Lb*917504+876544")
ap("Uq=UqP; Vq=VqP; Uk=UkP; Vk=VkP; Uv=UvP; Vv=VvP; Uo=UoP; Vo=VoP; Ug=UgP; Vg=VgP; Uu=UuP; Vu=VuP; Ud=UdP; Vd=VdP")
# compact response-row loads
def rload(dst, f):
    el = rows.get(f, R2); cnt = RESP * el; ob = off[f] + NCTX * el * 8
    ap(f"cuda_launch4i(from<loc<CVoid>>(addr(dcp)),({cnt}+255)/256,1,1,256,1,1,{dst},ASLB+Lb*{SLAYOUT}+{ob},{cnt},0)")
for dst, f in [("x","XI"),("XN","XN"),("DQ","DQ"),("DK","DK"),("DV","DV"),("DQr","DQr"),("DKr","DKr"),
               ("DCtx","DCtx"),("X1","X1"),("X1N","X1N"),("Gr","Gr"),("Up","Up"),("Hu","Hu"),("M2","M2"),
               ("sq","sq"),("sk","sk"),("sv","sv"),("so","so"),("sg","sg"),("su","su"),("sd","sd")]:
    rload(dst, f)
ap(f"SB<Int> = {RB}; SF<Int> = {RF}; SNQ<Int> = {RNQ}; SNKV<Int> = {RNKV}; SR<Int> = {RR}; SHP<Int> = {RHP}; SKP<Int> = {RKP}; RR2<Int> = {R2}")
ap("ONE2<Float> = 1.0; AR2<Float> = 2.0")
ap("# ---- R-RESTRICTED dattn: response q attends COMBINED cache (CK/CV slab = context+response) ----")
# combined staging buffers must be allocated; we reuse dQr2/dKr2/dVu2 as [R] and build cQ/cdC/cQr.. below.
# The core backward body below is the S->R transformed committed body; its original dattn section still
# reads DQr/DKr/DV + dCtx2 at [R] and dattn-params with S. Replace that section with the combined one.

out = head + "\n".join(W) + "\n" + "####BODY####\n" + body + "\n                }\n" + tail
open(DST, "w").write(out)
print("wrote", DST)
print("SLAYOUT", SLAYOUT)
print("body len", len(body))
print("NOTE: dattn combined-staging + buffer allocs still to splice; THIS IS A PARTIAL BUILD for compile-checking the S->R transform only.")
