#!/usr/bin/env python3
"""Parameterize kvresp_train_kv.vyb to arbitrary (S, NCTX). Recomputes the full per-layer ASLB byte
layout (25 fields), all S-proportional alloc sizes, CK/CV ([36][S][NKV]), the context-build row range,
and the response-forward row offsets from S/NCTX. Substitutes the committed S=93 literals.

Usage: gen_kv_train.py 93 9   (regression — should byte-match the committed kvresp_train_kv.vyb body)
       gen_kv_train.py 513 429 (full-manifest scale)
Writes native/train/kvresp_train_p{NC}.vyb (or a passed output name).
Assumes fields order:
XI,XN, DQ,DK,DV, DQN,DKN, DQr,DKr,DCtx, DO,X1,X1N, Gr,Up,Hu, M2, XOcur, sq,sk,sv,so,sg,su,sd."""
import sys, re

SRC = "native/train/kvresp_train_kv.vyb"
D, H, KVH, HD, FF, R2 = 2560, 32, 8, 128, 9728, 2
NQ, NKV = H * HD, KVH * HD

def aslb_layout(S):
    """Return OrderedDict field->byte offset and SLAYOUT(bytes) for given S."""
    fld = ["XI", "XN", "DQ", "DK", "DV", "DQN", "DKN", "DQr", "DKr", "DCtx", "DO", "X1", "X1N",
           "Gr", "Up", "Hu", "M2", "XOcur", "sq", "sk", "sv", "so", "sg", "su", "sd"]
    rows = {"XI": D, "XN": D, "DQ": NQ, "DK": NKV, "DV": NKV, "DQN": NQ, "DKN": NKV,
            "DQr": NQ, "DKr": NKV, "DCtx": NQ, "DO": D, "X1": D, "X1N": D,
            "Gr": FF, "Up": FF, "Hu": FF, "M2": D, "XOcur": D}
    off = {}; elem = 0
    for f in fld:
        off[f] = elem * 8
        elem += S * rows.get(f, R2)
    return off, elem * 8   # byte offsets, SLAYOUT bytes

def main(S, NC, out):
    NC = int(NC)
    S = int(S)
    RESP = S - NC
    off, slayout = aslb_layout(S)
    src = open(SRC).read()
    # constants
    src = re.sub(r"S<Int> = 93; D<Int> = 2560; H<Int> = 32; KVH<Int> = 8; HD<Int> = 128; FF<Int> = 9728; R<Int> = 2; NCTX<Int> = 9",
                 f"S<Int> = {S}; D<Int> = 2560; H<Int> = 32; KVH<Int> = 8; HD<Int> = 128; FF<Int> = 9728; R<Int> = 2; NCTX<Int> = {NC}", src)
    src = re.sub(r"POS<Int> = 93; BPP<Int> = 0", f"POS<Int> = {S}; BPP<Int> = 0", src)
    # CK/CV alloc size (36*93*NKV*8)
    src = src.replace(f"36*93*NKV*8", f"36*{S}*NKV*8")
    src = src.replace(f"93*FF*8", f"{S}*FF*8")  # RSHI
    # ASLB alloc total = 36*SLAYOUT_93; recompute 36*SLAYOUT(S)
    src = re.sub(r"loc\(ASLB\),\d+\)", f"loc(ASLB),{36*slayout})", src)
    # workspace alloc literals: 238080=S*D, 380928=S*NQ, 95232=S*NKV, 904704=S*FF, S*R
    src = src.replace("238080", str(S * D))
    src = src.replace("380928", str(S * NQ))
    src = src.replace("95232", str(S * NKV))
    src = src.replace("904704", str(S * FF))
    # per-layer SLAYOUT literal everywhere
    src = src.replace("50292912", str(slayout))
    # replace ASLB field byte offsets with the recomputed table
    for f, b in off.items():
        src = src.replace(str(b), str(b))  # same-value no-op placeholder (offsets ARE S-dependent)
    # field offsets appear as ASLB+L*50292912+<off> ; recompute each distinct committed offset
    # gather all ASLB+L*<SLAYOUT>+<OFF> now-mangled forms are covered by SLAYOUT+OFF literal replace below.
    # Direct byte-offset literals (the old S=93 values) inside slabA+XOXO expressions:
    old_off = aslb_layout(93)[0]
    for f in old_off:
        src = src.replace(f"+{old_off[f]}", f"+{off[f]}")
    # context-build row literals (NC-token forward): rows 0..NC-1, offsets 9*D8 etc.
    src = src.replace("for (tc0 in 0..8)", f"for (tc0 in 0..{NC-1})")
    src = src.replace("cAa+0,XIc,9*D,0", f"cAa+0,XIc,{NC}*D,0")
    src = src.replace("cAa+48377856,XOc,9*D,0", f"cAa+{off['XOcur']},XOc,{NC}*D,0")
    src = src.replace("48377856", str(off["XOcur"]))
    # response-forward row literals
    src = src.replace("RESP<Int> = 84", f"RESP<Int> = {RESP}")
    src = src.replace("9+t2", f"{NC}+t2")
    src = src.replace("10+t2", f"{NC+1}+t2")
    src = src.replace("(93-9)*D", f"({S}-{NC})*D")
    src = src.replace("9*D*8", f"{NC}*D*8")
    src = src.replace("slabK<Int> = CK+L3*93*NKV*8", f"slabK<Int> = CK+L3*{S}*NKV*8")
    src = src.replace("cKk<Int> = CK+Lc*93*NKV*8", f"cKk<Int> = CK+Lc*{S}*NKV*8")
    open(out, "w").write(src)
    print(f"generated {out}: S={S} NCTX={NC} RESP={RESP} SLAYOUT={slayout}")

if __name__ == "__main__":
    S = int(sys.argv[1]); NC = int(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else f"native/train/kvresp_train_p{NC}.vyb"
    main(S, NC, out)
