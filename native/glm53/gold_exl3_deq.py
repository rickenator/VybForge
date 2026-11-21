"""Gold: EXL3 expert dequant for the Vyb module check (native/glm53/exl3_load.vyb).

Dequantizes layer-3 expert 0's gate_proj through the P1 oracle path and writes the
digest in the same (v*2^40) as Int format the Vyb side prints.

Usage: env -u PYTHONPATH ~/venvs/exl3/bin/python native/glm53/gold_exl3_deq.py
"""
import json, sys
import numpy as np
sys.path.insert(0, "/home/rick/Projects/VybForge/native/glm53")
import verify_layer3_moe as B

S40 = 1099511627776.0
OUT = "native/out"


def dig(v):
    f = np.ascontiguousarray(v, dtype=np.float64).ravel()
    return {"n": len(f), "sum": int(f.sum() * S40), "abs": int(np.abs(f).sum() * S40),
            "sq": int((f * f).sum() * S40), "max": int(np.abs(f).max() * S40),
            **{f"v{k}": int(f[k] * S40) if k < len(f) else 0 for k in range(8)}}


def main():
    idx = B.index()
    w = B.read_expert_exl3(0, "gate_proj", idx).astype(np.float64)
    print(f"gate_proj expert0 shape {w.shape}")
    d = dig(w)
    line = " ".join(f"{k}={v}" for k, v in d.items())
    open(f"{OUT}/gold_exl3_deq.txt", "w").write("gp0 " + line + "\n")
    np.save(f"{OUT}/gold_exl3_gp0.npy", w)
    print("gold_gp0 n=%d sum=%d sq=%d max=%d" % (d["n"], d["sum"], d["sq"], d["max"]))
    # down_proj [in=2048, out=4096]: cw (out) 4096, ntiles = (2048/16)*(4096/16)
    wd = B.read_expert_exl3(0, "down_proj", idx).astype(np.float64)
    print(f"down_proj expert0 shape {wd.shape}")
    dd = dig(wd)
    dline = " ".join(f"{k}={v}" for k, v in dd.items())
    open(f"{OUT}/gold_exl3_deq.txt", "a").write("dp0 " + dline + "\n")
    np.save(f"{OUT}/gold_exl3_dp0.npy", wd)
    print("gold_dp0 n=%d sum=%d sq=%d max=%d" % (dd["n"], dd["sum"], dd["sq"], dd["max"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
