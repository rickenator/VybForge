"""Gate: EXL3_LOAD_VYB — the reusable EXL3 dequant on a real expert tensor.

Compares the Vyb dequant of layer-3 expert 0's gate_proj [4096, 2048] against the P1
oracle (loader.dequant_exl3_fast). Digest-level (sum/abs/sq/max + first eight values),
because the oracle itself carries fp16-butterfly noise: the P1 gate accepted max abs
4.4e-5 and 0/8.4M elements beyond 3e-3, so the tolerances here mirror that band.

Usage: env -u PYTHONPATH ~/venvs/exl3/bin/python native/glm53/verify_exl3_load.py
"""
import sys

OUT = "native/out"


def parse(path):
    out = {}
    for line in open(path):
        parts = line.split()
        if not parts:
            continue
        f = {}
        for p in parts[1:]:
            if "=" in p:
                k, v = p.split("=", 1)
                f[k] = int(v)
        if "n" in f:
            out[parts[0]] = f
    return out


def main():
    try:
        vyb = parse(f"{OUT}/exl3_load_vyb.txt")
        gold = parse(f"{OUT}/gold_exl3_deq.txt")
    except FileNotFoundError as e:
        print(f"EXL3_LOAD_VYB: FAIL ({e.filename})")
        return 1
    if "gp0" not in vyb or "gp0" not in gold:
        print("EXL3_LOAD_VYB: FAIL (missing gp0)")
        return 1
    v, g = vyb["gp0"], gold["gp0"]
    ok = v["n"] == g["n"] == 4096 * 2048
    print(f"elements: vyb {v['n']} gold {g['n']}")
    for k in ("sum", "abs", "sq", "max"):
        rel = abs(v[k] - g[k]) / max(1.0, abs(float(g[k])))
        print(f"  {k:4s} rel {rel:.2e}")
        if k == "sq":
            ok = ok and rel <= 1e-5
        if k == "max":
            ok = ok and rel <= 1e-3
    print("EXL3_LOAD_VYB:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
