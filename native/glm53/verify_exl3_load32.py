"""Gate: EXL3_LOAD_VYB32 — the f32 EXL3 dequant+Hadamard path on a real expert tensor.

Compares the f32-path dequant of layer-3 expert 0's gate_proj [4096, 2048] against the
same P1 oracle digest used by the f64 gate. The f32 transform introduces real rounding,
so tolerances are WIDER than the f64 gate's and are chosen from the measured spread:
f32 has ~6e-8 relative machine epsilon, the 128-tap MAC accumulates ~128 of those, and
the scales multiply; the band below (sum/sq rel <= 1e-3, max rel <= 3e-2) sits two
orders above the expected noise floor and two below any wiring bug (a wrong row gather
or orientation error is O(0.1+) — see the skill: a genuine wiring bug is O(0.1+)).

Usage: env -u PYTHONPATH ~/venvs/exl3/bin/python native/glm53/verify_exl3_load32.py
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
        vyb = parse(f"{OUT}/exl3_load_vyb32.txt")
        gold = parse(f"{OUT}/gold_exl3_deq.txt")
    except FileNotFoundError as e:
        print(f"EXL3_LOAD_VYB32: FAIL ({e.filename})")
        return 1
    if "gp0" not in vyb or "gp0" not in gold:
        print("EXL3_LOAD_VYB32: FAIL (missing gp0)")
        return 1
    v, g = vyb["gp0"], gold["gp0"]
    ok = v["n"] == g["n"] == 4096 * 2048
    print(f"elements: vyb {v['n']} gold {g['n']}")
    for k in ("sum", "abs", "sq", "max"):
        rel = abs(v[k] - g[k]) / max(1.0, abs(float(g[k])))
        print(f"  {k:4s} rel {rel:.2e}")
        if k == "sum":
            ok = ok and rel <= 1e-3
        if k == "abs":
            ok = ok and rel <= 1e-3
        if k == "sq":
            ok = ok and rel <= 1e-3
        if k == "max":
            ok = ok and rel <= 3e-2
    print("EXL3_LOAD_VYB32:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
