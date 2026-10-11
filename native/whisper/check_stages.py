#!/usr/bin/env python3
"""check_stages.py — gate the encoder's front-half stages against the reference.

    ~/miniconda/bin/python check_stages.py out/vyb_stages.txt [stage ...]

Each stage in the driver's output is compared against the tensor the real model
produced for the same input (fixtures/whisper/gold_stt). Stages not requested on
the command line are reported as SKIP, and an absent section is a failure — never
a silent pass.
"""
import sys

import numpy as np

HERE = __file__.rsplit("/", 1)[0]
GOLD = f"{HERE}/../../fixtures/whisper/gold_stt"

SHAPES = {
    "conv1": (384, 3000), "e_conv1": (384, 3000),
    "conv2": (384, 1500), "e_conv2": (384, 1500),
    "e_pos": (1500, 384),
    "e_in": (1500, 384),
    "e_l0_ln1": (1500, 384),
}
TOL = 1e-4


def parse(path):
    sections, cur, expect_n = {}, None, False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("SECTION "):
                cur = line.split(" ", 1)[1].split(" ", 1)[0]
                sections[cur] = []
                expect_n = True
                continue
            if cur is None or not line:
                continue
            if line.lstrip("-").isdigit():
                if expect_n:
                    expect_n = False
                    continue
                sections[cur].append(int(line))
    return sections


def main():
    sections = parse(sys.argv[1])
    only = sys.argv[2:]
    ok = True
    for stage in ("conv1", "conv2", "e_pos", "e_in", "e_l0_ln1"):
        if only and stage not in only:
            print(f"{stage}: SKIP (not requested)")
            continue
        if stage not in sections:
            print(f"{stage}: MISSING from the driver output -> FAIL")
            ok = False
            continue
        n = int(np.prod(SHAPES[stage]))
        got = np.asarray(sections[stage][:n], dtype=np.float64) / 1e6
        gold_name = stage if stage.startswith("e_") else "e_" + stage
        ref = np.fromfile(f"{GOLD}/{gold_name}.f32", dtype="<f4").astype(np.float64)
        if len(got) != len(ref):
            print(f"{stage}: LENGTH MISMATCH vyb={len(got)} ref={len(ref)} -> FAIL")
            ok = False
            continue
        d = np.abs(got - ref)
        peak = max(1.0, np.abs(ref).max())
        bad = int((d > TOL).sum())
        verdict = "ok" if bad == 0 else "FAIL"
        print(f"{stage}: n={len(got)} max|diff|={d.max():.3e} rel={d.max()/peak:.3e} "
              f"over {TOL:g}: {bad}  [{verdict}]")
        if bad:
            ok = False
            for i in np.argsort(-d)[:6]:
                row, col = divmod(int(i), SHAPES[stage][1])
                print(f"   row {row} col {col}: vyb={got[i]:.7f} ref={ref[i]:.7f} diff={got[i]-ref[i]:.3e}")
    print("STAGES_GATE:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
