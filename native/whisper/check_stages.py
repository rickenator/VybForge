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
    # ---- decoder (native/whisper/host/decoder_driver.vyb) -------------------
    "d_embed": (2, 384),
    "d_l0_ln1": (2, 384),
    "d_l0_selfattn": (2, 384),
    "d_l0_res1": (2, 384),
    "d_l0_crossattn": (2, 384),
    "d_l0_mlp": (2, 384),
    # d_layernorm_on_l0_out is the final decoder norm applied to LAYER 0's output
    # (a one-layer probe the oracle captured). The real four-layer path is covered
    # by d_step0_logits, which is the same norm feeding the tied head.
    "d_layernorm_on_l0_out": (2, 384),
    "d_step0_logits": (51864,),
    "conv1": (384, 3000), "e_conv1": (384, 3000),
    "conv2": (384, 1500), "e_conv2": (384, 1500),
    "e_pos": (1500, 384),
    "e_in": (1500, 384),
    "e_l0_ln1": (1500, 384),
    "e_l0_attn": (1500, 384),
    "e_l0_res1": (1500, 384),
    "e_l0_ln2": (1500, 384),
    "e_l0_mlp": (1500, 384),
    "e_layer0": (1500, 384),
    "e_layer1": (1500, 384),
    "e_layer2": (1500, 384),
    "e_layer3": (1500, 384),
    "e_final": (1500, 384),
}
# Scale-aware tolerance, and the numbers are MEASURED rather than guessed.
# The reference is the model running in float32; this is float64. On the last
# encoder block (activations to ~630) an INDEPENDENT float64 numpy implementation
# of the same block differs from that float32 reference by up to 1.46e-02 absolute
# and 2.3e-05 relative, on exactly the same three entries this gate used to flag —
# i.e. the residual is the reference's accumulation noise, not an implementation
# error. The bar is therefore an order of magnitude above that measured noise and
# still three orders below the signature of a real bug (the wrong-transpose bug
# here put 99.98% of values out by O(1)).
TOL_ABS = 1e-3
TOL_REL = 3e-4


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
    for stage in ("conv1", "conv2", "e_pos", "e_in", "e_l0_ln1", "e_l0_attn",
                  "e_l0_res1", "e_l0_ln2", "e_l0_mlp",
                  "e_layer0", "e_layer1", "e_layer2", "e_layer3", "e_final",
                  "d_embed", "d_l0_ln1", "d_l0_selfattn", "d_l0_res1", "d_l0_crossattn",
                  "d_l0_mlp", "d_step0_logits"):
        if only and stage not in only:
            print(f"{stage}: SKIP (not requested)")
            continue
        if stage not in sections:
            print(f"{stage}: MISSING from the driver output -> FAIL")
            ok = False
            continue
        n = int(np.prod(SHAPES[stage]))
        got = np.asarray(sections[stage][:n], dtype=np.float64) / 1e6
        if stage.startswith("d_"):
            gold_name = stage
        else:
            gold_name = stage if stage.startswith("e_") else "e_" + stage
        ref = np.fromfile(f"{GOLD}/{gold_name}.f32", dtype="<f4").astype(np.float64)
        if len(got) != len(ref):
            print(f"{stage}: LENGTH MISMATCH vyb={len(got)} ref={len(ref)} -> FAIL")
            ok = False
            continue
        d = np.abs(got - ref)
        peak = max(1.0, np.abs(ref).max())
        allowed = np.maximum(TOL_ABS, TOL_REL * np.abs(ref))
        bad = int((d > allowed).sum())
        verdict = "ok" if bad == 0 else "FAIL"
        print(f"{stage}: n={len(got)} max|diff|={d.max():.3e} rel={d.max()/peak:.3e} "
              f"ref_peak={peak:.3g} over abs {TOL_ABS:g}/rel {TOL_REL:g}: {bad}  [{verdict}]")
        if bad:
            ok = False
            # print the actual FAILURES (not the largest diffs — a scale-aware bar
            # means the largest diffs are usually legitimate float32 noise)
            for i in np.argsort(-(d - allowed))[:8]:
                if d[i] <= allowed[i]:
                    break
                row, col = divmod(int(i), SHAPES[stage][1])
                print(f"   row {row} col {col}: vyb={got[i]:.7f} ref={ref[i]:.7f} "
                      f"diff={got[i]-ref[i]:.3e} allowed={allowed[i]:.3e} "
                      f"(|ref|={abs(ref[i]):.3g})")
    print("STAGES_GATE:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
