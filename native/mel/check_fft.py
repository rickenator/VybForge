#!/usr/bin/env python3
"""check_fft.py — compare fft_probe.vyb's output against numpy's rfft.

    ~/miniconda/bin/python check_fft.py vyb_fft.txt

The probe prints values as integers scaled by 1e6 (that is the only exact
channel out of Vyb), so every comparison here is against integers/1e6 within a
tolerance that covers the print channel (5e-7) plus the reference's float32
rounding.
"""
import sys

import numpy as np

HERE = __file__.rsplit("/", 1)[0]


def load_gold(name):
    return np.fromfile(f"{HERE}/gold/{name}.f32", dtype="<f4").astype(np.float64)


def parse(path):
    sections = {}
    cur = None
    expect_n = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("SECTION "):
                cur = line.split(" ", 1)[1]
                sections[cur] = {"n": 0, "vals": []}
            elif line == "n=":
                expect_n = True
            elif cur and line and (line.lstrip("-").isdigit()):
                if expect_n:
                    sections[cur]["n"] = int(line)
                    expect_n = False
                else:
                    sections[cur]["vals"].append(int(line))
            elif line.startswith(("LOAD_FAIL", "BAD_LEN", "FFT_PROBE")):
                pass
    return sections


def report(tag, got, ref_re, ref_im, tol):
    ref = np.empty(2 * len(ref_re))
    ref[0::2] = ref_re
    ref[1::2] = ref_im
    got = np.asarray(got, dtype=np.float64)
    if len(got) != len(ref):
        print(f"{tag}: LENGTH MISMATCH got {len(got)} want {len(ref)}")
        return False
    diff = np.abs(got - ref)
    scale = max(1.0, np.abs(ref).max())
    print(f"{tag}: n={len(got)//2} max|diff|={diff.max():.3e} "
          f"rel={diff.max()/scale:.3e} ref_peak={np.abs(ref).max():.4f} "
          f"vyb_peak={np.abs(got).max():.4f}")
    bad = int((diff > tol).sum())
    print(f"{tag}: over tol={tol:g}: {bad}")
    if bad:
        idx = np.argsort(-diff)[:6]
        for i in idx:
            print(f"   bin {i//2} {'re' if i%2==0 else 'im'}: vyb={got[i]:.9f} "
                  f"ref={ref[i]:.9f} diff={got[i]-ref[i]:.3e}")
    return bad == 0


def main():
    sections = parse(sys.argv[1])
    tol = 2e-5      # absolute: covers the 5e-7 print quantum and float32 reference rounding
    ok = True
    for tag, (gname, imname) in (("random", ("fft_random_q_re", "fft_random_q_im")),
                                 ("frame", ("frame_spec_q_re", "frame_spec_q_im"))):
        if tag not in sections:
            print(f"{tag}: MISSING from probe output")
            ok = False
            continue
        scale = 1e6
        got = np.asarray(sections[tag]["vals"], dtype=np.float64) / scale
        gh = load_gold(gname)
        gi = load_gold(imname)
        # the probe emits re,im interleaved; fold the gold the same way
        ref = np.empty(2 * len(gh))
        ref[0::2] = gh
        ref[1::2] = gi
        if len(got) < len(ref):
            print(f"{tag}: LENGTH MISMATCH got {len(got)} want {len(ref)}")
            ok = False
            continue
        if len(got) > len(ref):
            # the Vyb runtime prints main()'s return value after the last line,
            # so one stray integer can land in the final section
            print(f"{tag}: note: {len(got) - len(ref)} trailing integer(s) ignored")
            got = got[:len(ref)]
        d = np.abs(got - ref)
        peak = max(1.0, np.abs(ref).max())
        print(f"{tag}: bins={len(gh)} max|diff|={d.max():.3e} rel={d.max()/peak:.3e} "
              f"ref_peak={np.abs(ref).max():.4f} vyb_peak={np.abs(got).max():.4f}")
        bad = int((d > tol).sum())
        print(f"{tag}: values over tol={tol:g}: {bad}")
        if bad:
            ok = False
            for i in np.argsort(-d)[:6]:
                print(f"   bin {i//2} {'re' if i % 2 == 0 else 'im'}: "
                      f"vyb={got[i]:.9f} ref={ref[i]:.9f} diff={got[i]-ref[i]:.3e}")
    print("FFT_GATE:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
