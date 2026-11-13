#!/usr/bin/env python3
"""check_mel.py — compare mel_probe.vyb's output against the reference artifacts.

    ~/miniconda/bin/python check_mel.py vyb_mel.txt

Two gates:
  filters — the 80x201 Slaney filterbank Vyb computed vs the one transformers
            built (transposed, and built with librosa's algorithm)
  mel     — the 80x3000 log-mel vs WhisperFeatureExtractor's own output on the
            same PCM, plus the distribution shape (per-band means) so a systematic
            scaling or offset error cannot hide behind a tolerance.
"""
import sys

import numpy as np

HERE = __file__.rsplit("/", 1)[0]
SECTIONS = ("filters", "mel")
SCALES = {"filters": 1e12, "mel": 1e6}


def parse(path):
    sections = {}
    cur = None
    expect_n = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("SECTION "):
                cur = line.split(" ", 1)[1].split(" ", 1)[0]
                sections[cur] = {"n": None, "vals": []}
                expect_n = line.endswith("n=")
            elif line == "n=" or line.endswith(" n="):
                expect_n = True
            elif cur and line and (line.lstrip("-").isdigit()):
                if expect_n:
                    sections[cur]["n"] = int(line)
                    expect_n = False
                else:
                    sections[cur]["vals"].append(int(line))
    return sections


def main():
    sections = parse(sys.argv[1])
    ok = True

    # --- filters -------------------------------------------------------------
    if "filters" not in sections:
        print("filters: MISSING")
        ok = False
    else:
        got = np.asarray(sections["filters"]["vals"], dtype=np.float64) / SCALES["filters"]
        got = got[:80 * 201]      # drop the trailing integer(s) main()'s return adds
        ref = np.load(f"{HERE}/gold/mel_filters.npy").T.reshape(-1).astype(np.float64)
        if len(got) != len(ref):
            print(f"filters: LENGTH MISMATCH got {len(got)} want {len(ref)}")
            ok = False
        else:
            d = np.abs(got - ref)
            print(f"filters: {len(got)} values  max|diff|={d.max():.3e} "
                  f"ref_peak={np.abs(ref).max():.6f} vyb_peak={np.abs(got).max():.6f}")
            # the print quantum at 1e12 is 5e-13, so anything above ~1e-9 is real
            bad = int((d > 1e-9).sum())
            print(f"filters: values over 1e-9: {bad}")
            if bad:
                ok = False
                for i in np.argsort(-d)[:6]:
                    print(f"   m={i//201} b={i%201}: vyb={got[i]:.9e} ref={ref[i]:.9e}")

    # --- mel -----------------------------------------------------------------
    if "mel" not in sections:
        print("mel: MISSING")
        ok = False
    else:
        raw = np.asarray(sections["mel"]["vals"], dtype=np.float64) / SCALES["mel"]
        raw = raw[:80 * 3000]     # drop the trailing integer(s) main()'s return adds
        # Both sides are band-major (band m at m*3000 + t) now that log_mel emits
        # the layout the model consumes — no transpose here on purpose: a
        # transposed comparison would hide exactly the bug this gate exists for.
        got = raw
        ref = np.load(f"{HERE}/gold/mel.npy").astype(np.float64).reshape(-1)
        if len(got) != len(ref):
            print(f"mel: LENGTH MISMATCH got {len(got)} want {len(ref)}")
            ok = False
        else:
            d = np.abs(got - ref)
            print(f"mel: {len(got)} values  max|diff|={d.max():.3e} rel={d.max()/max(1.0, np.abs(ref).max()):.3e}")
            print(f"mel: ref range [{ref.min():.4f}, {ref.max():.4f}]  "
                  f"vyb range [{got.min():.4f}, {got.max():.4f}]")
            # tolerance: 1e-6 print quantum + float64-vs-float32 reference rounding
            tol = 2e-5
            bad = int((d > tol).sum())
            print(f"mel: values over {tol:g}: {bad} ({100.0*bad/len(d):.4f}%)")
            if bad:
                ok = False
                worst = np.argsort(-d)[:8]
                for i in worst:
                    t, m = divmod(int(i), 80)
                    print(f"   frame {t} band {m}: vyb={got[i]:.7f} ref={ref[i]:.7f} diff={got[i]-ref[i]:.3e}")
            # shape check: per-band mean, because a uniform offset would be caught
            # by the max-diff but a small systematic one is easier to read here
            gr = raw.reshape(80, -1).mean(1)
            rr = np.load(f"{HERE}/gold/mel.npy").astype(np.float64).mean(1)
            print(f"mel: per-band mean max|diff|={np.abs(gr-rr).max():.3e} "
                  f"(freq resolution = {np.abs(rr).mean():.4f})")

    print("MEL_GATE:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
