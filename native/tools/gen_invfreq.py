#!/usr/bin/env python3
"""Generate a model's rope inverse-frequency table — the artifact the engine's rope kernel reads.

The table is one f64 per PAIR: `inv[k] = base ** (-2k / n_dims)` for k in 0 .. n_dims/2 - 1. The engine
reads it and the kernel indexes it as `FR[i % (n_rot/2)]`, so the entry count IS n_dims/2 and a table that
disagrees with the model's metadata is a silent rope corruption — the model would still run, with a subtly
wrong rotation, and every downstream number would be plausible.

Provenance, for the record: nothing in this repo generated `native/out/layer0_invfreq.bin`; it was
identified by measurement as `1/5e6^(2k/128)` (rope base 5,000,000, n_dims 128) at maxrel 0.0. This tool
reproduces it BYTE-FOR-BYTE from those two numbers, which is the check that matters: the generator is
validated against the artifact the engine already runs, not against my reading of a formula.

Usage:
  gen_invfreq.py BASE N_DIMS OUT [--expect-existing PATH]
    BASE      rope.freq_base from the model's GGUF metadata (required — no default; a wrong base is the
              failure this tool exists to prevent)
    N_DIMS    rope.dimension_count
    OUT       where to write (f64 little-endian, n_dims/2 entries)
    --expect-existing PATH   require the output to match this existing file byte-for-byte
"""
import os
import sys

import numpy as np


def table(base, n_dims):
    if n_dims <= 0 or n_dims % 2:
        raise SystemExit(f"GEN_INVFREQ_FAIL n_dims must be positive and even, got {n_dims}")
    return np.array([float(base) ** (-2.0 * k / n_dims) for k in range(n_dims // 2)], dtype="<f8")


def main(argv):
    if len(argv) < 4:
        print(__doc__)
        return 2
    base = float(argv[1])
    n_dims = int(argv[2])
    out = argv[3]
    expect = None
    if len(argv) >= 6 and argv[4] == "--expect-existing":
        expect = argv[5]

    t = table(base, n_dims)
    # self-checks that would catch a wrong base or a mis-scaled dim before anything is written
    implied = np.exp((n_dims / 2.0) * np.log(1.0 / (t[1] / t[0]))) if len(t) > 1 else base
    ok_implied = abs(implied - base) / base < 1e-9
    ok_count = len(t) == n_dims // 2
    print(f"GEN_INVFREQ base={base:g} n_dims={n_dims} entries={len(t)} "
          f"implied_base_from_ratio={implied:.6g} first={t[0]:.10g} last={t[-1]:.6e}")
    print(f"GEN_INVFREQ self-check implied_base_round_trip={ok_implied} entries==n_dims/2={ok_count}")
    if not (ok_implied and ok_count):
        print("GEN_INVFREQ_FAIL the table does not round-trip to the base it was given")
        return 1

    if expect:
        want = np.fromfile(expect, dtype="<f8")
        if want.shape != t.shape:
            print(f"GEN_INVFREQ_FAIL shape mismatch vs {expect}: {len(want)} vs {len(t)}")
            return 1
        if not np.array_equal(want, t):
            d = np.max(np.abs(want - t)) / max(float(np.max(np.abs(want))), 1e-30)
            print(f"GEN_INVFREQ_FAIL not byte-identical to {expect} (maxrel {d:.3e})")
            return 1
        print(f"GEN_INVFREQ_MATCH byte-identical to {expect} ({t.nbytes} bytes)")

    with open(out, "wb") as fh:
        fh.write(t.tobytes())
    print(f"GEN_INVFREQ_DONE wrote {out} ({t.nbytes} bytes, {len(t)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
