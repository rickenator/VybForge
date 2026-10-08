#!/usr/bin/env python3
"""Compare a driver dump against a reference dump, element by element, as BITS.

Both sides print each value as the signed 64-bit integer its f64 bits spell (the drivers copy the
8 bytes of the output slot into an Int; the references do struct.unpack('<q')). Comparing those
integers is EXACT: no decimal formatting, no tolerance standing in for a rounding floor.

Why this exists: the drivers used to print six significant decimal digits, so every gate measured
the printing rather than the arithmetic — a difference of ~4.5e-6 that was exactly the reference
rounded to six digits (caught when BF16, which does no arithmetic at all, showed the same figure).
With bits, agreement is exact or it is counted, in ulps.

Usage: compare_bits.py <driver_dump> <reference_dump> <tag> [outfile]

Writes (and prints):
    TENSOR <name> n=<count> ndiff=<n> max_ulp=<n>
    MISSING_IN_DRIVER <name> / EXTRA_IN_DRIVER <name> / LENGTH_MISMATCH <name> a=<n> b=<n>
    TENSORS_COMPARED <n>
    TOTAL_ELEMENTS <n>
    ELEMENTS_DIFFERING <n>
    WORST_ULP <n>
    EXACT_TENSORS <n>/<total>
"""
import sys


def ordered(bits):
    """Map an f64 bit pattern (read as a SIGNED 64-bit int) to a monotone key, so ulp distance is
    just a difference of keys. The standard transform, in signed terms:

        non-negative  ->  x + 2^63      (positives, in order)
        negative      ->  -x - 1        (negatives, reversed: -0 next to +0, then -epsilon, ...)

    Without the two branches, -0.0 (the most negative pattern) and +0.0 (zero) look 2^63 apart.
    """
    if bits < 0:
        return -bits - 1
    return bits + (1 << 63)


def parse(path, tag):
    out = {}
    try:
        fh = open(path)
    except OSError as exc:
        # A missing dump must FAIL the gate, not read as "nothing compared": a gate that passes
        # because its inputs vanished is worse than no gate.
        return {"__UNREADABLE__": [f"{path}: {exc.strerror or exc}"]}
    with fh:
        for line in fh:
            if not line.startswith(tag + " "):
                continue
            head, _, vals = line.partition("->")
            name = head.split()[1].split("@")[0]
            toks = vals.split()
            try:
                out[name] = [int(t) for t in toks]
            except ValueError:
                # A pre-upgrade dump (decimals) cannot be compared as bits; say so, loudly.
                out[name] = None
    return out


def main(argv):
    if len(argv) < 4:
        print("usage: compare_bits.py <driver_dump> <ref_dump> <tag> [outfile]")
        return 2
    dp, rp, tag = argv[1], argv[2], argv[3]
    outp = argv[4] if len(argv) > 4 else None

    drv = parse(dp, tag)
    ref = parse(rp, tag)
    lines = []

    unreadable = []
    for src, d in (("driver", drv), ("reference", ref)):
        if "__UNREADABLE__" in d:
            unreadable.append(f"UNREADABLE {src} dump {d['__UNREADABLE__'][0]}")
    if unreadable:
        lines += unreadable
        lines.append("TENSORS_COMPARED 0")
        text = "\n".join(lines) + "\n"
        print(text, end="")
        if outp:
            open(outp, "w").write(text)
        return 1

    decimal = [k for k, v in list(drv.items()) + list(ref.items()) if v is None]
    if decimal:
        lines.append("STALE_DUMP decimal values present in " + " ".join(sorted(set(decimal))))
        lines.append("TENSORS_COMPARED 0")
        text = "\n".join(lines) + "\n"
        print(text, end="")
        if outp:
            open(outp, "w").write(text)
        return 1

    missing = [k for k in ref if k not in drv]
    extra = [k for k in drv if k not in ref]
    for k in missing:
        lines.append(f"MISSING_IN_DRIVER {k}")
    for k in extra:
        lines.append(f"EXTRA_IN_DRIVER {k}")

    ncmp = 0
    total = 0
    differing = 0
    worst_ulp = 0
    exact_tensors = 0
    for k in sorted(ref):
        if k not in drv:
            continue
        a, b = drv[k], ref[k]
        if len(a) != len(b):
            lines.append(f"LENGTH_MISMATCH {k} a={len(a)} b={len(b)}")
            continue
        ncmp += 1
        total += len(a)
        ndiff = 0
        mulp = 0
        for x, y in zip(a, b):
            if x == y:
                continue
            ndiff += 1
            d = abs(ordered(x) - ordered(y))
            if d > mulp:
                mulp = d
        differing += ndiff
        if mulp == 0:
            exact_tensors += 1
        if mulp > worst_ulp:
            worst_ulp = mulp
        lines.append(f"TENSOR {k} n={len(a)} ndiff={ndiff} max_ulp={mulp}")

    lines.append(f"TENSORS_COMPARED {ncmp}")
    lines.append(f"TOTAL_ELEMENTS {total}")
    lines.append(f"ELEMENTS_DIFFERING {differing}")
    lines.append(f"WORST_ULP {worst_ulp}")
    lines.append(f"EXACT_TENSORS {exact_tensors}/{ncmp}")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if outp:
        open(outp, "w").write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
