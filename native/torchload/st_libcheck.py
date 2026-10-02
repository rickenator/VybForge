#!/usr/bin/env python3
"""Third opinion for the S0.3 safetensors gate: the upstream `safetensors` library.

Reads the listing produced by native/torchload/st_probe.vyb and checks it against the library's
own view of the same file — per tensor NAME: presence, dtype, shape. Order is deliberately NOT a
criterion here: the library's key order is normalized (it is not the container's header order), so
comparing positionally produces false failures on files whose header order happens not to be
sorted; the Vyb-vs-independent-parse byte-exact listing parity is what pins order. Key order is
still *reported*, as information.

Uses the lazy Slice API so bfloat16 tensors are described, not materialized (the numpy framework
cannot convert bf16 and torch is not in this venv).

Usage:  .venv/bin/python native/torchload/st_libcheck.py [listing]
Exit:   0 ok, 1 mismatch, 2 library unavailable (caller decides whether that is fatal).
"""
import os
import sys

LISTING = sys.argv[1] if len(sys.argv) > 1 else "native/out/st_listing_vyb.txt"

try:
    from safetensors import safe_open
except Exception as exc:  # library absent -> not a failure of the reader under test
    print(f"ST_LIBCHECK_UNAVAILABLE ({exc})")
    sys.exit(2)


def parse_listing(path):
    """-> (meta, ordered rows, name -> (dtype, shape, nbytes))"""
    rows, meta = [], {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if "|" not in line:
                k, _, v = line.partition(" ")
                meta[k] = v
                continue
            name, dtype, shape, nbytes, sha = line.split("|")
            shape = [int(x) for x in shape.split(",") if x != ""]
            rows.append((name, dtype, shape))
    return meta, rows, {n: (d, s) for n, d, s in rows}


def main():
    meta, rows, byname = parse_listing(LISTING)
    path = os.environ.get("VYBFORGE_ST_FILE", "artifacts/vybos-configurator-lora/adapter_model.safetensors")
    bad = []

    with safe_open(path, framework="np") as sf:
        lib_keys = list(sf.keys())
        lib = {}
        for k in lib_keys:
            sl = sf.get_slice(k)
            lib[k] = (str(sl.get_dtype()).replace("torch.", "").upper(), list(sl.get_shape()))

        missing = [k for k in lib_keys if k not in byname]
        extra = [n for n, _, _ in rows if n not in lib]
        if missing:
            bad.append(f"{len(missing)} in library but not listing (e.g. {missing[0]})")
        if extra:
            bad.append(f"{len(extra)} in listing but not library (e.g. {extra[0]})")

        for k in lib_keys:
            if k not in byname:
                continue
            ldt, lsh = lib[k]
            ddt, dsh = byname[k]
            if lsh != dsh:
                bad.append(f"shape {k} lib={lsh} listing={dsh}")
            if ddt != ldt:
                bad.append(f"dtype {k} lib={ldt} listing={ddt}")

        order_note = "same order" if lib_keys == [n for n, _, _ in rows] else "order differs (not a criterion)"

    if bad:
        for b in bad[:10]:
            print(f"   {b}")
        print(f"ST_LIBCHECK_FAIL {len(bad)} mismatch(es) vs {LISTING}")
        return 1
    print(f"ST_LIBCHECK_OK {len(rows)} tensors agree with the safetensors library by name "
          f"(count/dtype/shape); key {order_note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
