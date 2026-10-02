#!/usr/bin/env python3
"""Independent reference side of the S0.3 safetensors gate.

Parses the safetensors container itself with the stdlib (8-byte LE header length, JSON map, raw
LE data region) and emits the same canonical listing as native/torchload/st_probe.vyb. It is
deliberately NOT a wrapper around the Vyb reader: two independent implementations of the same
container, compared byte-for-byte. native/torchload/st_libcheck.py adds a third opinion from the
upstream `safetensors` library.

SAMPLING RULE (identical in st_probe.vyb — change both together): a tensor's sha column is the
sha256 of its first `hlen` bytes where hlen = its byte length if the file is <= 300,000,000 bytes,
else min(byte length, 1,048,576); files over the threshold hash only indices
{0, 1, n-2, n-1, argmin(bytes), argmax(bytes)}, deduped ascending, and unhashed tensors print "-".
"""
import hashlib
import json
import os
import struct
import sys

BIG = 300_000_000
WIN = 1_048_576

# mirrored from st_dtsize(): 0 means "unknown", counted separately rather than guessed at
DTSIZE = {
    "F64": 8, "I64": 8, "U64": 8,
    "F32": 4, "I32": 4, "U32": 4,
    "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
    "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1,
}


def shape_str(shape):
    return ",".join(str(int(d)) for d in shape)


def main():
    path = os.environ.get("VYBFORGE_ST_FILE", "artifacts/vybos-configurator-lora/adapter_model.safetensors")
    out = os.environ.get("VYBFORGE_ST_OUT", "native/out/st_listing_ref.txt")

    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) != 8:
            print(f"ST_REF_FAIL short prefix ({len(head)} bytes) path={path}")
            return 1
        hlen = struct.unpack("<Q", head)[0]
        hdr_raw = f.read(hlen)
        if len(hdr_raw) != hlen:
            print(f"ST_REF_FAIL short header ({len(hdr_raw)}/{hlen}) path={path}")
            return 1
        hdr = json.loads(hdr_raw.decode("utf-8"))
        base = 8 + hlen
        names = [k for k in hdr if k != "__metadata__"]
        if not names:
            print(f"ST_REF_FAIL no tensors path={path}")
            return 1

        data_end = base + max(hdr[n]["data_offsets"][1] for n in names)
        big = data_end > BIG

        picked = set()
        if big:
            n = len(names)
            sizes = {i: hdr[names[i]]["data_offsets"][1] - hdr[names[i]]["data_offsets"][0] for i in range(n)}
            imin = min(range(n), key=lambda i: sizes[i])          # first occurrence, like the Vyb scan
            imax = max(range(n), key=lambda i: sizes[i])
            for i in (0, 1, n - 2, n - 1, imin, imax):
                if 0 <= i < n:
                    picked.add(i)

        lines = [f"TENSORS {len(names)}", f"HEADER {hlen}", f"DATABASE {base}", f"DATAEND {data_end}",
                 f"BIG {'true' if big else 'false'}"]
        bad = nosize = hc = 0
        for i, name in enumerate(names):
            e = hdr[name]
            dtype = e["dtype"]
            shape = e["shape"]
            off, end = e["data_offsets"]
            nbytes = end - off
            size = DTSIZE.get(dtype, 0)
            if size:
                numel = 1
                for d in shape:
                    numel *= int(d)
                if numel * size != nbytes:
                    bad += 1
            else:
                nosize += 1
            go = (not big) or (i in picked)
            hlen_i = min(nbytes, WIN) if big else nbytes
            sha = "-"
            if go:
                f.seek(base + off)
                b = f.read(hlen_i)
                sha = hashlib.sha256(b).hexdigest() if len(b) == hlen_i else "ERR"
                if sha != "ERR":
                    hc += 1
            lines.append(f"{name}|{dtype}|{shape_str(shape)}|{nbytes}|{sha}")
        lines += [f"HASHED {hc}", f"MISMATCH {bad}", f"NOSIZE {nosize}"]

    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"ST_REF_OK tensors={len(names)} header={hlen} hashed={hc} mismatch={bad} nosize={nosize} out={out}")
    return 1 if bad > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
