#!/usr/bin/env python3
# Reference verification for the Vyb-native GGUF parser (parse_gguf.vyb).
# Reads native/gguf/parsed.txt (the parser's numeric dump) and checks it against
# the known synthetic fixture layout.
import os
base = os.path.join(os.path.dirname(__file__), "parsed.txt")
lines = [l.strip() for l in open(base) if l.strip()]
it = iter(lines)
def nxt(tag):
    v = next(it)
    assert v.startswith(tag), f"expected {tag!r}, got {v!r}"
    return v

# header: MAGIC ok VER ver NET nt NKV nkv
h = nxt("MAGIC")
tokens = h.split()
assert tokens[1] == "1" and tokens[3] == "3" and tokens[5] == "3" and tokens[7] == "1", tokens

def read_bytes(n):
    return bytes(int(next(it)) for _ in range(n))

assert nxt("KEYLEN").split()[1] == "12"
assert read_bytes(12) == b"general.name"
assert nxt("KEYTYPE").split()[1] == "8"
assert nxt("VALLEN").split()[1] == "4"
assert read_bytes(4) == b"tiny"

# tensor 0
assert nxt("TN").endswith("0 NAMELEN 2")
assert read_bytes(2) == b"t0"
assert nxt("TDIM") == "TDIM 0 ND 2 d2 d3"
assert nxt("TINFO") == "TINFO 0 ty 0 off 170"
# tensor 1
assert nxt("TN").endswith("1 NAMELEN 2")
assert read_bytes(2) == b"t1"
assert nxt("TDIM") == "TDIM 1 ND 1 d4"
assert nxt("TINFO") == "TINFO 1 ty 6 off 174"
# tensor 2 (q4_0)
assert nxt("TN").endswith("2 NAMELEN 2")
assert read_bytes(2) == b"t2"
assert nxt("TDIM") == "TDIM 2 ND 1 d32"
assert nxt("TINFO") == "TINFO 2 ty 2 off 190"
assert nxt("END") == "END"

print("GGUF_PARSE_VERIFY: OK  (header, kv, tensor index all match fixture)")
