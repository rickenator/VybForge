#!/usr/bin/env python3
"""Oracle probe: locate kv_a_proj_with_mqa.weight in shard 00005 and compare its
first words against what the device decoded (word 0x2918 instead of 0xBC30).
Read-only; no Vyb code touched. Diagnostic only."""
import json, struct, sys, glob, os

SNAP = "/srv/models/hf/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-4bpw-TensorFold/snapshots/078455ffe6472f9a52fbc1139f58b9db2881b25c"
NAME = "model.language_model.layers.3.self_attn.kv_a_proj_with_mqa.weight"

shards = sorted(glob.glob(os.path.join(SNAP, "*.safetensors")))
target = None
for s in shards:
    with open(s, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        if NAME in hdr:
            target = (s, n, hdr[NAME]); break
if not target:
    print("tensor not found in any shard"); sys.exit(1)
path, hlen, info = target
off = 8 + hlen + info["data_offsets"][0]
size = info["data_offsets"][1] - info["data_offsets"][0]
print("shard %s  header %d  data_off %d  size %d  dtype %s" % (os.path.basename(path), hlen, off, size, info["dtype"]))
with open(path, "rb") as f:
    f.seek(off)
    raw = f.read(256)
words = list(struct.unpack("<%dH" % (len(raw)//2), raw))
print("first 16 words: " + " ".join("0x%04X" % w for w in words[:16]))
print("first 4 as bf16: " + " ".join("%.8g" % struct.unpack('<f', struct.pack('<I', w << 16))[0] for w in words[:4]))
want = 0x2918
if want in words:
    print("0x2918 found at word index %d (byte %d) within the first 128 words" % (words.index(want), words.index(want)*2))
else:
    print("0x2918 NOT in the first 128 words of this tensor")
    # scan further
    with open(path, "rb") as f:
        f.seek(off); blob = f.read(size)
    allw = struct.unpack("<%dH" % (len(blob)//2), blob)
    hits = [i for i, w in enumerate(allw) if w == want]
    print("0x2918 occurrences in the whole tensor: %d (first few: %s)" % (len(hits), hits[:5]))
