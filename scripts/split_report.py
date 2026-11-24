#!/usr/bin/env python3
"""Report the real per-layer byte split of the packed checkpoint.

Reads each safetensors shard's JSON header (no weight data) to get exact tensor
byte sizes, groups them by decoder layer, and reports what the two-node split
actually costs -- so the quantisation question is answered with numbers.

Run on the node that holds the checkpoint.
"""
import json
import os
import re
import struct
import sys
from collections import defaultdict

D = sys.argv[1] if len(sys.argv) > 1 else "."

with open(os.path.join(D, "model.safetensors.index.json")) as fh:
    idx = json.load(fh)
weight_map = idx["weight_map"]

shards = sorted(set(weight_map.values()))
tensor_bytes = {}
shard_bytes = {}

for shard in shards:
    path = os.path.join(D, shard)
    with open(path, "rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(hlen))
    total = 0
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        end = meta["data_offsets"][1]
        start = meta["data_offsets"][0]
        tensor_bytes[name] = end - start
        total += end - start
    shard_bytes[shard] = total

layer_re = re.compile(r"layers\.(\d+)\.")
per_layer = defaultdict(int)
non_layer = defaultdict(int)
for name, nb in tensor_bytes.items():
    m = layer_re.search(name)
    if m:
        per_layer[int(m.group(1))] += nb
    else:
        # bucket by the part before "layers."
        tag = re.sub(r"\.\d+\..*", "", name)[:60]
        non_layer[tag] += nb

total_all = sum(tensor_bytes.values())
n_layers = len(per_layer)
print(f"shards              : {len(shards)}")
print(f"total tensor bytes  : {total_all/1e9:.2f} GB  ({total_all/2**30:.2f} GiB)")
print(f"decoder layers      : {n_layers}")
print(f"non-layer tensors   : {sum(non_layer.values())/1e9:.2f} GB")
for tag, nb in sorted(non_layer.items(), key=lambda kv: -kv[1])[:6]:
    print(f"    {nb/1e9:8.2f} GB  {tag}")

print("\nper-layer (GB):")
for i in range(n_layers):
    print(f"  layer {i:2d}: {per_layer[i]/1e9:6.2f}")

# best contiguous split minimising the max half
best = None  # (max_half, split_index, front, back)
for k in range(1, n_layers):
    front = sum(per_layer[i] for i in range(k))
    back = sum(per_layer[i] for i in range(k, n_layers))
    mx = max(front, back)
    if best is None or mx < best[0]:
        best = (mx, k, front, back)
assert best is not None, "no split found"
mx, k, front, back = best
print(f"\nbest contiguous split at layer {k}:")
print(f"  node A (0..{k-1}) : {front/1e9:.2f} GB + non-layer share")
print(f"  node B ({k}..{n_layers-1}) : {back/1e9:.2f} GB + non-layer share")
print(f"  max layer-half     : {mx/1e9:.2f} GB")

# what each node must hold if non-layer tensors are replicated on one side
nl = sum(non_layer.values())
print(f"\nwith non-layer tensors on node A only:")
print(f"  node A: {(front+nl)/1e9:.2f} GB    node B: {back/1e9:.2f} GB")
