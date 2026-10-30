"""Oracle for native/host/gx_plan.vyb — GX_PLAN_MATCH.

Independent parse of the same checkpoint: weight_map order from the index JSON, each
tensor's byte range from its shard's own header, then the same `name|shard|off|nbytes|
dtype` lines and a sha256 digest. The digest must equal the Vyb side's.

Also reports how many tensors and how many bytes the plan covers (the loader's budget).

Usage: env -u PYTHONPATH ~/venvs/exl3/bin/python native/glm53/gx_plan_ref.py
"""

import hashlib
import json
import os
import struct
import sys

CKPT = "/mnt/apex/rick/vybforge/glm53-checkpoint"
OUT = "native/out/gx_plan_ref.txt"
DT = {"BF16": 2, "F16": 2, "F32": 4, "I16": 2, "I32": 4, "F64": 8, "I8": 1, "U8": 1}


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def main():
    ckpt = os.environ.get("VYBFORGE_GX_DIR", CKPT)
    out = os.environ.get("VYBFORGE_GX_OUT_REF", OUT)
    weight_map = json.load(open(f"{ckpt}/model.safetensors.index.json"))["weight_map"]
    want = set(weight_map)

    # Same traversal order as the Vyb driver: shards 1..83 ascending, each shard's
    # header entries in header order. (weight_map order would be a different digest.)
    total = 83
    lines = []
    per_layer = {}
    seen = 0
    total_bytes = 0
    pre = "model.language_model.layers."
    for s in range(1, total + 1):
        path = f"{ckpt}/model-{s:05d}-of-{total:05d}.safetensors"
        h, base = header(path)
        for name, meta in h.items():
            if name not in want:
                continue
            o0, o1 = meta["data_offsets"]
            nb = o1 - o0
            total_bytes += nb
            seen += 1
            lines.append(f"{name}|{s}|{base + o0}|{nb}|{meta['dtype']}")
            slot = 45
            if name.startswith(pre):
                num = name[len(pre):].split(".")[0]
                if num.isdigit() and int(num) < 45:
                    slot = int(num)
            per_layer[slot] = per_layer.get(slot, 0) + 1

    body = "\n".join(lines) + "\n"
    dig = hashlib.sha256(body.encode()).hexdigest()
    missing = len(want) - seen

    r = [f"PLAN tensors={len(want)} found={seen} missing={missing} ms=0"]
    for l in range(45):
        r.append(f"LAYER {l} tensors={per_layer.get(l, 0)}")
    r.append(f"OTHER tensors={per_layer.get(45, 0)}")
    r.append(f"DIGEST {dig}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(r) + "\n")
    print(f"GX_PLAN_REF_OK tensors={len(want)} found={seen} missing={missing} "
          f"bytes={total_bytes} ({total_bytes / 2**30:.1f} GiB) digest={dig}")
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
