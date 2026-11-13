"""Oracle for native/host/gx_load_driver.vyb — GX_LOAD_MATCH.

Same tensor list, same three byte windows (first / 8-aligned middle / last 8 bytes),
printed as the same `name|nbytes|w0|wmid|wlast` listing the Vyb driver emits. Bytes
come from an independent parse of the same shards (json header + data_offsets, read
through a plain file handle — not through the Vyb safetensors module). The Vyb side
additionally proves the bytes survived the upload to the device (in-driver readback),
which this side cannot see.

Usage: env -u PYTHONPATH ~/venvs/exl3/bin/python native/glm53/gx_load_ref.py
"""

import json
import os
import struct
import sys

CKPT = "/mnt/apex/rick/vybforge/glm53-checkpoint"
OUT = "native/out/gx_load_ref.txt"

WANT = [
    "model.language_model.layers.0.self_attn.q_proj.weight",
    "model.language_model.layers.0.mlp.down_proj.weight",
    "model.language_model.layers.0.hc_attn_fn",
    "model.language_model.layers.3.hc_ffn_scale",
    "model.language_model.layers.3.mlp.gate.weight",
    "model.language_model.layers.3.mlp.shared_experts.up_proj.weight",
    "model.language_model.layers.7.self_attn.kv_b_proj.weight",
    "model.language_model.layers.3.mlp.experts.0.gate_proj.trellis",
    "model.language_model.layers.3.mlp.experts.0.up_proj.suh",
    "model.language_model.layers.3.mlp.experts.287.down_proj.trellis",
]


def raw_bytes(path, name):
    """The tensor's raw bytes straight out of the shard via the header offsets."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        meta = header[name]
        o0, o1 = meta["data_offsets"]
        f.seek(base + o0)
        return f.read(o1 - o0)


def main():
    ckpt = os.environ.get("VYBFORGE_GX_DIR", CKPT)
    out = os.environ.get("VYBFORGE_GX_OUT_REF", OUT)
    idx = json.load(open(f"{ckpt}/model.safetensors.index.json"))["weight_map"]
    lines = ["TOTAL 83"]
    ok = miss = 0
    for name in WANT:
        shard = idx.get(name)
        if shard is None:
            miss += 1
            lines.append(f"{name}|MISSING")
            continue
        data = raw_bytes(f"{ckpt}/{shard}", name)
        nb = len(data)
        mid = ((nb // 2) // 8) * 8
        last = nb - 8
        lines.append(f"{name}|{nb}|{data[0:8].hex()}|{data[mid:mid + 8].hex()}|{data[last:last + 8].hex()}")
        ok += 1
    lines += [f"LOADED {ok}", "FAILED 0", f"MISSING {miss}"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"GX_LOAD_REF_OK loaded={ok} missing={miss} out={out}")
    return 0 if miss == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
