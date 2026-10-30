"""Oracle for gx_check.vyb — the same listing from the `safetensors` library.

Independent parse: builds each shard's header with the upstream library, resolves the
same names, and prints shard/offset/length/dtype plus the sha256 of the first 1 MiB.
The Vyb listing must match this byte for byte (GX_INDEX_MATCH).

Usage: env -u PYTHONPATH ~/venvs/exl3/bin/python native/glm53/gx_ref.py
"""

import hashlib
import json
import os
import sys

from safetensors import safe_open

DIR = os.environ.get("VYBFORGE_GX_DIR", "/mnt/apex/rick/vybforge/glm53-checkpoint")
OUT = os.environ.get("VYBFORGE_GX_OUT", "native/out/gx_listing_ref.txt")
TOTAL = 83
WIN = 1048576

WANT = [
    "model.language_model.layers.0.self_attn.q_proj.weight",
    "model.language_model.layers.0.mlp.down_proj.weight",
    "model.language_model.layers.0.hc_attn_fn",
    "model.language_model.layers.3.mlp.gate.weight",
    "model.language_model.layers.3.mlp.shared_experts.up_proj.weight",
    "model.language_model.layers.3.hc_ffn_scale",
    "model.language_model.layers.3.mlp.experts.0.gate_proj.trellis",
    "model.language_model.layers.3.mlp.experts.0.up_proj.suh",
    "model.language_model.layers.3.mlp.experts.0.down_proj.svh",
    "model.language_model.layers.3.mlp.experts.287.down_proj.trellis",
    "model.language_model.nonexistent.tensor",
]


def shard_path(shard):
    return f"{DIR}/model-{shard:05d}-of-{TOTAL:05d}.safetensors"


def main():
    # locate each name: header JSON per shard (8-byte LE length, then JSON)
    loc = {}
    for s in range(1, TOTAL + 1):
        p = shard_path(s)
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            import struct
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        base = 8 + n
        for name in WANT:
            if name in loc or "__metadata__" not in hdr and name not in hdr:
                continue
            e = hdr.get(name)
            if e is None:
                continue
            off, end = e["data_offsets"]
            loc[name] = (s, base + off, end - off, e["dtype"])

    lines = [f"TOTAL {TOTAL}"]
    hashed = 0
    missing = 0
    for name in WANT:
        if name not in loc:
            lines.append(f"{name}|MISSING")
            missing += 1
            continue
        s, off, nb, dt = loc[name]
        hl = min(nb, WIN)
        with open(shard_path(s), "rb") as f:
            f.seek(off)
            sha = hashlib.sha256(f.read(hl)).hexdigest()
        hashed += 1
        lines.append(f"{name}|{s}|{off}|{nb}|{dt}|{sha}")
    lines.append(f"HASHED {hashed}")
    lines.append(f"MISSING {missing}")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"GX_REF_OK hashed={hashed} missing={missing} out={OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
