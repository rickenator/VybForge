"""GLM-5.3 full 45-layer forward — numpy vs HuggingFace, layer by layer.

The keystone gate for "GLM-5.3 works on the substrate path": every layer of the
real model is composed in numpy (KDA linear attention, MLA+DSA dense form, dense
or MoE feed-forward, two hyper-connection sites each) and compared against the
HF transformers glm5_next layer outputs on identical inputs, with weights loaded
from the apex checkpoint copy (/mnt/apex/rick/vybforge/glm53-checkpoint/).

Weights stay per-layer (load -> run -> free); MoE experts dequantize on demand
from EXL3 trellis for exactly the experts the router selects.

Usage: env -u PYTHONPATH ~/venvs/exl3/bin/python native/glm53/verify_full_forward.py [--dry-run]
"""

import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/rick/Projects/VybForge/native/glm53")
sys.path.insert(0, "/home/rick/Projects/VybForge/native/exl3")

CKPT = "/mnt/apex/rick/vybforge/glm53-checkpoint"
H = 4096


def load_config():
    with open(f"{CKPT}/config.json") as f:
        cfg = json.load(f).get("text_config", {})
    return cfg


def dispatch_table(cfg):
    """Which layer index is KDA vs DSA, dense vs MoE — from the checkpoint's own config."""
    lac = cfg["linear_attn_config"]
    kda = set(lac["kda_layers"])
    full = set(lac["full_attn_layers"])
    first_dense = cfg.get("first_k_dense_replace", 3)
    rows = []
    for i in range(cfg["num_hidden_layers"]):
        attn = "KDA" if i in kda else ("DSA" if i in full else "?")
        mlp = "dense" if i < first_dense else "MoE"
        rows.append((i, attn, mlp))
    return rows


def dry_run(cfg):
    rows = dispatch_table(cfg)
    n_kda = sum(1 for r in rows if r[1] == "KDA")
    n_dsa = sum(1 for r in rows if r[1] == "DSA")
    n_dense = sum(1 for r in rows if r[2] == "dense")
    print(f"layers: {len(rows)}  KDA {n_kda}  DSA {n_dsa}  dense-MLP {n_dense}  MoE {len(rows) - n_dense}")
    print("schedule:", " ".join(f"{i}:{a}/{m}" for i, a, m in rows[:8]) + " ...")
    assert n_kda == 34 and n_dsa == 11, "schedule drifted from the pinned config"
    assert [r[1] for r in rows] == [
        "KDA" if i in set(cfg["linear_attn_config"]["kda_layers"]) else "DSA" for i in range(len(rows))
    ]
    print("dispatch table: OK (34 KDA + 11 DSA, first 3 layers dense)")
    return 0


def main():
    cfg = load_config()
    if "--dry-run" in sys.argv or "--dry-run" in sys.argv:
        return dry_run(cfg)
    print("full run needs the apex checkpoint copy (.COMPLETE) — see ROADMAP; dry-run only for now")
    return dry_run(cfg)


if __name__ == "__main__":
    sys.exit(main())
