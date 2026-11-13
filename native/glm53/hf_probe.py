"""Explore the HF glm5_next text model so P4 can load layer 0's real weights into it.

Usage: env -u PYTHONPATH ~/venvs/exl3/bin/python native/glm53/hf_probe.py
"""

import json
import sys

import torch
from transformers import AutoConfig

CFG = "/home/rick/Projects/VybForge/native/glm53/layer0/config.json"


def main():
    d = json.load(open(CFG))
    cfg = AutoConfig.from_dict(d) if "text_config" not in d else \
        __import__("transformers.models.glm5_next.configuration_glm5_next",
                   fromlist=["Glm5NextConfig"]).Glm5NextConfig.from_dict(d)
    print("config class:", type(cfg).__name__)
    tc = getattr(cfg, "text_config", None)
    print("text_config class:", type(tc).__name__ if tc is not None else None)
    if tc is None:
        return 1

    # one layer, dense mlp, no MTP heads
    tc.num_hidden_layers = 1
    tc.layer_types = [tc.layer_types[0]]
    tc.mlp_layer_types = [tc.mlp_layer_types[0]]
    tc.indexer_types = [tc.indexer_types[0]]
    if hasattr(tc, "num_nextn_predict_layers"):
        tc.num_nextn_predict_layers = 0
    if hasattr(tc, "linear_attn_config") and hasattr(tc.linear_attn_config, "kda_layers"):
        tc.linear_attn_config.kda_layers = [0]
        tc.linear_attn_config.full_attn_layers = []
    print("layer_types:", tc.layer_types, "mlp:", tc.mlp_layer_types)
    print("hc_mult:", getattr(tc, "hc_mult", None), "hidden:", tc.hidden_size)

    import transformers.models.glm5_next.modeling_glm5_next as M
    cls = getattr(M, "Glm5NextTextModel", None) or getattr(M, "Glm5NextModel", None)
    print("model class:", cls.__name__)
    model = cls(tc).to(torch.float32)
    model.eval()

    print("\n--- parameters (layer 0 + top level) ---")
    n = 0
    for name, p in model.named_parameters():
        if "layers.1" in name:
            continue
        print(f"  {name:59s} {tuple(p.shape)} {p.dtype}")
        n += 1
    print(f"{n} params shown")
    total = sum(p.numel() for p in model.parameters())
    print(f"total params in 1-layer model: {total:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
