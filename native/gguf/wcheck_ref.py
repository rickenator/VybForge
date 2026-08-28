#!/usr/bin/env python3
"""Reference for wcheck_driver.vyb: first 6 values of attn_v / ffn_down for
layers 0,4,35 from the real GGUF via the shared layer0_ref loader."""
import os, importlib.util, numpy as np
repo = "/home/rick/Projects/VybAIConf"
spec = importlib.util.spec_from_file_location("l0", os.path.join(repo, "native/gguf/layer0_ref.py"))
l0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l0)
tens = l0.parse_tsv()
def show(L, n):
    te = tens[f"blk.{L}.{n}.weight"]
    shape = tuple(te["shape"])
    w = np.asarray(l0.read_weight(te, shape)).reshape(-1)
    print(f"WREF {n} L{L} -> {[float(v) for v in w[:6]]}")
for L in (0, 4, 35):
    show(L, "attn_v")
    show(L, "ffn_down")
