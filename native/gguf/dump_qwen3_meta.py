#!/usr/bin/env python3
# Reference (verification-only) dump of the real Qwen3-4B GGUF used to define
# the Vyb-native loader target and later cross-check it (llama.cpp gguf-py).
# Usage: PYTHONPATH=/home/rick/Projects/llama.cpp/gguf-py python3 dump_qwen3_meta.py
import sys, os
sys.path.insert(0, "/home/rick/Projects/llama.cpp/gguf-py")
import numpy as np
from gguf import GGUFReader

MODEL = os.environ.get("VF_QWEN3", "/home/rick/Models/qwen3/Qwen3-4B-Q4_K_M.gguf")
r = GGUFReader(MODEL)

def scalar(name):
    f = r.get_field(name)
    if f is None or f.parts is None:
        return None
    arr = f.parts[f.data[0]]
    return arr.item() if arr.size == 1 else arr.tolist()

# --- config ---
keys = [
    "general.architecture", "general.name", "general.file_type",
    "qwen3.block_count", "qwen3.embedding_length", "qwen3.feed_forward_length",
    "qwen3.attention.head_count", "qwen3.attention.head_count_kv",
    "qwen3.attention.rope_freq_base", "qwen3.attention.layer_norm_rms_epsilon",
    "qwen3.context_length", "tokenizer.ggml.model",
]
print("=== CONFIG ===")
for k in keys:
    print(f"{k} = {scalar(k)}")
vocab = r.get_field("tokenizer.ggml.tokens")
print(f"tokenizer.ggml.tokens len = {None if vocab is None else vocab.parts[vocab.data[0]].size}")

# --- full tensor index ---
print("=== n_tensors: %d ===" % len(r.tensors))
lines = []
for t in r.tensors:
    lines.append("%s\t%s\t%s\toff=%d\tsize=%d" % (
        t.name, "x".join(str(x) for x in t.shape), t.tensor_type,
        t.data_offset, (t.data_size_bytes if hasattr(t, "data_size_bytes") else 0)))
os.makedirs("native/out", exist_ok=True)
with open("native/out/qwen3_4b_tensors.tsv", "w") as fh:
    fh.write("\n".join(lines) + "\n")
print("\n".join(lines[:18]))
print("... (%d total, written to native/out/qwen3_4b_tensors.tsv)" % len(lines))
sys.exit(0)
