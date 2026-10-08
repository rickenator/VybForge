#!/usr/bin/env python3
"""Ridge GGUF tensor inventory — VybForge#10 phase 3 groundwork.

Parses a GGUF v3 header (metadata KV + tensor table) and emits:
  * a checked-in inventory TSV: name, dims, ggml type id, type name, offset, bytes
  * a metadata dump of the architecture-relevant keys
  * a per-role type summary and a coverage report against the quant types this
    engine implements today, plus a recomputed bpw to check the published figure

The point is reproducibility: the inventory is a function of the file, so anyone
can re-derive it rather than trusting a hand-written table.

Usage:
  env -u PYTHONPATH python3 native/gguf/ridge_inventory.py [GGUF] \
      [--tsv native/gguf/ridge-3.7bpw-inventory.tsv] [--meta /tmp/ridge_meta.txt]
"""
import argparse
import os
import struct
import sys
from collections import defaultdict

# ggml type id -> (name, bytes per block, elements per block)
GGML_TYPES = {
    0: ("F32", 4, 1), 1: ("F16", 2, 1), 2: ("Q4_0", 18, 32), 3: ("Q4_1", 20, 32),
    6: ("Q5_0", 22, 32), 7: ("Q5_1", 24, 32), 8: ("Q8_0", 34, 32), 9: ("Q8_1", 36, 32),
    10: ("Q2_K", 84, 256), 11: ("Q3_K", 110, 256), 12: ("Q4_K", 144, 256),
    13: ("Q5_K", 176, 256), 14: ("Q6_K", 210, 256), 15: ("Q8_K", 292, 256),
    16: ("IQ2_XXS", 66, 256), 17: ("IQ2_XS", 74, 256), 18: ("IQ3_XXS", 98, 256),
    19: ("IQ1_S", 50, 256), 20: ("IQ4_NL", 18, 32), 21: ("IQ3_S", 110, 256),
    22: ("IQ2_S", 82, 256), 23: ("IQ4_XS", 136, 256), 24: ("I8", 1, 1), 25: ("I16", 2, 1),
    26: ("I32", 4, 1), 27: ("I64", 8, 1), 28: ("F64", 8, 1), 29: ("IQ1_M", 56, 256),
    30: ("BF16", 2, 1),
}

# GGUF metadata value type ids
KV_UINT8, KV_INT8, KV_UINT16, KV_INT16, KV_UINT32, KV_INT32 = 0, 1, 2, 3, 4, 5
KV_FLOAT32, KV_BOOL, KV_STRING, KV_ARRAY, KV_UINT64, KV_INT64, KV_FLOAT64 = 6, 7, 8, 9, 10, 11, 12
KV_FIXED = {
    KV_UINT8: ("<B", 1), KV_INT8: ("<b", 1), KV_UINT16: ("<H", 2), KV_INT16: ("<h", 2),
    KV_UINT32: ("<I", 4), KV_INT32: ("<i", 4), KV_FLOAT32: ("<f", 4), KV_BOOL: ("<?", 1),
    KV_UINT64: ("<Q", 8), KV_INT64: ("<q", 8), KV_FLOAT64: ("<d", 8),
}

# Quant types this engine implements today (native/gguf/*, qwen3.vyb kernels).
IMPLEMENTED = {"F32", "Q4_0", "Q4_K", "Q6_K"}

ARCH_KEYS = (
    "general.architecture", "general.name", "general.size_label", "general.file_type",
    "context_length", "block_count", "embedding_length", "feed_forward_length",
    "attention.head_count", "attention.head_count_kv", "attention.key_length",
    "attention.value_length", "attention.layer_norm_rms_epsilon", "rope.freq_base",
    "ssm.", "nextn.", "vision.", "clip.", "mtp", "draft", "tensor_data_offset",
)


class Reader:
    def __init__(self, fh):
        self.fh = fh

    def raw(self, n):
        b = self.fh.read(n)
        if len(b) != n:
            raise EOFError(f"wanted {n} bytes, got {len(b)}")
        return b

    def u32(self):
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.raw(8))[0]

    def i32(self):
        return struct.unpack("<i", self.raw(4))[0]

    def f32(self):
        return struct.unpack("<f", self.raw(4))[0]

    def string(self):
        return self.raw(self.u64()).decode("utf-8", "replace")

    def kv_value(self, vtype, depth=0):
        if vtype in KV_FIXED:
            fmt, size = KV_FIXED[vtype]
            return struct.unpack(fmt, self.raw(size))[0]
        if vtype == KV_STRING:
            return self.string()
        if vtype == KV_ARRAY:
            etype, n = self.u32(), self.u64()
            vals = [self.kv_value(etype, depth + 1) for _ in range(n)]
            if depth == 0 and len(vals) > 8:          # tokenizer vocab etc.
                return f"[{etype}:{n} items] {vals[:4]} ..."
            return vals
        raise ValueError(f"unknown KV value type {vtype}")


def parse(path):
    with open(path, "rb") as fh:
        r = Reader(fh)
        magic = r.raw(4)
        if magic != b"GGUF":
            sys.exit(f"not a GGUF file: magic {magic!r}")
        version = r.u32()
        n_tensors, n_kv = r.u64(), r.u64()
        meta = {}
        for _ in range(n_kv):
            key = r.string()
            meta[key] = r.kv_value(r.u32())
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            nd = r.u32()
            dims = [r.u64() for _ in range(nd)]
            ttype = r.u32()
            off = r.u64()
            tensors.append((name, dims, ttype, off))
        data_start = fh.tell()
    return version, meta, tensors, data_start


def role_of(name):
    if name.startswith("blk.64.") or name.startswith("nextn."):
        return "mtp"
    if name.startswith("v.") or name.startswith("mm.") or name.startswith("vision"):
        return "vision"
    if "ssm_" in name:
        return "gdn_state"
    if "attn_qkv" in name or "attn_gate" in name or "ssm_out" in name:
        return "gdn_mixer"
    if "attn_" in name:
        return "attention"
    if "ffn_" in name:
        return "ffn"
    if "token_embd" in name or "output.weight" in name:
        return "embed_head"
    if "norm" in name:
        return "norm"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", nargs="?",
                    default=os.path.expanduser("~/Models/qwen38-27b-ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf"))
    ap.add_argument("--tsv", default="native/gguf/ridge-3.7bpw-inventory.tsv")
    ap.add_argument("--meta", default="")
    args = ap.parse_args()
    if not os.path.exists(args.gguf):
        sys.exit(f"missing GGUF: {args.gguf}")

    version, meta, tensors, data_start = parse(args.gguf)
    print(f"file      : {args.gguf} ({os.path.getsize(args.gguf):,} bytes)")
    print(f"gguf ver  : {version}   tensors: {len(tensors)}   kv: {len(meta)}")

    # ---- inventory TSV ----------------------------------------------------
    rows, per_type, per_role = [], defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    total_elems = total_bytes = 0
    for name, dims, ttype, off in tensors:
        tname, blk_b, blk_e = GGML_TYPES.get(ttype, (f"TYPE{ttype}", 0, 1))
        nelem = 1
        for d in dims:
            nelem *= d
        nbytes = (nelem // blk_e) * blk_b if blk_e > 1 else nelem * blk_b
        rows.append((name, "x".join(str(d) for d in dims), ttype, tname, off, nbytes))
        per_type[tname][0] += nbytes
        per_type[tname][1] += 1
        role = role_of(name)
        per_role[role][0] += nbytes
        per_role[role][1] += 1
        total_elems += nelem
        total_bytes += nbytes

    os.makedirs(os.path.dirname(args.tsv) or ".", exist_ok=True)
    with open(args.tsv, "w", encoding="utf-8") as fh:
        fh.write("# Ridge GGUF tensor inventory — generated by native/gguf/ridge_inventory.py\n")
        fh.write("# name\tdims\ttype_id\ttype\trole\toffset\tbytes\n")
        for (name, dims, ttype, tname, off, nbytes) in rows:
            fh.write(f"{name}\t{dims}\t{ttype}\t{tname}\t{role_of(name)}\t{off}\t{nbytes}\n")
    print(f"\nwrote inventory: {args.tsv}  ({len(rows)} tensors)")
    print(f"tensor bytes  : {total_bytes:,}  ({total_bytes / 2**30:.2f} GiB)  "
          f"bpw = {total_bytes * 8 / total_elems:.3f}")

    print("\n--- per type ---")
    for t, (b, n) in sorted(per_type.items(), key=lambda kv: -kv[1][0]):
        mark = "ok " if t in IMPLEMENTED else "MISSING"
        print(f"  {t:9s} {n:4d} tensors  {b / 2**30:7.3f} GiB   {mark}")
    print("\n--- per role ---")
    for r, (b, n) in sorted(per_role.items(), key=lambda kv: -kv[1][0]):
        print(f"  {r:11s} {n:4d} tensors  {b / 2**30:7.3f} GiB")

    missing = sorted(set(per_type) - IMPLEMENTED)
    print("\n--- coverage ---")
    print(f"  implemented: {', '.join(sorted(IMPLEMENTED & set(per_type))) or '(none)'}")
    print(f"  MISSING    : {', '.join(missing) or '(none)'}")
    for t in missing:
        print(f"    {t}: {per_type[t][1]} tensors, {per_type[t][0] / 2**30:.3f} GiB")

    # ---- architecture metadata ------------------------------------------
    lines = [f"{k} = {v}" for k, v in meta.items()
             if any(s in k for s in ARCH_KEYS)]
    if args.meta:
        with open(args.meta, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"\nwrote metadata: {args.meta}")
    print("\n--- architecture metadata (subset) ---")
    for ln in lines[:40]:
        print("  " + ln[:150])


if __name__ == "__main__":
    main()
