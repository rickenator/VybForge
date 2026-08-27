#!/usr/bin/env python3
# Phase-1 verification: cross-check the Vyb-native GGUF metadata reader
# (native/gguf/read_real_meta.vyb -> native/out/qwen3_4b_vyb.txt) against the
# ground truth from llama.cpp gguf-py (native/out/qwen3_4b_tensors.tsv).
# Verification-only; never in the runtime.
import os

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ground truth: name \t shape (x-joined) \t type \t off=<abs>
gt = {}
for line in open(os.path.join(repo, "native/out/qwen3_4b_tensors.tsv")):
    if not line.strip():
        continue
    name, shape, ty, offs, *_ = line.rstrip("\n").split("\t")
    off = int(offs.removeprefix("off="))
    gt[name] = (shape, int(ty), off)

# Vyb output: name \t dims (x-joined) \t type \t rel_off
vyb = {}
for line in open(os.path.join(repo, "native/out/qwen3_4b_vyb.txt")):
    if not line.strip():
        continue
    name, dims, ty, rel = line.rstrip("\n").split("\t")
    vyb[name] = (dims, int(ty), int(rel))

names_gt, names_vyb = set(gt), set(vyb)
if names_gt != names_vyb:
    print(f"NAME SET MISMATCH: only_gt={sorted(names_gt-names_vyb)[:3]} "
          f"only_vyb={sorted(names_vyb-names_gt)[:3]}")
    raise SystemExit(1)

# data-section base = gt_abs(output_norm) - vyb_rel(output_norm)  (first tensor)
first = "output_norm.weight"
data_base = gt[first][2] - vyb[first][2]

bad = []
for name in gt:
    gs, gt_ty, gt_abs = gt[name]
    vs, vy_ty, vy_rel = vyb[name]
    if gs != vs:
        bad.append((name, "shape", gs, vs))
    if gt_ty != vy_ty:
        bad.append((name, "type", gt_ty, vy_ty))
    if gt_abs != vy_rel + data_base:
        bad.append((name, "offset", gt_abs, vy_rel + data_base))

print(f"tensors compared: {len(gt)}   data_base={data_base}")
if bad:
    print("MISMATCHES:", len(bad))
    for b in bad[:10]:
        print("  ", b)
    raise SystemExit(1)
print("GGUF_META_VERIFY: OK (name/shape/type/offset exact vs llama.cpp)")
