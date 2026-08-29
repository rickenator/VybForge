#!/usr/bin/env python3
# Reference check for T1: the Vyb native corpus encoder (native/train/encode_corpus.vyb)
# must tokenize real corpus assistant responses EXACTLY like transformers. Verification
# only (the encoder pipeline itself is all Vyb). Reads the same JSONL records, tokenizes
# the assistant content with the Qwen3 tokenizer, and compares to the ids the Vyb program
# printed (ASST <i>: <text>  =>  [id...]; LABELS lines are the shifted CE targets).
import os, re, json, subprocess, sys
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
try:
    from transformers import AutoTokenizer
except Exception:
    print("ENCODE_VERIFY: SKIP (transformers not available)")
    sys.exit(0)

tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "artifacts", "vybos-configurator-lora"))
out = subprocess.run(
    ["/home/rick/Projects/Vyb/build/vyb", "native/train/encode_corpus.vyb",
     "--module-path", "native/tokenizer", "--module-path", "native/json"],
    capture_output=True, text=True, cwd=ROOT)
if out.returncode != 0:
    print("ENCODE_VERIFY: FAIL (vyb exit", out.returncode, ")"); print(out.stderr); sys.exit(1)

# parse ASSISTANT id lists from the Vyb output (in record order)
vyb_ids = []
for line in out.stdout.splitlines():
    if line.startswith("ASST "):
        ids = [int(x) for x in line.split("=>", 1)[1].split("[", 1)[1].split("]", 1)[0].split()]
        vyb_ids.append(ids)

# reference from the same first-4 train records
ref_ids = []
with open(os.path.join(ROOT, "data/vybos-configurator-train.jsonl")) as fh:
    lines = [json.loads(l) for l in fh][:4]
for rec in lines:
    content = rec["messages"][-1]["content"]
    ref_ids.append(tok.encode(content, add_special_tokens=False))

n = min(len(vyb_ids), len(ref_ids))
ok = 0
for i in range(n):
    match = (vyb_ids[i] == ref_ids[i])
    # labels = ref shifted by one (predict next), last = ignore
    lbl = ref_ids[i][1:] + [-1]
    print(f"record {i}: tokens={len(ref_ids[i])} id-match={'OK' if match else 'X'}  label0={lbl[0]}")
    if match: ok += 1
    else:
        print("  vyb head:", vyb_ids[i][:15])
        print("  ref head:", ref_ids[i][:15])
print(f"ENCODE_CORPUS_VERIFY: {'OK' if ok == n else 'FAIL'}  ({ok}/{n} exact match, ids+lables derived)")
