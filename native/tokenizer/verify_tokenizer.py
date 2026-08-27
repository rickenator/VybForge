#!/usr/bin/env python3
# Reference verification for the Vyb-native Qwen3 tokenizer: runs test_tokenizer.vyb
# with build/vyb, parses each "<label>: [ids]" line, and compares to the
# transformers reference. Verification only (the tokenizer itself is all Vyb).
import subprocess, re, os, sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."))
try:
    from transformers import AutoTokenizer
except Exception:
    print("TOKENIZER_VERIFY: SKIP (transformers/.venv not available)")
    sys.exit(0)

tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "artifacts", "vybos-configurator-lora"))
out = subprocess.run(
    ["/home/rick/Projects/Vyb/build/vyb",
     "native/tokenizer/test_tokenizer.vyb", "--module-path", "native/tokenizer"],
    capture_output=True, text=True, cwd=ROOT)
n = ok = 0
for line in out.stdout.splitlines():
    if "[" not in line:
        continue
    label = line.split(": [", 1)[0]
    ids = [int(x) for x in re.findall(r"\d+", line.split("[", 1)[1].split("]", 1)[0])]
    ref = tok.encode(label, add_special_tokens=False)
    n += 1
    if ids == ref:
        ok += 1
    else:
        print("X ", repr(label), "gpu=", ids, "ref=", ref)
print(f"TOKENIZER_VERIFY: {'OK' if ok == n else 'FAIL'}  ({ok}/{n} exact match vs transformers)")
