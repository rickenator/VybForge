#!/usr/bin/env python3
"""Gate for Checkpoint B — arbitrary-prompt llm.chat (make chat-prompt).

Runs the full 36-layer forward over a real prompt (ids from the stdlib/vllm
tokenizer, embedded via the GPU token-embd gather), then compares the GPU's
per-position lm_head argmax ids against the numpy reference (prompt_ref.py)
token-for-token. Proves the closed loop over an arbitrary prompt, not just the
fixed [0,1] prefix.

Reads native/out/prompt_top1_ref.txt (numpy gold) and
native/out/prefill_top1_vyb.txt (overwritten by the prompt run's argmax).
"""
import os, numpy as np

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ref = os.path.join(repo, "native/out/prompt_top1_ref.txt")
vyb = os.path.join(repo, "native/out/prefill_top1_vyb.txt")
pids = os.path.join(repo, "native/out/prompt_ids.txt")
ok = True

if not (os.path.exists(ref) and os.path.exists(vyb)):
    print("MISSING prompt_top1 ref/vyb files — run prompt_ref + model_driver first")
    raise SystemExit(2)

gold = np.loadtxt(ref, dtype=int)
got = np.loadtxt(vyb, dtype=int)
if gold.size != got.size:
    print(f"FAIL: size mismatch gold={gold.size} gpu={got.size}")
    raise SystemExit(1)
prompt_ids = [int(x) for x in open(pids).read().strip().split(",")]
print(f"prompt ids = {prompt_ids}  S = {gold.size}")
match = (gold == got).all()
print(f"GPU per-position top1 = {got.tolist()}")
print(f"numpy gold   top1     = {gold.tolist()}")
print("PROMPT_TOP1_MATCH:", "OK" if match else "FAIL")
if not match:
    for i in range(gold.size):
        mark = "" if gold[i] == got[i] else "   <-- MISMATCH"
        print(f"  tok{i}: gold={gold[i]} gpu={got[i]}{mark}")
    ok = False

print("\nCHAT-PROMPT:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
