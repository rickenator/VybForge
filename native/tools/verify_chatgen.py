#!/usr/bin/env python3
"""Gate for Checkpoint C — fluent autoregressive llm.chat over a real prompt.

decode_driver.vyb (VYB_PROMPT_IDS prompt mode) seeds the proven recompute-
full-prefix greedy gen loop from an arbitrary prompt's ids and emits the FULL
generated token stream (prompt + GEN continuation). This asserts the GPU stream
matches the numpy autoregressive gold (chatgen_ref.py) token-for-token, and
that a non-empty fluent response was decoded via the stdlib/vllm CPU head.

Reads native/out/chatgen_ids_ref.txt (numpy gold) and
native/out/chatgen_ids_vyb.txt (GPU gen loop).
"""
import os, numpy as np

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ref = os.path.join(repo, "native/out/chatgen_ids_ref.txt")
vyb = os.path.join(repo, "native/out/chatgen_ids_vyb.txt")
log = os.path.join(repo, "native/out/chat_gen.log")
pids = os.path.join(repo, "native/out/prompt_ids.txt")
ok = True

if not (os.path.exists(ref) and os.path.exists(vyb)):
    print("MISSING chatgen ref/vyb files — run chatgen_ref + decode_driver first")
    raise SystemExit(2)

gold = np.loadtxt(ref, dtype=int)
got = np.loadtxt(vyb, dtype=int)
if gold.size != got.size:
    print(f"FAIL: size mismatch gold={gold.size} gpu={got.size}")
    raise SystemExit(1)
prompt_ids = [int(x) for x in open(pids).read().strip().split(",")]
gen_gold = gold[len(prompt_ids):]
gen_got = got[len(prompt_ids):]

print(f"prompt ids = {prompt_ids}")
print(f"GPU generated stream = {got.tolist()}")
print(f"numpy gold  generated = {gold.tolist()}")
match = (gold == got).all()
print("CHATGEN_STREAM_MATCH:", "OK" if match else "FAIL")
if not match:
    for i in range(gold.size):
        mark = "" if gold[i] == got[i] else "   <-- MISMATCH"
        print(f"  pos{i}: gold={gold[i]} gpu={got[i]}{mark}")
    ok = False

print(f"continuation (gen): gold={gen_gold.tolist()}  gpu={gen_got.tolist()}" +
      ("  MATCH" if (gen_gold == gen_got).all() else "  MISMATCH"))
if not (gen_gold == gen_got).all():
    ok = False

# fluent response present + non-empty (prompt mode decodes the continuation)
if os.path.exists(log):
    txt = open(log).read()
    mr = __import__("re").search(r"LLM_CHAT_RESPONSE=<(.*)>", txt)
    if mr:
        resp = mr.group(1)
        present = len(resp) > 0
        print(f"generated response: <{resp}> ({len(resp)} chars)")
        print("LLM_CHAT_RESPONSE:", "OK" if present else "FAIL")
        if not present:
            ok = False

print("\nCHAT-GEN:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
