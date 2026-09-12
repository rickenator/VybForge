#!/usr/bin/env python3
"""Gate for the real-model llm.chat closed loop (make -f native/Makefile chat-real).

Reads native/out/chat_real.log produced by model_driver.vyb (full 36-layer
Qwen3-4B forward + tied lm_head argmax, argmax ids re-decoded to text through
the stdlib/vllm CPU head). Asserts:
  - prefill top1 argmax ids == the committed gold [31784, 31784]
  - a non-empty LLM_CHAT_RESPONSE was decoded (the real model's continuation)
  - prefill_hidden still matches the numpy reference (maxrel gate)
"""
import os, re, numpy as np

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
log = os.path.join(repo, "native/out/chat_real.log")
ok = True

# 0. Step-1 embed probe: the GPU gather of ids [0,1] must reproduce the committed
#    layer0_input.bin exactly (proves the token-embd path, not a stale artifact).
ev = os.path.join(repo, "native/out/prefill_embedded_vyb.txt")
l0 = os.path.join(repo, "native/out/layer0_input.bin")
if os.path.exists(ev) and os.path.exists(l0):
    emb_gpu = np.loadtxt(ev)
    l0ref = np.fromfile(l0, dtype=np.float64)
    if emb_gpu.size == l0ref.size:
        d = np.abs(emb_gpu - l0ref).max()
        ok_match = d < 1e-5   # fp-level dequant-ordering noise; a wrong-row gather ~O(0.1)
        print(f"  embed(ids=[0,1]) vs layer0_input.bin: max abs diff = {d:.3e} (tol 1e-5)")
        print("EMBED_GOLD_MATCH:", "OK" if ok_match else "FAIL")
        if not ok_match:
            ok = False
    else:
        print(f"  EMBED size mismatch: gpu={emb_gpu.size} gold={l0ref.size}")
        ok = False


if not os.path.exists(log):
    print("MISSING chat_real.log — run the driver first")
    raise SystemExit(2)
txt = open(log).read()

# 1. top1 ids from the closed-loop decode must be the committed gold [31784,31784]
m = re.search(r"LLM_CHAT_TOP1_IDS=(\d+),(\d+)", txt)
if not m:
    print("FAIL: no LLM_CHAT_TOP1_IDS line in log")
    ok = False
else:
    a, b = int(m.group(1)), int(m.group(2))
    gold = (31784, 31784)
    match = (a, b) == gold
    print(f"  llm.chat top1 ids = ({a}, {b})  gold = {gold}  -> {'MATCH' if match else 'MISMATCH'}")
    print("LLM_CHAT_TOP1_MATCH:", "OK" if match else "FAIL")
    if not match:
        ok = False

# 2. a real decoded response must be present
mr = re.search(r"LLM_CHAT_RESPONSE=<(.*)>", txt)
if not mr:
    print("FAIL: no LLM_CHAT_RESPONSE line in log")
    ok = False
else:
    resp = mr.group(1)
    present = len(resp) > 0
    print(f"  response decoded: <{resp}> ({len(resp)} chars) -> {'PRESENT' if present else 'EMPTY'}")
    print("LLM_CHAT_RESPONSE:", "OK" if present else "FAIL")
    if not present:
        ok = False

# 3. prefill hidden still matches numpy (forward correctness preserved)
hv = os.path.join(repo, "native/out/prefill_hidden_vyb.txt")
hr = os.path.join(repo, "native/out/prefill_hidden_ref.txt")
if os.path.exists(hv) and os.path.exists(hr):
    vyb = np.loadtxt(hv); ref = np.loadtxt(hr)
    rel = np.abs(vyb - ref) / np.maximum(np.abs(ref), 1e-6)
    bad = int((rel > 2e-3).sum())
    print(f"  prefill hidden preserved: bad(rel>2e-3) = {bad}  maxrel = {rel.max():.3e}")
    print("PREFILL_HIDDEN_MATCH:", "OK" if bad == 0 else "FAIL")
    if bad:
        ok = False

print("\nCHAT-REAL:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
