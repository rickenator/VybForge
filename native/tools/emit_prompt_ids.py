#!/usr/bin/env python3
"""Parse the llm_encode_probe output (PROMPT_IDS line) into native/out/prompt_ids.txt.

Used by `make chat-prompt-ref` so the numpy gold embeds the SAME deterministic
ids the GPU driver embeds. Reads native/out/chat_prompt_encode.log.
"""
import os

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
log = os.path.join(repo, "native/out/chat_prompt_encode.log")
for line in open(log):
    if line.startswith("PROMPT_IDS="):
        ids = line.strip().split("=", 1)[1]
        with open(os.path.join(repo, "native/out/prompt_ids.txt"), "w") as f:
            f.write(ids + "\n")
        print(f"wrote prompt_ids.txt: {ids} (N={len(ids.split(','))})")
        raise SystemExit(0)
print("FAIL: no PROMPT_IDS line in", log)
raise SystemExit(1)
