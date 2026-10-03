#!/usr/bin/env python3
"""Parse the llm_encode_probe output (PROMPT_IDS line) into native/out/prompt_ids.txt.

Used by the chat-prompt / chat-real targets so the numpy gold embeds the SAME
deterministic ids the GPU driver embeds. Reads native/out/chat_prompt_encode.log.

Two guards, both earned:
  - the log must carry a PROMPT=<...> line, and when the expected prompt is passed
    on the command line it must MATCH it. The shared log file is overwritten by any
    run with the same name, and a stale one silently poisons the gold: the checked-in
    prompt_ids.txt held newline-separated ids for "Add a package to a QEMU x86_64
    boot test." while CHAT_PROMPT was "The capital of France is".
  - separators are accepted as commas or whitespace, because writers disagree, but
    the file is always rewritten in the canonical comma-separated single-line form.

Usage: emit_prompt_ids.py [expected-prompt]
"""
import os, sys

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
log = os.path.join(repo, "native/out/chat_prompt_encode.log")
expected = sys.argv[1] if len(sys.argv) > 1 else None

if not os.path.exists(log):
    print(f"FAIL: {log} missing — run the encode probe first")
    raise SystemExit(1)

seen_prompt, ids_line = None, None
for line in open(log):
    if line.startswith("PROMPT_IDS="):
        ids_line = line.strip().split("=", 1)[1]
    elif line.startswith("PROMPT=<"):
        seen_prompt = line.strip()[len("PROMPT=<"):-1]

if ids_line is None:
    print(f"FAIL: no PROMPT_IDS line in {log}")
    print("      (an empty log usually means the driver died before printing it)")
    raise SystemExit(1)

if seen_prompt is None:
    print("FAIL: log has no PROMPT=<...> line — refusing to trust ids of unknown provenance")
    raise SystemExit(1)

if expected is not None and seen_prompt != expected:
    print(f"FAIL: log was produced for PROMPT=<{seen_prompt}>, expected <{expected}>")
    print("      the shared encode log was clobbered by another run; re-run the target")
    raise SystemExit(1)

ids = [t for t in ids_line.replace(",", " ").split() if t]
if not ids or not all(t.lstrip("-").isdigit() for t in ids):
    print(f"FAIL: unparsable ids: {ids_line!r}")
    raise SystemExit(1)

with open(os.path.join(repo, "native/out/prompt_ids.txt"), "w") as f:
    f.write(",".join(ids) + "\n")
print(f"wrote prompt_ids.txt: prompt=<{seen_prompt}> N={len(ids)} ids={','.join(ids)}")
