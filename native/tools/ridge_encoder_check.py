#!/usr/bin/env python3
"""Check OUR tokenizer against the captured Ridge oracle (VybForge#10 phase 4, W2/W3 prerequisite).

Why this exists, and why it runs BEFORE any forward gate. A whole-model comparison is only as good
as the token stream it starts from: if our encoder emits different ids than llama.cpp for the same
text, every hidden and every top1 downstream is comparing a different input, and the failure looks
like a model bug for as long as it takes to notice. So the oracle fixtures (captured by
native/tools/ridge_oracle_capture.py) carry the oracle's OWN tokenizer output, and this script
requires ours to reproduce it exactly.

It also exists because of a silent upstream trap found while setting this up. `build_vocab_from` in
stdlib/vllm (the CPU tokenizer) walks vocab.json with a hand-rolled reader whose `read_int` stops at
the first non-digit; a PRETTY-PRINTED vocab.json (a space after the `:`) therefore parses every key
to id 0 — no error, no short read, and the TOKEN COUNT still comes out right (measured: 5 ids for
"The capital of France is", all 0). The official Qwen/Qwen3.8-27B repo ships vocab.json
pretty-printed, so a fresh download hits it immediately. This script compacts the JSON before use and
asserts a known token maps to a non-zero id, so the pathology fails loudly here instead of looking
like a wrong model.

Run from the repo root:
    env -u PYTHONPATH .venv/bin/python native/tools/ridge_encoder_check.py

Env: VYBFORGE_RIDGE_TOK_DIR (default artifacts/ridge-tokenizer), VYBFORGE_RIDGE_GGUF,
     VYB / VYBHOME (the toolchain), VYBFORGE_ORACLE_OUT (fixtures dir).
Exit codes: 0 pass (or skip), 1 fail. SKIPs when the toolchain or the tokenizer dir is absent — a
fresh clone has neither, and a gate that proved nothing must not read as PASS.
"""
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VYBHOME = os.environ.get("VYBHOME", os.path.expanduser("~/Projects/Vyb"))
VYB = os.environ.get("VYB", os.path.join(VYBHOME, "build", "vyb"))
STDLIB = os.environ.get("VYB_STDLIB", os.path.join(VYBHOME, "stdlib"))
TOKDIR = os.environ.get("VYBFORGE_RIDGE_TOK_DIR", os.path.join(ROOT, "artifacts/ridge-tokenizer"))
FIXDIR = os.environ.get("VYBFORGE_ORACLE_OUT", os.path.join(ROOT, "native/legit/fixtures/llama_ridge"))

# A known token and the id it must have: "," is 11 in the Qwen byte-level vocab (the first 256 ids are
# the bytes in GPT-2's bytes_to_unicode order). This is the assertion that catches the silent-zero
# parse — a map that maps everything to 0 also maps "," to 11-wrong.
PROBE_TOKEN, PROBE_ID = ",", 11


def compact_vocab(path):
    """Return (changed, note). Compacts a pretty-printed vocab.json in place.

    The reader in stdlib/vllm only skips forward to the next key on a parse it cannot use, so a
    whitespace-formatted file yields a map of all-zero ids rather than an error (see the docstring).
    Compacting is idempotent: a file already written compact is left untouched.
    """
    raw = open(path, "rb").read()
    if b": " not in raw and b"\n" not in raw:
        return False, "already compact"
    d = json.loads(raw.decode("utf-8"))
    if PROBE_TOKEN not in d:
        return False, "compact-check withheld: %r not in the map" % PROBE_TOKEN
    with open(path, "w") as fh:
        json.dump(d, fh, separators=(",", ":"), ensure_ascii=False)
    return True, "compacted to %d bytes, %d entries" % (os.path.getsize(path), len(d))


def field(path, key):
    for line in open(path):
        if line.startswith(key + " "):
            return line[len(key) + 1:].rstrip("\n")
    return None


def main():
    print("RIDGE_ENCODER_CHECK tokdir=%s" % TOKDIR)
    if not os.path.isfile(os.path.join(TOKDIR, "vocab.json")) or \
       not os.path.isfile(os.path.join(TOKDIR, "merges.txt")):
        # One short reason on the prefixed line (the gate's step line shows only that), then the fix.
        print("RIDGE_ENCODER_CHECK_SKIP no vocab.json/merges.txt in %s" % TOKDIR)
        print("RIDGE_ENCODER_CHECK fetch them with: hf download Qwen/Qwen3.8-27B vocab.json "
              "merges.txt --local-dir %s" % TOKDIR)
        return 0
    if not os.access(VYB, os.X_OK):
        print("RIDGE_ENCODER_CHECK_SKIP no Vyb toolchain at %s" % VYB)
        return 0
    fixtures = sorted(glob.glob(os.path.join(FIXDIR, "*.fix")))
    if not fixtures:
        print("RIDGE_ENCODER_CHECK_SKIP no fixtures in %s (run native/tools/ridge_oracle_capture.py)"
              % FIXDIR)
        return 0

    changed, note = compact_vocab(os.path.join(TOKDIR, "vocab.json"))
    print("RIDGE_ENCODER_CHECK vocab.json %s (%s)" % ("rewritten" if changed else "checked", note))
    # The guard: a vocab map that lost its ids answers 0 for everything, including the byte tokens.
    d = json.load(open(os.path.join(TOKDIR, "vocab.json")))
    if d.get(PROBE_TOKEN) != PROBE_ID:
        print("RIDGE_ENCODER_CHECK_FAIL the vocab map maps %r to %r, want %d — the tokenizer dir is "
              "not usable (is vocab.json pretty-printed?)" % (PROBE_TOKEN, d.get(PROBE_TOKEN), PROBE_ID))
        return 1
    print("RIDGE_ENCODER_CHECK vocab map sanity: %r -> %d, %d entries" % (PROBE_TOKEN, PROBE_ID, len(d)))

    bad = []
    for fx in fixtures:
        text = field(fx, "prompt_text")
        want = [int(x) for x in (field(fx, "prompt_ids") or "").split()]
        if text is None or not want:
            print("RIDGE_ENCODER_CHECK %s SKIP (no prompt_text/prompt_ids)" % os.path.basename(fx))
            continue
        env = dict(os.environ, VYB_STDLIB=STDLIB, VYB_LLM_DIR=TOKDIR, VYB_PROMPT=text)
        r = subprocess.run([VYB, "native/llm/llm_encode_probe.vyb", "--module-path", "native/llm"],
                           cwd=ROOT, capture_output=True, text=True, env=env)
        out = r.stdout + r.stderr
        got = None
        for line in out.splitlines():
            if line.startswith("PROMPT_IDS="):
                got = [int(x) for x in line.split("=", 1)[1].split(",") if x.strip() != ""]
        name = os.path.basename(fx)
        if got is None:
            print("RIDGE_ENCODER_CHECK %s FAIL (the probe printed no PROMPT_IDS): %s"
                  % (name, out[-300:]))
            bad.append(name)
            continue
        if got and all(v == 0 for v in got):
            print("RIDGE_ENCODER_CHECK %s FAIL all %d ids are 0 — the vocab map is empty to the "
                  "tokenizer (the pretty-printed vocab.json pathology)" % (name, len(got)))
            bad.append(name)
            continue
        if got != want:
            print("RIDGE_ENCODER_CHECK %s FAIL ours=%s oracle=%s" % (name, got, want))
            bad.append(name)
        else:
            print("RIDGE_ENCODER_CHECK %s OK <%s> %d ids == the oracle's" % (name, text, len(got)))
    if bad:
        print("RIDGE_ENCODER_CHECK_FAIL %d fixture(s) disagree with the oracle's tokenizer: %s"
              % (len(bad), ", ".join(bad)))
        return 1
    print("RIDGE_ENCODER_CHECK_DONE our encoder reproduces llama.cpp's ids on every fixture "
          "(%d prompts) — the prompt stream a forward gate starts from is the oracle's" % len(fixtures))
    return 0


if __name__ == "__main__":
    sys.exit(main())
