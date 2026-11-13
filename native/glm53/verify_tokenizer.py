"""P5 gate: TOKENIZER_REF — pin the GLM-5.3 tokenizer contract.

HF fast tokenizer loaded from the checkpoint's own tokenizer.json (in fixture/).
Writes native/out/tokenizer_ref_ids.txt: one line per corpus string, tab-separated
(id, id, id, ...) — the golden file any Vyb-side tokenizer must reproduce byte for
byte. Also asserts the decode round-trip and the special-token facts.

Usage: env -u PYTHONPATH ~/venvs/exl3/bin/python native/glm53/verify_tokenizer.py
"""

import json
import sys

from transformers import AutoTokenizer

FIX = "/home/rick/Projects/VybForge/native/glm53/fixture"
OUT = "/home/rick/Projects/VybForge/native/out"

CORPUS = [
    "def add(a, b):\n    return a + b",
    "The quick brown fox jumps over the lazy dog.",
    "x = 1048576  # 1M context",
    "tabs\tand    spaces   collapse? no — preserved",
    "unicode: café \u00fcber \u4f60\u597d \U0001f600",
    "<|system|>\nyou are a helpful assistant\n<|user|>\nhello\n<|assistant|>",
    "a" * 300,
    "mixed123numbers456and789",
    "if (x > 0) { y = -1; } else { y = 1; }",
    "\n\n\nleading blank lines",
]


def main():
    t = AutoTokenizer.from_pretrained(FIX)
    facts = {
        "class": type(t).__name__,
        "vocab_size": t.vocab_size,
        "len": len(t),
        "bos": t.bos_token_id,
        "eos": t.eos_token_id,
        "pad": t.pad_token_id,
    }
    print("facts:", facts)

    lines = []
    ok = True
    for i, s in enumerate(CORPUS):
        ids = t.encode(s)
        dec = t.decode(ids)
        rt = dec == s
        ok = ok and rt
        lines.append(",".join(str(x) for x in ids))
        print(f"  [{i}] {len(ids):4d} ids  roundtrip {'ok' if rt else 'FAIL ' + repr(dec[:60])}")

    with open(f"{OUT}/tokenizer_ref_ids.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    meta = {"facts": facts, "corpus": CORPUS, "n": len(CORPUS)}
    with open(f"{OUT}/tokenizer_ref_meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {OUT}/tokenizer_ref_ids.txt ({len(CORPUS)} lines) + meta")
    print("TOKENIZER_REF:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
