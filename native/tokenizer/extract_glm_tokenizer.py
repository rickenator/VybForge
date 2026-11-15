#!/usr/bin/env python3
# Extract GLM-5.3's BPE artifacts from the checkpoint's tokenizer.json into the
# flat vocab.json + merges.txt form the Vyb-native tokenizer reads
# (native/tokenizer/tokenizer.vyb). GLM-5.3 uses the same GPT-2 byte-level
# scheme as Qwen (same bytes_to_unicode mapping); differences handled by the
# tokenizer itself:
#   - pre-tokenizer regex uses \p{N}{1,3} (Qwen uses \p{N})  -> VYBFORGE_DIGMAX=3
#   - model.ignore_merges = true (Qwen: false)               -> VYBFORGE_TOKIGNORE=1
import json, os, sys

src = sys.argv[1] if len(sys.argv) > 1 else "/mnt/apex/rick/vybforge/glm53-checkpoint"
dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "artifacts", "glm53-tokenizer")
os.makedirs(dst, exist_ok=True)

t = json.load(open(os.path.join(src, "tokenizer.json")))
m = t["model"]
assert m["type"] == "BPE"
with open(os.path.join(dst, "vocab.json"), "w", encoding="utf-8") as f:
    f.write(json.dumps(m["vocab"], ensure_ascii=False, separators=(",", ":")))
with open(os.path.join(dst, "merges.txt"), "w", encoding="utf-8") as f:
    f.write("#version: 0.2\n")
    for a, b in m["merges"]:
        f.write(f"{a} {b}\n")
with open(os.path.join(dst, "added_tokens.json"), "w", encoding="utf-8") as f:
    json.dump({a["content"]: a["id"] for a in t["added_tokens"]}, f,
              ensure_ascii=False, indent=0)
print(f"GLM_TOKENIZER_EXTRACT: vocab={len(m['vocab'])} merges={len(m['merges'])} "
      f"added={len(t['added_tokens'])} -> {dst}")
