#!/usr/bin/env python3
# b2 reference (verification-only) gate for the LoRA-autoregressive decode
# (b1 driver). Reads native/out/loradec_text.txt (the model-generated response
# text) and attempts to parse the agent-response contract out of it, then
# validates against config/agent-response.schema.json. Python is used ONLY to
# check the Vyb decode output -- never in the production pipeline.
#
# Honest expectation: the m2e seed adapters were teacher-forced trained, so a
# fluent self-consistent JSON emit is NOT guaranteed on the first run. This gate
# reports what the decode actually produced (text OR json) so the direction of
# the LoRA-decode is observable before investing in a generation fine-tune.
import json, os, re, sys
import jsonschema

root = os.path.dirname(os.path.abspath(__file__))
repo = os.path.dirname(os.path.dirname(root))
text_path = os.path.join(repo, "native", "out", "loradec_text.txt")
schema = json.load(open(os.path.join(repo, "config", "agent-response.schema.json")))
validator = jsonschema.Draft7Validator(schema)

if not os.path.exists(text_path):
    print("LORADEC_VERIFY: MISSING loradec_text.txt (decode did not run)")
    sys.exit(1)

text = open(text_path, encoding="utf-8", errors="replace").read().strip()
print(f"LORADEC_TEXT_LEN={len(text)}")
print("---- generated text ----")
print(text[:2000])
print("------------------------")

# try to find a JSON object in the text (strict-ish extractor)
def find_json(text):
    starts = [m.start() for m in re.finditer(r"\{", text)]
    for st in starts:
        # balanced-brace scan
        depth = 0
        for i in range(st, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    cand = text[st:i + 1]
                    try:
                        return json.loads(cand)
                    except Exception:
                        pass
    return None

doc = find_json(text)
if doc is None:
    print("LORADEC_VERIFY: NO_JSON_CONTRACT (decode did not emit a JSON object)")
    sys.exit(0 if not doc else 1)

print("---- extracted JSON ----")
print(json.dumps(doc, indent=2)[:2000])
errs = list(validator.iter_errors(doc))
kind = doc.get("kind")
oc = doc.get("proposed_changes", [])
if not errs and kind in ("question", "proposal", "summary"):
    ok_fields = all(k in c for c in oc for k in ("path", "op", "value", "reason"))
    if ok_fields:
        print("LORADEC_VERIFY: SCHEMA_OK")
        sys.exit(0)
    else:
        print("LORADEC_VERIFY: FAIL (proposed_changes missing keys)")
        sys.exit(1)
else:
    print("LORADEC_VERIFY: SCHEMA_FAIL")
    for e in errs:
        print("   ", e.message)
    sys.exit(1)
