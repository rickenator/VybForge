#!/usr/bin/env python3
# Reference (verification-only) harness for G-decode's Vyb-emitted agent-response
# contract. Validates each native/out/contract_<kind>.json against
# config/agent-response.schema.json and reports per-kind pass/fail. Python is
# used ONLY to check the Vyb output — never in the production pipeline.
import json, glob, os, sys
import jsonschema

root = os.path.dirname(os.path.abspath(__file__))
repo = os.path.dirname(os.path.dirname(root))   # native/.. -> repo root
schema = json.load(open(os.path.join(repo, "config", "agent-response.schema.json")))
validator = jsonschema.Draft7Validator(schema)

files = {
    "question": ("native/out/contract_question.json", "question"),
    "summary":  ("native/out/contract_summary.json", "summary"),
    "proposal": ("native/out/contract_proposal.json", "proposal"),
    "pipeline": ("native/out/contract_pipeline.json", "summary"),
}
allok = True
for label, (rel, expkind) in files.items():
    path = os.path.join(repo, rel)
    if not os.path.exists(path):
        print(f"CONTRACT_VERIFY: {label} MISSING ({rel})")
        allok = False
        continue
    doc = json.load(open(path))
    errs = list(validator.iter_errors(doc))
    ok = (doc.get("kind") == expkind) and not errs
    if errs:
        print(f"CONTRACT_VERIFY: {label} FAIL")
        for e in errs:
            print("   ", e.message)
    elif doc.get("kind") != expkind:
        print(f"CONTRACT_VERIFY: {label} FAIL (kind mismatch: {doc.get('kind')})")
    else:
        print(f"CONTRACT_VERIFY: {label} OK")
    allok = allok and ok

print("CONTRACT_VERIFY:", "ALL_OK" if allok else "FAIL")
sys.exit(0 if allok else 1)
