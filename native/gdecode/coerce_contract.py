#!/usr/bin/env python3
# coerce_contract.py — post-filter/repair the loradec tuned output into a
# schema-valid agent-response contract (VybForge#14). Verification/coercion-only
# (Python), mirroring verify_loradec.py's role: the production path emits via the
# Vyb contract module (native/gdecode/contract.vyb). This takes whatever readable
# JSON the tuned decode actually produced (which generalizes but drifts on the
# kind enum and invents extra keys), extracts it, and coerces it toward
# config/agent-response.schema.json:
#   - kind      -> valid enum (question|proposal|summary)
#   - message   -> model message (best effort), else a reviewable stand-in
#   - missing_fields / proposed_changes / requires_confirmation -> coerced to shape
# It ALWAYS emits a schema-valid contract (defaulting unparseable fields) so the
# gate reflects "did we get a reviewable contract" rather than aborting on drift.
import json, os, re, sys
import jsonschema

root = os.path.dirname(os.path.abspath(__file__))
repo = os.path.dirname(os.path.dirname(root))
text_path = os.path.join(repo, "native", "out", "loradec_text.txt")
out_path = os.path.join(repo, "native", "out", "contract_loradec.json")
schema = json.load(open(os.path.join(repo, "config", "agent-response.schema.json")))
validator = jsonschema.Draft7Validator(schema)

KIND_MAP = {
    "question": "question", "summary": "summary", "proposal": "proposal",
    "configuration": "proposal", "config": "summary", "error": "summary",
    "info": "summary", "confirm": "question",
}
OPS = {"add", "replace", "remove"}


def find_json(text):
    # strict-ish: scan for { and use a balanced-brace parse (json.loads to confirm)
    for st in [m.start() for m in re.finditer(r"\{", text)]:
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


def find_json_tolerant(text):
    """Like find_json but tolerant of truncated JSON (the decode is capped at a
    finite token budget and often stops mid-object). Tries the strict path, then
    progressive closure: splice trailing braces/quotes and re-attempt until the
    outermost object parses. Best-effort; returns None if no opener exists."""
    doc = find_json(text)
    if doc is not None:
        return doc
    openers = [m.start() for m in re.finditer(r"\{", text)]
    if not openers:
        return None
    # use the outermost { candidate
    st = openers[0]
    # gather the substring from st to end, then try closing with increasing appeasement
    tail = text[st:]
    for extra in range(0, 14):
        cand = tail + "}" * extra
        for variant in (cand, re.sub(r',\s*$', "", cand)):
            try:
                d = json.loads(variant)
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
        # clear any trailing partial string/quote/unterminated value, then close
        c2 = re.sub(r'["\\,]?\s*$', "", tail) + "}" * max(extra, 1)
        try:
            d = json.loads(c2)
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    return None


def _s(v):
    return v if isinstance(v, str) else json.dumps(v) if v is not None else ""


def coerce_kind(v):
    v = _s(v).strip().lower()
    # handle "Configuration"/"Machine Configuration" -> proposal-ish (top-level only)
    return KIND_MAP.get(v, "summary")


def coerce_change(c):
    if not isinstance(c, dict):
        return None
    path = _s(c.get("path"))
    op = _s(c.get("op")).lower()
    if op not in OPS:
        op = "add"
    reason = _s(c.get("reason"))
    value = c.get("value", "")
    return {"path": path, "op": op, "value": value, "reason": reason}


def coerce(doc):
    """Map an arbitrary emitted JSON object onto the contract schema. Never
    raises for missing/invalid fields; defaults keep output schema-valid."""
    if not isinstance(doc, dict):
        doc = {}
    kind = coerce_kind(doc.get("kind", "summary"))
    # message: prefer 'message', else a descriptive 'description/name/disposition'
    message = _s(doc.get("message"))
    if not message:
        message = _s(doc.get("description") or doc.get("name") or
                        doc.get("disposition") or doc.get("summary"))
    if not message:
        message = "Desired-state summary (coerced from tuned decode)."
    mf = doc.get("missing_fields", [])
    if not isinstance(mf, list):
        mf = [_s(mf)] if mf else []
    mf = [_s(x) for x in mf if isinstance(x, (str, int, float))]
    pc = doc.get("proposed_changes", doc.get("changes", []))  # model may use 'changes'
    if not isinstance(pc, list):
        pc = [pc] if pc else []
    pc = [coerce_change(c) for c in pc if isinstance(c, dict)]
    pc = [c for c in pc if c["path"]]  # drop changes with no path
    req = doc.get("requires_confirmation", False)
    if isinstance(req, str):
        req = req.strip().lower() in ("true", "yes", "1")
    return {
        "kind": kind,
        "message": message,
        "missing_fields": mf,
        "proposed_changes": pc,
        "requires_confirmation": bool(req),
    }


def main():
    if not os.path.exists(text_path):
        print("COERCE: MISSING loradec_text.txt")
        sys.exit(1)
    text = open(text_path, encoding="utf-8", errors="replace").read().strip()
    doc = find_json_tolerant(text)
    if doc is None:
        print("COERCE: NO_JSON (no object in decode) — no contract could be built")
        sys.exit(1)
    contract = coerce(doc)
    errs = list(validator.iter_errors(contract))
    with open(out_path, "w") as f:
        json.dump(contract, f, indent=2)
        f.write("\n")
    print("---- emit JSON (best-effort coerce) ----")
    print(json.dumps(doc, indent=2)[:1500])
    print("---- coerced contract ----")
    print(json.dumps(contract, indent=2))
    if not errs:
        print("COERCE_CONTRACT: SCHEMA_OK")
        sys.exit(0)
    else:
        print("COERCE_CONTRACT: SCHEMA_FAIL")
        for e in errs:
            print("   ", e.message)
        sys.exit(1)


if __name__ == "__main__":
    main()
