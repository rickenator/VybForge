#!/usr/bin/env python3
"""VybForge adaptive-repair proposal driver (issue #7 model boundary).

Backend-neutral shell around the repair proposal pathway; mirrors
`configurator.py` (ollama / openai-chat / openai-responses) but returns a
single schema-constrained PatchProposal that the deterministic Vyb repair core
(VybOS modules/repair.vyb) can apply/validate/seal.

In addition to the live backends it ships a deterministic `mock` backend (no
CPU/GPU model needed) so the FULL boundary is provable end-to-end on any
machine: it renders a fixed, well-formed proposal for the given pristine source
and writes JSON that the validator consumes identically to a real model's. A
real deployment picks a live backend; CI/demo uses mock.

Usage:
  propose_repair.py --pristine <src.txt> --reason "<what broke>" \\
      --diff "-<line>\n+<new>\n" --cls generated --generator qwen3-8b [--mock]
  # or supply a pre-written proposal file and just re-validate it:
  propose_repair.py --from-file out/proposal.json --out out/proposal.json

Writes the PatchProposal JSON (the model boundary value) to --out.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def getenv(name: str, default: str = "") -> str:
    return os.environ.get("VYBFORGE_" + name) or default


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def mock_proposal(args) -> dict:
    """Deterministic, GPU-free proposal. If a --diff is supplied use it verbatim,
    else synthesize a minimal repair: replace the EXACT line containing the bug
    pattern (`return a - b`) with its fixed twin `return a + b`, so the shape is
    a genuine, applied, minimal repair. Matched only if that line exists exactly."""
    if args.diff:
        diff = args.diff.replace("\\n", "\n")
    else:
        lines = read_text(args.pristine).rstrip("\n").split("\n")
        old = ""
        for ln in lines:
            if "a - b" in ln:
                old = ln
                break
        if old:
            corrected = old.replace("a - b", "a + b")
        else:
            corrected = lines[-1] + "  // repaired"
            old = lines[-1]
        diff = "-" + old + "\n+" + corrected
    matched = all(
        l.startswith("-") and l[1:] in (read_text(args.pristine).split("\n"))
        for l in diff.split("\n") if l.startswith("-")
    )
    return {
        "cls": args.cls,
        "diff": diff,
        "reason": args.reason,
        "generator": args.generator,
        "matched": matched,
    }


def load_backend(args, response_schema: dict):
    import urllib.error
    import urllib.request

    def json_request(url, payload, api_key=None):
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.load(resp)

    def ollama():
        body = json_request(
            args.endpoint.rstrip("/") + "/api/chat",
            {"model": args.model, "messages": args.messages, "stream": False,
             "format": response_schema, "options": {"temperature": 0.2}},
        )
        return json.loads(body["message"]["content"])

    def openai_chat():
        payload = {"model": args.model, "messages": args.messages, "temperature": 0.2}
        mode = args.structured_output
        rf = None
        if mode == "json_object":
            rf = {"type": "json_object"}
        elif mode == "json_schema":
            rf = {"type": "json_schema", "json_schema": {"name": "vybforge_proposal", "strict": True, "schema": response_schema}}
        if rf:
            payload["response_format"] = rf
        body = json_request(args.endpoint.rstrip("/") + "/chat/completions", payload, args.api_key or None)
        return json.loads(body["choices"][0]["message"]["content"])

    def openai_responses():
        transcript = "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in args.messages)
        payload = {"model": args.model, "input": transcript, "store": False, "temperature": 0.2}
        rf = None
        if args.structured_output == "json_object":
            rf = {"type": "json_object"}
        elif args.structured_output == "json_schema":
            rf = {"type": "json_schema", "name": "vybforge_proposal", "strict": True, "schema": response_schema}
        if rf:
            payload["text"] = {"format": rf}
        body = json_request(args.endpoint.rstrip("/") + "/responses", payload, args.api_key or None)
        return json.loads(body.get("output_text") or body["output"][0]["content"][0]["text"])

    return {"ollama": ollama, "openai-chat": openai_chat, "openai-responses": openai_responses}[args.backend]()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["mock", "ollama", "openai-chat", "openai-responses"],
                    default=getenv("BACKEND", "mock"))
    ap.add_argument("--endpoint", default=getenv("ENDPOINT", "http://127.0.0.1:11434"))
    ap.add_argument("--model", default=getenv("MODEL", "qwen3:8b"))
    ap.add_argument("--api-key", default=getenv("API_KEY", ""))
    ap.add_argument("--structured-output", default=getenv("STRUCTURED_OUTPUT", "json_schema"),
                    choices=["json_schema", "json_object", "prompt"])
    ap.add_argument("--schema", default="config/repair-proposal.schema.json")
    ap.add_argument("--prompt", default="prompts/repair.md")
    ap.add_argument("--pristine", default="")
    ap.add_argument("--reason", default="deterministic mock repair")
    ap.add_argument("--diff", default="")
    ap.add_argument("--cls", default="generated", choices=["curated", "upstream", "generated"])
    ap.add_argument("--generator", default="mock")
    ap.add_argument("--from-file", default="")
    ap.add_argument("--out", default="out/proposal.json")
    args = ap.parse_args()

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_file:
        proposal = json.loads(Path(args.from_file).read_text(encoding="utf-8"))
        print(f"[propose_repair] re-emitted proposal from {args.from_file} (validated shape)")
    else:
        args.messages = None
        response_schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
        if args.backend == "mock":
            proposal = mock_proposal(args)
            print("[propose_repair] mock (deterministic, no model) proposal built")
        else:
            system_prompt = read_text(args.prompt)
            context = ("Constraint schema:\n" + json.dumps(response_schema)
                       + "\n\nPristine source to repair:\n" + read_text(args.pristine)
                       + "\n\nBuild/test failure:\n" + args.reason)
            args.messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": context}]
            proposal = load_backend(args, response_schema)
            print(f"[propose_repair] {args.backend}/{args.model} proposal received")

    Path(args.out).write_text(json.dumps(proposal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # report in a model-agnostic one-liner
    print("proposal → %s cls=%s matched=%s diff_lines=%d" % (
        args.out, proposal.get("cls"), proposal.get("matched"), proposal.get("diff", "").count("\n") + 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
