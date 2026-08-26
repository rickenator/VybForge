#!/usr/bin/env python3
"""Schema-constrained, non-mutating VybOS configuration interview client."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def read_json(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def request_ollama(endpoint: str, model: str, messages: list[dict], response_schema: dict) -> dict:
    payload = json.dumps({"model": model, "messages": messages, "stream": False, "format": response_schema, "options": {"temperature": 0.2}}).encode()
    request = urllib.request.Request(endpoint.rstrip("/") + "/api/chat", data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        body = json.load(response)
    return json.loads(body["message"]["content"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["ollama"], default="ollama")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--schema", required=True)
    parser.add_argument("--response-schema", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt", default="prompts/system.md")
    args = parser.parse_args()

    config_schema = read_json(args.schema)
    response_schema = read_json(args.response_schema)
    config = read_json(args.config)
    system_prompt = Path(args.prompt).read_text(encoding="utf-8")
    context = "Desired-state schema:\n" + json.dumps(config_schema) + "\n\nCurrent mock configuration:\n" + json.dumps(config)
    messages = [{"role": "system", "content": system_prompt + "\n\n" + context}]
    print(f"VybOS configurator using {args.model}. Ctrl-D exits; no changes are applied.")
    while True:
        try:
            user = input("you> ").strip()
        except EOFError:
            print()
            return 0
        if not user:
            continue
        messages.append({"role": "user", "content": user})
        try:
            answer = request_ollama(args.endpoint, args.model, messages, response_schema)
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as error:
            print(f"backend error: {error}", file=sys.stderr)
            return 2
        print(json.dumps(answer, indent=2, sort_keys=True))
        messages.append({"role": "assistant", "content": json.dumps(answer)})


if __name__ == "__main__":
    raise SystemExit(main())
