#!/usr/bin/env python3
"""Schema-constrained, non-mutating VybOS configuration interview client."""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def read_json(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def endpoint_url(endpoint: str, suffix: str) -> str:
    """Join a base endpoint such as https://host/v1 to an API suffix."""
    return endpoint.rstrip("/") + suffix


def json_request(url: str, payload: dict, api_key: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def structured_format(response_schema: dict, mode: str) -> dict | None:
    if mode == "prompt":
        return None
    if mode == "json_object":
        return {"type": "json_object"}
    return {"type": "json_schema", "json_schema": {"name": "vybos_configurator_response", "strict": True, "schema": response_schema}}


def responses_structured_format(response_schema: dict, mode: str) -> dict | None:
    if mode == "prompt":
        return None
    if mode == "json_object":
        return {"type": "json_object"}
    return {"type": "json_schema", "name": "vybos_configurator_response", "strict": True, "schema": response_schema}


def request_ollama(endpoint: str, model: str, messages: list[dict], response_schema: dict, _: str, __: str) -> dict:
    payload = {"model": model, "messages": messages, "stream": False, "format": response_schema, "options": {"temperature": 0.2}}
    body = json_request(endpoint_url(endpoint, "/api/chat"), payload)
    return json.loads(body["message"]["content"])


def request_openai_chat(endpoint: str, model: str, messages: list[dict], response_schema: dict, mode: str, api_key: str) -> dict:
    payload = {"model": model, "messages": messages, "temperature": 0.2}
    response_format = structured_format(response_schema, mode)
    if response_format:
        payload["response_format"] = response_format
    body = json_request(endpoint_url(endpoint, "/chat/completions"), payload, api_key)
    return json.loads(body["choices"][0]["message"]["content"])


def response_text(body: dict) -> str:
    if isinstance(body.get("output_text"), str):
        return body["output_text"]
    for item in body.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                return content["text"]
    raise KeyError("Responses reply has no output_text")


def request_openai_responses(endpoint: str, model: str, messages: list[dict], response_schema: dict, mode: str, api_key: str) -> dict:
    transcript = "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in messages)
    payload = {"model": model, "input": transcript, "store": False, "temperature": 0.2}
    response_format = responses_structured_format(response_schema, mode)
    if response_format:
        payload["text"] = {"format": response_format}
    body = json_request(endpoint_url(endpoint, "/responses"), payload, api_key)
    return json.loads(response_text(body))


def api_key_from_environment(name: str) -> str | None:
    return os.environ.get(name) or (os.environ.get("OPENAI_API_KEY") if name != "OPENAI_API_KEY" else None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["ollama", "openai-chat", "openai-responses"], default=os.environ.get("VYBAICONF_BACKEND", "ollama"))
    parser.add_argument("--endpoint", default=os.environ.get("VYBAICONF_ENDPOINT", "http://127.0.0.1:11434"))
    parser.add_argument("--model", default=os.environ.get("VYBAICONF_MODEL", "qwen3:8b"))
    parser.add_argument("--api-key-env", default=os.environ.get("VYBAICONF_API_KEY_ENV", "VYBAICONF_API_KEY"), help="environment variable holding a hosted-provider API key")
    parser.add_argument("--structured-output", choices=["json_schema", "json_object", "prompt"], default=os.environ.get("VYBAICONF_STRUCTURED_OUTPUT", "json_schema"), help="fall back to json_object or prompt for compatible gateways without JSON Schema support")
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
    api_key = api_key_from_environment(args.api_key_env)
    if args.backend != "ollama" and not api_key:
        parser.error(f"{args.backend} requires ${args.api_key_env} (or $OPENAI_API_KEY); credentials are never stored by VybAIConf")

    request_backend = {"ollama": request_ollama, "openai-chat": request_openai_chat, "openai-responses": request_openai_responses}[args.backend]
    print(f"VybOS configurator using {args.backend} / {args.model}. Ctrl-D exits; no changes are applied.")
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
            answer = request_backend(args.endpoint, args.model, messages, response_schema, args.structured_output, api_key or "")
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as error:
            print(f"backend error: {error}", file=sys.stderr)
            return 2
        print(json.dumps(answer, indent=2, sort_keys=True))
        messages.append({"role": "assistant", "content": json.dumps(answer)})


if __name__ == "__main__":
    raise SystemExit(main())
