#!/usr/bin/env python3
"""Run the trained configurator adapter and stream its proposed_changes to a
patches file that the deterministic applier consumes.

Usage:
  interview_infer.py --adapter <dir> --prompt "<goal>" [--out out/patches.jsonl]

Appends one compact JSON line per proposed_changes item (treated as the user-
confirmed request set for the demo). Requires CUDA.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="artifacts/vybos-configurator-lora-v2")
    ap.add_argument("--base-model", default="Qwen/Qwen3-4B")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", default="out/patches.jsonl")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for adapter inference.")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, device_map={"": 0}, dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(model, args.adapter)

    system = ("You are the VybOS configuration interviewer. Return exactly one JSON object with kind, "
              "message, missing_fields, proposed_changes, and requires_confirmation. VybOS builds a "
              "machine from a minimal SystemSpec: system, hostname, pkgs, services. Each proposed_changes "
              "item is {path, op, value, reason}; paths are system|hostname|pkgs|services and values must "
              "be schema-conformant. For pkgs, op add has value {\"name\":..,\"version\":..,\"source\":..}; "
              "for services, op add has value {\"name\":..,\"command\":..}; hostname/system use op replace "
              "with a plain string value. Emit only valid JSON.")
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": args.prompt}]
    enc = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        enable_thinking=False, return_dict=True, return_tensors="pt")
    inputs = enc.to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    print("RAW:", text)

    resp = json.loads(text)
    changes = resp.get("proposed_changes", [])
    if not changes:
        print("no proposed_changes in response (kind=%s); nothing appended" % resp.get("kind"))
        return
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for c in changes:
            canonical = {"path": c["path"], "op": c["op"], "value": c["value"], "reason": c["reason"]}
            fh.write(json.dumps(canonical, separators=(",", ":")) + "\n")
            n += 1
    print(f"appended {n} proposed_change(s) -> {out_path}")


if __name__ == "__main__":
    main()
