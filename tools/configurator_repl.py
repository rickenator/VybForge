#!/usr/bin/env python3
"""Interactive VybOS configurator REPL using the retrained LoRA adapter on GPU.

Loads the base model + the v2 adapter ONCE, then runs the schema-constrained
interview loop on the RTX 3090. Type a VybOS goal; the model replies with the
configurator JSON (question/proposal/summary). /quit exits.

Run:  . .venv/bin/activate && python tools/configurator_repl.py [--adapter DIR]
"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default="artifacts/vybos-configurator-lora-v2")
    ap.add_argument("--base-model", default="Qwen/Qwen3-4B")
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this REPL (we're on the 3090).")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    print("loading base model + adapter on GPU (one-time)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, device_map={"": 0}, dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    print("ready. type a VybOS goal (/quit to exit)\n", flush=True)

    system = ("You are the VybOS configuration interviewer. Return exactly one JSON object with kind, "
              "message, missing_fields, proposed_changes, and requires_confirmation. VybOS builds a "
              "machine from a minimal SystemSpec: system, hostname, pkgs, services. Each proposed_changes "
              "item is {path, op, value, reason}. paths are system|hostname|pkgs|services. "
              "hostname/system use op replace with a plain string value; pkgs add has value "
              '{"name":..,"version":..,"source":..}; services add has value {"name":..,"command":..}. '
              "Emit only valid JSON.")

    while True:
        try:
            user = input("config> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user or user == "/quit":
            return
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        enc = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=True, return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=args.max_tokens, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        try:
            print(json.dumps(json.loads(text), indent=2, sort_keys=True))
        except json.JSONDecodeError:
            print("RAW (not valid JSON):", text)
        print()


if __name__ == "__main__":
    main()
