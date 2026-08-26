#!/usr/bin/env python3
"""Load a locally trained adapter and verify one JSON-only response."""
import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', default='Qwen/Qwen3-4B')
    parser.add_argument('--adapter', default='artifacts/vybos-configurator-lora')
    parser.add_argument('--prompt', required=True)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for this smoke test.')
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, device_map='auto', dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    messages = [
        {'role': 'system', 'content': 'You are the VybOS configuration interviewer. Return exactly one JSON object with kind, message, missing_fields, proposed_changes, and requires_confirmation.'},
        {'role': 'user', 'content': args.prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=True, return_tensors='pt',
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=192, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(
        output[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True
    ).strip()
    parsed = json.loads(text)
    required = {'kind', 'message', 'missing_fields', 'proposed_changes', 'requires_confirmation'}
    if set(parsed) != required:
        raise SystemExit(f'unexpected response keys: {set(parsed)}')
    print(json.dumps(parsed, indent=2))


if __name__ == '__main__':
    main()
