#!/usr/bin/env python3
"""Oracle for native/train/render_chat.vyb (P2.2a).

Renders every record of the configurator corpus through the chat template carried in the
Qwen3-4B GGUF metadata, using llama.cpp's own Jinja implementation
(llama_chat_format.Jinja2ChatFormatter) with add_generation_prompt=False — the same call
shape as training/train_lora.py's tokenizer.apply_chat_template(..., tokenize=False,
add_generation_prompt=False).

Writes the same artifacts as the Vyb program: the concatenated rendered text and the
cumulative byte-offset index. Usage:

    .venv/bin/python native/train/render_chat_ref.py [--src data/...] [--out X] [--idx Y]
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

GGUF = os.environ.get("VYBFORGE_QWEN3_GGUF", os.path.expanduser("~/Models/qwen3/Qwen3-4B-Q4_K_M.gguf"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/vybos-configurator-all.jsonl')
    ap.add_argument('--out', default='native/out/chat_render.txt')
    ap.add_argument('--idx', default='native/out/chat_render.idx')
    ap.add_argument('--gguf', default=GGUF)
    ap.add_argument('--template-sha', default=None,
                    help='also print/verify the GGUF template sha256 against this value')
    args = ap.parse_args()

    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter

    md = Llama(model_path=args.gguf, vocab_only=True, verbose=False).metadata
    template = md.get('tokenizer.chat_template')
    if not template:
        print('oracle: no tokenizer.chat_template in ' + args.gguf, file=sys.stderr)
        return 2
    sha = hashlib.sha256(template.encode()).hexdigest()
    print(f'template chars: {len(template)} sha256: {sha[:16]}')
    if args.template_sha and sha != args.template_sha:
        print(f'oracle: template sha256 {sha[:16]} != pinned {args.template_sha[:16]}',
              file=sys.stderr)
        return 3

    fmt = Jinja2ChatFormatter(template=template, eos_token='<|im_end|>', bos_token='',
                              add_generation_prompt=False)
    txt, idx, n, bad = [], [], 0, 0
    total = 0
    for lineno, line in enumerate(Path(args.src).read_text().splitlines(), 1):
        if len(line) <= 2:
            continue
        rec = json.loads(line)
        try:
            prompt = fmt(messages=rec['messages'], functions=None, function_call=None,
                         tools=None, tool_choice=None).prompt
        except Exception as exc:  # the Vyb side refuses what it does not implement
            print(f'oracle: line {lineno}: {exc}', file=sys.stderr)
            bad += 1
            continue
        txt.append(prompt)
        total += len(prompt.encode())
        idx.append(str(total))
        n += 1

    if bad:
        print(f'oracle: {bad} record(s) failed to render', file=sys.stderr)
        return 4
    Path(args.out).write_text(''.join(txt))
    Path(args.idx).write_text('\n'.join(idx) + '\n')
    print(f'rendered: {n} records -> {args.out}')
    print(f'bytes: {total}; idx: {args.idx}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
