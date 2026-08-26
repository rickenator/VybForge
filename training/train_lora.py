#!/usr/bin/env python3
"""Explicitly-invoked QLoRA training for the VybOS configurator dataset."""
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen3-4B')
    parser.add_argument('--train', default='data/vybos-configurator-train.jsonl')
    parser.add_argument('--eval', default='data/vybos-configurator-eval.jsonl')
    parser.add_argument('--output', default='artifacts/vybos-configurator-lora')
    parser.add_argument('--epochs', type=float, default=3)
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required; do not fall back to CPU training.')
    for path in (args.train, args.eval):
        if not Path(path).is_file():
            raise SystemExit(f'missing dataset: {path}; run training/generate-data.sh first')

    dataset = load_dataset('json', data_files={'train': args.train, 'eval': args.eval})
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def render(example):
        return {'text': tokenizer.apply_chat_template(example['messages'], tokenize=False, add_generation_prompt=False)}

    dataset = dataset.map(render)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4'),
        trust_remote_code=True,
    )
    training = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        logging_steps=5,
        eval_strategy='steps',
        eval_steps=25,
        save_strategy='steps',
        save_steps=25,
        bf16=True,
        report_to='none',
        dataset_text_field='text',
        max_length=2048,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['eval'],
        args=training,
        peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules='all-linear', task_type='CAUSAL_LM'),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == '__main__':
    main()
