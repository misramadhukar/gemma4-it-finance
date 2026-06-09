#!/usr/bin/env python3
"""Run a quick generation test with the fine-tuned adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from hub_utils import adapter_model_revision
from sft_utils import build_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="google/gemma-4-E2B-it")
    parser.add_argument(
        "--adapter",
        default="outputs/gemma4-e2b-finance-qlora/final_adapter",
    )
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--model-revision", default="auto")
    parser.add_argument("--article-file", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for 4-bit adapter inference.")
    if not args.article_file.is_file():
        raise SystemExit(f"Article file does not exist: {args.article_file}")
    adapter_path = Path(args.adapter)
    if not adapter_path.is_dir():
        raise SystemExit(f"Adapter directory does not exist: {adapter_path}")

    article = args.article_file.read_text(encoding="utf-8")
    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_model = args.base_model or peft_config.base_model_name_or_path
    model_revision = adapter_model_revision(
        adapter_path,
        base_model,
        args.model_revision,
    )
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        revision=model_revision,
        dtype=dtype,
        device_map="auto",
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    inputs = tokenizer.apply_chat_template(
        build_prompt(article),
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    ).to(model.device)
    input_length = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    print(tokenizer.decode(output[0][input_length:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
