#!/usr/bin/env python3
"""Merge a trained LoRA adapter into a full-precision Gemma checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from hub_utils import adapter_model_revision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--model-revision", default="auto")
    parser.add_argument(
        "--adapter",
        default="outputs/gemma4-e2b-finance-qlora/final_adapter",
    )
    parser.add_argument("--output-dir", default="outputs/gemma4-e2b-finance-merged")
    parser.add_argument(
        "--device-map",
        default="cpu",
        help='Use "cpu" for a 32 GB RAM VM or "auto" when GPU memory is available.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_path = Path(args.adapter)
    if not adapter_path.is_dir():
        raise SystemExit(f"Adapter directory does not exist: {adapter_path}")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_model = args.base_model or peft_config.base_model_name_or_path
    model_revision = adapter_model_revision(
        adapter_path,
        base_model,
        args.model_revision,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        revision=model_revision,
        dtype=torch.bfloat16,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="5GB",
    )

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved merged model and tokenizer to {output_dir}")


if __name__ == "__main__":
    main()
