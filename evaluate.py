#!/usr/bin/env python3
"""Evaluate a trained adapter on the deterministic held-out split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from dataset_utils import load_and_split_dataset
from hub_utils import adapter_model_revision, load_run_manifest, resolve_revision
from sft_utils import (
    build_completion,
    build_prompt,
    fit_article_to_length,
    parse_response,
    rouge_l_f1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        default="outputs/gemma4-e2b-finance-qlora/final_adapter",
    )
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--model-revision", default="auto")
    parser.add_argument("--dataset-id", default="kdave/Indian_Financial_News")
    parser.add_argument("--dataset-revision", default="auto")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--no-deduplicate", action="store_true")
    parser.add_argument("--min-accuracy", type=float, default=None)
    parser.add_argument("--min-parse-rate", type=float, default=None)
    parser.add_argument("--min-rouge-l", type=float, default=None)
    return parser.parse_args()


def macro_f1(references: list[str], predictions: list[str | None]) -> float:
    labels = ("Negative", "Neutral", "Positive")
    scores: list[float] = []
    for label in labels:
        true_positive = sum(
            reference == label and prediction == label
            for reference, prediction in zip(references, predictions)
        )
        false_positive = sum(
            reference != label and prediction == label
            for reference, prediction in zip(references, predictions)
        )
        false_negative = sum(
            reference == label and prediction != label
            for reference, prediction in zip(references, predictions)
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        scores.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return sum(scores) / len(scores)


def enforce_threshold(name: str, value: float, minimum: float | None) -> None:
    if minimum is not None and value < minimum:
        raise SystemExit(f"{name}={value:.4f} is below required minimum {minimum:.4f}.")


def main() -> None:
    args = parse_args()
    if args.max_examples <= 0:
        raise SystemExit("--max-examples must be positive.")
    if args.max_length <= 0 or args.max_new_tokens <= 0:
        raise SystemExit("Token limits must be positive.")
    if not 0 < args.test_size < 1:
        raise SystemExit("--test-size must be between 0 and 1.")
    for name, threshold in (
        ("--min-accuracy", args.min_accuracy),
        ("--min-parse-rate", args.min_parse_rate),
        ("--min-rouge-l", args.min_rouge_l),
    ):
        if threshold is not None and not 0 <= threshold <= 1:
            raise SystemExit(f"{name} must be between 0 and 1.")
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for 4-bit adapter evaluation.")

    adapter_path = Path(args.adapter)
    if not adapter_path.is_dir():
        raise SystemExit(f"Adapter directory does not exist: {adapter_path}")

    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_model = args.base_model or peft_config.base_model_name_or_path
    model_revision = adapter_model_revision(
        adapter_path,
        base_model,
        args.model_revision,
    )
    manifest = load_run_manifest(adapter_path)
    if not manifest:
        raise SystemExit("Evaluation requires run_manifest.json beside the adapter.")
    if manifest.get("status") != "completed":
        raise SystemExit(
            f"Training manifest status is {manifest.get('status')!r}, not 'completed'."
        )
    saved_args = manifest.get("command_arguments", {})
    evaluation_keys = (
        "dataset_id",
        "dataset_split",
        "limit",
        "test_size",
        "seed",
        "max_length",
        "no_deduplicate",
    )
    mismatches = [
        f"{key}: trained={saved_args.get(key)!r}, evaluation={getattr(args, key)!r}"
        for key in evaluation_keys
        if key in saved_args and saved_args[key] != getattr(args, key)
    ]
    if mismatches:
        raise SystemExit(
            "Evaluation split configuration does not match training:\n  "
            + "\n  ".join(mismatches)
        )
    saved_dataset_revision = manifest.get("dataset", {}).get("resolved_revision")
    dataset_revision = (
        saved_dataset_revision
        if args.dataset_revision == "auto" and saved_dataset_revision
        else resolve_revision(
            args.dataset_id,
            "dataset",
            "main" if args.dataset_revision == "auto" else args.dataset_revision,
        )
    )
    args.model_revision = model_revision
    args.dataset_revision = dataset_revision

    split, label_names = load_and_split_dataset(args)
    evaluation_dataset = split["test"].shuffle(seed=args.seed)
    if args.max_examples and args.max_examples < len(evaluation_dataset):
        evaluation_dataset = evaluation_dataset.select(range(args.max_examples))

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    references: list[str] = []
    predictions: list[str | None] = []
    rouge_scores: list[float] = []
    parsed_summaries = 0

    with predictions_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(evaluation_dataset):
            reference_sentiment = label_names[int(row["Sentiment"])]
            reference_summary = str(row["Summary"]).strip()
            completion = build_completion(reference_summary, reference_sentiment)
            article, _, was_truncated = fit_article_to_length(
                tokenizer,
                str(row["Content"]),
                completion,
                args.max_length,
            )
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
            response = tokenizer.decode(
                output[0][input_length:],
                skip_special_tokens=True,
            )
            predicted_summary, predicted_sentiment = parse_response(response)
            references.append(reference_sentiment)
            predictions.append(predicted_sentiment)
            if predicted_summary is not None:
                parsed_summaries += 1
            rouge = rouge_l_f1(reference_summary, predicted_summary or "")
            rouge_scores.append(rouge)

            record = {
                "index": index,
                "reference_sentiment": reference_sentiment,
                "predicted_sentiment": predicted_sentiment,
                "reference_summary": reference_summary,
                "predicted_summary": predicted_summary,
                "rouge_l_f1": rouge,
                "article_truncated": was_truncated,
                "raw_response": response,
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            if (index + 1) % 10 == 0:
                print(f"Evaluated {index + 1}/{len(evaluation_dataset)} examples.")

    count = len(references)
    correct = sum(
        reference == prediction
        for reference, prediction in zip(references, predictions)
    )
    parsed_sentiments = sum(prediction is not None for prediction in predictions)
    metrics = {
        "examples": count,
        "sentiment_accuracy": correct / count,
        "sentiment_macro_f1": macro_f1(references, predictions),
        "sentiment_parse_rate": parsed_sentiments / count,
        "summary_parse_rate": parsed_summaries / count,
        "mean_rouge_l_f1": sum(rouge_scores) / count,
        "reference_class_counts": dict(sorted(Counter(references).items())),
        "predicted_class_counts": dict(
            sorted(Counter(prediction or "UNPARSED" for prediction in predictions).items())
        ),
        "base_model": base_model,
        "model_revision": model_revision,
        "dataset_id": args.dataset_id,
        "dataset_revision": dataset_revision,
        "adapter": str(adapter_path),
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Predictions: {predictions_path}")
    print(f"Metrics: {metrics_path}")

    enforce_threshold(
        "sentiment_accuracy",
        metrics["sentiment_accuracy"],
        args.min_accuracy,
    )
    enforce_threshold(
        "sentiment_parse_rate",
        metrics["sentiment_parse_rate"],
        args.min_parse_rate,
    )
    enforce_threshold(
        "mean_rouge_l_f1",
        metrics["mean_rouge_l_f1"],
        args.min_rouge_l,
    )


if __name__ == "__main__":
    main()
