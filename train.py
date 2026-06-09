#!/usr/bin/env python3
"""QLoRA fine-tuning for Gemma 4 E2B on financial news."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer

from dataset_utils import format_dataset, load_and_split_dataset
from hub_utils import resolve_revision


TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "bitsandbytes",
    "accelerate",
    "datasets",
    "huggingface-hub",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="google/gemma-4-E2B-it")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--dataset-id", default="kdave/Indian_Financial_News")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--output-dir", default="outputs/gemma4-e2b-finance-qlora")
    parser.add_argument("--limit", type=int, default=5000, help="0 uses the full dataset.")
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Positive values override --epochs; useful for smoke tests.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-targets",
        default="all-linear",
        help='Use "all-linear" or a comma-separated module list.',
    )
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--dataloader-workers", type=int, default=2)
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help='Checkpoint path, or "latest" to resume from the newest checkpoint.',
    )
    parser.add_argument(
        "--overwrite-output-dir",
        action="store_true",
        help="Delete a non-empty output directory before starting a new run.",
    )
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--no-deduplicate", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.test_size < 1:
        raise SystemExit("--test-size must be between 0 and 1.")
    if args.limit < 0:
        raise SystemExit("--limit must be zero or positive.")
    if args.max_length <= 0:
        raise SystemExit("--max-length must be positive.")
    if args.epochs <= 0 and args.max_steps <= 0:
        raise SystemExit("--epochs must be positive unless --max-steps is positive.")
    if args.learning_rate <= 0:
        raise SystemExit("--learning-rate must be positive.")
    if args.train_batch_size <= 0 or args.eval_batch_size <= 0:
        raise SystemExit("Batch sizes must be positive.")
    if args.gradient_accumulation_steps <= 0:
        raise SystemExit("--gradient-accumulation-steps must be positive.")
    if args.lora_r <= 0 or args.lora_alpha <= 0:
        raise SystemExit("LoRA rank and alpha must be positive.")
    if not 0 <= args.lora_dropout < 1:
        raise SystemExit("--lora-dropout must be in [0, 1).")
    if args.save_steps <= 0 or args.logging_steps <= 0:
        raise SystemExit("Save and logging intervals must be positive.")
    if args.save_total_limit <= 0:
        raise SystemExit("--save-total-limit must be positive.")
    if args.dataloader_workers < 0:
        raise SystemExit("--dataloader-workers must be zero or positive.")
    if args.push_to_hub and not args.hub_model_id:
        raise SystemExit("--hub-model-id is required when --push-to-hub is set.")
    if args.overwrite_output_dir and args.resume_from_checkpoint:
        raise SystemExit("Do not combine --overwrite-output-dir with resume.")


def prepare_output_dir(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).expanduser().resolve()
    protected = {
        Path.cwd().resolve(),
        Path.home().resolve(),
        Path(output_dir.anchor).resolve(),
    }
    if output_dir in protected:
        raise SystemExit(f"Unsafe output directory: {output_dir}")

    has_content = output_dir.exists() and any(output_dir.iterdir())
    if args.resume_from_checkpoint:
        if not output_dir.exists():
            raise SystemExit(f"Cannot resume; output directory does not exist: {output_dir}")
        if args.resume_from_checkpoint == "latest" and not list(
            output_dir.glob("checkpoint-*")
        ):
            raise SystemExit(f"No checkpoints found in {output_dir}")
        if (
            args.resume_from_checkpoint != "latest"
            and not Path(args.resume_from_checkpoint).is_dir()
        ):
            raise SystemExit(
                f"Checkpoint directory does not exist: {args.resume_from_checkpoint}"
            )
    elif has_content:
        if not args.overwrite_output_dir:
            raise SystemExit(
                f"Output directory is not empty: {output_dir}. "
                "Use --resume-from-checkpoint or --overwrite-output-dir."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_lora_targets(value: str) -> str | list[str]:
    if value.strip() == "all-linear":
        return "all-linear"
    targets = [item.strip() for item in value.split(",") if item.strip()]
    if not targets:
        raise ValueError("--lora-targets must not be empty.")
    return targets


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def class_counts(dataset, label_names: list[str]) -> dict[str, int]:
    return dict(
        sorted(Counter(label_names[int(value)] for value in dataset["Sentiment"]).items())
    )


def load_resume_manifest(
    args: argparse.Namespace,
    manifest_path: Path,
) -> dict:
    if not args.resume_from_checkpoint:
        return {}
    if not manifest_path.is_file():
        raise SystemExit(f"Cannot resume without run manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    saved_args = manifest.get("command_arguments", {})
    compatibility_keys = (
        "model_id",
        "dataset_id",
        "dataset_split",
        "limit",
        "test_size",
        "seed",
        "max_length",
        "epochs",
        "max_steps",
        "learning_rate",
        "train_batch_size",
        "eval_batch_size",
        "gradient_accumulation_steps",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "lora_targets",
        "no_deduplicate",
    )
    mismatches = [
        f"{key}: saved={saved_args.get(key)!r}, current={getattr(args, key)!r}"
        for key in compatibility_keys
        if key in saved_args and saved_args[key] != getattr(args, key)
    ]
    if mismatches:
        raise SystemExit(
            "Resume configuration does not match the original run:\n  "
            + "\n  ".join(mismatches)
        )
    return manifest


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for this 4-bit QLoRA training script.")

    output_dir = prepare_output_dir(args)
    manifest_path = output_dir / "run_manifest.json"
    existing_manifest = load_resume_manifest(args, manifest_path)
    requested_model_revision = args.model_revision
    requested_dataset_revision = args.dataset_revision
    if existing_manifest:
        args.model_revision = existing_manifest["model"]["resolved_revision"]
        args.dataset_revision = existing_manifest["dataset"]["resolved_revision"]
    else:
        args.model_revision = resolve_revision(
            args.model_id,
            "model",
            requested_model_revision,
        )
        args.dataset_revision = resolve_revision(
            args.dataset_id,
            "dataset",
            requested_dataset_revision,
        )

    set_seed(args.seed)
    gpu_properties = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(
        f"GPU: {gpu_properties.name}; capability={capability}; "
        f"compute dtype={compute_dtype}"
    )
    print(f"Model revision: {args.model_revision}")
    print(f"Dataset revision: {args.dataset_revision}")

    training_args = SFTConfig(
        output_dir=str(output_dir),
        run_name=output_dir.name,
        max_length=args.max_length,
        completion_only_loss=True,
        packing=False,
        loss_type="chunked_nll",
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        bf16=use_bf16,
        fp16=not use_bf16,
        tf32=capability[0] >= 8,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to=["tensorboard"],
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=True,
        seed=args.seed,
        data_seed=args.seed,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_strategy="end",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.model_revision,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    split, label_names = load_and_split_dataset(args)
    if existing_manifest:
        saved_dataset = existing_manifest.get("dataset", {})
        fingerprints = {
            "train_fingerprint": split["train"]._fingerprint,
            "eval_fingerprint": split["test"]._fingerprint,
        }
        fingerprint_mismatches = [
            f"{key}: saved={saved_dataset.get(key)!r}, current={value!r}"
            for key, value in fingerprints.items()
            if saved_dataset.get(key) != value
        ]
        if fingerprint_mismatches:
            raise SystemExit(
                "Dataset fingerprints changed; refusing unsafe resume:\n  "
                + "\n  ".join(fingerprint_mismatches)
            )
    current_environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {package: version(package) for package in TRACKED_PACKAGES},
        "gpu": gpu_properties.name,
        "gpu_vram_gib": gpu_properties.total_memory / 1024**3,
        "cuda_capability": capability,
        "torch_cuda_runtime": torch.version.cuda,
        "bf16": use_bf16,
    }
    if existing_manifest:
        manifest = existing_manifest
        manifest["status"] = "preparing"
        manifest.setdefault("resume_events", []).append(
            {
                "resumed_at": utc_now(),
                "checkpoint": args.resume_from_checkpoint,
                "environment": current_environment,
            }
        )
    else:
        manifest = {
            "status": "preparing",
            "created_at": utc_now(),
            "command_arguments": vars(args),
            "model": {
                "id": args.model_id,
                "requested_revision": requested_model_revision,
                "resolved_revision": args.model_revision,
            },
            "dataset": {
                "id": args.dataset_id,
                "requested_revision": requested_dataset_revision,
                "resolved_revision": args.dataset_revision,
                "split": args.dataset_split,
                "train_rows": len(split["train"]),
                "eval_rows": len(split["test"]),
                "train_fingerprint": split["train"]._fingerprint,
                "eval_fingerprint": split["test"]._fingerprint,
                "train_class_counts": class_counts(split["train"], label_names),
                "eval_class_counts": class_counts(split["test"], label_names),
            },
            "environment": current_environment,
        }
    write_json(manifest_path, manifest)
    dataset = format_dataset(split, tokenizer, label_names, args.max_length)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        dtype=compute_dtype,
        device_map={"": local_rank},
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=resolve_lora_targets(args.lora_targets),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
    )

    resume: str | bool | None = args.resume_from_checkpoint
    if resume == "latest":
        resume = True

    manifest["status"] = "training"
    manifest["training_started_at"] = utc_now()
    write_json(manifest_path, manifest)
    try:
        train_result = trainer.train(resume_from_checkpoint=resume)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        eval_metrics = trainer.evaluate()
        trainer.save_metrics("eval", eval_metrics)

        final_dir = output_dir / "final_adapter"
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(final_dir)
        print(f"Saved final adapter and tokenizer to {final_dir}")

        if args.push_to_hub:
            trainer.push_to_hub()

        manifest.update(
            {
                "status": "completed",
                "completed_at": utc_now(),
                "train_metrics": train_result.metrics,
                "eval_metrics": eval_metrics,
                "final_adapter": str(final_dir),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
            }
        )
        write_json(manifest_path, manifest)
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "failed_at": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
            }
        )
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
