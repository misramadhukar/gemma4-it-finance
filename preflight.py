#!/usr/bin/env python3
"""Verify the Ubuntu GPU environment, Hugging Face access, and dataset schema."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from datasets import load_dataset
from packaging.requirements import Requirement
from transformers import AutoConfig, AutoTokenizer

from hub_utils import resolve_revision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="google/gemma-4-E2B-it")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--dataset-id", default="kdave/Indian_Financial_News")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument(
        "--requirements",
        type=Path,
        nargs="+",
        default=[Path("requirements-cuda.txt"), Path("requirements.txt")],
    )
    parser.add_argument("--output", type=Path, default=Path("preflight_report.json"))
    return parser.parse_args()


def check_requirements(paths: list[Path]) -> dict[str, str]:
    installed: dict[str, str] = {}
    errors: list[str] = []
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"Requirements file does not exist: {path}")
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("--"):
                continue
            requirement = Requirement(line)
            try:
                installed_version = version(requirement.name)
            except PackageNotFoundError:
                errors.append(f"{requirement.name} is not installed")
                continue
            installed[requirement.name] = installed_version
            if requirement.specifier and installed_version not in requirement.specifier:
                errors.append(
                    f"{requirement.name}=={installed_version} does not satisfy "
                    f"{requirement.specifier}"
                )
    if errors:
        raise SystemExit("Dependency check failed:\n  " + "\n  ".join(errors))
    return installed


def main() -> None:
    args = parse_args()

    print("Package versions")
    packages = check_requirements(args.requirements)
    for package, installed_version in sorted(packages.items()):
        print(f"  {package}: {installed_version}")

    if not torch.cuda.is_available():
        raise SystemExit("PyTorch cannot access a CUDA GPU.")

    properties = torch.cuda.get_device_properties(0)
    vram_gib = properties.total_memory / 1024**3
    print("\nGPU")
    print(f"  name: {properties.name}")
    print(f"  capability: {torch.cuda.get_device_capability(0)}")
    print(f"  VRAM: {vram_gib:.1f} GiB")
    print(f"  bf16: {torch.cuda.is_bf16_supported()}")
    if vram_gib < 20:
        raise SystemExit("At least 20 GiB VRAM is recommended for the default run.")

    free_gib = shutil.disk_usage(".").free / 1024**3
    print(f"\nFree disk: {free_gib:.1f} GiB")
    if free_gib < 60:
        raise SystemExit("At least 60 GiB free disk is required before downloading weights.")

    model_revision = resolve_revision(args.model_id, "model", args.model_revision)
    dataset_revision = resolve_revision(
        args.dataset_id,
        "dataset",
        args.dataset_revision,
    )

    print(f"\nChecking model access: {args.model_id}@{model_revision}")
    config = AutoConfig.from_pretrained(args.model_id, revision=model_revision)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=model_revision)
    print(f"  architecture: {config.architectures}")
    print(f"  tokenizer: {tokenizer.__class__.__name__}")
    if tokenizer.chat_template is None:
        raise SystemExit("The tokenizer does not provide a chat template.")

    print(f"\nChecking dataset: {args.dataset_id}@{dataset_revision}")
    sample = load_dataset(
        args.dataset_id,
        split="train[:1]",
        revision=dataset_revision,
    )
    required = {"Content", "Summary", "Sentiment"}
    missing = sorted(required - set(sample.column_names))
    if missing:
        raise SystemExit(f"Dataset is missing required columns: {missing}")
    print(f"  columns: {sample.column_names}")

    report = {
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "packages": packages,
        "gpu": {
            "name": properties.name,
            "capability": torch.cuda.get_device_capability(0),
            "vram_gib": vram_gib,
            "bf16": torch.cuda.is_bf16_supported(),
            "torch_cuda_runtime": torch.version.cuda,
        },
        "free_disk_gib": free_gib,
        "model": {
            "id": args.model_id,
            "resolved_revision": model_revision,
            "architectures": config.architectures,
        },
        "dataset": {
            "id": args.dataset_id,
            "resolved_revision": dataset_revision,
            "columns": sample.column_names,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"\nPreflight passed. Report: {args.output}")
    print("Next: bash gcp/smoke_test.sh")


if __name__ == "__main__":
    main()
