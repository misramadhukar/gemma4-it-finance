"""Hugging Face Hub helpers."""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi


def resolve_revision(repo_id: str, repo_type: str, revision: str) -> str:
    api = HfApi()
    if repo_type == "model":
        info = api.model_info(repo_id, revision=revision)
    elif repo_type == "dataset":
        info = api.dataset_info(repo_id, revision=revision)
    else:
        raise ValueError(f"Unsupported repo type: {repo_type}")
    if not info.sha:
        raise RuntimeError(f"Could not resolve {repo_type} revision for {repo_id}.")
    return info.sha


def load_run_manifest(adapter_path: str | Path) -> dict:
    manifest_path = Path(adapter_path).resolve().parent / "run_manifest.json"
    if not manifest_path.is_file():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def adapter_model_revision(
    adapter_path: str | Path,
    base_model: str,
    requested_revision: str,
) -> str:
    if requested_revision != "auto":
        return resolve_revision(base_model, "model", requested_revision)
    manifest = load_run_manifest(adapter_path)
    saved_revision = manifest.get("model", {}).get("resolved_revision")
    return saved_revision or resolve_revision(base_model, "model", "main")
