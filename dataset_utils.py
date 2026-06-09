"""Deterministic dataset loading, splitting, and formatting."""

from __future__ import annotations

from collections import Counter

from datasets import Dataset, DatasetDict, load_dataset

from sft_utils import (
    EXPECTED_SENTIMENTS,
    build_completion,
    build_prompt,
    content_fingerprint,
    fit_article_to_length,
    normalize_sentiment,
)


def deduplicate_by_content(dataset: Dataset) -> Dataset:
    seen: set[str] = set()
    keep_indices: list[int] = []
    for index, content in enumerate(dataset["Content"]):
        fingerprint = content_fingerprint(content)
        if fingerprint not in seen:
            seen.add(fingerprint)
            keep_indices.append(index)
    removed = len(dataset) - len(keep_indices)
    print(f"Removed {removed:,} duplicate content rows.")
    return dataset.select(keep_indices)


def load_and_split_dataset(args) -> tuple[DatasetDict, list[str]]:
    dataset = load_dataset(
        args.dataset_id,
        split=args.dataset_split,
        revision=args.dataset_revision,
    )
    required = ("Content", "Summary", "Sentiment")
    missing = sorted(set(required) - set(dataset.column_names))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    dataset = dataset.filter(
        lambda row: all(str(row[column]).strip() for column in required),
        desc="Dropping empty rows",
    )
    dataset = dataset.map(
        lambda row: {"Sentiment": normalize_sentiment(row["Sentiment"])},
        desc="Normalizing labels",
    )

    labels = set(dataset.unique("Sentiment"))
    unexpected = sorted(labels - EXPECTED_SENTIMENTS)
    if unexpected:
        raise ValueError(f"Unexpected sentiment labels: {unexpected}")

    if not args.no_deduplicate:
        dataset = deduplicate_by_content(dataset)

    if args.limit and args.limit < len(dataset):
        dataset = dataset.shuffle(seed=args.seed).select(range(args.limit))

    print(f"Selected {len(dataset):,} rows.")
    print(f"Class counts: {dict(sorted(Counter(dataset['Sentiment']).items()))}")

    dataset = dataset.class_encode_column("Sentiment")
    label_names = dataset.features["Sentiment"].names
    try:
        split = dataset.train_test_split(
            test_size=args.test_size,
            seed=args.seed,
            stratify_by_column="Sentiment",
        )
    except ValueError as exc:
        print(f"Stratified split was not possible ({exc}); using a seeded random split.")
        split = dataset.train_test_split(test_size=args.test_size, seed=args.seed)
    return split, label_names


def format_dataset(
    split: DatasetDict,
    tokenizer,
    label_names: list[str],
    max_length: int,
) -> DatasetDict:
    def format_row(row: dict) -> dict:
        sentiment = label_names[int(row["Sentiment"])]
        completion = build_completion(str(row["Summary"]), sentiment)
        article, sequence_length, was_truncated = fit_article_to_length(
            tokenizer,
            str(row["Content"]),
            completion,
            max_length,
        )
        return {
            "prompt": build_prompt(article),
            "completion": completion,
            "sequence_length": sequence_length,
            "was_truncated": was_truncated,
        }

    formatted = DatasetDict()
    for split_name, split_dataset in split.items():
        mapped = split_dataset.map(
            format_row,
            remove_columns=split_dataset.column_names,
            desc=f"Formatting {split_name}",
        )
        truncated = sum(mapped["was_truncated"])
        lengths = mapped["sequence_length"]
        print(
            f"{split_name}: {len(mapped):,} rows, {truncated:,} article(s) truncated, "
            f"max rendered length {max(lengths):,} tokens."
        )
        formatted[split_name] = mapped.remove_columns(
            ["sequence_length", "was_truncated"]
        )
    return formatted
