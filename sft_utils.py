"""Shared prompt, truncation, parsing, and text-metric utilities."""

from __future__ import annotations

import hashlib
import re


SYSTEM_PROMPT = """You are a financial news analyst specializing in sentiment analysis.

For each article:
1. Write a brief, factual summary.
2. Classify the sentiment as exactly Positive, Negative, or Neutral.

Return only the requested XML-like format. Do not include reasoning."""

USER_PROMPT = """Analyze this financial news article:

<ARTICLE>
{article}
</ARTICLE>

Return:
<SUMMARY>
brief summary
</SUMMARY>
<SENTIMENT>
Positive, Negative, or Neutral
</SENTIMENT>"""

TRUNCATION_MARKER = "\n[... article truncated ...]\n"
EXPECTED_SENTIMENTS = {"Positive", "Negative", "Neutral"}
SUMMARY_RE = re.compile(r"<SUMMARY>\s*(.*?)\s*</SUMMARY>", re.IGNORECASE | re.DOTALL)
SENTIMENT_RE = re.compile(
    r"<SENTIMENT>\s*(Positive|Negative|Neutral)\s*</SENTIMENT>",
    re.IGNORECASE,
)


def normalize_sentiment(value: object) -> str:
    return str(value).strip().title()


def content_fingerprint(value: object) -> str:
    normalized = re.sub(r"\s+", " ", str(value)).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_prompt(article: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(article=article)},
    ]


def build_completion(summary: str, sentiment: str) -> list[dict[str, str]]:
    answer = (
        f"<SUMMARY>\n{summary.strip()}\n</SUMMARY>\n"
        f"<SENTIMENT>\n{sentiment}\n</SENTIMENT>"
    )
    return [{"role": "assistant", "content": answer}]


def rendered_length(
    tokenizer,
    prompt: list[dict[str, str]],
    completion: list[dict[str, str]],
) -> int:
    token_ids = tokenizer.apply_chat_template(
        prompt + completion,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return len(token_ids)


def head_tail_article(tokenizer, token_ids: list[int], keep_tokens: int) -> str:
    if keep_tokens >= len(token_ids):
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    if keep_tokens <= 0:
        return TRUNCATION_MARKER.strip()

    head_tokens = max(1, int(keep_tokens * 0.75))
    tail_tokens = max(0, keep_tokens - head_tokens)
    head = tokenizer.decode(token_ids[:head_tokens], skip_special_tokens=True).strip()
    tail = (
        tokenizer.decode(token_ids[-tail_tokens:], skip_special_tokens=True).strip()
        if tail_tokens
        else ""
    )
    return f"{head}{TRUNCATION_MARKER}{tail}".strip()


def fit_article_to_length(
    tokenizer,
    article: str,
    completion: list[dict[str, str]],
    max_length: int,
) -> tuple[str, int, bool]:
    prompt = build_prompt(article)
    full_length = rendered_length(tokenizer, prompt, completion)
    if full_length <= max_length:
        return article, full_length, False

    article_tokens = tokenizer.encode(article, add_special_tokens=False)
    low, high = 0, len(article_tokens)
    best_article: str | None = None
    best_length: int | None = None

    while low <= high:
        keep_tokens = (low + high) // 2
        candidate = head_tail_article(tokenizer, article_tokens, keep_tokens)
        candidate_length = rendered_length(
            tokenizer,
            build_prompt(candidate),
            completion,
        )
        if candidate_length <= max_length:
            best_article = candidate
            best_length = candidate_length
            low = keep_tokens + 1
        else:
            high = keep_tokens - 1

    if best_article is None or best_length is None:
        raise ValueError(
            "The prompt and completion exceed max_length even with an empty article. "
            "Increase max_length."
        )
    return best_article, best_length, True


def parse_response(text: str) -> tuple[str | None, str | None]:
    summary_match = SUMMARY_RE.search(text)
    sentiment_match = SENTIMENT_RE.search(text)
    summary = summary_match.group(1).strip() if summary_match else None
    sentiment = (
        normalize_sentiment(sentiment_match.group(1)) if sentiment_match else None
    )
    return summary, sentiment


def rouge_l_f1(reference: str, prediction: str) -> float:
    reference_tokens = re.findall(r"\w+", reference.lower())
    prediction_tokens = re.findall(r"\w+", prediction.lower())
    if not reference_tokens or not prediction_tokens:
        return 0.0

    previous = [0] * (len(prediction_tokens) + 1)
    for reference_token in reference_tokens:
        current = [0]
        for index, prediction_token in enumerate(prediction_tokens, start=1):
            if reference_token == prediction_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current

    lcs = previous[-1]
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if lcs else 0.0
