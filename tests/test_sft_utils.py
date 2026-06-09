from __future__ import annotations

import unittest

from sft_utils import (
    TRUNCATION_MARKER,
    build_completion,
    build_prompt,
    content_fingerprint,
    fit_article_to_length,
    normalize_sentiment,
    parse_response,
    rendered_length,
    rouge_l_f1,
)


class CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> list[int]:
        del tokenize, enable_thinking
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return self.encode(rendered)


class SftUtilsTests(unittest.TestCase):
    def test_normalize_sentiment(self) -> None:
        self.assertEqual(normalize_sentiment(" positive "), "Positive")

    def test_content_fingerprint_normalizes_case_and_whitespace(self) -> None:
        self.assertEqual(
            content_fingerprint("Revenue   Increased"),
            content_fingerprint(" revenue increased "),
        )

    def test_long_article_is_trimmed_without_losing_completion(self) -> None:
        tokenizer = CharacterTokenizer()
        completion = build_completion("Revenue increased.", "Positive")
        fitted, length, truncated = fit_article_to_length(
            tokenizer,
            "A" * 5000,
            completion,
            max_length=700,
        )
        self.assertTrue(truncated)
        self.assertIn(TRUNCATION_MARKER.strip(), fitted)
        self.assertLessEqual(length, 700)
        rendered = tokenizer.decode(
            tokenizer.apply_chat_template(
                build_prompt(fitted) + completion,
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        )
        self.assertIn("<SENTIMENT>\nPositive\n</SENTIMENT>", rendered)
        self.assertEqual(
            rendered_length(
                tokenizer,
                [{"role": "user", "content": "short"}],
                completion,
            ),
            len(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": "short"}] + completion,
                    tokenize=True,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
            ),
        )

    def test_parse_response(self) -> None:
        summary, sentiment = parse_response(
            "<SUMMARY>Revenue increased.</SUMMARY>"
            "<SENTIMENT>positive</SENTIMENT>"
        )
        self.assertEqual(summary, "Revenue increased.")
        self.assertEqual(sentiment, "Positive")
        self.assertEqual(parse_response("invalid"), (None, None))

    def test_rouge_l(self) -> None:
        self.assertEqual(rouge_l_f1("same summary", "same summary"), 1.0)
        self.assertEqual(rouge_l_f1("", "prediction"), 0.0)
        self.assertGreater(rouge_l_f1("revenue increased today", "revenue increased"), 0.7)


if __name__ == "__main__":
    unittest.main()
