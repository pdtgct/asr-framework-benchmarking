"""Word and character error rate under a named normalization policy.

Engine-versus-oracle is raw whitespace tokenization. Corpus reference
applies full normalization. The policy name is stored beside the rate.
A second hypothesis source is required before another accuracy policy
is added.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from asrbench.definitions import (
    NORMALIZATION_CORPUS_REFERENCE,
    NORMALIZATION_ENGINE_VS_OWN_ORACLE,
)

_TAG = re.compile(r"<[^>]*>|\[[^\]]*\]")
_ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = (
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)


class NormalizationError(ValueError):
    """The named policy is not one of the three comparisons."""


@dataclass(frozen=True)
class EditScore:
    """One alignment's edits and, when defined, its rates."""

    substitutions: int
    deletions: int
    insertions: int
    reference_tokens: int
    hypothesis_tokens: int
    reference_chars: int
    char_edits: int
    wer: float | None
    cer: float | None


def _english(number: int) -> tuple[str, ...]:
    """Return cardinal words for ``number`` in 0..999.

    Args:
        number: An integer in that range.

    Returns:
        Words, without the 'and' that some speakers insert.
    """
    if number < 20:
        return (_ONES[number],)
    if number < 100:
        ten, unit = divmod(number, 10)
        if unit == 0:
            return (_TENS[ten],)
        return (_TENS[ten], _ONES[unit])
    hundreds, rest = divmod(number, 100)
    head = (_ONES[hundreds], "hundred")
    if rest == 0:
        return head
    return head + _english(rest)


def _phrases() -> tuple[tuple[tuple[str, ...], str], ...]:
    """Build longest-first cardinal phrases for 0..999.

    Returns:
        ``(words, decimal)`` pairs, longest phrases first.
    """
    found: list[tuple[tuple[str, ...], str]] = []
    for number in range(1000):
        words = _english(number)
        found.append((words, str(number)))
        if number >= 100 and number % 100 != 0:
            hundreds, rest = divmod(number, 100)
            with_and = (_ONES[hundreds], "hundred", "and") + _english(rest)
            found.append((with_and, str(number)))
    found.sort(key=lambda item: (-len(item[0]), item[0]))
    return tuple(found)


_PHRASES = _phrases()


def _apply_cardinals(tokens: list[str]) -> list[str]:
    """Replace en_cardinals_v1 phrases with decimal strings.

    Args:
        tokens: Case-folded tokens.

    Returns:
        Tokens with cardinal phrases collapsed.
    """
    output: list[str] = []
    index = 0
    while index < len(tokens):
        matched = False
        for words, digits in _PHRASES:
            width = len(words)
            if tokens[index : index + width] == list(words):
                output.append(digits)
                index += width
                matched = True
                break
        if not matched:
            output.append(tokens[index])
            index += 1
    return output


def _full_normalization(text: str) -> list[str]:
    """Case-fold, strip tags and punctuation, collapse, fold cardinals.

    Args:
        text: Raw transcript or reference text.

    Returns:
        Tokens ready for alignment.
    """
    folded = text.casefold()
    without_tags = _TAG.sub(" ", folded)
    chars: list[str] = []
    for char in without_tags:
        category = unicodedata.category(char)
        if category.startswith("P") or category in {"Cc", "Cf"}:
            chars.append(" ")
        else:
            chars.append(char)
    tokens = "".join(chars).split()
    return _apply_cardinals(tokens)


def _oracle_tokens(text: str) -> list[str]:
    """Whitespace-tokenize without otherwise changing the text.

    Args:
        text: Raw text.

    Returns:
        Tokens split on whitespace. Case and punctuation stay.
    """
    return text.split()


def _corpus_tokens(text: str) -> list[str]:
    """Full normalization for engine-versus-corpus-reference.

    Args:
        text: Raw text.

    Returns:
        Tokens ready for alignment.
    """
    return _full_normalization(text)


_TOKENIZERS = {
    NORMALIZATION_ENGINE_VS_OWN_ORACLE: _oracle_tokens,
    NORMALIZATION_CORPUS_REFERENCE: _corpus_tokens,
}


def normalize(text: str, policy: str) -> list[str]:
    """Tokenize ``text`` under the named policy.

    Args:
        text: Raw text.
        policy: One of the named normalization policies.

    Returns:
        Tokens.

    Raises:
        NormalizationError: ``policy`` is not one of the three.
    """
    try:
        tokenizer = _TOKENIZERS[policy]
    except KeyError as error:
        raise NormalizationError(
            f"unknown normalization policy {policy!r}"
        ) from error
    return tokenizer(text)


def edit_counts(
    reference: list[str], hypothesis: list[str]
) -> tuple[int, int, int]:
    """Return substitutions, deletions, and insertions.

    Args:
        reference: Reference tokens or characters.
        hypothesis: Hypothesis tokens or characters.

    Returns:
        A triple ``(substitutions, deletions, insertions)``.
    """
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    costs = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        costs[i][0] = i
    for j in range(1, cols):
        costs[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                costs[i][j] = costs[i - 1][j - 1]
            else:
                costs[i][j] = 1 + min(
                    costs[i - 1][j - 1],
                    costs[i - 1][j],
                    costs[i][j - 1],
                )
    substitutions = 0
    deletions = 0
    insertions = 0
    i = len(reference)
    j = len(hypothesis)
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and reference[i - 1] == hypothesis[j - 1]
            and costs[i][j] == costs[i - 1][j - 1]
        ):
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and costs[i][j] == costs[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and costs[i][j] == costs[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    return substitutions, deletions, insertions


def _rate(edits: int, denominator: int) -> float | None:
    if denominator == 0:
        if edits == 0:
            return 0.0
        return None
    return edits / denominator


def score_texts(reference: str, hypothesis: str, policy: str) -> EditScore:
    """Score one pair under ``policy``.

    Args:
        reference: Reference text. Empty is a real reference.
        hypothesis: Hypothesis text. An empty string is a real empty
            hypothesis and is aligned. A missing transcript is not
            passed here.
        policy: Named normalization policy.

    Returns:
        Edits, denominators, and rates. A rate is ``None`` when its
        denominator is zero and any edit remains.
    """
    ref_tokens = normalize(reference, policy)
    hyp_tokens = normalize(hypothesis, policy)
    substitutions, deletions, insertions = edit_counts(ref_tokens, hyp_tokens)
    ref_chars = list(" ".join(ref_tokens))
    hyp_chars = list(" ".join(hyp_tokens))
    char_s, char_d, char_i = edit_counts(ref_chars, hyp_chars)
    return EditScore(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_tokens=len(ref_tokens),
        hypothesis_tokens=len(hyp_tokens),
        reference_chars=len(ref_chars),
        char_edits=char_s + char_d + char_i,
        wer=_rate(substitutions + deletions + insertions, len(ref_tokens)),
        cer=_rate(char_s + char_d + char_i, len(ref_chars)),
    )
