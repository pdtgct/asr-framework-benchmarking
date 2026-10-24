"""Named normalization policies stay distinct, and oracle stays raw."""

import pytest

from asrbench.definitions import (
    NORMALIZATION_CORPUS_REFERENCE,
    NORMALIZATION_ENGINE_VS_OWN_ORACLE,
)
from asrbench.wer import (
    NormalizationError,
    edit_counts,
    normalize,
    score_texts,
)


def test_oracle_keeps_case_and_corpus_folds_it() -> None:
    oracle = score_texts("Hello", "hello", NORMALIZATION_ENGINE_VS_OWN_ORACLE)
    corpus = score_texts("Hello", "hello", NORMALIZATION_CORPUS_REFERENCE)
    assert oracle.substitutions == 1
    assert oracle.wer == 1.0
    assert corpus.substitutions == 0
    assert corpus.wer == 0.0


def test_cardinals_apply_only_under_full_normalization() -> None:
    oracle = score_texts("1", "one", NORMALIZATION_ENGINE_VS_OWN_ORACLE)
    corpus = score_texts("1", "one", NORMALIZATION_CORPUS_REFERENCE)
    assert oracle.wer == 1.0
    assert corpus.wer == 0.0
    assert normalize("one hundred and one", NORMALIZATION_CORPUS_REFERENCE) == [
        "101"
    ]


def test_tags_are_stripped_only_by_full_normalization() -> None:
    assert normalize("<unk> hello", NORMALIZATION_CORPUS_REFERENCE) == ["hello"]
    assert normalize("[noise] hello", NORMALIZATION_CORPUS_REFERENCE) == [
        "hello"
    ]
    assert normalize("<unk> hello", NORMALIZATION_ENGINE_VS_OWN_ORACLE) == [
        "<unk>",
        "hello",
    ]


def test_substitution_is_counted_separately() -> None:
    score = score_texts(
        "the cat sat", "the dog sat", NORMALIZATION_ENGINE_VS_OWN_ORACLE
    )
    assert (score.substitutions, score.deletions, score.insertions) == (1, 0, 0)
    assert score.wer == pytest.approx(1 / 3)


def test_apostrophe_is_punctuation_under_corpus_normalization() -> None:
    tokens = normalize("don't", NORMALIZATION_CORPUS_REFERENCE)
    assert tokens == ["don", "t"]
    assert normalize("don't", NORMALIZATION_ENGINE_VS_OWN_ORACLE) == ["don't"]


def test_empty_hypothesis_is_deletions() -> None:
    score = score_texts("hello", "", NORMALIZATION_ENGINE_VS_OWN_ORACLE)
    assert score.deletions == 1
    assert score.wer == 1.0


def test_both_empty_is_zero_not_undefined() -> None:
    score = score_texts("", "", NORMALIZATION_ENGINE_VS_OWN_ORACLE)
    assert score.wer == 0.0


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(NormalizationError):
        normalize("hello", "lowercase")


def test_insertion_and_mid_sentence_deletion_are_separate() -> None:
    inserted = score_texts("a", "a b", NORMALIZATION_ENGINE_VS_OWN_ORACLE)
    deleted = score_texts("a b c", "a c", NORMALIZATION_ENGINE_VS_OWN_ORACLE)
    inserted_counts = (
        inserted.substitutions,
        inserted.deletions,
        inserted.insertions,
    )
    assert inserted_counts == (0, 0, 1)
    assert (deleted.substitutions, deleted.deletions, deleted.insertions) == (
        0,
        1,
        0,
    )
    reference = ["the", "cat", "sat", "down"]
    hypothesis = ["a", "cat", "sat"]
    score = score_texts(
        "the cat sat down", "a cat sat", NORMALIZATION_ENGINE_VS_OWN_ORACLE
    )
    assert (
        score.substitutions,
        score.deletions,
        score.insertions,
    ) == edit_counts(reference, hypothesis)
    assert score.reference_chars == 16
    assert score.char_edits == 8
    assert score.cer == 0.5


def test_tie_break_prefers_substitution_then_deletion() -> None:
    # Equal cost paths exist; the backtrace must be stable.
    substitutions, deletions, insertions = edit_counts(["a"], ["b"])
    assert (substitutions, deletions, insertions) == (1, 0, 0)
