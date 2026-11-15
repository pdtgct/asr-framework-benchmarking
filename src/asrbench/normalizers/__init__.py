"""Named corpus-reference text normalizers.

``en_cardinals_v1`` is stdlib and lives in :mod:`asrbench.wer`. The
Open ASR Leaderboard normalizers are vendored under ``_vendor`` and need
the ``asrbench[wer-multilingual]`` extra (``regex``, ``num2words``). They
are imported only when a run names them, so the core stays stdlib-only
and a missing extra is a configuration error, not an import failure.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from functools import cache
from typing import Any

from asrbench.definitions import (
    CORPUS_NORMALIZER_EN_CARDINALS,
    CORPUS_NORMALIZER_OPEN_ASR_ENGLISH,
    CORPUS_NORMALIZER_OPEN_ASR_MULTILINGUAL,
    OPEN_ASR_NORMALIZER_COMMIT,
)

#: Every accepted ``pass_policy.corpus_normalizer``.
CORPUS_NORMALIZERS = frozenset(
    {
        CORPUS_NORMALIZER_EN_CARDINALS,
        CORPUS_NORMALIZER_OPEN_ASR_ENGLISH,
        CORPUS_NORMALIZER_OPEN_ASR_MULTILINGUAL,
    }
)
#: The optional extra the open_asr normalizers import.
EXTRA = "wer-multilingual"
#: Languages the leaderboard scores with CER (``eval_utils.CER_LANGUAGES``).
CER_LANGUAGES = frozenset({"zh"})
#: Leaderboard language normalizers that are not vendored here.
NOT_VENDORED = frozenset({"zh", "hy"})

_ENGLISH_ONLY = frozenset(
    {CORPUS_NORMALIZER_EN_CARDINALS, CORPUS_NORMALIZER_OPEN_ASR_ENGLISH}
)
_VENDOR = "asrbench.normalizers._vendor.open_asr"
_TAG = re.compile(r"<[^>]*>|\[[^\]]*\]")


class NormalizerUnavailable(ValueError):
    """A named normalizer cannot run: unknown, wrong language, or no extra."""


def primary_language(tag: str | None) -> str | None:
    """Return the lower-case primary subtag: ``de-DE`` -> ``de``."""
    if not tag or not tag.strip():
        return None
    return re.split(r"[-_]", tag.strip(), maxsplit=1)[0].lower()


def check(name: str, language: str | None) -> None:
    """Refuse a normalizer that cannot score this language here.

    Args:
        name: A :data:`CORPUS_NORMALIZERS` value.
        language: Workload language tag.

    Raises:
        NormalizerUnavailable: Unknown name, an English-only normalizer
            on another language, or the extra is not installed.
    """
    if name not in CORPUS_NORMALIZERS:
        raise NormalizerUnavailable(
            f"unknown corpus normalizer {name!r}; one of "
            f"{', '.join(sorted(CORPUS_NORMALIZERS))}"
        )
    lang = primary_language(language)
    if name in _ENGLISH_ONLY and lang not in {None, "en"}:
        raise NormalizerUnavailable(
            f"{name} normalizes English only; language {language!r} needs "
            f"--corpus-normalizer {CORPUS_NORMALIZER_OPEN_ASR_MULTILINGUAL}"
        )
    if name == CORPUS_NORMALIZER_EN_CARDINALS:
        return
    _load_vendored("normalizer")
    if not _is_english(name, language):
        _load_vendored("multilingual")


def _load_vendored(module: str) -> Any:
    try:
        return importlib.import_module(f"{_VENDOR}.{module}")
    except ImportError as error:
        raise NormalizerUnavailable(
            f"the Open ASR normalizers need the {EXTRA} extra: "
            f"pip install 'asrbench[{EXTRA}]' ({error})"
        ) from error


@cache
def _english() -> Callable[[str], str]:
    normalizer: Callable[[str], str] = _load_vendored(
        "normalizer"
    ).EnglishTextNormalizer()
    return normalizer


@cache
def _multilingual() -> Callable[..., str]:
    normalizer: Callable[..., str] = _load_vendored(
        "multilingual"
    ).MultilingualNormalizer(remove_diacritics=False)
    return normalizer


def _is_english(name: str, language: str | None) -> bool:
    if name == CORPUS_NORMALIZER_OPEN_ASR_ENGLISH:
        return True
    return primary_language(language) in {None, "en"}


def open_asr_text(name: str, language: str | None, text: str) -> str:
    """Normalize one text with a vendored Open ASR normalizer.

    Args:
        name: ``open_asr_english_v1`` or ``open_asr_multilingual_v1``.
        language: Workload language tag.
        text: Raw reference or hypothesis.

    Returns:
        Normalized text, space separated.
    """
    stripped = _TAG.sub(" ", text)
    if _is_english(name, language):
        return _english()(stripped)
    return _multilingual()(stripped, lang=primary_language(language))


def open_asr_pair(
    name: str, language: str | None, reference: str, hypothesis: str
) -> tuple[list[str], list[str]]:
    """Normalize a pair and, for non-English, align compound boundaries.

    Args:
        name: ``open_asr_english_v1`` or ``open_asr_multilingual_v1``.
        language: Workload language tag.
        reference: Raw reference.
        hypothesis: Raw hypothesis.

    Returns:
        Reference and hypothesis tokens.
    """
    ref = open_asr_text(name, language, reference)
    hyp = open_asr_text(name, language, hypothesis)
    if not _is_english(name, language):
        compounds = _load_vendored("multilingual").normalize_compound_pairs
        (ref,), (hyp,) = compounds([ref], [hyp])
    return ref.split(), hyp.split()


def provenance(name: str, language: str | None) -> dict[str, Any]:
    """Describe the normalizer a WER block was scored with.

    Args:
        name: A :data:`CORPUS_NORMALIZERS` value.
        language: Workload language tag.

    Returns:
        A JSON-ish block: name, language, source, and parity with the
        Open ASR Leaderboard scorer.
    """
    lang = primary_language(language)
    block: dict[str, Any] = {"name": name, "language": lang}
    if name == CORPUS_NORMALIZER_EN_CARDINALS:
        block.update(source="asrbench.wer", parity="asrbench_native")
        return block
    english = _is_english(name, language)
    if lang in NOT_VENDORED and not english:
        parity = "basic_multilingual_fallback"
    else:
        parity = "normalizer_exact_without_merge_compounds"
    block.update(
        source="huggingface/open_asr_leaderboard normalizer/",
        commit=OPEN_ASR_NORMALIZER_COMMIT,
        normalizer=(
            "EnglishTextNormalizer"
            if english
            else "MultilingualNormalizer(remove_diacritics=False)"
        ),
        compound_pairs=not english,
        leaderboard_metric="cer" if lang in CER_LANGUAGES else "wer",
        parity=parity,
    )
    return block
