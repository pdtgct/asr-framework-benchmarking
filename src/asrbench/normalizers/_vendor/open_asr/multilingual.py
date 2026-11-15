# Vendored from huggingface/open_asr_leaderboard,
# commit c21a412f2dd7cbe3af658d69fbf3d0dd536a264e:
#   - FILLER_WORDS and MultilingualNormalizer from normalizer/data_utils.py
#   - CER_LANGUAGES and normalize_compound_pairs from normalizer/eval_utils.py
# Licensed under the Apache License, Version 2.0 (see the upstream LICENSE and
# this repository's LICENSE and THIRD_PARTY.md).
# Modified: excerpted from larger modules, imports changed to this package,
# and the Armenian ("hy") and Chinese ("zh") language normalizers are not
# vendored, so _language_normalizers is empty (marked "asrbench:" below) and
# those languages take the basic multilingual path. Otherwise unmodified.
#
import re
from difflib import SequenceMatcher

import num2words

from .normalizer import BasicMultilingualTextNormalizer

CER_LANGUAGES = {"zh"}

# Language-specific filler words / hesitations, removed when the matching
# lang= is passed to the multilingual normalizer. Multi-word entries are
# supported (matched on whitespace boundaries). Currently empty
FILLER_WORDS = {}


class MultilingualNormalizer(BasicMultilingualTextNormalizer):
    """BasicMultilingualTextNormalizer with optional number normalization.

    Call with just text for standard normalization (backward-compatible).
    Pass lang= to also convert digits to words via num2words and remove
    language-specific filler words (see FILLER_WORDS).
    """

    def __init__(self, remove_diacritics: bool = True):
        super().__init__(remove_diacritics)
        # asrbench: the hy and zh normalizers are not vendored.
        self._language_normalizers = {}
        # Pre-compile filler patterns. Each filler word is passed through the
        # base normalization itself, so the pattern matches the normalized
        # text exactly (base normalization may strip punctuation such as "…"
        # or combining marks). Longest-first so that multi-word and longer
        # variants match before their prefixes. Matched on whitespace
        # boundaries ((?<!\S) / (?!\S)) rather than \b, which is unreliable
        # next to combining marks.
        self._filler_patterns = {}
        base_normalize = super().__call__
        for lang, words in FILLER_WORDS.items():
            normalized_words = {base_normalize(w) for w in words}
            normalized_words.discard("")
            self._filler_patterns[lang] = re.compile(
                r"(?<!\S)(?:"
                + "|".join(re.escape(w) for w in sorted(normalized_words, key=len, reverse=True))
                + r")(?!\S)"
            )

    def _remove_fillers(self, text, lang):
        pattern = self._filler_patterns.get(lang)
        if pattern is None:
            return text
        text = pattern.sub("", text)
        return re.sub(r"\s+", " ", text).strip()

    def _normalize_numbers(self, text, lang):
        # Join space-separated thousand groups (e.g. "10 000" -> "10000")
        text = re.sub(r"(\d)\s+(\d{3})\b", r"\1\2", text)

        # Convert remaining digit sequences to words
        def _replace(m):
            try:
                return num2words.num2words(int(m.group()), lang=lang)
            except Exception:
                return m.group()

        return re.sub(r"\d+", _replace, text)

    def __call__(self, s, lang=None):
        language_normalizer = self._language_normalizers.get(lang)
        if language_normalizer is not None:
            return language_normalizer(s)

        s = super().__call__(s)
        if lang is not None:
            s = self._remove_fillers(s, lang)
            s = self._normalize_numbers(s, lang)
        return s


def normalize_compound_pairs(refs, preds):
    """Align compound word boundaries between ref/pred pairs.

    When a mismatch region has identical characters ignoring whitespace,
    normalize both sides to the joined form.
    """
    new_refs, new_preds = [], []
    for ref_text, pred_text in zip(refs, preds):
        ref_words = ref_text.split()
        pred_words = pred_text.split()

        sm = SequenceMatcher(None, ref_words, pred_words)
        new_rw, new_pw = [], []

        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                new_rw.extend(ref_words[i1:i2])
                new_pw.extend(pred_words[j1:j2])
            else:
                rc = "".join(ref_words[i1:i2])
                pc = "".join(pred_words[j1:j2])
                if rc == pc:
                    new_rw.append(rc)
                    new_pw.append(pc)
                else:
                    new_rw.extend(ref_words[i1:i2])
                    new_pw.extend(pred_words[j1:j2])

        new_refs.append(" ".join(new_rw))
        new_preds.append(" ".join(new_pw))
    return new_refs, new_preds
