"""Locale-aware corpus normalizers: dispatch, refusals, identity."""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from asrbench.config import BINDINGS, ConfigError, resolve
from asrbench.fingerprint import _software_block, observation_fingerprint
from asrbench.normalizers import NormalizerUnavailable, check, provenance
from asrbench.wer import score_texts

_POLICY = "corpus_reference"
EN = "open_asr_english_v1"
ML = "open_asr_multilingual_v1"
CARD = "en_cardinals_v1"


def _wer(ref: str, hyp: str, name: str, language: str | None = None) -> Any:
    return score_texts(
        ref, hyp, _POLICY, normalizer=name, language=language
    ).wer


def test_default_normalizer_is_unchanged() -> None:
    assert (
        _wer("Two hundred and five apples.", "205 apples", "en_cardinals_v1")
        == 0
    )
    assert score_texts("a b", "a c", _POLICY).wer == 0.5


def test_english_leaderboard_normalizer() -> None:
    pytest.importorskip("regex")
    ref = "Mr. Smith paid twenty dollars, OK?"
    hyp = "mister smith paid $20 okay"
    assert _wer(ref, hyp, "open_asr_english_v1") == 0.0
    assert _wer(ref, hyp, "en_cardinals_v1") > 0
    # Spelling variants fold as the leaderboard folds them.
    assert (
        _wer("the colour grey", "the color gray", "open_asr_english_v1") == 0.0
    )


def test_multilingual_numbers_compounds_and_tags() -> None:
    pytest.importorskip("regex")
    pytest.importorskip("num2words")
    name = "open_asr_multilingual_v1"
    assert (
        _wer("Er hat 21 Äpfel.", "er hat einundzwanzig äpfel", name, "de-DE")
        == 0.0
    )
    # Diacritics are kept, so a missing umlaut is an error.
    assert _wer("Äpfel", "Apfel", name, "de-DE") == 1.0
    assert _wer("die Haustür", "die Haus tür", name, "de-DE") == 0.0
    assert (
        _wer("<fr-FR> Il a 3 pommes", "il a trois pommes", name, "fr-FR") == 0.0
    )
    # English under the multilingual name uses the English normalizer.
    assert _wer("Mr. Smith", "mister smith", name, "en-US") == 0.0


@pytest.mark.parametrize(
    ("name", "language", "message"),
    [
        ("en_cardinals_v1", "de-DE", "English only"),
        ("open_asr_english_v1", "fr-FR", "English only"),
        ("bogus", "en-US", "unknown corpus normalizer"),
    ],
)
def test_unfit_normalizers_are_refused(
    name: str, language: str, message: str
) -> None:
    with pytest.raises(NormalizerUnavailable, match=message):
        check(name, language)


def test_a_missing_extra_names_it(monkeypatch: pytest.MonkeyPatch) -> None:
    real = importlib.import_module

    def without_vendor(module: str, *args: Any) -> Any:
        if module.startswith("asrbench.normalizers._vendor"):
            raise ImportError("No module named 'regex'")
        return real(module, *args)

    monkeypatch.setattr(importlib, "import_module", without_vendor)
    with pytest.raises(
        NormalizerUnavailable, match=r"asrbench\[wer-multilingual\]"
    ):
        check("open_asr_multilingual_v1", "de-DE")
    check("en_cardinals_v1", "en-US")


def test_provenance_states_parity() -> None:
    assert provenance("en_cardinals_v1", "en-US")["parity"] == "asrbench_native"
    german = provenance("open_asr_multilingual_v1", "de-DE")
    assert german["normalizer"].startswith("MultilingualNormalizer")
    assert german["compound_pairs"] is True
    assert german["parity"] == "normalizer_exact_without_merge_compounds"
    chinese = provenance("open_asr_multilingual_v1", "zh-CN")
    assert chinese["parity"] == "basic_multilingual_fallback"
    assert chinese["leaderboard_metric"] == "cer"


def _resolve(**overrides: Any) -> Any:
    base: dict[tuple[str, ...], Any] = {
        ("run", "output_path"): "receipt.json",
        ("run", "pass_policy", "normalization_policy"): _POLICY,
    }
    base.update(
        {tuple(key.split(".")): value for key, value in overrides.items()}
    )
    return resolve(env={}, overrides=base)


def test_config_refusals() -> None:
    with pytest.raises(ConfigError, match="English only"):
        _resolve(**{"workload.language": "de-DE"})
    with pytest.raises(ConfigError, match="only under"):
        _resolve(
            **{
                "run.pass_policy.normalization_policy": "engine_vs_own_oracle",
                "run.pass_policy.corpus_normalizer": "open_asr_english_v1",
            }
        )


def test_identity_ignores_the_default_and_keys_the_rest() -> None:
    pytest.importorskip("regex")
    default = _resolve()
    explicit = _resolve(
        **{"run.pass_policy.corpus_normalizer": "en_cardinals_v1"}
    )
    leaderboard = _resolve(
        **{"run.pass_policy.corpus_normalizer": "open_asr_english_v1"}
    )
    assert observation_fingerprint(default) == observation_fingerprint(explicit)
    assert observation_fingerprint(default) != observation_fingerprint(
        leaderboard
    )
    versions = {"schema_version": "s", "measurement_semantics_version": "m"}
    plain = _software_block(versions, _POLICY, "floor_index")
    assert "corpus_normalizer" not in plain
    assert (
        _software_block(versions, _POLICY, "floor_index", "en_cardinals_v1")
        == plain
    )
    keyed = _software_block(
        versions, _POLICY, "floor_index", "open_asr_english_v1"
    )
    assert keyed["corpus_normalizer"] == "open_asr_english_v1"


def test_cli_flag_is_bound() -> None:
    assert "--corpus-normalizer" in {binding.flag for binding in BINDINGS}


def test_a_german_corpus_run_records_its_normalizer(tmp_path: Any) -> None:
    pytest.importorskip("regex")
    pytest.importorskip("num2words")
    from tests.test_corpus_run import _config, _manifest, _run, _scripted

    references = {"d0": "Er hat 21 Äpfel.", "d1": "Die Haustür ist zu."}
    manifest = _manifest(tmp_path / "m", references)
    raw = manifest.read_text().replace('"en-US"', '"de-DE"')
    manifest.write_text(raw)
    config = _scripted(
        _config(
            manifest,
            iterations=2,
            extra={
                ("workload", "language"): "de-DE",
                ("run", "pass_policy", "corpus_normalizer"): ML,
            },
        ),
        {
            "d0.wav": "er hat einundzwanzig äpfel",
            "d1.wav": "die haus tür ist zu",
        },
    )
    receipt = _run(config, tmp_path)
    wer = receipt["summary"]["pooled"]["wer"]
    assert wer["corpus"]["corpus_wer"] == 0.0
    assert wer["normalizer"]["name"] == ML
    assert wer["normalizer"]["language"] == "de"
    software = receipt["identity"]["software"]
    assert software["corpus_normalizer"] == ML
