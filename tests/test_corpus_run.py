"""Corpus runs: per-clip references, round robin, corpus WER, 8 kHz."""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from asrbench.audio import read_pcm16
from asrbench.canonical import sha256_bytes
from asrbench.config import ConfigError, resolve
from asrbench.fingerprint import workload_fingerprint
from asrbench.identity import (
    FixedProbe,
    GitState,
    HardwareRecord,
    collect_hardware,
)
from asrbench.manifest import write_manifest
from asrbench.resample import decimate_16k_to_8k, interpolate_8k_to_16k
from asrbench.runner import AdvancingClock, _refuse_unsent_rate, run_benchmark
from asrbench.spec import ResolvedConfig
from tests.corpus_fixtures import tone_pcm, write_wav

_CLIPS = {
    "c0": "hello world",
    "c1": "the quick brown fox",
    "c2": "one two three",
}


def _manifest(root: Path, references: dict[str, str] = _CLIPS) -> Path:
    clips = []
    for index, (name, text) in enumerate(references.items()):
        pcm = tone_pcm(0.32 + 0.16 * index, hz=500.0 + 200 * index)
        write_wav(root / "clips" / f"{name}.wav", pcm)
        clips.append(
            {
                "id": name,
                "audio": f"clips/{name}.wav",
                "reference": text,
                "pcm_sha256": sha256_bytes(pcm),
                "duration_s": len(pcm) / 2 / 16_000,
            }
        )
    write_manifest(
        root / "manifest.json",
        manifest_id="tiny-corpus",
        language="en-US",
        clips=clips,
    )
    return root / "manifest.json"


def _config(
    manifest: Path,
    *,
    lanes: int = 1,
    iterations: int = 3,
    extra: dict[tuple[str, ...], Any] | None = None,
) -> ResolvedConfig:
    overrides: dict[tuple[str, ...], Any] = {
        ("workload", "manifest", "path"): str(manifest),
        ("workload", "warmup"): 0,
        ("workload", "iterations"): iterations,
        ("workload", "trials"): 1,
        ("workload", "lanes"): lanes,
        ("run", "output_path"): "receipt.json",
        ("run", "pass_policy", "normalization_policy"): "corpus_reference",
        ("run", "pass_policy", "sustainability_policy"): "completion_fraction",
    }
    overrides.update(extra or {})
    return resolve(env={}, overrides=overrides)


def _scripted(
    config: ResolvedConfig, scripts: dict[str, str]
) -> ResolvedConfig:
    options = dict(config.target.transport_options)
    options["transcripts"] = scripts
    return replace(
        config, target=replace(config.target, transport_options=options)
    )


def _run(config: ResolvedConfig, tmp_path: Path) -> dict[str, Any]:
    hardware = collect_hardware(
        FixedProbe(None),
        cpu_model="injected-cpu",
        logical_cpu_count=2,
        architecture="x86_64",
        system="Linux",
        hostname="injected-host",
        collected_at="2026-01-01T00:00:00Z",
    )
    assert isinstance(hardware, HardwareRecord)
    return asyncio.run(
        run_benchmark(
            config,
            clock=AdvancingClock(1.0),
            hardware=hardware,
            git_state=GitState(commit="unknown", dirty=True, available=False),
            package_version="0.0.1",
            python_version="3.12.0",
            run_id="corpus-test",
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
            materialize_dir=tmp_path / "materialized",
            env={},
        )
    )


def _scripts(**wrong: str) -> dict[str, str]:
    return {
        f"{name}.wav": wrong.get(name, text) for name, text in _CLIPS.items()
    }


def test_corpus_wer_pools_over_corpus_references(tmp_path: Path) -> None:
    config = _scripted(
        _config(_manifest(tmp_path / "m"), lanes=1, iterations=3),
        _scripts(c1="the quick brown dog", c2=""),
    )
    receipt = _run(config, tmp_path)
    wer = receipt["summary"]["pooled"]["wer"]
    corpus = wer["corpus"]
    # 1 substitution in c1, 3 deletions in c2, over 2 + 4 + 3 words.
    assert corpus["edits"] == 4
    assert corpus["reference_tokens"] == 9
    assert corpus["corpus_wer"] == pytest.approx(4 / 9)
    assert corpus["reduction"] == "pooled_errors_over_reference_words"
    assert corpus["complete"] is True and corpus["balanced"] is True
    per_clip = {row["clip_id"]: row["wer"] for row in corpus["clips"]}
    assert per_clip == {"c0": 0.0, "c1": 0.25, "c2": 1.0}
    assert corpus["per_clip_mean_wer"] == pytest.approx((0 + 0.25 + 1) / 3)
    dataset = receipt["dataset"]
    assert dataset["manifest"]["id"] == "tiny-corpus"
    assert dataset["manifest"]["clip_count"] == 3
    assert [clip["reference"] for clip in dataset["manifest"]["clips"]] == list(
        _CLIPS.values()
    )
    assert "audio_condition" not in dataset
    # The preset's single-clip reference is superseded, not scored.
    assert (
        receipt["resolved_config"]["workload"]["reference_transcript"] is None
    )


def test_round_robin_spreads_concurrent_lanes(tmp_path: Path) -> None:
    config = _scripted(
        _config(_manifest(tmp_path / "m"), lanes=3, iterations=2), _scripts()
    )
    receipt = _run(config, tmp_path)
    outcomes = receipt["trials"][0]["outcomes"]
    first = [o["clip_name"] for o in outcomes if o["label"].endswith("-i0")]
    assert sorted(first) == ["c0.wav", "c1.wav", "c2.wav"]
    corpus = receipt["summary"]["pooled"]["wer"]["corpus"]
    assert corpus["corpus_wer"] == 0.0
    assert corpus["plays_per_clip_min"] == corpus["plays_per_clip_max"] == 2


def test_short_runs_record_incomplete_coverage(tmp_path: Path) -> None:
    config = _scripted(
        _config(_manifest(tmp_path / "m"), lanes=1, iterations=2), _scripts()
    )
    corpus = _run(config, tmp_path)["summary"]["pooled"]["wer"]["corpus"]
    assert corpus["complete"] is False
    assert corpus["clips_scored"] == 2 and corpus["clip_count"] == 3


@pytest.mark.parametrize(
    ("condition", "rate"),
    [("telephony_8k", 8_000), ("telephony_8k_in_16k", 16_000)],
)
def test_narrowband_conditions_send_the_declared_rate(
    tmp_path: Path, condition: str, rate: int
) -> None:
    manifest = _manifest(tmp_path / "m")
    config = _scripted(
        _config(manifest, extra={("workload", "audio_condition"): condition}),
        _scripts(),
    )
    receipt = _run(config, tmp_path)
    block = receipt["dataset"]["audio_condition"]
    assert block["name"] == condition
    assert block["sent_sample_rate_hertz"] == rate
    assert block["source_sample_rate_hertz"] == 16_000
    durations = [
        o["audio_duration_s"] for o in receipt["trials"][0]["outcomes"]
    ]
    assert durations == pytest.approx([0.32, 0.48, 0.64], abs=1e-3)
    assert receipt["summary"]["pooled"]["wer"]["corpus"]["corpus_wer"] == 0.0
    # The staged manifest audio is never rewritten.
    staged = read_pcm16(tmp_path / "m" / "clips" / "c0.wav")
    assert (
        sha256_bytes(staged)
        == json.loads(manifest.read_text())["clips"][0]["pcm_sha256"]
    )


def test_identity_tracks_digest_condition_and_rate(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "m")
    native = workload_fingerprint(_config(manifest))
    narrow = workload_fingerprint(
        _config(
            manifest, extra={("workload", "audio_condition"): "telephony_8k"}
        )
    )
    in16 = workload_fingerprint(
        _config(
            manifest,
            extra={("workload", "audio_condition"): "telephony_8k_in_16k"},
        )
    )
    assert len({native, narrow, in16}) == 3
    edited = _manifest(tmp_path / "e", {**_CLIPS, "c2": "one two four"})
    assert workload_fingerprint(_config(edited)) != native
    moved = _manifest(tmp_path / "elsewhere")
    assert workload_fingerprint(_config(moved)) == native


def test_manifest_needs_corpus_policy_and_matching_language(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path / "m")
    with pytest.raises(ConfigError, match="corpus_reference"):
        _config(
            manifest,
            extra={
                (
                    "run",
                    "pass_policy",
                    "normalization_policy",
                ): "engine_vs_own_oracle"
            },
        )
    with pytest.raises(ConfigError, match="language"):
        _config(manifest, extra={("workload", "language"): "de-DE"})


def test_a_changed_manifest_cannot_replay_a_receipt(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "m")
    config = _scripted(_config(manifest), _scripts())
    receipt = _run(config, tmp_path)
    saved = tmp_path / "receipt.json"
    saved.write_text(json.dumps(receipt))
    again = resolve(from_receipt=saved, env={})
    assert workload_fingerprint(again) == workload_fingerprint(config)
    _manifest(tmp_path / "m", {**_CLIPS, "c0": "hello there"})
    with pytest.raises(ConfigError, match="digest"):
        resolve(from_receipt=saved, env={})


def test_sixteen_khz_only_adapters_refuse_8k() -> None:
    class Wideband:
        name = "realtime-ws"

    _refuse_unsent_rate(Wideband(), 16_000)  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="telephony_8k_in_16k"):
        _refuse_unsent_rate(Wideband(), 8_000)  # type: ignore[arg-type]


def _rms(pcm: bytes) -> float:
    values = [
        int.from_bytes(pcm[i : i + 2], "little", signed=True)
        for i in range(0, len(pcm), 2)
    ]
    core = values[len(values) // 4 : -len(values) // 4]
    return math.sqrt(sum(v * v for v in core) / len(core))


def test_resampler_keeps_the_telephone_band_and_drops_above_it() -> None:
    passband = tone_pcm(0.5, hz=1_000.0)
    narrow = decimate_16k_to_8k(passband)
    assert len(narrow) == len(passband) // 2
    assert _rms(narrow) == pytest.approx(_rms(passband), rel=0.02)
    wide = interpolate_8k_to_16k(narrow)
    assert len(wide) == len(passband)
    assert _rms(wide) == pytest.approx(_rms(passband), rel=0.02)
    stopband = tone_pcm(0.5, hz=6_000.0)
    assert _rms(interpolate_8k_to_16k(decimate_16k_to_8k(stopband))) < (
        0.01 * _rms(stopband)
    )
