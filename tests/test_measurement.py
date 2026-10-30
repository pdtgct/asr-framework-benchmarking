"""Comparison axis and the #177 headline metrics.

Engine may differ. Checkpoint, operating point, GPU, workload, and decode
features may not. Precision is recorded. Frontier RTFx is invalid when
``audio_processed`` decreases and unavailable when the field is absent.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from asrbench.adapters.mock import MOCK_E2E_CLOCK
from asrbench.compare import compare_receipts
from asrbench.config import ConfigError, resolve
from asrbench.identity import FixedProbe, GitState, collect_hardware
from asrbench.outcome import AudioProgressSample, WordTimingSample
from asrbench.reducers import reduce_trial
from asrbench.runner import AdvancingClock, run_benchmark
from asrbench.spec import PassPolicy
from tests.support import session


def _reduce(outcomes: list[Any], **kwargs: Any) -> dict[str, Any]:
    return reduce_trial(
        outcomes,
        wall_span_s=kwargs.get("wall_span_s", 2.0),
        mode="streaming",
        pacing=True,
        lanes=len(outcomes) or 1,
        reference="hello",
        pass_policy=PassPolicy(),
        percentile_method="numpy_linear",
        warmup_excluded=0,
    )


def _progress(
    points: list[tuple[float, float, float]],
) -> list[AudioProgressSample]:
    return [
        {
            "audio_processed_s": processed,
            "received_s": received,
            "audio_sent_s": sent,
        }
        for processed, received, sent in points
    ]


def _run(
    tmp_path: Path,
    run_id: str,
    overrides: Mapping[tuple[str, ...], Any] | None = None,
) -> dict[str, Any]:
    chosen: dict[tuple[str, ...], Any] = dict(overrides or {})
    chosen.setdefault(("run", "output_path"), "receipt.json")
    config = resolve(env={}, overrides=chosen)
    hardware = collect_hardware(
        FixedProbe(None),
        cpu_model="injected-cpu",
        logical_cpu_count=2,
        architecture="x86_64",
        system="Linux",
        hostname="injected-host",
        collected_at="2026-01-01T00:00:00Z",
    )
    receipt: dict[str, Any] = asyncio.run(
        run_benchmark(
            config,
            clock=AdvancingClock(1.0),
            hardware=hardware,
            git_state=GitState(commit="unknown", dirty=False, available=False),
            package_version="0.0.1",
            python_version="3.12.0",
            run_id=run_id,
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
            materialize_dir=tmp_path / run_id,
            env={},
        )
    )
    return receipt


def _checkpoint(**fields: str) -> dict[tuple[str, ...], Any]:
    base: dict[tuple[str, ...], Any] = {
        ("target", "checkpoint_id"): "nvidia/nemotron-3.5-asr-streaming-0.6b",
        ("target", "checkpoint_revision"): "abc123",
        ("target", "checkpoint_source"): "huggingface",
    }
    for key, value in fields.items():
        base[("target", key)] = value
    return base


def test_engine_may_differ_and_checkpoint_may_not(tmp_path: Path) -> None:
    shared = _checkpoint()
    left = _run(
        tmp_path,
        "nim",
        {
            **shared,
            ("target", "model"): "cache-aware-parakeet",
            ("target", "server_version"): "nim-image",
            ("target", "engine_name"): "nim",
        },
    )
    right = _run(
        tmp_path,
        "omni",
        {
            **shared,
            ("target", "model"): "nemotron-omni",
            ("target", "server_version"): "omni-commit",
            ("target", "engine_name"): "vllm-omni",
            ("target", "engine_version"): "omni-commit",
        },
    )
    result = compare_receipts(left, right)
    assert result.status == "comparable"
    assert result.comparison_key_left == result.comparison_key_right
    assert any(item.field == "model" for item in result.warnings)
    assert any(item.field == "engine" for item in result.recorded)
    assert "server_version" not in {item.field for item in result.differences}
    assert "model" not in left["identity"]["target"]
    other = _run(
        tmp_path,
        "other-weights",
        {
            **_checkpoint(checkpoint_revision="def456"),
            ("target", "model"): "cache-aware-parakeet",
            ("target", "engine_name"): "nim",
        },
    )
    refused = compare_receipts(left, other)
    assert refused.status == "refused"
    assert any("revision" in item.field for item in refused.differences)


def test_transport_is_recorded_with_the_engine(tmp_path: Path) -> None:
    left = _run(tmp_path, "mock", _checkpoint())
    right = json.loads(json.dumps(left))
    right["resolved_config"]["target"]["transport"] = "riva-grpc"
    right["identity"]["recorded"]["adapter"] = "riva-grpc"
    right["identity"]["recorded"]["engine"]["transport"] = "riva-grpc"
    right["identity"]["software"].pop("adapter", None)
    result = compare_receipts(left, right)
    assert result.status == "comparable"
    assert result.comparison_key_left == result.comparison_key_right
    engine = next(item for item in result.recorded if item.field == "engine")
    assert engine.left["transport"] == "mock"
    assert engine.right["transport"] == "riva-grpc"


def test_precision_is_recorded_and_not_a_key(tmp_path: Path) -> None:
    left = _run(tmp_path, "fp16", {("target", "precision"): "FP16"})
    right = json.loads(json.dumps(left))
    right["resolved_config"]["target"]["precision"] = "FP16+INT8"
    right["identity"]["recorded"]["precision"] = "FP16+INT8"
    result = compare_receipts(left, right)
    assert result.status == "comparable"
    assert result.comparison_key_left == result.comparison_key_right
    assert any(item.field == "precision" for item in result.recorded)
    assert not any(item.field == "precision" for item in result.warnings)
    assert not any("precision" in item.field for item in result.differences)
    with pytest.raises(ConfigError, match="precision"):
        resolve(env={}, overrides={("target", "precision"): "BF16"})


def test_batch_cuda_and_client_location_warn(tmp_path: Path) -> None:
    left = _run(
        tmp_path,
        "box",
        {
            ("target", "batch_size"): 8,
            ("target", "max_num_seqs"): 16,
            ("target", "cuda_version"): "12.4",
            ("target", "torch_version"): "2.4.0",
            ("target", "client_location"): "loopback",
        },
    )
    right = _run(
        tmp_path,
        "wan",
        {
            ("target", "batch_size"): 1,
            ("target", "max_num_seqs"): 32,
            ("target", "cuda_version"): "12.6",
            ("target", "torch_version"): "2.5.0",
            ("target", "client_location"): "wan",
        },
    )
    result = compare_receipts(left, right)
    assert result.status == "comparable"
    assert result.comparison_key_left == result.comparison_key_right
    fields = {item.field for item in result.warnings}
    assert "batch_size" in fields
    assert "max_num_seqs" in fields
    assert "cuda_version" in fields
    assert "torch_version" in fields
    assert "client_location" in fields


def test_operating_point_and_decoder_refuse(tmp_path: Path) -> None:
    left = _run(tmp_path, "r1", {("workload", "att_context_right"): 1})
    right = json.loads(json.dumps(left))
    right["resolved_config"]["workload"]["att_context_right"] = 6
    right["resolved_config"]["workload"]["population"][0][
        "att_context_right"
    ] = 6
    refused = compare_receipts(left, right)
    assert refused.status == "refused"
    assert any(
        "att_context_right" in item.field for item in refused.differences
    )
    decoded = json.loads(json.dumps(left))
    decoded["resolved_config"]["workload"]["decoder"] = "beam"
    decoded_result = compare_receipts(left, decoded)
    assert decoded_result.status == "refused"
    assert any("decoder" in item.field for item in decoded_result.differences)


def test_audio_processed_monotonicity_gate() -> None:
    steady = session(label="steady", audio_duration_s=2.0)
    steady.first_audio_sent_s = 0.5
    # Speech, a hold across silence, then a second utterance.
    steady.audio_progress = _progress(
        [
            (0.0, 1.00, 0.16),
            (0.8, 1.40, 0.96),
            (0.8, 2.10, 1.60),
            (1.6, 2.80, 2.00),
        ]
    )
    block = _reduce([steady])["frontier_rtfx"]
    assert block["availability"] == "measured"
    assert block["n_invalid_sessions"] == 0
    assert block["n_samples"] == 3
    assert block["n_total"] == 1
    assert block["minimum"] == 0.0
    assert block["percentile_method"] == "numpy_linear"

    decreased = session(label="drop")
    decreased.audio_progress = _progress([(1.0, 1.0, 0.2), (0.4, 1.2, 0.4)])
    bad = _reduce([decreased])["frontier_rtfx"]
    assert bad["availability"] == "invalid"
    assert bad["reason"] == "audio_processed_decreased"
    assert bad["p50"] is None
    assert bad["n_with_measure"] == 0
    assert bad["n_total"] == 1

    missing = _reduce([session(label="none")])["frontier_rtfx"]
    assert missing["availability"] == "unavailable"
    assert missing["n_total"] == 1
    assert missing["p50"] is None


def test_word_latency_keeps_a_negative_and_e2e_uses_the_span() -> None:
    outcome = session(audio_duration_s=1.6)
    outcome.first_audio_sent_s = 0.5
    outcome.final_received_s = 2.5
    timing: WordTimingSample = {
        "end_time_s": 1.0,
        "final_received_s": 1.2,
    }
    outcome.word_timings = [timing]
    reduction = _reduce([outcome])
    words = reduction["word_latency_s"]
    assert words["availability"] == "measured"
    assert words["minimum"] == pytest.approx(-0.3)
    assert words["p50"] == pytest.approx(-0.3)
    e2e = reduction["e2e_rtfx"]
    assert e2e["mean"] == pytest.approx(1.6 / 2.0)
    assert e2e["n_with_measure"] == 1
    assert e2e["n_total"] == 1


def test_aggregate_rtfx_keeps_failed_audio_in_the_numerator() -> None:
    ok = session(label="ok", audio_duration_s=1.6)
    failed = session(
        label="timeout",
        transcript=None,
        ttfs_s=None,
        final_latency_s=None,
        response_latencies_s=[],
        interim_texts=None,
        timed_out=True,
        audio_duration_s=1.6,
    )
    block = _reduce([ok, failed], wall_span_s=2.0)["aggregate_rtfx"]
    assert block["value"] == pytest.approx(3.2 / 2.0)
    assert block["offered_audio_s"] == pytest.approx(3.2)
    assert block["successful_audio_s"] == pytest.approx(1.6)


def test_riva_client_compat_requires_matching_counts() -> None:
    paired = session(label="paired")
    paired.chunk_send_times_s = [1.0, 2.0]
    paired.response_receive_times_s = [1.1, 2.3]
    measured = _reduce([paired])["riva_client_compat_s"]
    assert measured["label"] == "Riva client compat"
    assert measured["availability"] == "measured"
    assert measured["n_samples"] == 2
    assert measured["minimum"] == pytest.approx(0.1)

    extra = session(label="extra")
    extra.chunk_send_times_s = [1.0, 2.0]
    extra.response_receive_times_s = [1.1, 2.3, 3.0]
    plus_one = _reduce([extra])["riva_client_compat_s"]
    assert plus_one["availability"] == "measured"
    assert plus_one["n_samples"] == 2

    short = session(label="short")
    short.chunk_send_times_s = [1.0, 2.0]
    short.response_receive_times_s = [1.1]
    invalid = _reduce([short])["riva_client_compat_s"]
    assert invalid["availability"] == "invalid"
    assert invalid["p50"] is None
    assert invalid["n_total"] == 1

    absent = _reduce([session(label="none")])["riva_client_compat_s"]
    assert absent["availability"] == "unavailable"


def test_frontier_lag_does_not_clamp() -> None:
    outcome = session(label="lag")
    outcome.first_audio_sent_s = 5.0
    outcome.audio_progress = _progress([(0.2, 5.1, 0.4)])
    lag = _reduce([outcome])["frontier_lag_s"]
    # 5.1 - (5.0 + 0.4) = -0.3
    assert lag["availability"] == "measured"
    assert lag["minimum"] == pytest.approx(-0.3)


def test_adapter_and_transport_do_not_refuse(tmp_path: Path) -> None:
    left = _run(tmp_path, "riva", _checkpoint())
    right = json.loads(json.dumps(left))
    right["resolved_config"]["target"]["transport"] = "realtime-ws"
    right["identity"]["recorded"]["adapter"] = "realtime-ws"
    right["identity"]["recorded"]["engine"]["transport"] = "realtime-ws"
    right["identity"]["software"].pop("adapter", None)
    result = compare_receipts(left, right)
    assert result.status == "comparable"
    assert result.comparison_key_left == result.comparison_key_right
    assert not any(item.field == "adapter" for item in result.differences)
    assert not any("transport" in item.field for item in result.differences)
    adapter = next(item for item in result.recorded if item.field == "adapter")
    assert adapter.left == "mock"
    assert adapter.right == "realtime-ws"


def test_non_mock_checkpoint_id_and_verified_revision() -> None:
    with pytest.raises(ConfigError, match="checkpoint_id"):
        resolve(env={}, overrides={("target", "transport"): "riva-grpc"})
    with pytest.raises(ConfigError, match="checkpoint_revision"):
        resolve(
            env={},
            overrides={
                ("target", "checkpoint_id"): "org/model",
                ("target", "checkpoint_source"): "huggingface",
            },
        )
    resolved = resolve(
        env={},
        overrides={
            ("target", "transport"): "realtime-ws",
            ("target", "checkpoint_id"): "nemotron-asr-streaming",
        },
    )
    assert resolved.target.checkpoint_id == "nemotron-asr-streaming"
    assert resolved.target.checkpoint_revision is None
    with pytest.raises(ConfigError, match="server-reported-model-name"):
        resolve(
            env={},
            overrides={
                ("target", "transport"): "riva-grpc",
                ("target", "checkpoint_id"): "server-reported-model-name",
            },
        )
    with pytest.raises(ConfigError, match="server-reported-model-name"):
        resolve(
            env={},
            overrides={
                ("target", "transport"): "riva-grpc",
                ("target", "checkpoint_id"): "nemotron-asr-streaming",
                ("target", "model"): "server-reported-model-name",
            },
        )


def test_mock_run_measures_e2e_rtfx(tmp_path: Path) -> None:
    receipt = _run(tmp_path, "e2e")
    block = receipt["summary"]["pooled"]["e2e_rtfx"]
    assert block["availability"] == "measured"
    assert block["clock"] == MOCK_E2E_CLOCK
    assert block["n_with_measure"] == block["n_total"]
    assert block["n_total"] >= 1
    assert block["mean"] > 0


def test_e2e_rtfx_excludes_a_failed_early_final() -> None:
    ok = session(label="ok", audio_duration_s=1.6)
    ok.first_audio_sent_s = 0.0
    ok.final_received_s = 2.0
    failed = session(
        label="fail",
        transcript="early",
        error_code="UNAVAILABLE",
        terminal_status="errored",
        audio_duration_s=1.6,
    )
    failed.first_audio_sent_s = 0.0
    failed.final_received_s = 0.2
    block = _reduce([ok, failed])["e2e_rtfx"]
    assert block["mean"] == pytest.approx(1.6 / 2.0)
    assert block["n_with_measure"] == 1
    assert block["n_total"] == 2


def test_finalization_uses_last_audio_and_keeps_a_negative() -> None:
    outcome = session(final_latency_s=0.05)
    outcome.last_audio_sent_s = 1.0
    outcome.final_received_s = 1.4
    measured = _reduce([outcome])["finalization_s"]
    assert measured["mean"] == pytest.approx(0.4)
    early = session(label="early", final_latency_s=0.05)
    early.last_audio_sent_s = 2.0
    early.final_received_s = 1.5
    negative = _reduce([early])["finalization_s"]
    assert negative["mean"] == pytest.approx(-0.5)
    failed = session(
        label="fail",
        transcript="early",
        error_code="UNAVAILABLE",
        terminal_status="errored",
        final_latency_s=0.01,
    )
    failed.last_audio_sent_s = 1.0
    failed.final_received_s = 0.5
    pooled = _reduce([outcome, failed])["finalization_s"]
    assert pooled["mean"] == pytest.approx(0.4)
    assert pooled["n_with_measure"] == 1
    assert pooled["n_total"] == 2
