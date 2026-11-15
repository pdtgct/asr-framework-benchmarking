"""Measurement-review reducers: chunk coverage latency, server RTFx.

Chunk coverage runs against 1:1 and non-1:1 stand-in servers.
"""

import asyncio
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("grpc")
pytest.importorskip("riva.client.proto.riva_asr_pb2")
pytest.importorskip("riva.client.proto.riva_asr_pb2_grpc")

from asrbench.outcome import SessionOutcome
from asrbench.reducers import reduce_trial
from asrbench.spec import PassPolicy
from tests.riva_fake import FakeRiva, RunningFake
from tests.support import session
from tests.test_riva_grpc import _adapter, _request, _wav


def _stream(
    tmp_path: Path, mode: str, *, seconds: float = 0.6
) -> SessionOutcome:
    path = _wav(tmp_path, seconds)
    server = RunningFake(FakeRiva(mode=mode))
    try:
        return asyncio.run(
            _adapter(server.endpoint).run(
                _request(path, pace=True, interim_results=True)
            )
        )
    finally:
        server.stop()


def _reduce(outcomes: list[SessionOutcome], *, pacing: bool = True) -> Any:
    return reduce_trial(
        outcomes,
        wall_span_s=1.0,
        mode="streaming",
        pacing=pacing,
        lanes=len(outcomes),
        reference=None,
        pass_policy=PassPolicy(),
        percentile_method="floor_index",
        warmup_excluded=0,
    )


def test_one_to_one_server_matches_riva_compat_pair_for_pair(
    tmp_path: Path,
) -> None:
    outcome = _stream(tmp_path, "per-chunk")
    reduced = _reduce([outcome])
    compat = reduced["riva_client_compat_s"]
    chunk = reduced["chunk_latency_s"]
    assert compat["availability"] == "measured"
    assert chunk["n_samples"] == compat["n_samples"] == 6
    sends = outcome.chunk_send_times_s or []
    receives = outcome.response_receive_times_s or []
    pairs = sorted(receives[i] - sends[i] for i in range(len(sends)))
    assert chunk["minimum"] == pytest.approx(pairs[0])
    assert chunk["maximum"] == pytest.approx(pairs[-1])
    assert chunk["p50"] == pytest.approx(compat["p50"])
    assert chunk["mean"] == pytest.approx(compat["mean"])
    # No audio_processed on this server: a labelled lower bound.
    assert chunk["coverage_source"] == "send_watermark"
    assert chunk["availability"] == "measured_lower_bound"
    assert chunk["label"] == "per-chunk coverage latency"


def test_final_only_server_publishes_no_number(tmp_path: Path) -> None:
    outcome = _stream(tmp_path, "final")
    reduced = _reduce([outcome])
    chunk = reduced["chunk_latency_s"]
    assert reduced["riva_client_compat_s"]["availability"] == "invalid"
    assert chunk["availability"] == "final_only"
    assert chunk["reason"] == "server_sent_finals_only"
    assert chunk["p50"] is None
    assert chunk["n_samples"] == 0
    assert chunk["n_total"] == 1
    assert chunk["n_sessions_by_coverage"]["final_only"] == 1
    # finalization is still reported.
    assert reduced["finalization_s"]["n_with_measure"] == 1


def test_paired_interims_cannot_move_the_latency(tmp_path: Path) -> None:
    outcome = _stream(tmp_path, "paired-interims")
    receives = outcome.response_receive_times_s or []
    sends = outcome.chunk_send_times_s or []
    # 6 chunks, two interims each, one final: neither N nor N + 1.
    assert len(sends) == 6
    assert len(receives) == 13
    reduced = _reduce([outcome])
    assert reduced["riva_client_compat_s"]["availability"] == "invalid"
    chunk = reduced["chunk_latency_s"]
    assert chunk["availability"] == "measured_lower_bound"
    assert chunk["n_samples"] == 6
    assert chunk["n_bursts"] >= 1
    assert chunk["n_responses"] == 13
    # Drop the second of every pair: the first covering response wins,
    # so the statistic does not change.
    flags = outcome.response_final_flags or []
    keep = list(range(0, 13, 2))
    thinned = session(label="thinned", audio_duration_s=0.6)
    thinned.chunk_ms = outcome.chunk_ms
    thinned.chunk_send_times_s = list(sends)
    thinned.response_receive_times_s = [receives[i] for i in keep]
    thinned.response_final_flags = [flags[i] for i in keep]
    again = _reduce([thinned])["chunk_latency_s"]
    assert again["mean"] == pytest.approx(chunk["mean"])
    assert again["maximum"] == pytest.approx(chunk["maximum"])


def test_trusted_audio_processed_shows_the_server_step(tmp_path: Path) -> None:
    outcome = _stream(tmp_path, "cadence-frontier")
    reduced = _reduce([outcome])
    assert reduced["riva_client_compat_s"]["availability"] == "invalid"
    chunk = reduced["chunk_latency_s"]
    assert chunk["coverage_source"] == "audio_processed"
    assert chunk["availability"] == "measured"
    assert chunk["n_samples"] == 6
    # Odd chunks wait for the next chunk's send: about one chunk (100 ms).
    assert chunk["maximum"] >= 0.09


def test_untrusted_audio_processed_falls_back_to_the_watermark() -> None:
    outcome = session(label="final-frontier-only", audio_duration_s=0.3)
    outcome.chunk_ms = 100
    outcome.chunk_send_times_s = [1.1, 1.2, 1.3]
    outcome.response_receive_times_s = [1.15, 1.25, 1.35, 1.4]
    outcome.response_final_flags = [False, False, False, True]
    # Only the final carried audio_processed, as the bring-up frontend.
    outcome.audio_progress = [
        {"audio_processed_s": 0.3, "received_s": 1.4, "audio_sent_s": 0.3}
    ]
    chunk = _reduce([outcome])["chunk_latency_s"]
    assert chunk["coverage_source"] == "send_watermark"
    assert chunk["mean"] == pytest.approx(0.05)


def test_v5_outcome_without_final_flags_is_inferred() -> None:
    outcome = session(label="old", audio_duration_s=0.3, interim_texts=[])
    outcome.interim_count = 0
    outcome.chunk_ms = 100
    outcome.chunk_send_times_s = [1.1, 1.2, 1.3]
    outcome.response_receive_times_s = [1.5]
    chunk = _reduce([outcome])["chunk_latency_s"]
    assert chunk["availability"] == "final_only"
    assert chunk["n_sessions_final_flags_inferred"] == 1


def test_offline_mode_is_not_applicable() -> None:
    reduced = reduce_trial(
        [session(label="offline")],
        wall_span_s=1.0,
        mode="offline",
        pacing=False,
        lanes=1,
        reference=None,
        pass_policy=PassPolicy(),
        percentile_method="floor_index",
        warmup_excluded=0,
    )
    assert reduced["chunk_latency_s"]["availability"] == "not_applicable"
    assert reduced["riva_client_compat_s"]["availability"] == "not_applicable"


def _offline(
    label: str, clip_s: float, server_s: float | None
) -> SessionOutcome:
    outcome = session(label=label, audio_duration_s=clip_s)
    outcome.terminal_status = "completed"
    if server_s is not None:
        outcome.protocol_extras["server_audio_processed_s"] = server_s
    return outcome


def _reduce_offline(outcomes: list[SessionOutcome]) -> Any:
    return reduce_trial(
        outcomes,
        wall_span_s=2.0,
        mode="offline",
        pacing=False,
        lanes=len(outcomes),
        reference=None,
        pass_policy=PassPolicy(),
        percentile_method="floor_index",
        warmup_excluded=0,
    )


def test_server_audio_rtfx_and_its_gap_vs_clip() -> None:
    same = _reduce_offline([_offline("a", 1.0, 1.0), _offline("b", 1.0, 1.0)])
    block = same["server_audio_rtfx"]
    assert block["availability"] == "measured"
    assert block["value"] == pytest.approx(1.0)
    assert block["gap_vs_clip"] == pytest.approx(0.0)
    assert block["mismatch"] is False
    off = _reduce_offline([_offline("a", 1.0, 1.0), _offline("b", 1.0, 1.02)])
    block = off["server_audio_rtfx"]
    assert block["gap_vs_clip"] == pytest.approx(0.01)
    assert block["mismatch"] is True
    missing = _reduce_offline([_offline("a", 1.0, None)])
    assert missing["server_audio_rtfx"]["availability"] == "unavailable"
    assert missing["server_audio_rtfx"]["reason"] == (
        "server_audio_processed_absent"
    )
    streaming = _reduce([session(label="s")])["server_audio_rtfx"]
    assert streaming["availability"] == "not_applicable"
