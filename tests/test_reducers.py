"""Failures stay in every denominator, and unavailable metrics stay null."""

import pytest

from asrbench.definitions import (
    FINALIZATION_CLOCK,
    RESPONSE_LATENCY_CLOCK,
    SUSTAINED_STREAMS_UNAVAILABLE_REASON,
    TTFS_CLOCK,
)
from asrbench.outcome import SessionOutcome
from asrbench.reducers import reduce_trial
from asrbench.spec import PassPolicy
from tests.support import session


def _reduce(
    outcomes: list[SessionOutcome],
    *,
    reference: str | None = "hello",
    mode: str = "streaming",
    pacing: bool = True,
    wall_span_s: float = 1.0,
    pass_policy: PassPolicy | None = None,
    warmup_excluded: int = 0,
) -> dict[str, object]:
    return reduce_trial(
        outcomes,
        wall_span_s=wall_span_s,
        mode=mode,
        pacing=pacing,
        lanes=len(outcomes) or 1,
        reference=reference,
        pass_policy=pass_policy or PassPolicy(),
        percentile_method="numpy_linear",
        warmup_excluded=warmup_excluded,
    )


def test_old_outcome_constructor_still_works() -> None:
    outcome = SessionOutcome("s", "full", "clip.wav", 160, 1.0)
    assert outcome.interim_texts is None
    assert outcome.timed_out is False


def test_failures_stay_in_the_population_and_the_audio_denominator() -> None:
    success = session(label="ok")
    timed_out = session(
        label="timeout",
        transcript=None,
        ttfs_s=None,
        final_latency_s=None,
        response_latencies_s=[],
        interim_texts=None,
        timed_out=True,
        error_detail="TimeoutError: slow",
        protocol_extras={"timed_out": True, "exception": "TimeoutError: slow"},
    )
    errored = session(
        label="error",
        transcript=None,
        ttfs_s=0.3,
        final_latency_s=None,
        response_latencies_s=[0.3],
        interim_texts=None,
        protocol_errors=[{"message": "boom"}],
    )
    empty = session(
        label="empty",
        transcript="",
        ttfs_s=0.4,
        response_latencies_s=[0.4],
        interim_texts=[],
    )
    reduction = _reduce([success, timed_out, errored, empty])
    counts = reduction["counts"]
    assert isinstance(counts, dict)
    assert counts["attempted"] == 4
    assert counts["admitted"] == 4
    assert counts["completed"] == 2
    assert counts["productive"] == 1
    assert counts["errored"] == 1
    assert counts["timed_out"] == 1
    assert counts["no_first_partial"] == 1
    ttfs = reduction["ttfs_s"]
    assert isinstance(ttfs, dict)
    assert ttfs["n_with_measure"] == 3
    assert ttfs["n_total"] == 4
    assert ttfs["p50"] == 0.3
    assert ttfs["p50"] != 0
    wer = reduction["wer"]
    assert isinstance(wer, dict)
    assert wer["micro_wer"] == 0.75
    assert wer["n_with_measure"] == 2
    assert wer["n_total"] == 4
    assert wer["deletions"] == 1
    assert wer["deletions_from_missing_transcript"] == 2
    assert wer["n_failed"] == 2
    audio = reduction["audio"]
    assert isinstance(audio, dict)
    assert audio["attempted_audio_s"] == 6.4
    assert audio["successful_audio_s"] == 1.6
    conjunction = reduction["conjunction"]
    assert isinstance(conjunction, dict)
    assert "first_partial_presence" in conjunction["failed_clauses"]
    assert "sustainability" in conjunction["unevaluable_clauses"]
    assert conjunction["passed"] is False


def test_empty_percentile_is_none_and_warmup_is_not_in_n_total() -> None:
    quiet = session(
        transcript=None,
        ttfs_s=None,
        final_latency_s=None,
        response_latencies_s=[],
        interim_texts=None,
    )
    reduction = _reduce([quiet], warmup_excluded=2)
    ttfs = reduction["ttfs_s"]
    counts = reduction["counts"]
    assert isinstance(ttfs, dict)
    assert isinstance(counts, dict)
    assert ttfs["p50"] is None
    assert ttfs["availability"] == "no_samples"
    assert ttfs["n_total"] == 1
    assert counts["warmup_excluded"] == 2
    assert counts["attempted"] == 1


def test_no_outcomes_do_not_invent_a_zero_percentile() -> None:
    reduction = _reduce([])
    ttfs = reduction["ttfs_s"]
    assert isinstance(ttfs, dict)
    assert ttfs["p50"] is None
    assert ttfs["n_total"] == 0


def test_streaming_and_offline_rtfx_are_not_relabeled() -> None:
    streaming = _reduce([session()])
    offline = _reduce([session()], mode="offline", pacing=False)
    streaming_rtfx = streaming["streaming_rtfx"]
    offline_on_stream = streaming["offline_rtfx"]
    offline_rtfx = offline["offline_rtfx"]
    streaming_on_offline = offline["streaming_rtfx"]
    assert isinstance(streaming_rtfx, dict)
    assert isinstance(offline_on_stream, dict)
    assert isinstance(offline_rtfx, dict)
    assert isinstance(streaming_on_offline, dict)
    assert streaming_rtfx["value"] == 1.6
    assert streaming_rtfx["validity"] == "pacing_fidelity_unavailable"
    assert offline_on_stream["value"] is None
    assert offline_on_stream["reason"] == "mode_is_streaming"
    assert streaming_on_offline["value"] is None
    assert streaming_on_offline["reason"] == "mode_is_offline"
    assert offline_rtfx["validity"] == "offline_cell"
    assert offline_rtfx["value"] == 1.6


def test_non_positive_span_does_not_become_infinity() -> None:
    reduction = _reduce([session()], wall_span_s=0.0)
    rtfx = reduction["streaming_rtfx"]
    assert isinstance(rtfx, dict)
    assert rtfx["value"] is None
    assert rtfx["reason"] == "cell_wall_span_not_positive"


def test_mixed_clocks_are_not_pooled() -> None:
    left = session(label="a", ttfs_s=0.2, latency_clock=TTFS_CLOCK)
    right = session(label="b", ttfs_s=0.4, latency_clock="server_clock/other")
    reduction = _reduce([left, right])
    ttfs = reduction["ttfs_s"]
    assert isinstance(ttfs, dict)
    assert ttfs["availability"] == "mixed_clocks"
    assert ttfs["p50"] is None
    assert ttfs["n_with_measure"] == 2
    assert ttfs["n_total"] == 2
    assert ttfs["clock"] is None


def test_response_latency_is_not_substituted_for_ttfs() -> None:
    policy = PassPolicy(latency_metric="response_latency", latency_slo_s=0.1)
    reduction = _reduce(
        [session(response_latencies_s=[], ttfs_s=0.01)],
        pass_policy=policy,
    )
    conjunction = reduction["conjunction"]
    assert isinstance(conjunction, dict)
    clauses = conjunction["clauses"]
    assert isinstance(clauses, list)
    latency = next(item for item in clauses if item["name"] == "latency_slo")
    assert latency["status"] == "unevaluable"
    assert "not substituted" in latency["detail"]
    assert "response_latency" in latency["detail"]


def test_prefix_is_unavailable_until_interim_text_is_recorded() -> None:
    reduction = _reduce(
        [session(interim_texts=None, transcript="hello", ttfs_s=0.2)]
    )
    stability = reduction["stability"]
    assert isinstance(stability, dict)
    prefix = stability["prefix_violations"]
    assert isinstance(prefix, dict)
    assert prefix["availability"] == "unavailable"
    assert prefix["reason"] == "interim_texts_not_recorded"
    assert prefix["count"] is None


def test_prefix_violations_use_raw_strings() -> None:
    reduction = _reduce(
        [
            session(
                transcript="a b c",
                interim_texts=["a b", "a"],
                ttfs_s=0.2,
            )
        ]
    )
    stability = reduction["stability"]
    assert isinstance(stability, dict)
    prefix = stability["prefix_violations"]
    revision = stability["revision_rate"]
    assert isinstance(prefix, dict)
    assert isinstance(revision, dict)
    assert prefix["availability"] == "measured"
    assert prefix["count"] == 1
    assert revision["availability"] == "measured"
    assert revision["rate"] is not None


def test_lane_unavailable_stays_in_the_population() -> None:
    blocked = session(
        configured=False,
        transcript=None,
        ttfs_s=None,
        final_latency_s=None,
        response_latencies_s=[],
        interim_texts=None,
        terminal_status="lane_unavailable",
    )
    reduction = _reduce([blocked])
    counts = reduction["counts"]
    assert isinstance(counts, dict)
    assert counts["attempted"] == 1
    assert counts["admitted"] == 0
    assert counts["lane_unavailable"] == 1


def test_unavailable_metrics_are_emitted_and_not_zero() -> None:
    reduction = _reduce([session()])
    unavailable = reduction["unavailable"]
    assert isinstance(unavailable, list)
    by_name = {item["name"]: item for item in unavailable}
    assert by_name["emission_lag"]["availability"] == "unavailable"
    assert by_name["lag_excess"]["reason"] == "emission_lag_unavailable"
    assert (
        by_name["pacing_fidelity"]["reason"]
        == "client_send_trace_not_in_outcome"
    )
    assert (
        by_name["interim_cadence_conformance"]["reason"]
        == "interim_arrival_timestamps_not_in_outcome"
    )
    for item in unavailable:
        assert "value" not in item or item.get("value") is None
    sustained = reduction["sustained_streams_at_slo"]
    assert isinstance(sustained, dict)
    assert sustained["availability"] == "unavailable"
    assert sustained["reason"] == SUSTAINED_STREAMS_UNAVAILABLE_REASON
    assert "value" not in sustained


def test_completion_fraction_can_pass_when_steady_interval_cannot() -> None:
    policy = PassPolicy(
        sustainability_policy="completion_fraction",
        latency_percentile=50,
        latency_slo_s=1.0,
    )
    reduction = _reduce([session()], pass_policy=policy)
    conjunction = reduction["conjunction"]
    goodput = reduction["goodput"]
    assert isinstance(conjunction, dict)
    assert isinstance(goodput, dict)
    assert conjunction["passed"] is True
    assert conjunction["unevaluable_clauses"] == []
    assert goodput["availability"] == "measured"
    assert goodput["per_second"] == 1.0
    closed = _reduce([session()])
    closed_goodput = closed["goodput"]
    assert isinstance(closed_goodput, dict)
    assert closed_goodput["availability"] == "unavailable"
    assert closed_goodput["per_second"] is None


def test_protocol_residue_alone_is_not_a_timeout() -> None:
    residue = session(
        protocol_extras={"timed_out": True, "exception": "TimeoutError: slow"}
    )
    reduction = _reduce([residue])
    counts = reduction["counts"]
    assert isinstance(counts, dict)
    assert counts["timed_out"] == 0
    assert counts["errored"] == 0


def test_revision_rate_uses_newly_emitted_tokens() -> None:
    reduction = _reduce(
        [
            session(
                transcript="a b c",
                interim_texts=["a", "a x", "a b c"],
            )
        ]
    )
    stability = reduction["stability"]
    assert isinstance(stability, dict)
    revision = stability["revision_rate"]
    assert isinstance(revision, dict)
    assert revision["revised_tokens"] == 1
    assert revision["emitted_tokens"] == 4
    assert revision["rate"] == 0.25


def test_one_interim_can_fail_both_prefix_checks() -> None:
    reduction = _reduce(
        [session(transcript="a c", interim_texts=["a b", "a c"])]
    )
    stability = reduction["stability"]
    assert isinstance(stability, dict)
    prefix = stability["prefix_violations"]
    assert isinstance(prefix, dict)
    assert prefix["count"] == 2


def test_missing_transcript_is_not_an_alignment_deletion() -> None:
    reduction = _reduce(
        [
            session(label="ok", transcript="hello"),
            session(
                label="empty",
                transcript="",
                interim_texts=[],
            ),
            session(
                label="missing",
                transcript=None,
                ttfs_s=None,
                final_latency_s=None,
                response_latencies_s=[],
                interim_texts=None,
            ),
        ]
    )
    wer = reduction["wer"]
    assert isinstance(wer, dict)
    assert wer["micro_wer"] == pytest.approx(2 / 3)
    per = wer["per_utterance_wer"]
    assert isinstance(per, dict)
    assert per["mean"] == 0.5
    assert wer["n_with_measure"] == 2
    assert wer["n_failed"] == 1


def test_non_finite_latency_is_a_measurement_error() -> None:
    from asrbench.reducers import MeasurementError

    with pytest.raises(MeasurementError):
        _reduce([session(ttfs_s=float("nan"))])
    with pytest.raises(MeasurementError):
        _reduce([session(response_latencies_s=[float("inf")])])


def test_native_clock_constants_travel_with_the_distribution() -> None:
    reduction = _reduce([session()])
    ttfs = reduction["ttfs_s"]
    response = reduction["response_latency_s"]
    final = reduction["finalization_s"]
    assert isinstance(ttfs, dict)
    assert isinstance(response, dict)
    assert isinstance(final, dict)
    assert ttfs["clock"] == TTFS_CLOCK
    assert ttfs["clock_matches_native"] is True
    assert response["clock"] == RESPONSE_LATENCY_CLOCK
    assert response["n_samples"] == 1
    assert response["n_with_measure"] == 1
    assert final["clock"] == FINALIZATION_CLOCK
    assert ttfs["definition"]
    assert response["definition"] != ttfs["definition"]
