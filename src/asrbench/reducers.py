"""Reducers for the four metric families.

A reducer reads outcomes and returns numbers. It does not drop a session
that lacks a measure: the session stays in ``n_total``, and the percentile
is computed only over ``n_with_measure``. Metrics that need ground truth
this outcome shape does not carry are emitted as unavailable. They are
never filled with a zero.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable
from typing import Any

from asrbench import definitions as d
from asrbench.outcome import SessionOutcome
from asrbench.percentile import percentile
from asrbench.spec import PassPolicy
from asrbench.wer import score_texts

_NORM_TEXT = {
    "engine_vs_own_oracle": d.ORACLE_NORMALIZATION_DEFINITION,
    "corpus_reference": d.CORPUS_NORMALIZATION_DEFINITION,
}
_ACROSS = (
    ("ttfs_p50_s", "ttfs_s", "p50"),
    ("ttfs_p99_s", "ttfs_s", "p99"),
    ("response_latency_p50_s", "response_latency_s", "p50"),
    ("response_latency_p99_s", "response_latency_s", "p99"),
    ("finalization_p50_s", "finalization_s", "p50"),
    ("chunk_latency_p50_s", "chunk_latency_s", "p50"),
    ("chunk_latency_p99_s", "chunk_latency_s", "p99"),
    ("e2e_rtfx_mean", "e2e_rtfx", "mean"),
    ("aggregate_rtfx", "aggregate_rtfx", "value"),
    ("micro_wer", "wer", "micro_wer"),
    ("streaming_rtfx", "streaming_rtfx", "value"),
    ("offline_rtfx", "offline_rtfx", "value"),
    ("goodput_per_second", "goodput", "per_second"),
)
_WER_SUMS = (
    "substitutions",
    "deletions",
    "deletions_from_missing_transcript",
    "n_failed",
    "chars_from_missing_transcript",
    "insertions",
    "reference_tokens",
    "hypothesis_tokens",
    "reference_chars",
)
_RTFX = (
    (
        "streaming_rtfx",
        d.STREAMING_RTFX_DEFINITION,
        d.MODE_STREAMING,
        d.MODE_IS_OFFLINE_REASON,
        False,
    ),
    (
        "offline_rtfx",
        d.OFFLINE_RTFX_DEFINITION,
        d.MODE_OFFLINE,
        "mode_is_streaming",
        True,
    ),
)
_UNAVAILABLE = (
    (
        "emission_lag",
        d.EMISSION_LAG_UNAVAILABLE_REASON,
        d.EMISSION_LAG_CLOCK,
        d.EMISSION_LAG_DEFINITION,
    ),
    (
        "lag_excess",
        d.LAG_EXCESS_UNAVAILABLE_REASON,
        d.EMISSION_LAG_CLOCK,
        d.LAG_EXCESS_DEFINITION,
    ),
    (
        "interim_cadence_conformance",
        d.INTERIM_CADENCE_UNAVAILABLE_REASON,
        d.INTERIM_CADENCE_CLOCK,
        d.INTERIM_CADENCE_DEFINITION,
    ),
)
_STATS = ("mean", "minimum", "maximum", "p50", "p90", "p95", "p99")
_Reader = Callable[[SessionOutcome], tuple[list[float], str, bool]]


class MeasurementError(ValueError):
    """An outcome carried a non-finite latency into a reduction."""


def is_timed_out(outcome: SessionOutcome) -> bool:
    """Return whether ``timed_out`` is set. Protocol residue is not read."""
    return outcome.timed_out


def is_errored(outcome: SessionOutcome) -> bool:
    """Return whether the session errored and did not time out."""
    if outcome.timed_out:
        return False
    return bool(
        outcome.protocol_errors or outcome.error_code or outcome.error_detail
    )


def is_completed(outcome: SessionOutcome) -> bool:
    """Return whether a transcript arrived without error or timeout."""
    return (
        outcome.transcript is not None
        and not is_timed_out(outcome)
        and not is_errored(outcome)
    )


def is_productive(outcome: SessionOutcome) -> bool:
    """Return whether the session completed with a non-empty transcript."""
    return is_completed(outcome) and bool(outcome.transcript)


def has_first_partial(outcome: SessionOutcome) -> bool:
    """Return whether a non-empty partial was observed."""
    return outcome.ttfs_s is not None


def reduce_trial(
    outcomes: list[SessionOutcome],
    *,
    wall_span_s: float,
    mode: str,
    pacing: bool,
    lanes: int,
    reference: str | None,
    pass_policy: PassPolicy,
    percentile_method: str,
    warmup_excluded: int,
    span_pooling: str = "measured_cell",
    wer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reduce one trial's measured outcomes.

    Args:
        outcomes: Measured sessions only. Warmup is not in this list.
        wall_span_s: Measured cell span, warmup excluded.
        mode: ``streaming`` or ``offline``.
        pacing: Whether the workload asked for real-time pacing.
        lanes: Offered concurrency, reported beside sustained-streams.
        reference: Reference transcript, or ``None``.
        pass_policy: Conjunction thresholds and named policies.
        percentile_method: Named percentile policy.
        warmup_excluded: How many warmup sessions were left out.
        span_pooling: ``measured_cell`` or ``sum_of_trial_spans``.
        wer: Precomputed WER block. The pooled reduction passes the
            concatenation of trial rows so utterances are not scored twice.

    Returns:
        A JSON-ish reduction. Unavailable metrics are present and null,
        not omitted and not zero.

    Raises:
        MeasurementError: A latency sample is not finite.
    """
    _check_finite(outcomes)
    population = len(outcomes)
    ttfs, ttfs_samples = _measure(
        outcomes,
        name="ttfs_s",
        native_clock=d.TTFS_CLOCK,
        definition=d.TTFS_DEFINITION,
        method=percentile_method,
        population=population,
        read=_scalar_reader("ttfs_s", "latency_clock", d.TTFS_CLOCK),
    )
    response, response_samples = _measure(
        outcomes,
        name="response_latency_s",
        native_clock=d.RESPONSE_LATENCY_CLOCK,
        definition=d.RESPONSE_LATENCY_DEFINITION,
        method=percentile_method,
        population=population,
        read=_read_response,
    )
    finalization, _final_samples = _measure(
        outcomes,
        name="finalization_s",
        native_clock=d.FINALIZATION_CLOCK,
        definition=d.FINALIZATION_DEFINITION,
        method=percentile_method,
        population=population,
        read=_read_finalization,
    )
    if wer is None:
        wer = _wer(
            outcomes,
            reference,
            pass_policy.normalization_policy,
            percentile_method,
        )
    attempted, successful = _audio(outcomes)
    e2e, _e2e_samples = _measure(
        outcomes,
        name="e2e_rtfx",
        native_clock=d.E2E_RTFX_CLOCK,
        definition=d.E2E_RTFX_DEFINITION,
        method=percentile_method,
        population=population,
        read=_read_e2e,
    )
    e2e["unit"] = "ratio"
    frontier_rtfx = _frontier_rtfx(outcomes, percentile_method)
    frontier_lag = _frontier_lag(outcomes, percentile_method, pacing=pacing)
    word_latency = _word_latency(
        outcomes, percentile_method, pacing=pacing, mode=mode
    )
    aggregate = _aggregate_rtfx(
        attempted=attempted,
        successful=successful,
        wall_span_s=wall_span_s,
    )
    paced_streaming = pacing and mode != d.MODE_OFFLINE
    riva_compat = _riva_compat(
        outcomes, percentile_method, paced=paced_streaming, mode=mode
    )
    riva_intermediate = _riva_compat(
        outcomes,
        percentile_method,
        split="intermediate",
        paced=paced_streaming,
        mode=mode,
    )
    riva_final = _riva_compat(
        outcomes,
        percentile_method,
        split="final",
        paced=paced_streaming,
        mode=mode,
    )
    chunk_latency = _chunk_latency(
        outcomes, percentile_method, mode=mode, pacing=pacing
    )
    server_audio_rtfx = _server_audio_rtfx(
        outcomes, wall_span_s=wall_span_s, mode=mode
    )
    pacing_fidelity = _pacing_fidelity(
        outcomes, pacing=pacing, method=percentile_method, mode=mode
    )
    rtfx = _rtfx_blocks(
        mode=mode,
        pacing=pacing,
        attempted=attempted,
        successful=successful,
        wall_span_s=wall_span_s,
        span_pooling=span_pooling,
        simulated=_clocks_are_simulated(outcomes),
        pacing_validity=pacing_fidelity["validity"],
    )
    counts = _counts(outcomes, warmup_excluded)
    conjunction = _conjunction(
        outcomes,
        counts=counts,
        metrics={
            d.LATENCY_METRIC_TTFS: (ttfs, ttfs_samples),
            d.LATENCY_METRIC_RESPONSE: (response, response_samples),
            d.LATENCY_METRIC_FINALIZATION: (
                finalization,
                _final_samples,
            ),
            d.LATENCY_METRIC_FRONTIER_LAG: (
                frontier_lag,
                _lag_samples(outcomes, pacing=pacing),
            ),
        },
        wer=wer,
        reference=reference,
        pass_policy=pass_policy,
        percentile_method=percentile_method,
    )
    return {
        "counts": counts,
        "ttfs_s": ttfs,
        "response_latency_s": response,
        "finalization_s": finalization,
        "e2e_rtfx": e2e,
        "frontier_rtfx": frontier_rtfx,
        "frontier_lag_s": frontier_lag,
        "word_latency_s": word_latency,
        "aggregate_rtfx": aggregate,
        "riva_client_compat_s": riva_compat,
        "riva_client_compat_intermediate_s": riva_intermediate,
        "riva_client_compat_final_s": riva_final,
        "chunk_latency_s": chunk_latency,
        "server_audio_rtfx": server_audio_rtfx,
        "pacing_fidelity": pacing_fidelity,
        "wer": wer,
        "streaming_rtfx": rtfx["streaming_rtfx"],
        "offline_rtfx": rtfx["offline_rtfx"],
        "stability": {
            "prefix_violations": _prefix(outcomes),
            "revision_rate": _revision(outcomes),
        },
        "goodput": _goodput(
            outcomes,
            wall_span_s=wall_span_s,
            reference=reference,
            pass_policy=pass_policy,
            percentile_method=percentile_method,
            utterances=wer["utterances"],
        ),
        "sustained_streams_at_slo": {
            "name": "sustained_streams_at_slo",
            "availability": "unavailable",
            "reason": d.SUSTAINED_STREAMS_UNAVAILABLE_REASON,
            "definition": d.SUSTAINED_STREAMS_DEFINITION,
            "offered_lanes": lanes,
            "cell_conjunction_passed": conjunction["passed"],
            "clock": "",
        },
        "unavailable": _unavailable(),
        "conjunction": conjunction,
        "audio": {
            "definition": d.AUDIO_DENOMINATOR_DEFINITION,
            "attempted_audio_s": attempted,
            "successful_audio_s": successful,
            "productive_definition": d.PRODUCTIVE_DEFINITION,
        },
    }


def summarize_trials(
    reductions: list[dict[str, Any]],
    outcomes: list[SessionOutcome],
    *,
    wall_span_s: float,
    mode: str,
    pacing: bool,
    lanes: int,
    reference: str | None,
    pass_policy: PassPolicy,
    percentile_method: str,
    warmup_excluded: int,
) -> dict[str, Any]:
    """Summarize trials without dropping one that lacks a measure.

    The across-trial block is the mean of trial-level scalars. ``pooled``
    re-reduces every measured session. Its per-utterance rows are the
    trial rows concatenated, and its micro totals are the sums of the
    trial totals.

    Args:
        reductions: Per-trial reductions, in trial order.
        outcomes: Measured sessions from every trial, in the same order.
        wall_span_s: Sum of measured trial spans.
        mode: Workload mode.
        pacing: Workload pacing flag.
        lanes: Offered lanes.
        reference: Reference transcript, or ``None``.
        pass_policy: Conjunction policy.
        percentile_method: Named percentile policy.
        warmup_excluded: Warmup sessions across all trials.

    Returns:
        The receipt ``summary`` object.
    """
    across = {
        key: _across([item[section][field] for item in reductions])
        for key, section, field in _ACROSS
    }
    failed = list(
        dict.fromkeys(
            name
            for reduction in reductions
            for name in reduction["conjunction"]["failed_clauses"]
        )
    )
    unevaluable = list(
        dict.fromkeys(
            name
            for reduction in reductions
            for name in reduction["conjunction"]["unevaluable_clauses"]
        )
    )
    pooled = reduce_trial(
        outcomes,
        wall_span_s=wall_span_s,
        mode=mode,
        pacing=pacing,
        lanes=lanes,
        reference=reference,
        pass_policy=pass_policy,
        percentile_method=percentile_method,
        warmup_excluded=warmup_excluded,
        span_pooling="sum_of_trial_spans",
        wer=_pooled_wer(reductions, reference, pass_policy, percentile_method),
    )
    return {
        "n_trials": len(reductions),
        "conjunction_passed": bool(reductions)
        and all(item["conjunction"]["passed"] for item in reductions),
        "failed_clauses": failed,
        "unevaluable_clauses": unevaluable,
        "across_trials": across,
        "pooled": pooled,
    }


def _counts(
    outcomes: list[SessionOutcome], warmup_excluded: int
) -> dict[str, Any]:
    return {
        "definition": d.OUTCOME_ACCOUNTING_DEFINITION,
        "attempted": len(outcomes),
        "admitted": sum(1 for item in outcomes if item.configured),
        "completed": sum(1 for item in outcomes if is_completed(item)),
        "productive": sum(1 for item in outcomes if is_productive(item)),
        "errored": sum(1 for item in outcomes if is_errored(item)),
        "timed_out": sum(1 for item in outcomes if is_timed_out(item)),
        "no_first_partial": sum(
            1 for item in outcomes if not has_first_partial(item)
        ),
        "lane_unavailable": sum(
            1 for item in outcomes if item.terminal_status == "lane_unavailable"
        ),
        "warmup_excluded": warmup_excluded,
    }


def _audio(outcomes: list[SessionOutcome]) -> tuple[float, float]:
    attempted = sum(item.audio_duration_s for item in outcomes)
    successful = sum(
        item.audio_duration_s for item in outcomes if is_productive(item)
    )
    return attempted, successful


def _check_finite(outcomes: list[SessionOutcome]) -> None:
    for outcome in outcomes:
        for name, value in (
            ("ttfs_s", outcome.ttfs_s),
            ("final_latency_s", outcome.final_latency_s),
        ):
            if value is not None and not math.isfinite(value):
                raise MeasurementError(f"{outcome.label} {name} is not finite")
        for sample in outcome.response_latencies_s:
            if not math.isfinite(sample):
                raise MeasurementError(
                    f"{outcome.label} response latency is not finite"
                )
        for name, value in (
            ("first_audio_sent_s", outcome.first_audio_sent_s),
            ("last_audio_sent_s", outcome.last_audio_sent_s),
            ("final_received_s", outcome.final_received_s),
        ):
            if value is not None and not math.isfinite(value):
                raise MeasurementError(f"{outcome.label} {name} is not finite")
        _check_progress(outcome)
        _check_words(outcome)
        _check_trace(outcome, "chunk_send_times_s", outcome.chunk_send_times_s)
        _check_trace(
            outcome,
            "response_receive_times_s",
            outcome.response_receive_times_s,
        )


def _finite(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MeasurementError(f"{where} is not finite")
    number = float(value)
    if not math.isfinite(number):
        raise MeasurementError(f"{where} is not finite")
    return number


def _check_progress(outcome: SessionOutcome) -> None:
    if outcome.audio_progress is None:
        return
    for index, item in enumerate(outcome.audio_progress):
        where = f"{outcome.label} audio_progress[{index}]"
        for key in ("audio_processed_s", "received_s", "audio_sent_s"):
            if key not in item:
                raise MeasurementError(f"{where} missing {key}")
            _finite(item[key], f"{where}.{key}")


def _check_words(outcome: SessionOutcome) -> None:
    if outcome.word_timings is None:
        return
    for index, item in enumerate(outcome.word_timings):
        where = f"{outcome.label} word_timings[{index}]"
        for key in ("end_time_s", "final_received_s"):
            if key not in item:
                raise MeasurementError(f"{where} missing {key}")
            _finite(item[key], f"{where}.{key}")


def _check_trace(
    outcome: SessionOutcome, name: str, values: list[float] | None
) -> None:
    if values is None:
        return
    for index, value in enumerate(values):
        _finite(value, f"{outcome.label} {name}[{index}]")


def _e2e_clock(outcome: SessionOutcome) -> str:
    """Name the E2E clock. Mock stamps are simulated, not monotonic."""
    if (outcome.latency_clock or "").startswith("mock/"):
        return "mock/simulated_first_audio_sent_to_final_received"
    return d.E2E_RTFX_CLOCK


def _read_e2e(outcome: SessionOutcome) -> tuple[list[float], str, bool]:
    """E2E RTFx for a completed session. Failures stay in n_total only."""
    clock = _e2e_clock(outcome)
    if not is_completed(outcome):
        return [], clock, False
    start = outcome.first_audio_sent_s
    finish = outcome.final_received_s
    if start is None or finish is None:
        return [], clock, False
    span = finish - start
    if span <= 0:
        return [], clock, False
    return [outcome.audio_duration_s / span], clock, True


def _finalization_seconds(outcome: SessionOutcome) -> float | None:
    """Return t_f minus t_last when both stamps exist.

    The legacy ``final_latency_s`` is the fallback. A negative value is
    kept: the final arrived before the last audio sample.
    """
    last = outcome.last_audio_sent_s
    final = outcome.final_received_s
    if last is not None and final is not None:
        return final - last
    if outcome.final_latency_s is None:
        return None
    return float(outcome.final_latency_s)


def _read_finalization(
    outcome: SessionOutcome,
) -> tuple[list[float], str, bool]:
    """Finalization for a completed session. Failures stay in n_total."""
    if not is_completed(outcome):
        return [], d.FINALIZATION_CLOCK, False
    value = _finalization_seconds(outcome)
    if value is None:
        return [], d.FINALIZATION_CLOCK, False
    explicit = str(outcome.final_latency_clock or "")
    return [value], explicit or d.FINALIZATION_CLOCK, bool(explicit)


def _decreased(samples: list[float]) -> bool:
    return any(
        samples[index] < samples[index - 1] for index in range(1, len(samples))
    )


def _frontier_pairs(
    outcome: SessionOutcome,
) -> tuple[str, list[float]]:
    """Return ``invalid``, ``unavailable``, or ``measured`` and RTFx samples."""
    progress = outcome.audio_progress
    if progress is None:
        return "unavailable", []
    processed = [float(item["audio_processed_s"]) for item in progress]
    if _decreased(processed):
        return "invalid", []
    ratios: list[float] = []
    for index in range(1, len(progress)):
        delta_t = float(progress[index]["received_s"]) - float(
            progress[index - 1]["received_s"]
        )
        if delta_t <= 0:
            continue
        delta_p = processed[index] - processed[index - 1]
        ratios.append(delta_p / delta_t)
    return "measured", ratios


def _frontier_rtfx(
    outcomes: list[SessionOutcome], method: str
) -> dict[str, Any]:
    invalid = 0
    absent = 0
    measured_sessions = 0
    samples: list[float] = []
    for outcome in outcomes:
        status, ratios = _frontier_pairs(outcome)
        if status == "invalid":
            invalid += 1
            continue
        if status == "unavailable":
            absent += 1
            continue
        if not ratios:
            continue
        measured_sessions += 1
        samples.extend(ratios)
    publish = invalid == 0
    if invalid:
        availability = "invalid"
        reason: str | None = d.FRONTIER_RTFX_INVALID_REASON
        shown: list[float] = []
        with_measure = 0
    elif measured_sessions == 0 and absent == len(outcomes):
        availability = "unavailable"
        reason = d.FRONTIER_RTFX_UNAVAILABLE_REASON
        shown = []
        with_measure = 0
    elif measured_sessions == 0:
        availability = "no_samples"
        reason = (
            "audio_processed_absent_on_some_sessions"
            if absent
            else "audio_processed_has_no_interval"
        )
        shown = []
        with_measure = 0
    elif absent:
        availability = "partial"
        reason = "audio_processed_absent_on_some_sessions"
        shown = samples
        with_measure = measured_sessions
    else:
        availability = "measured"
        reason = None
        shown = samples
        with_measure = measured_sessions
    stats = _percentiles(shown, method) if shown else _blank_stats()
    return {
        "name": "frontier_rtfx",
        "clock": d.FRONTIER_RTFX_CLOCK,
        "definition": d.FRONTIER_RTFX_DEFINITION,
        "percentile_method": method,
        "availability": availability,
        "reason": reason,
        "unit": "ratio",
        **stats,
        "n_with_measure": with_measure,
        "n_total": len(outcomes),
        "n_samples": len(shown),
        "n_invalid_sessions": invalid,
        "n_unavailable_sessions": absent,
        "population_definition": d.POPULATION_DEFINITION,
        "published": publish,
    }


def _audio_origin(
    outcome: SessionOutcome, *, pacing: bool
) -> tuple[float | None, str]:
    """Return audio t=0 on the client clock and how it was found.

    See :data:`asrbench.definitions.AUDIO_CLOCK_ORIGIN_DEFINITION`.
    """
    if outcome.audio_clock_origin_s is not None:
        return outcome.audio_clock_origin_s, "schedule_origin"
    first = outcome.first_audio_sent_s
    if first is None:
        return None, "absent"
    if pacing:
        first_chunk = min(outcome.chunk_ms / 1000, outcome.audio_duration_s)
        return first - max(first_chunk, 0.0), "first_send_minus_first_chunk"
    return first, "first_send"


def _origin_counts(
    outcomes: list[SessionOutcome], *, pacing: bool
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for outcome in outcomes:
        _origin, source = _audio_origin(outcome, pacing=pacing)
        counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def _lags_for(
    outcome: SessionOutcome, *, pacing: bool = True
) -> tuple[str, list[float]]:
    """Return ``invalid``, ``unavailable``, or ``measured`` and lags.

    Lag is receive time minus the sum of the audio clock origin and
    ``audio_processed``. A decrease is invalid. A missing frontier or
    origin is unavailable. The audio duration sent is not this frontier.
    """
    progress = outcome.audio_progress
    if progress is None:
        return "unavailable", []
    processed = [float(item["audio_processed_s"]) for item in progress]
    if _decreased(processed):
        return "invalid", []
    origin, _source = _audio_origin(outcome, pacing=pacing)
    if origin is None:
        return "unavailable", []
    return "measured", [
        float(item["received_s"]) - (origin + processed[index])
        for index, item in enumerate(progress)
    ]


def _lag_samples(
    outcomes: list[SessionOutcome], *, pacing: bool = True
) -> list[float]:
    found: list[float] = []
    for outcome in outcomes:
        status, lags = _lags_for(outcome, pacing=pacing)
        if status == "measured":
            found.extend(lags)
    return found


def _pooled_clock(
    *,
    name: str,
    clock: str,
    definition: str,
    method: str,
    outcomes: list[SessionOutcome],
    read: Callable[[SessionOutcome], list[float] | None],
    absent_reason: str,
) -> dict[str, Any]:
    """Pool per-session sample lists. ``None`` from ``read`` is absent."""
    absent = 0
    with_measure = 0
    samples: list[float] = []
    for outcome in outcomes:
        values = read(outcome)
        if values is None:
            absent += 1
            continue
        if not values:
            continue
        with_measure += 1
        samples.extend(values)
    if with_measure == 0 and absent == len(outcomes):
        availability = "unavailable"
        reason: str | None = absent_reason
    elif with_measure == 0:
        availability = "no_samples"
        reason = None if absent == 0 else absent_reason
    elif absent:
        availability = "partial"
        reason = absent_reason
    else:
        availability = "measured"
        reason = None
    stats = _percentiles(samples, method) if samples else _blank_stats()
    return {
        "name": name,
        "clock": clock,
        "clock_attribution": "outcome" if samples else "none",
        "clock_matches_native": bool(samples),
        "definition": definition,
        "unit": "s",
        "percentile_method": method,
        "availability": availability,
        "reason": reason,
        **stats,
        "n_with_measure": with_measure,
        "n_total": len(outcomes),
        "n_samples": len(samples),
        "n_unavailable_sessions": absent,
        "population_definition": d.POPULATION_DEFINITION,
    }


def _frontier_lag(
    outcomes: list[SessionOutcome], method: str, *, pacing: bool = True
) -> dict[str, Any]:
    """Pool frontier lag. One decreasing stream publishes nothing."""
    invalid = 0
    absent = 0
    measured_sessions = 0
    samples: list[float] = []
    for outcome in outcomes:
        status, lags = _lags_for(outcome, pacing=pacing)
        if status == "invalid":
            invalid += 1
            continue
        if status == "unavailable":
            absent += 1
            continue
        if not lags:
            continue
        measured_sessions += 1
        samples.extend(lags)
    if invalid:
        availability = "invalid"
        reason: str | None = d.FRONTIER_LAG_INVALID_REASON
        shown: list[float] = []
        with_measure = 0
    elif measured_sessions == 0 and absent == len(outcomes):
        availability = "unavailable"
        reason = d.FRONTIER_LAG_UNAVAILABLE_REASON
        shown = []
        with_measure = 0
    elif measured_sessions == 0:
        availability = "no_samples"
        reason = d.FRONTIER_LAG_UNAVAILABLE_REASON if absent else None
        shown = []
        with_measure = 0
    elif absent:
        availability = "partial"
        reason = d.FRONTIER_LAG_UNAVAILABLE_REASON
        shown = samples
        with_measure = measured_sessions
    else:
        availability = "measured"
        reason = None
        shown = samples
        with_measure = measured_sessions
    stats = _percentiles(shown, method) if shown else _blank_stats()
    return {
        "name": "frontier_lag_s",
        "clock": d.FRONTIER_LAG_CLOCK,
        "clock_attribution": "outcome" if shown else "none",
        "clock_matches_native": bool(shown),
        "definition": d.FRONTIER_LAG_DEFINITION,
        "unit": "s",
        "percentile_method": method,
        "availability": availability,
        "reason": reason,
        **stats,
        "n_with_measure": with_measure,
        "n_total": len(outcomes),
        "n_samples": len(shown),
        "n_unavailable_sessions": absent,
        "population_definition": d.POPULATION_DEFINITION,
        "audio_clock_origin": _origin_counts(outcomes, pacing=pacing),
        "audio_clock_origin_definition": d.AUDIO_CLOCK_ORIGIN_DEFINITION,
    }


def _word_offsets(
    outcome: SessionOutcome, *, pacing: bool = True
) -> list[float] | None:
    if outcome.word_timings is None:
        return None
    origin, _source = _audio_origin(outcome, pacing=pacing)
    if origin is None:
        return None
    return [
        float(item["final_received_s"]) - (origin + float(item["end_time_s"]))
        for item in outcome.word_timings
    ]


def _word_latency(
    outcomes: list[SessionOutcome],
    method: str,
    *,
    pacing: bool = True,
    mode: str = d.MODE_STREAMING,
) -> dict[str, Any]:
    offline = mode == d.MODE_OFFLINE
    block = _pooled_clock(
        name="word_latency_s",
        clock=d.WORD_LATENCY_CLOCK,
        definition=d.WORD_LATENCY_DEFINITION,
        method=method,
        outcomes=outcomes,
        # Offline sends all audio at the call start, so there is no paced
        # audio clock to measure word ends against. Word timings stay on
        # the outcomes as raw data.
        read=(
            (lambda _outcome: None)
            if offline
            else (lambda outcome: _word_offsets(outcome, pacing=pacing))
        ),
        absent_reason=d.WORD_LATENCY_UNAVAILABLE_REASON,
    )
    block["audio_clock_origin"] = _origin_counts(outcomes, pacing=pacing)
    block["audio_clock_origin_definition"] = d.AUDIO_CLOCK_ORIGIN_DEFINITION
    if offline:
        block["availability"] = "not_applicable"
        block["reason"] = d.MODE_IS_OFFLINE_REASON
    return block


def _aggregate_rtfx(
    *,
    attempted: float,
    successful: float,
    wall_span_s: float,
) -> dict[str, Any]:
    if wall_span_s <= 0:
        value = None
        availability = "unavailable"
        reason: str | None = "benchmark_span_not_positive"
    else:
        value = attempted / wall_span_s
        availability = "measured"
        reason = None
    return {
        "name": "aggregate_rtfx",
        "availability": availability,
        "reason": reason,
        "value": value,
        "offered_audio_s": attempted,
        "successful_audio_s": successful,
        "benchmark_span_s": wall_span_s,
        "clock": d.AGGREGATE_RTFX_CLOCK,
        "definition": d.AGGREGATE_RTFX_DEFINITION,
    }


def _riva_pairs(
    outcome: SessionOutcome,
) -> tuple[str, list[float], list[bool] | None, bool]:
    """Pair response i with chunk i, as riva_streaming_asr_client does.

    Returns:
        Status (``measured``, ``invalid``, or ``unavailable``), one
        latency per chunk, the matching final flags or ``None`` when they
        were not recorded, and whether a trailing extra response was
        ignored.
    """
    sends = outcome.chunk_send_times_s
    receives = outcome.response_receive_times_s
    if sends is None or receives is None:
        return "unavailable", [], None, False
    if len(receives) not in (len(sends), len(sends) + 1):
        return "invalid", [], None, False
    pairs = [receives[index] - sends[index] for index in range(len(sends))]
    flags = outcome.response_final_flags
    paired_flags = (
        list(flags[: len(sends)])
        if flags is not None and len(flags) == len(receives)
        else None
    )
    return "measured", pairs, paired_flags, len(receives) == len(sends) + 1


def _riva_compat(
    outcomes: list[SessionOutcome],
    method: str,
    *,
    split: str | None = None,
    paced: bool = True,
    mode: str = d.MODE_STREAMING,
) -> dict[str, Any]:
    """Reduce Riva compat, pooled or one side of the is_final split.

    Args:
        outcomes: Measured sessions.
        method: Percentile policy.
        split: ``None`` for every pair, or ``intermediate`` / ``final``
            for the pairs whose response carried that is_final value.
        paced: Whether the streaming workload was paced in real time.
            ``riva_streaming_asr_client`` prints latency only under
            ``--simulate_realtime``, so an unpaced cell is
            ``not_applicable``.
        mode: Workload mode. Offline is ``not_applicable``.
    """
    invalid = 0
    absent = 0
    trailing = 0
    measured_sessions = 0
    samples: list[float] = []
    for outcome in outcomes:
        status, pairs, flags, extra = _riva_pairs(outcome)
        if status == "invalid":
            invalid += 1
            continue
        if status == "unavailable" or (split is not None and flags is None):
            absent += 1
            continue
        measured_sessions += 1
        trailing += int(extra)
        if split is None or flags is None:
            samples.extend(pairs)
            continue
        wanted = split == "final"
        samples.extend(
            value
            for value, final in zip(pairs, flags, strict=True)
            if final is wanted
        )
    absent_reason = (
        d.RIVA_CLIENT_COMPAT_UNAVAILABLE_REASON
        if split is None
        else d.RIVA_CLIENT_COMPAT_SPLIT_UNAVAILABLE_REASON
    )
    if mode == d.MODE_OFFLINE or not paced:
        availability = "not_applicable"
        reason: str | None = (
            d.MODE_IS_OFFLINE_REASON
            if mode == d.MODE_OFFLINE
            else d.RIVA_CLIENT_COMPAT_NOT_PACED_REASON
        )
        shown: list[float] = []
        with_measure = 0
    elif invalid:
        availability = "invalid"
        reason = d.RIVA_CLIENT_COMPAT_INVALID_REASON
        shown = []
        with_measure = 0
    elif measured_sessions == 0:
        availability = "unavailable"
        reason = absent_reason
        shown = []
        with_measure = 0
    elif absent:
        availability = "partial" if samples else "no_samples"
        reason = absent_reason
        shown = samples
        with_measure = measured_sessions
    elif samples:
        availability = "measured"
        reason = None
        shown = samples
        with_measure = measured_sessions
    else:
        availability = "no_samples"
        reason = None
        shown = []
        with_measure = measured_sessions
    stats = _percentiles(shown, method) if shown else _blank_stats()
    name = "riva_client_compat_s"
    definition = d.RIVA_CLIENT_COMPAT_DEFINITION
    if split is not None:
        name = f"riva_client_compat_{split}_s"
        definition = d.RIVA_CLIENT_COMPAT_SPLIT_DEFINITION
    return {
        "name": name,
        "label": d.RIVA_CLIENT_COMPAT_LABEL,
        "clock": d.RIVA_CLIENT_COMPAT_CLOCK,
        "definition": definition,
        "percentile_method": method,
        "availability": availability,
        "reason": reason,
        "unit": "s",
        **stats,
        "n_with_measure": with_measure,
        "n_total": len(outcomes),
        "n_samples": len(shown),
        "n_invalid_sessions": invalid,
        "n_unavailable_sessions": absent,
        "n_trailing_extra_ignored": trailing,
        "population_definition": d.POPULATION_DEFINITION,
    }


_COVER_EPSILON_S = 1e-5


def _response_flags(
    outcome: SessionOutcome, count: int
) -> tuple[list[bool], bool]:
    """Per-response final flags, and whether they had to be inferred."""
    flags = outcome.response_final_flags
    if flags is not None and len(flags) == count:
        return list(flags), False
    return [outcome.interim_count == 0] * count, True


def _trusted_frontier(
    outcome: SessionOutcome, receives: list[float], flags: list[bool]
) -> dict[float, float] | None:
    """Map receive time to ``audio_processed`` when it can be trusted.

    Trusted means present on every non-final response, never decreasing,
    and never beyond the audio sent by then plus the slack.
    """
    progress = outcome.audio_progress
    if not progress:
        return None
    slack = float(d.CHUNK_LATENCY_AUDIO_PROCESSED_SLACK_S)
    processed = [float(item["audio_processed_s"]) for item in progress]
    if _decreased(processed):
        return None
    by_time: dict[float, float] = {}
    for item in progress:
        value = float(item["audio_processed_s"])
        if value > float(item["audio_sent_s"]) + slack:
            return None
        by_time[float(item["received_s"])] = value
    for received, final in zip(receives, flags, strict=True):
        if not final and received not in by_time:
            return None
    return by_time


def _chunk_coverage(outcome: SessionOutcome) -> dict[str, Any]:
    """Per-chunk coverage latency for one session.

    Returns a dict with ``status`` (``unavailable``, ``no_responses``,
    ``final_only``, or ``measured``), ``source``, ``latencies``, and
    diagnostics.
    """
    sends = outcome.chunk_send_times_s
    receives = outcome.response_receive_times_s
    found: dict[str, Any] = {
        "status": "unavailable",
        "source": None,
        "latencies": [],
        "n_chunks": 0,
        "n_uncovered": 0,
        "n_by_final": 0,
        "n_responses": 0,
        "n_bursts": 0,
        "n_text_repeats": 0,
        "flags_inferred": False,
    }
    if sends is None or receives is None or not sends:
        return found
    burst_gap = float(d.CHUNK_LATENCY_BURST_GAP_S)
    texts = outcome.interim_texts or []
    found["n_responses"] = len(receives)
    found["n_bursts"] = sum(
        1
        for index in range(1, len(receives))
        if receives[index] - receives[index - 1] < burst_gap
    )
    found["n_text_repeats"] = sum(
        1 for index in range(1, len(texts)) if texts[index] == texts[index - 1]
    )
    found["n_chunks"] = len(sends)
    if not receives:
        # Chunks went out and nothing came back (an error or timeout before
        # the first response). That says nothing about where coverage came
        # from, so it is its own status, not a coverage source.
        found["status"] = "no_responses"
        found["n_uncovered"] = len(sends)
        return found
    flags, inferred = _response_flags(outcome, len(receives))
    found["flags_inferred"] = inferred
    if receives and all(flags):
        found["status"] = "final_only"
        return found
    ends = _chunk_ends(outcome, len(sends))
    frontier = _trusted_frontier(outcome, receives, flags)
    covered: list[float] = []
    for received in receives:
        value = frontier.get(received) if frontier is not None else None
        if value is None:
            last = bisect.bisect_right(sends, received) - 1
            value = ends[last] if last >= 0 else 0.0
        covered.append(value)
    latencies: list[float] = []
    for index, sent in enumerate(sends):
        hit = next(
            (
                j
                for j, received in enumerate(receives)
                if received >= sent
                and covered[j] >= ends[index] - _COVER_EPSILON_S
            ),
            None,
        )
        if hit is None:
            found["n_uncovered"] += 1
            continue
        latencies.append(receives[hit] - sent)
        found["n_by_final"] += int(flags[hit])
    found["status"] = "measured"
    found["source"] = (
        "audio_processed" if frontier is not None else "send_watermark"
    )
    found["latencies"] = latencies
    return found


def _chunk_latency(
    outcomes: list[SessionOutcome],
    method: str,
    *,
    mode: str = d.MODE_STREAMING,
    pacing: bool = True,
) -> dict[str, Any]:
    """Reduce per-chunk coverage latency (``chunk_latency_s``)."""
    sessions = [_chunk_coverage(outcome) for outcome in outcomes]
    by_source: dict[str, int] = {
        "audio_processed": 0,
        "send_watermark": 0,
        "final_only": 0,
        "no_responses": 0,
        "unavailable": 0,
    }
    samples: list[float] = []
    with_measure = 0
    for item in sessions:
        if item["status"] == "measured":
            by_source[item["source"]] += 1
            if item["latencies"]:
                with_measure += 1
                samples.extend(item["latencies"])
        else:
            by_source[item["status"]] += 1
    measured = by_source["audio_processed"] + by_source["send_watermark"]
    if mode == d.MODE_OFFLINE:
        availability = "not_applicable"
        reason: str | None = d.MODE_IS_OFFLINE_REASON
        samples = []
    elif not pacing:
        # Unpaced, every chunk leaves at about once, so coverage collapses
        # to first-response-minus-send. NVIDIA's client prints no latency
        # without --simulate_realtime; mirror riva_client_compat_s.
        availability = "not_applicable"
        reason = d.RIVA_CLIENT_COMPAT_NOT_PACED_REASON
        samples = []
    elif measured == 0 and by_source["final_only"]:
        availability = "final_only"
        reason = d.CHUNK_LATENCY_FINAL_ONLY_REASON
        samples = []
    elif measured == 0:
        availability = "unavailable"
        reason = d.CHUNK_LATENCY_UNAVAILABLE_REASON
    elif not samples:
        availability = "no_samples"
        reason = None
    elif by_source["send_watermark"]:
        availability = "measured_lower_bound"
        reason = None
    else:
        availability = "measured"
        reason = None
    if reason is None and by_source["no_responses"]:
        reason = d.CHUNK_LATENCY_NO_RESPONSES_REASON
    elif reason is None and (
        by_source["final_only"] or by_source["unavailable"]
    ):
        reason = (
            d.CHUNK_LATENCY_FINAL_ONLY_REASON
            if by_source["final_only"]
            else d.CHUNK_LATENCY_UNAVAILABLE_REASON
        )
    if by_source["audio_processed"] and by_source["send_watermark"]:
        source: str | None = "mixed"
    elif by_source["audio_processed"]:
        source = "audio_processed"
    elif by_source["send_watermark"]:
        source = "send_watermark"
    else:
        source = None
    stats = _percentiles(samples, method) if samples else _blank_stats()
    return {
        "name": "chunk_latency_s",
        "label": d.CHUNK_LATENCY_LABEL,
        "clock": d.CHUNK_LATENCY_CLOCK,
        "definition": d.CHUNK_LATENCY_DEFINITION,
        "percentile_method": method,
        "availability": availability,
        "reason": reason,
        "coverage_source": source,
        "unit": "s",
        **stats,
        "n_with_measure": with_measure,
        "n_total": len(outcomes),
        "n_samples": len(samples),
        "n_sessions_by_coverage": by_source,
        "n_chunks": sum(item["n_chunks"] for item in sessions),
        "n_chunks_uncovered": sum(item["n_uncovered"] for item in sessions),
        "n_chunks_covered_by_final": sum(
            item["n_by_final"] for item in sessions
        ),
        "n_responses": sum(item["n_responses"] for item in sessions),
        "n_bursts": sum(item["n_bursts"] for item in sessions),
        "n_text_repeats": sum(item["n_text_repeats"] for item in sessions),
        "n_sessions_final_flags_inferred": sum(
            int(item["flags_inferred"]) for item in sessions
        ),
        "final_flags_inferred_definition": (
            d.CHUNK_LATENCY_FINAL_FLAGS_INFERRED
        ),
        "population_definition": d.POPULATION_DEFINITION,
    }


def _server_audio_rtfx(
    outcomes: list[SessionOutcome], *, wall_span_s: float, mode: str
) -> dict[str, Any]:
    """Offline throughput on the server's ``audio_processed`` sum."""
    block: dict[str, Any] = {
        "name": "server_audio_rtfx",
        "clock": d.SERVER_AUDIO_RTFX_CLOCK,
        "definition": d.SERVER_AUDIO_RTFX_DEFINITION,
        "availability": "not_applicable",
        "reason": "mode_is_streaming",
        "value": None,
        "server_audio_s": None,
        "clip_audio_s": None,
        "gap_vs_clip": None,
        "mismatch": None,
        "mismatch_tolerance": float(d.SERVER_AUDIO_RTFX_MISMATCH_TOLERANCE),
        "n_completed": 0,
        "n_without_server_audio": 0,
        "benchmark_span_s": wall_span_s,
    }
    if mode != d.MODE_OFFLINE:
        return block
    completed = [item for item in outcomes if is_completed(item)]
    values = [
        item.protocol_extras.get("server_audio_processed_s")
        for item in completed
    ]
    present = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    block["n_completed"] = len(completed)
    block["n_without_server_audio"] = len(completed) - len(present)
    if not completed or len(present) != len(completed):
        block["availability"] = "unavailable"
        block["reason"] = d.SERVER_AUDIO_RTFX_UNAVAILABLE_REASON
        return block
    if wall_span_s <= 0:
        block["availability"] = "unavailable"
        block["reason"] = "benchmark_span_not_positive"
        return block
    server = sum(present)
    clip = sum(item.audio_duration_s for item in completed)
    block["availability"] = "measured"
    block["reason"] = None
    block["value"] = server / wall_span_s
    block["server_audio_s"] = server
    block["clip_audio_s"] = clip
    if clip > 0:
        gap = server / clip - 1.0
        block["gap_vs_clip"] = gap
        block["mismatch"] = abs(gap) > block["mismatch_tolerance"]
    return block


def _chunk_ends(outcome: SessionOutcome, count: int) -> list[float]:
    """Cumulative audio end of each of ``count`` chunks, in seconds."""
    step = outcome.chunk_ms / 1000
    total = outcome.audio_duration_s
    return [min(total, (index + 1) * step) for index in range(count)]


def _pacing_lags(outcome: SessionOutcome) -> list[float] | None:
    """Return every chunk's send lag behind origin + E_i, or ``None``."""
    sends = outcome.chunk_send_times_s
    if sends is None or not sends:
        return None
    origin, _source = _audio_origin(outcome, pacing=True)
    if origin is None:
        return None
    ends = _chunk_ends(outcome, len(sends))
    return [
        sends[index] - (origin + ends[index]) for index in range(len(sends))
    ]


def _pacing_tolerance_s(chunk_ms: int) -> float:
    return max(
        float(d.PACING_FIDELITY_TOLERANCE_FLOOR_S),
        float(d.PACING_FIDELITY_TOLERANCE_FRACTION) * chunk_ms / 1000,
    )


def _pacing_fidelity(
    outcomes: list[SessionOutcome],
    *,
    pacing: bool,
    method: str,
    mode: str = d.MODE_STREAMING,
) -> dict[str, Any]:
    """Measure how closely paced sessions kept to the chunk schedule."""
    offline = mode == d.MODE_OFFLINE
    pacing = pacing and not offline
    lags: list[float] = []
    session_max: list[float] = []
    tolerance: float | None = None
    rank = 0
    absent = 0
    for outcome in outcomes:
        found = _pacing_lags(outcome) if pacing else None
        if found is None:
            absent += 1
            continue
        lags.extend(found)
        worst_here = max(found) if found else 0.0
        session_max.append(worst_here)
        allowed = _pacing_tolerance_s(outcome.chunk_ms)
        tolerance = allowed if tolerance is None else min(tolerance, allowed)
        if worst_here > outcome.chunk_ms / 1000:
            rank = max(rank, 2)
        elif worst_here > allowed:
            rank = max(rank, 1)
    measured = len(session_max)
    reason: str | None = None
    if offline:
        availability = "not_applicable"
        reason = d.MODE_IS_OFFLINE_REASON
        validity = d.RTFX_VALIDITY_NOT_APPLICABLE
    elif not pacing:
        availability = "not_applicable"
        reason = d.PACING_FIDELITY_NOT_PACED_REASON
        validity = d.RTFX_VALIDITY_PACING_DISABLED
    elif measured == 0:
        availability = "unavailable"
        reason = d.PACING_FIDELITY_UNAVAILABLE_REASON
        validity = d.RTFX_VALIDITY_UNQUALIFIED
    else:
        availability = "partial" if absent else "measured"
        if absent:
            reason = d.PACING_FIDELITY_UNAVAILABLE_REASON
        validity = (
            d.RTFX_VALIDITY_PACED,
            d.RTFX_VALIDITY_PACING_LAGGED,
            d.RTFX_VALIDITY_UNPACED,
        )[rank]
    stats = _percentiles(lags, method) if lags else _blank_stats()
    return {
        "name": "pacing_fidelity",
        "availability": availability,
        "reason": reason,
        "validity": validity,
        "clock": d.PACING_FIDELITY_CLOCK,
        "definition": d.PACING_FIDELITY_DEFINITION,
        "tolerance_definition": d.PACING_FIDELITY_TOLERANCE_DEFINITION,
        "tolerance_s": tolerance,
        "audio_clock_origin": _origin_counts(outcomes, pacing=True)
        if pacing
        else {},
        "percentile_method": method,
        "unit": "s",
        "lag_s": stats,
        "max_session_lag_s": max(session_max) if session_max else None,
        "n_with_measure": measured,
        "n_total": len(outcomes),
        "n_samples": len(lags),
        "population_definition": d.POPULATION_DEFINITION,
    }


def _clocks_are_simulated(outcomes: list[SessionOutcome]) -> bool:
    clocks = [
        outcome.latency_clock for outcome in outcomes if outcome.latency_clock
    ]
    return bool(clocks) and all(clock.startswith("mock/") for clock in clocks)


def _common_prefix(left: list[str], right: list[str]) -> int:
    count = 0
    for first, second in zip(left, right, strict=False):
        if first != second:
            break
        count += 1
    return count


def _blank_stats() -> dict[str, None]:
    return {name: None for name in _STATS}


def _percentiles(samples: list[float], method: str) -> dict[str, Any]:
    return {
        "mean": sum(samples) / len(samples),
        "minimum": min(samples),
        "maximum": max(samples),
        "p50": percentile(samples, 50, method),
        "p90": percentile(samples, 90, method),
        "p95": percentile(samples, 95, method),
        "p99": percentile(samples, 99, method),
    }


def _distribution(
    *,
    name: str,
    clock: str | None,
    clock_attribution: str,
    clock_matches_native: bool,
    definition: str,
    unit: str,
    method: str,
    availability: str,
    samples: list[float],
    n_with_measure: int,
    n_total: int,
    n_samples: int,
) -> dict[str, Any]:
    stats = (
        _blank_stats()
        if availability != "measured"
        else _percentiles(samples, method)
    )
    return {
        "name": name,
        "clock": clock,
        "clock_attribution": clock_attribution,
        "clock_matches_native": clock_matches_native,
        "definition": definition,
        "unit": unit,
        "percentile_method": method,
        "availability": availability,
        **stats,
        "n_with_measure": n_with_measure,
        "n_total": n_total,
        "n_samples": n_samples,
        "population_definition": d.POPULATION_DEFINITION,
    }


def _clocked_distribution(
    *,
    name: str,
    native_clock: str,
    definition: str,
    method: str,
    entries: list[tuple[float, str, bool]],
    n_with_measure: int,
    n_total: int,
) -> dict[str, Any]:
    clocks = {clock for _, clock, _ in entries}
    samples = [value for value, _, _ in entries]
    n_samples = len(entries)
    if len(clocks) > 1:
        clock: str | None = None
        attribution = "mixed"
        matches = False
        availability = "mixed_clocks"
        samples = []
    elif not entries:
        clock = native_clock
        attribution = "none"
        matches = False
        availability = "no_samples"
        n_samples = 0
    else:
        clock = next(iter(clocks))
        explicit = all(flag for _, _, flag in entries)
        attribution = "outcome" if explicit else "native_default"
        matches = clock == native_clock
        availability = "measured"
    return _distribution(
        name=name,
        clock=clock,
        clock_attribution=attribution,
        clock_matches_native=matches,
        definition=definition,
        unit="s",
        method=method,
        availability=availability,
        samples=samples,
        n_with_measure=n_with_measure,
        n_total=n_total,
        n_samples=n_samples,
    )


def _scalar_reader(value_name: str, clock_name: str, native: str) -> _Reader:
    def read(outcome: SessionOutcome) -> tuple[list[float], str, bool]:
        value = getattr(outcome, value_name)
        if value is None:
            return [], native, False
        explicit = str(getattr(outcome, clock_name) or "")
        return [float(value)], explicit or native, bool(explicit)

    return read


def _read_response(outcome: SessionOutcome) -> tuple[list[float], str, bool]:
    if not outcome.response_latencies_s:
        return [], d.RESPONSE_LATENCY_CLOCK, False
    explicit = outcome.response_latency_clock or ""
    clock = explicit or d.RESPONSE_LATENCY_CLOCK
    values = [float(value) for value in outcome.response_latencies_s]
    return values, clock, bool(explicit)


def _measure(
    outcomes: list[SessionOutcome],
    *,
    name: str,
    native_clock: str,
    definition: str,
    method: str,
    population: int,
    read: _Reader,
) -> tuple[dict[str, Any], list[float]]:
    entries: list[tuple[float, str, bool]] = []
    sessions = 0
    for outcome in outcomes:
        values, clock, explicit = read(outcome)
        if not values:
            continue
        sessions += 1
        entries.extend((value, clock, explicit) for value in values)
    block = _clocked_distribution(
        name=name,
        native_clock=native_clock,
        definition=definition,
        method=method,
        entries=entries,
        n_with_measure=sessions,
        n_total=population,
    )
    return block, [value for value, _, _ in entries]


def _row(
    label: str,
    *,
    scored: bool,
    reason: str | None,
    wer: float | None = None,
    cer: float | None = None,
    substitutions: int = 0,
    deletions: int = 0,
    insertions: int = 0,
    reference_tokens: int = 0,
    missing: int | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "label": label,
        "scored": scored,
        "reason": reason,
        "wer": wer,
        "cer": cer,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "reference_tokens": reference_tokens,
    }
    if missing is not None:
        row["deletions_from_missing_transcript"] = missing
    return row


def _wer_block(
    *,
    policy: str,
    availability: str,
    reason: str | None,
    micro_wer: float | None,
    micro_cer: float | None,
    substitutions: int,
    deletions: int,
    missing: int,
    n_failed: int,
    chars_missing: int,
    insertions: int,
    reference_tokens: int,
    hypothesis_tokens: int,
    reference_chars: int,
    samples: list[float],
    n_total: int,
    method: str,
    utterances: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "policy": policy,
        "definition": d.WER_DEFINITION,
        "normalization_definition": _NORM_TEXT[policy],
        "availability": availability,
        "reason": reason,
        "micro_wer": micro_wer,
        "micro_cer": micro_cer,
        "cer_definition": d.CER_DEFINITION,
        "sdi_definition": d.SDI_DEFINITION,
        "substitutions": substitutions,
        "deletions": deletions,
        "deletions_from_missing_transcript": missing,
        "n_failed": n_failed,
        "chars_from_missing_transcript": chars_missing,
        "insertions": insertions,
        "reference_tokens": reference_tokens,
        "hypothesis_tokens": hypothesis_tokens,
        "reference_chars": reference_chars,
        "n_with_measure": len(samples),
        "n_total": n_total,
        "per_utterance_wer": _rate_distribution(
            name="per_utterance_wer",
            definition=d.PER_UTTERANCE_WER_DEFINITION,
            method=method,
            samples=samples,
            n_total=n_total,
        ),
        "utterances": utterances,
    }


def _wer(
    outcomes: list[SessionOutcome],
    reference: str | None,
    policy: str,
    method: str,
) -> dict[str, Any]:
    utterances: list[dict[str, Any]] = []
    substitutions = deletions = insertions = 0
    reference_tokens = hypothesis_tokens = reference_chars = 0
    char_edits = missing = chars_missing = n_failed = 0
    samples: list[float] = []
    if reference is None:
        for outcome in outcomes:
            utterances.append(
                _row(
                    outcome.label,
                    scored=False,
                    reason="no_reference_transcript",
                )
            )
        block = _wer_block(
            policy=policy,
            availability="no_reference",
            reason="no_reference_transcript",
            micro_wer=None,
            micro_cer=None,
            substitutions=0,
            deletions=0,
            missing=0,
            n_failed=0,
            chars_missing=0,
            insertions=0,
            reference_tokens=0,
            hypothesis_tokens=0,
            reference_chars=0,
            samples=[],
            n_total=len(outcomes),
            method=method,
            utterances=utterances,
        )
        block["_char_edits"] = 0
        return block
    for outcome in outcomes:
        if outcome.transcript is None:
            blank = score_texts(reference, "", policy)
            n_failed += 1
            missing += blank.reference_tokens
            chars_missing += blank.reference_chars
            reference_tokens += blank.reference_tokens
            reference_chars += blank.reference_chars
            utterances.append(
                _row(
                    outcome.label,
                    scored=False,
                    reason="missing_transcript",
                    reference_tokens=blank.reference_tokens,
                    missing=blank.reference_tokens,
                )
            )
            continue
        scored = score_texts(reference, outcome.transcript, policy)
        substitutions += scored.substitutions
        deletions += scored.deletions
        insertions += scored.insertions
        reference_tokens += scored.reference_tokens
        hypothesis_tokens += scored.hypothesis_tokens
        reference_chars += scored.reference_chars
        char_edits += scored.char_edits
        if scored.wer is not None:
            samples.append(scored.wer)
        utterances.append(
            _row(
                outcome.label,
                scored=True,
                reason=None if scored.wer is not None else "undefined_rate",
                wer=scored.wer,
                cer=scored.cer,
                substitutions=scored.substitutions,
                deletions=scored.deletions,
                insertions=scored.insertions,
                reference_tokens=scored.reference_tokens,
                missing=0,
            )
        )
    micro_wer = _rate(
        substitutions + deletions + insertions + missing, reference_tokens
    )
    micro_cer = _rate(char_edits + chars_missing, reference_chars)
    if micro_wer is None and not outcomes:
        availability, reason = "no_samples", "no_measured_sessions"
    elif micro_wer is None:
        availability, reason = "undefined", "reference_word_denominator_is_zero"
    else:
        availability, reason = "measured", None
    block = _wer_block(
        policy=policy,
        availability=availability,
        reason=reason,
        micro_wer=micro_wer,
        micro_cer=micro_cer,
        substitutions=substitutions,
        deletions=deletions,
        missing=missing,
        n_failed=n_failed,
        chars_missing=chars_missing,
        insertions=insertions,
        reference_tokens=reference_tokens,
        hypothesis_tokens=hypothesis_tokens,
        reference_chars=reference_chars,
        samples=samples,
        n_total=len(outcomes),
        method=method,
        utterances=utterances,
    )
    block["_char_edits"] = char_edits
    return block


def _pooled_wer(
    reductions: list[dict[str, Any]],
    reference: str | None,
    pass_policy: PassPolicy,
    method: str,
) -> dict[str, Any]:
    policy = pass_policy.normalization_policy
    if not reductions:
        block = _wer([], reference, policy, method)
        block.pop("_char_edits", None)
        return block
    rows: list[dict[str, Any]] = []
    sums = {name: 0 for name in _WER_SUMS}
    char_edits = 0
    for reduction in reductions:
        block = reduction["wer"]
        char_edits += int(block.pop("_char_edits"))
        rows.extend(block["utterances"])
        for name in _WER_SUMS:
            sums[name] += int(block[name])
    samples = [
        float(row["wer"])
        for row in rows
        if isinstance(row["wer"], (int, float))
        and not isinstance(row["wer"], bool)
    ]
    if reference is None:
        return _wer_block(
            policy=policy,
            availability="no_reference",
            reason="no_reference_transcript",
            micro_wer=None,
            micro_cer=None,
            substitutions=0,
            deletions=0,
            missing=0,
            n_failed=0,
            chars_missing=0,
            insertions=0,
            reference_tokens=0,
            hypothesis_tokens=0,
            reference_chars=0,
            samples=[],
            n_total=len(rows),
            method=method,
            utterances=rows,
        )
    micro_wer = _rate(
        sums["substitutions"]
        + sums["deletions"]
        + sums["insertions"]
        + sums["deletions_from_missing_transcript"],
        sums["reference_tokens"],
    )
    micro_cer = _rate(
        char_edits + sums["chars_from_missing_transcript"],
        sums["reference_chars"],
    )
    if micro_wer is None and not rows:
        availability, reason = "no_samples", "no_measured_sessions"
    elif micro_wer is None:
        availability, reason = "undefined", "reference_word_denominator_is_zero"
    else:
        availability, reason = "measured", None
    return _wer_block(
        policy=policy,
        availability=availability,
        reason=reason,
        micro_wer=micro_wer,
        micro_cer=micro_cer,
        substitutions=sums["substitutions"],
        deletions=sums["deletions"],
        missing=sums["deletions_from_missing_transcript"],
        n_failed=sums["n_failed"],
        chars_missing=sums["chars_from_missing_transcript"],
        insertions=sums["insertions"],
        reference_tokens=sums["reference_tokens"],
        hypothesis_tokens=sums["hypothesis_tokens"],
        reference_chars=sums["reference_chars"],
        samples=samples,
        n_total=len(rows),
        method=method,
        utterances=rows,
    )


def _rate(edits: int, denominator: int) -> float | None:
    if denominator == 0:
        if edits == 0:
            return 0.0
        return None
    return edits / denominator


def _rate_distribution(
    *,
    name: str,
    definition: str,
    method: str,
    samples: list[float],
    n_total: int,
) -> dict[str, Any]:
    stats = _percentiles(samples, method) if samples else _blank_stats()
    return {
        "name": name,
        "definition": definition,
        "percentile_method": method,
        "availability": "measured" if samples else "no_samples",
        "mean": stats["mean"],
        "p50": stats["p50"],
        "p90": stats["p90"],
        "p95": stats["p95"],
        "p99": stats["p99"],
        "n_with_measure": len(samples),
        "n_total": n_total,
        "population_definition": d.POPULATION_DEFINITION,
    }


def _rtfx_blocks(
    *,
    mode: str,
    pacing: bool,
    attempted: float,
    successful: float,
    wall_span_s: float,
    span_pooling: str,
    simulated: bool,
    pacing_validity: str = d.RTFX_VALIDITY_UNQUALIFIED,
) -> dict[str, dict[str, Any]]:
    built: dict[str, dict[str, Any]] = {}
    for name, definition, applicable_mode, reason, offline in _RTFX:
        built[name] = _rtfx(
            name=name,
            definition=definition,
            applicable=mode == applicable_mode,
            inapplicable_reason=reason,
            pacing=pacing,
            attempted=attempted,
            successful=successful,
            wall_span_s=wall_span_s,
            span_pooling=span_pooling,
            offline=offline,
            simulated=simulated,
            pacing_validity=pacing_validity,
        )
    return built


def _rtfx(
    *,
    name: str,
    definition: str,
    applicable: bool,
    inapplicable_reason: str,
    pacing: bool,
    attempted: float,
    successful: float,
    wall_span_s: float,
    span_pooling: str,
    offline: bool,
    simulated: bool,
    pacing_validity: str,
) -> dict[str, Any]:
    availability = "unavailable"
    reason: str | None = inapplicable_reason
    value = None
    validity = d.RTFX_VALIDITY_NOT_APPLICABLE
    if applicable and wall_span_s <= 0:
        reason = "cell_wall_span_not_positive"
        validity = d.RTFX_VALIDITY_UNQUALIFIED
    elif applicable:
        availability = "measured"
        reason = None
        value = successful / wall_span_s
        if simulated:
            validity = d.RTFX_VALIDITY_SIMULATED
        elif offline:
            validity = "offline_cell"
        elif not pacing:
            validity = d.RTFX_VALIDITY_PACING_DISABLED
        else:
            validity = pacing_validity
    return {
        "name": name,
        "availability": availability,
        "reason": reason,
        "value": value,
        "successful_audio_s": successful,
        "attempted_audio_s": attempted,
        "cell_wall_span_s": wall_span_s,
        "span_pooling": span_pooling,
        "clock": d.STREAMING_RTFX_CLOCK,
        "definition": definition,
        "pacing": pacing,
        "validity": validity,
    }


def _prefix(outcomes: list[SessionOutcome]) -> dict[str, Any]:
    recorded = [item for item in outcomes if item.interim_texts is not None]
    if not recorded:
        return {
            "availability": "unavailable",
            "reason": "interim_texts_not_recorded",
            "count": None,
            "pairs_checked": 0,
            "final_checked": False,
            "definition": d.PREFIX_VIOLATION_DEFINITION,
        }
    count = 0
    pairs = 0
    final_checked = True
    for outcome in recorded:
        texts = outcome.interim_texts or []
        for left, right in zip(texts, texts[1:], strict=False):
            pairs += 1
            if not right.startswith(left):
                count += 1
        if outcome.transcript is None:
            final_checked = False
            continue
        for text in texts:
            pairs += 1
            if not outcome.transcript.startswith(text):
                count += 1
    partial = len(recorded) != len(outcomes) or not final_checked
    reason = None
    if len(recorded) != len(outcomes):
        reason = "some_sessions_did_not_record_interim_texts"
    elif not final_checked:
        reason = "final_transcript_missing"
    return {
        "availability": "partial" if partial else "measured",
        "reason": reason,
        "count": count,
        "pairs_checked": pairs,
        "final_checked": final_checked,
        "definition": d.PREFIX_VIOLATION_DEFINITION,
    }


def _revision(outcomes: list[SessionOutcome]) -> dict[str, Any]:
    if not outcomes or all(item.interim_texts is None for item in outcomes):
        return {
            "availability": "unavailable",
            "reason": "interim_texts_not_recorded",
            "rate": None,
            "revised_tokens": None,
            "emitted_tokens": None,
            "definition": d.REVISION_RATE_DEFINITION,
        }
    revised = 0
    emitted = 0
    final_missing = False
    for outcome in outcomes:
        if outcome.interim_texts is None:
            continue
        sequences = [text.split() for text in outcome.interim_texts]
        if outcome.transcript is None:
            final_tokens = None
        else:
            final_tokens = outcome.transcript.split()
        if final_tokens is None and sequences:
            final_missing = True
        previous: list[str] | None = None
        for index, sequence in enumerate(sequences):
            if previous is None:
                emitted += len(sequence)
            else:
                emitted += len(sequence) - _common_prefix(previous, sequence)
            previous = sequence
            if index + 1 < len(sequences):
                nxt = sequences[index + 1]
            elif final_tokens is None:
                continue
            else:
                nxt = final_tokens
            revised += len(sequence) - _common_prefix(sequence, nxt)
    rate = revised / emitted if emitted else None
    return {
        "availability": "partial" if final_missing else "measured",
        "reason": "final_transcript_missing" if final_missing else None,
        "rate": rate,
        "revised_tokens": revised,
        "emitted_tokens": emitted,
        "definition": d.REVISION_RATE_DEFINITION,
    }


def _unavailable() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "availability": "unavailable",
            "reason": reason,
            "clock": clock,
            "definition": definition,
        }
        for name, reason, clock, definition in _UNAVAILABLE
    ]


def _conjunction(
    outcomes: list[SessionOutcome],
    *,
    counts: dict[str, Any],
    metrics: dict[str, tuple[dict[str, Any], list[float]]],
    wer: dict[str, Any],
    reference: str | None,
    pass_policy: PassPolicy,
    percentile_method: str,
) -> dict[str, Any]:
    total = len(outcomes)
    clauses = [
        _fraction_clause(
            "admission_fraction",
            observed=None if total == 0 else counts["admitted"] / total,
            threshold=pass_policy.admission_fraction_min,
            empty=total == 0,
        ),
        _fraction_clause(
            "first_partial_presence",
            observed=None
            if total == 0
            else (total - counts["no_first_partial"]) / total,
            threshold=pass_policy.first_partial_presence_min,
            empty=total == 0,
        ),
        _latency_clause(
            metrics,
            pass_policy=pass_policy,
            percentile_method=percentile_method,
        ),
        _sustainability_clause(counts, total, pass_policy),
        _accuracy_clause(wer, reference, pass_policy),
    ]
    failed = [
        clause["name"] for clause in clauses if clause["status"] == "failed"
    ]
    unevaluable = [
        clause["name"]
        for clause in clauses
        if clause["status"] == "unevaluable"
    ]
    return {
        "definition": d.PASS_CONJUNCTION_DEFINITION,
        "passed": bool(clauses) and not failed and not unevaluable,
        "failed_clauses": failed,
        "unevaluable_clauses": unevaluable,
        "clauses": clauses,
    }


def _fraction_clause(
    name: str, *, observed: float | None, threshold: float, empty: bool
) -> dict[str, Any]:
    if empty or observed is None:
        return _clause(
            name, "unevaluable", observed, threshold, "no measured sessions"
        )
    status = "passed" if observed >= threshold else "failed"
    return _clause(
        name, status, observed, threshold, f"{observed} >= {threshold}"
    )


def _latency_clause(
    metrics: dict[str, tuple[dict[str, Any], list[float]]],
    *,
    pass_policy: PassPolicy,
    percentile_method: str,
) -> dict[str, Any]:
    metric = pass_policy.latency_metric
    chosen, samples = metrics[metric]
    if chosen["availability"] != "measured":
        return _clause(
            "latency_slo",
            "unevaluable",
            None,
            pass_policy.latency_slo_s,
            (
                f"{metric} unavailable ({chosen['availability']}); "
                "not substituted with the other latency"
            ),
        )
    observed = percentile(
        samples, pass_policy.latency_percentile, percentile_method
    )
    if observed is None:
        return _clause(
            "latency_slo",
            "unevaluable",
            None,
            pass_policy.latency_slo_s,
            f"{metric} has no samples",
        )
    status = "passed" if observed <= pass_policy.latency_slo_s else "failed"
    return _clause(
        "latency_slo",
        status,
        observed,
        pass_policy.latency_slo_s,
        (
            f"{metric} p{pass_policy.latency_percentile:g} on "
            f"{chosen['clock']} is {observed} against SLO "
            f"{pass_policy.latency_slo_s}"
        ),
    )


def _sustainability_clause(
    counts: dict[str, Any], total: int, pass_policy: PassPolicy
) -> dict[str, Any]:
    policy = pass_policy.sustainability_policy

    def steady() -> dict[str, Any]:
        return _clause(
            "sustainability",
            "unevaluable",
            None,
            pass_policy.sustainability_min,
            d.STEADY_INTERVAL_DEFINITION
            + " "
            + d.STEADY_INTERVAL_UNAVAILABLE_REASON,
        )

    def completion() -> dict[str, Any]:
        if total == 0:
            return _clause(
                "sustainability",
                "unevaluable",
                None,
                pass_policy.sustainability_min,
                "no measured sessions",
            )
        observed = counts["completed"] / total
        status = (
            "passed" if observed >= pass_policy.sustainability_min else "failed"
        )
        return _clause(
            "sustainability",
            status,
            observed,
            pass_policy.sustainability_min,
            d.COMPLETION_FRACTION_DEFINITION,
        )

    return {
        d.SUSTAINABILITY_STEADY_INTERVAL: steady,
        d.SUSTAINABILITY_COMPLETION_FRACTION: completion,
    }[policy]()


def _accuracy_clause(
    wer: dict[str, Any], reference: str | None, pass_policy: PassPolicy
) -> dict[str, Any]:
    if reference is None or wer["availability"] == "no_reference":
        return _clause(
            "accuracy",
            "unevaluable",
            None,
            pass_policy.accuracy_tolerance,
            f"no reference under policy {pass_policy.normalization_policy}",
        )
    observed = wer["micro_wer"]
    if observed is None:
        return _clause(
            "accuracy",
            "unevaluable",
            None,
            pass_policy.accuracy_tolerance,
            f"{wer['policy']} word error rate is undefined",
        )
    status = (
        "passed" if observed <= pass_policy.accuracy_tolerance else "failed"
    )
    return _clause(
        "accuracy",
        status,
        observed,
        pass_policy.accuracy_tolerance,
        (
            f"{wer['policy']} micro WER {observed} "
            f"<= {pass_policy.accuracy_tolerance}"
        ),
    )


def _clause(
    name: str,
    status: str,
    observed: float | None,
    threshold: float | None,
    detail: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "observed": observed,
        "threshold": threshold,
        "detail": detail,
    }


def _goodput(
    outcomes: list[SessionOutcome],
    *,
    wall_span_s: float,
    reference: str | None,
    pass_policy: PassPolicy,
    percentile_method: str,
    utterances: list[dict[str, Any]],
) -> dict[str, Any]:
    base = {
        "name": "goodput",
        "definition": d.GOODPUT_DEFINITION,
        "n_total": len(outcomes),
        "cell_wall_span_s": wall_span_s,
    }
    if pass_policy.sustainability_policy == d.SUSTAINABILITY_STEADY_INTERVAL:
        return {
            **base,
            "availability": "unavailable",
            "reason": d.STEADY_INTERVAL_UNAVAILABLE_REASON,
            "streams_meeting_slo": None,
            "per_second": None,
        }
    by_label = {str(row["label"]): row for row in utterances}
    meeting = sum(
        1
        for outcome in outcomes
        if _session_meets(
            outcome,
            reference=reference,
            pass_policy=pass_policy,
            percentile_method=percentile_method,
            scored=by_label.get(outcome.label),
        )
    )
    if wall_span_s <= 0:
        return {
            **base,
            "availability": "unavailable",
            "reason": "cell_wall_span_not_positive",
            "streams_meeting_slo": meeting,
            "per_second": None,
        }
    return {
        **base,
        "availability": "measured",
        "reason": None,
        "streams_meeting_slo": meeting,
        "per_second": meeting / wall_span_s,
    }


def _session_meets(
    outcome: SessionOutcome,
    *,
    reference: str | None,
    pass_policy: PassPolicy,
    percentile_method: str,
    scored: dict[str, Any] | None,
) -> bool:
    if not outcome.configured or not has_first_partial(outcome):
        return False
    if not is_completed(outcome):
        return False
    if pass_policy.latency_metric == d.LATENCY_METRIC_TTFS:
        if outcome.ttfs_s is None or outcome.ttfs_s > pass_policy.latency_slo_s:
            return False
    elif pass_policy.latency_metric == d.LATENCY_METRIC_FINALIZATION:
        latency = _finalization_seconds(outcome)
        if latency is None or latency > pass_policy.latency_slo_s:
            return False
    elif pass_policy.latency_metric == d.LATENCY_METRIC_FRONTIER_LAG:
        status, lags = _lags_for(outcome)
        if status != "measured" or not lags:
            return False
        observed = percentile(
            lags, pass_policy.latency_percentile, percentile_method
        )
        if observed is None or observed > pass_policy.latency_slo_s:
            return False
    else:
        if not outcome.response_latencies_s:
            return False
        value = percentile(
            outcome.response_latencies_s,
            pass_policy.latency_percentile,
            percentile_method,
        )
        if value is None or value > pass_policy.latency_slo_s:
            return False
    if reference is not None:
        rate = None if scored is None else scored.get("wer")
        if not isinstance(rate, (int, float)) or isinstance(rate, bool):
            return False
        if rate > pass_policy.accuracy_tolerance:
            return False
    return True


def _across(values: list[float | None]) -> dict[str, Any]:
    present = [value for value in values if value is not None]
    return {
        "values": values,
        "mean": sum(present) / len(present) if present else None,
        "minimum": min(present) if present else None,
        "maximum": max(present) if present else None,
        "n_trials": len(values),
        "n_trials_with_measure": len(present),
    }
