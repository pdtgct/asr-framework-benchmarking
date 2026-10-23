"""Reducers for the four metric families.

A reducer reads outcomes and returns numbers. It does not drop a session
that lacks a measure: the session stays in ``n_total``, and the percentile
is computed only over ``n_with_measure``. Metrics that need ground truth
this outcome shape does not carry are emitted as unavailable. They are
never filled with a zero.
"""

from __future__ import annotations

import math
from typing import Any

from asrbench.definitions import (
    AUDIO_DENOMINATOR_DEFINITION,
    CER_DEFINITION,
    COMPLETION_FRACTION_DEFINITION,
    CORPUS_NORMALIZATION_DEFINITION,
    EMISSION_LAG_CLOCK,
    EMISSION_LAG_DEFINITION,
    EMISSION_LAG_UNAVAILABLE_REASON,
    FINALIZATION_CLOCK,
    FINALIZATION_DEFINITION,
    GOODPUT_DEFINITION,
    INTERIM_CADENCE_CLOCK,
    INTERIM_CADENCE_DEFINITION,
    INTERIM_CADENCE_UNAVAILABLE_REASON,
    LAG_EXCESS_DEFINITION,
    LAG_EXCESS_UNAVAILABLE_REASON,
    LATENCY_METRIC_RESPONSE,
    LATENCY_METRIC_TTFS,
    MODE_OFFLINE,
    MODE_STREAMING,
    OFFLINE_RTFX_DEFINITION,
    ORACLE_NORMALIZATION_DEFINITION,
    OTHER_SERVER_NORMALIZATION_DEFINITION,
    OUTCOME_ACCOUNTING_DEFINITION,
    PACING_FIDELITY_CLOCK,
    PACING_FIDELITY_DEFINITION,
    PACING_FIDELITY_UNAVAILABLE_REASON,
    PASS_CONJUNCTION_DEFINITION,
    PER_UTTERANCE_WER_DEFINITION,
    POPULATION_DEFINITION,
    PREFIX_VIOLATION_DEFINITION,
    PRODUCTIVE_DEFINITION,
    RESPONSE_LATENCY_CLOCK,
    RESPONSE_LATENCY_DEFINITION,
    REVISION_RATE_DEFINITION,
    RTFX_VALIDITY_NOT_APPLICABLE,
    RTFX_VALIDITY_PACING_DISABLED,
    RTFX_VALIDITY_SIMULATED,
    RTFX_VALIDITY_UNQUALIFIED,
    SDI_DEFINITION,
    STEADY_INTERVAL_DEFINITION,
    STEADY_INTERVAL_UNAVAILABLE_REASON,
    STREAMING_RTFX_CLOCK,
    STREAMING_RTFX_DEFINITION,
    SUSTAINABILITY_COMPLETION_FRACTION,
    SUSTAINABILITY_STEADY_INTERVAL,
    SUSTAINED_STREAMS_DEFINITION,
    SUSTAINED_STREAMS_UNAVAILABLE_REASON,
    TTFS_CLOCK,
    TTFS_DEFINITION,
    WER_DEFINITION,
)
from asrbench.outcome import SessionOutcome
from asrbench.percentile import percentile
from asrbench.spec import PassPolicy
from asrbench.wer import score_texts

_NORM_TEXT = {
    "engine_vs_own_oracle": ORACLE_NORMALIZATION_DEFINITION,
    "corpus_reference": CORPUS_NORMALIZATION_DEFINITION,
    "engine_vs_other_server": OTHER_SERVER_NORMALIZATION_DEFINITION,
}


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
    """Return whether a non-empty partial was observed.

    Args:
        outcome: One session.

    Returns:
        True when ``ttfs_s`` is present.
    """
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

    Returns:
        A JSON-ish reduction. Unavailable metrics are present and null,
        not omitted and not zero.

    Raises:
        MeasurementError: A latency sample is not finite.
    """
    _check_finite(outcomes)
    counts = _counts(outcomes, warmup_excluded)
    ttfs = _latency(
        outcomes,
        name="ttfs_s",
        native_clock=TTFS_CLOCK,
        definition=TTFS_DEFINITION,
        method=percentile_method,
        value_of=lambda item: item.ttfs_s,
        clock_of=lambda item: item.latency_clock,
        population=len(outcomes),
    )
    response = _response_latency(outcomes, percentile_method)
    finalization = _latency(
        outcomes,
        name="finalization_s",
        native_clock=FINALIZATION_CLOCK,
        definition=FINALIZATION_DEFINITION,
        method=percentile_method,
        value_of=lambda item: item.final_latency_s,
        clock_of=lambda item: item.final_latency_clock,
        population=len(outcomes),
    )
    wer = _wer(
        outcomes, reference, pass_policy.normalization_policy, percentile_method
    )
    attempted, successful = _audio(outcomes)
    simulated = _clocks_are_simulated(outcomes)
    streaming = _rtfx(
        name="streaming_rtfx",
        definition=STREAMING_RTFX_DEFINITION,
        applicable=mode == MODE_STREAMING,
        inapplicable_reason="mode_is_offline",
        pacing=pacing,
        attempted=attempted,
        successful=successful,
        wall_span_s=wall_span_s,
        span_pooling=span_pooling,
        offline=False,
        simulated=simulated,
    )
    offline = _rtfx(
        name="offline_rtfx",
        definition=OFFLINE_RTFX_DEFINITION,
        applicable=mode == MODE_OFFLINE,
        inapplicable_reason="mode_is_streaming",
        pacing=pacing,
        attempted=attempted,
        successful=successful,
        wall_span_s=wall_span_s,
        span_pooling=span_pooling,
        offline=True,
        simulated=simulated,
    )
    stability = {
        "prefix_violations": _prefix(outcomes),
        "revision_rate": _revision(outcomes),
    }
    unavailable = _unavailable()
    conjunction = _conjunction(
        outcomes,
        counts=counts,
        ttfs=ttfs,
        response=response,
        wer=wer,
        reference=reference,
        pass_policy=pass_policy,
        percentile_method=percentile_method,
    )
    goodput = _goodput(
        outcomes,
        wall_span_s=wall_span_s,
        reference=reference,
        pass_policy=pass_policy,
        percentile_method=percentile_method,
        utterances=wer["utterances"],
    )
    return {
        "counts": counts,
        "ttfs_s": ttfs,
        "response_latency_s": response,
        "finalization_s": finalization,
        "wer": wer,
        "streaming_rtfx": streaming,
        "offline_rtfx": offline,
        "stability": stability,
        "goodput": goodput,
        "sustained_streams_at_slo": {
            "name": "sustained_streams_at_slo",
            "availability": "unavailable",
            "reason": SUSTAINED_STREAMS_UNAVAILABLE_REASON,
            "definition": SUSTAINED_STREAMS_DEFINITION,
            "offered_lanes": lanes,
            "cell_conjunction_passed": conjunction["passed"],
            "clock": "",
        },
        "unavailable": unavailable,
        "conjunction": conjunction,
        "audio": {
            "definition": AUDIO_DENOMINATOR_DEFINITION,
            "attempted_audio_s": attempted,
            "successful_audio_s": successful,
            "productive_definition": PRODUCTIVE_DEFINITION,
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

    The across-trial block is the mean of trial-level scalars. ``values``
    keeps a null where a trial had no measure. ``pooled`` re-reduces every
    measured session together and labels that choice.

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

    def values(reader: Any) -> list[float | None]:
        return [reader(reduction) for reduction in reductions]

    across = {
        "ttfs_p50_s": _across(values(lambda item: item["ttfs_s"]["p50"])),
        "ttfs_p99_s": _across(values(lambda item: item["ttfs_s"]["p99"])),
        "response_latency_p50_s": _across(
            values(lambda item: item["response_latency_s"]["p50"])
        ),
        "response_latency_p99_s": _across(
            values(lambda item: item["response_latency_s"]["p99"])
        ),
        "finalization_p50_s": _across(
            values(lambda item: item["finalization_s"]["p50"])
        ),
        "micro_wer": _across(values(lambda item: item["wer"]["micro_wer"])),
        "streaming_rtfx": _across(
            values(lambda item: item["streaming_rtfx"]["value"])
        ),
        "offline_rtfx": _across(
            values(lambda item: item["offline_rtfx"]["value"])
        ),
        "goodput_per_second": _across(
            values(lambda item: item["goodput"]["per_second"])
        ),
    }
    failed: list[str] = []
    unevaluable: list[str] = []
    for reduction in reductions:
        for name in reduction["conjunction"]["failed_clauses"]:
            if name not in failed:
                failed.append(name)
        for name in reduction["conjunction"]["unevaluable_clauses"]:
            if name not in unevaluable:
                unevaluable.append(name)
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
        "definition": OUTCOME_ACCOUNTING_DEFINITION,
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
    if availability != "measured":
        mean = minimum = maximum = None
        p50 = p90 = p95 = p99 = None
    else:
        mean = sum(samples) / len(samples)
        minimum = min(samples)
        maximum = max(samples)
        p50 = percentile(samples, 50, method)
        p90 = percentile(samples, 90, method)
        p95 = percentile(samples, 95, method)
        p99 = percentile(samples, 99, method)
    return {
        "name": name,
        "clock": clock,
        "clock_attribution": clock_attribution,
        "clock_matches_native": clock_matches_native,
        "definition": definition,
        "unit": unit,
        "percentile_method": method,
        "availability": availability,
        "mean": mean,
        "minimum": minimum,
        "maximum": maximum,
        "p50": p50,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "n_with_measure": n_with_measure,
        "n_total": n_total,
        "n_samples": n_samples,
        "population_definition": POPULATION_DEFINITION,
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
    if len(clocks) > 1:
        return _distribution(
            name=name,
            clock=None,
            clock_attribution="mixed",
            clock_matches_native=False,
            definition=definition,
            unit="s",
            method=method,
            availability="mixed_clocks",
            samples=[],
            n_with_measure=n_with_measure,
            n_total=n_total,
            n_samples=len(entries),
        )
    if not entries:
        return _distribution(
            name=name,
            clock=native_clock,
            clock_attribution="none",
            clock_matches_native=False,
            definition=definition,
            unit="s",
            method=method,
            availability="no_samples",
            samples=[],
            n_with_measure=0,
            n_total=n_total,
            n_samples=0,
        )
    clock = next(iter(clocks))
    if all(explicit for _, _, explicit in entries):
        attribution = "outcome"
    else:
        attribution = "native_default"
    return _distribution(
        name=name,
        clock=clock,
        clock_attribution=attribution,
        clock_matches_native=clock == native_clock,
        definition=definition,
        unit="s",
        method=method,
        availability="measured",
        samples=[value for value, _, _ in entries],
        n_with_measure=n_with_measure,
        n_total=n_total,
        n_samples=len(entries),
    )


def _latency(
    outcomes: list[SessionOutcome],
    *,
    name: str,
    native_clock: str,
    definition: str,
    method: str,
    value_of: Any,
    clock_of: Any,
    population: int,
) -> dict[str, Any]:
    entries: list[tuple[float, str, bool]] = []
    for outcome in outcomes:
        value = value_of(outcome)
        if value is None:
            continue
        explicit = str(clock_of(outcome) or "")
        entries.append((float(value), explicit or native_clock, bool(explicit)))
    return _clocked_distribution(
        name=name,
        native_clock=native_clock,
        definition=definition,
        method=method,
        entries=entries,
        n_with_measure=len(entries),
        n_total=population,
    )


def _response_latency(
    outcomes: list[SessionOutcome], method: str
) -> dict[str, Any]:
    entries: list[tuple[float, str, bool]] = []
    sessions = 0
    for outcome in outcomes:
        if not outcome.response_latencies_s:
            continue
        sessions += 1
        explicit = outcome.response_latency_clock or ""
        clock = explicit or RESPONSE_LATENCY_CLOCK
        for sample in outcome.response_latencies_s:
            entries.append((float(sample), clock, bool(explicit)))
    return _clocked_distribution(
        name="response_latency_s",
        native_clock=RESPONSE_LATENCY_CLOCK,
        definition=RESPONSE_LATENCY_DEFINITION,
        method=method,
        entries=entries,
        n_with_measure=sessions,
        n_total=len(outcomes),
    )


def _wer(
    outcomes: list[SessionOutcome],
    reference: str | None,
    policy: str,
    method: str,
) -> dict[str, Any]:
    utterances: list[dict[str, Any]] = []
    substitutions = deletions = insertions = 0
    reference_tokens = hypothesis_tokens = reference_chars = char_edits = 0
    missing = chars_missing = n_failed = 0
    samples: list[float] = []
    if reference is None:
        for outcome in outcomes:
            utterances.append(
                _unscored(outcome.label, "no_reference_transcript")
            )
        availability = "no_reference"
        reason: str | None = "no_reference_transcript"
        micro_wer = None
        micro_cer = None
    else:
        for outcome in outcomes:
            if outcome.transcript is None:
                blank = score_texts(reference, "", policy)
                n_failed += 1
                missing += blank.reference_tokens
                chars_missing += blank.reference_chars
                reference_tokens += blank.reference_tokens
                reference_chars += blank.reference_chars
                utterances.append(
                    {
                        "label": outcome.label,
                        "scored": False,
                        "reason": "missing_transcript",
                        "wer": None,
                        "cer": None,
                        "substitutions": 0,
                        "deletions": 0,
                        "insertions": 0,
                        "reference_tokens": blank.reference_tokens,
                        "deletions_from_missing_transcript": (
                            blank.reference_tokens
                        ),
                    }
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
            reason_row = None if scored.wer is not None else "undefined_rate"
            if scored.wer is not None:
                samples.append(scored.wer)
            utterances.append(
                {
                    "label": outcome.label,
                    "scored": True,
                    "reason": reason_row,
                    "wer": scored.wer,
                    "cer": scored.cer,
                    "substitutions": scored.substitutions,
                    "deletions": scored.deletions,
                    "insertions": scored.insertions,
                    "reference_tokens": scored.reference_tokens,
                    "deletions_from_missing_transcript": 0,
                }
            )
        micro_wer = _rate(
            substitutions + deletions + insertions + missing, reference_tokens
        )
        micro_cer = _rate(char_edits + chars_missing, reference_chars)
        if micro_wer is None and not outcomes:
            availability = "no_samples"
            reason = "no_measured_sessions"
        elif micro_wer is None:
            availability = "undefined"
            reason = "reference_word_denominator_is_zero"
        else:
            availability = "measured"
            reason = None
    per = _rate_distribution(
        name="per_utterance_wer",
        definition=PER_UTTERANCE_WER_DEFINITION,
        method=method,
        samples=samples,
        n_total=len(outcomes),
    )
    return {
        "policy": policy,
        "definition": WER_DEFINITION,
        "normalization_definition": _NORM_TEXT[policy],
        "availability": availability,
        "reason": reason,
        "micro_wer": micro_wer,
        "micro_cer": micro_cer,
        "cer_definition": CER_DEFINITION,
        "sdi_definition": SDI_DEFINITION,
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
        "n_total": len(outcomes),
        "per_utterance_wer": per,
        "utterances": utterances,
    }


def _unscored(label: str, reason: str) -> dict[str, Any]:
    return {
        "label": label,
        "scored": False,
        "reason": reason,
        "wer": None,
        "cer": None,
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
        "reference_tokens": 0,
    }


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
    if not samples:
        mean = p50 = p90 = p95 = p99 = None
        availability = "no_samples"
    else:
        mean = sum(samples) / len(samples)
        p50 = percentile(samples, 50, method)
        p90 = percentile(samples, 90, method)
        p95 = percentile(samples, 95, method)
        p99 = percentile(samples, 99, method)
        availability = "measured"
    return {
        "name": name,
        "definition": definition,
        "percentile_method": method,
        "availability": availability,
        "mean": mean,
        "p50": p50,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "n_with_measure": len(samples),
        "n_total": n_total,
        "population_definition": POPULATION_DEFINITION,
    }


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
) -> dict[str, Any]:
    availability = "unavailable"
    reason: str | None = inapplicable_reason
    value = None
    validity = RTFX_VALIDITY_NOT_APPLICABLE
    if applicable and wall_span_s <= 0:
        reason = "cell_wall_span_not_positive"
        validity = RTFX_VALIDITY_UNQUALIFIED
    elif applicable:
        availability = "measured"
        reason = None
        value = successful / wall_span_s
        if simulated:
            validity = RTFX_VALIDITY_SIMULATED
        elif offline:
            validity = "offline_cell"
        elif not pacing:
            validity = RTFX_VALIDITY_PACING_DISABLED
        else:
            validity = RTFX_VALIDITY_UNQUALIFIED
    return {
        "name": name,
        "availability": availability,
        "reason": reason,
        "value": value,
        "successful_audio_s": successful,
        "attempted_audio_s": attempted,
        "cell_wall_span_s": wall_span_s,
        "span_pooling": span_pooling,
        "clock": STREAMING_RTFX_CLOCK,
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
            "definition": PREFIX_VIOLATION_DEFINITION,
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
        "definition": PREFIX_VIOLATION_DEFINITION,
    }


def _revision(outcomes: list[SessionOutcome]) -> dict[str, Any]:
    if not outcomes or all(item.interim_texts is None for item in outcomes):
        return {
            "availability": "unavailable",
            "reason": "interim_texts_not_recorded",
            "rate": None,
            "revised_tokens": None,
            "emitted_tokens": None,
            "definition": REVISION_RATE_DEFINITION,
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
        "definition": REVISION_RATE_DEFINITION,
    }


def _unavailable() -> list[dict[str, Any]]:
    return [
        {
            "name": "emission_lag",
            "availability": "unavailable",
            "reason": EMISSION_LAG_UNAVAILABLE_REASON,
            "clock": EMISSION_LAG_CLOCK,
            "definition": EMISSION_LAG_DEFINITION,
        },
        {
            "name": "lag_excess",
            "availability": "unavailable",
            "reason": LAG_EXCESS_UNAVAILABLE_REASON,
            "clock": EMISSION_LAG_CLOCK,
            "definition": LAG_EXCESS_DEFINITION,
        },
        {
            "name": "pacing_fidelity",
            "availability": "unavailable",
            "reason": PACING_FIDELITY_UNAVAILABLE_REASON,
            "clock": PACING_FIDELITY_CLOCK,
            "definition": PACING_FIDELITY_DEFINITION,
        },
        {
            "name": "interim_cadence_conformance",
            "availability": "unavailable",
            "reason": INTERIM_CADENCE_UNAVAILABLE_REASON,
            "clock": INTERIM_CADENCE_CLOCK,
            "definition": INTERIM_CADENCE_DEFINITION,
        },
    ]


def _conjunction(
    outcomes: list[SessionOutcome],
    *,
    counts: dict[str, Any],
    ttfs: dict[str, Any],
    response: dict[str, Any],
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
            outcomes,
            pass_policy=pass_policy,
            percentile_method=percentile_method,
            ttfs=ttfs,
            response=response,
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
        "definition": PASS_CONJUNCTION_DEFINITION,
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
            name,
            "unevaluable",
            observed,
            threshold,
            "no measured sessions",
        )
    status = "passed" if observed >= threshold else "failed"
    return _clause(
        name, status, observed, threshold, f"{observed} >= {threshold}"
    )


def _latency_clause(
    outcomes: list[SessionOutcome],
    *,
    pass_policy: PassPolicy,
    percentile_method: str,
    ttfs: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    metric = pass_policy.latency_metric
    chosen = ttfs if metric == LATENCY_METRIC_TTFS else response
    if metric not in {LATENCY_METRIC_TTFS, LATENCY_METRIC_RESPONSE}:
        return _clause(
            "latency_slo",
            "unevaluable",
            None,
            pass_policy.latency_slo_s,
            f"unknown latency metric {metric}",
        )
    samples = _latency_samples(outcomes, metric)
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


def _latency_samples(
    outcomes: list[SessionOutcome], metric: str
) -> list[float]:
    if metric == LATENCY_METRIC_TTFS:
        return [
            float(item.ttfs_s) for item in outcomes if item.ttfs_s is not None
        ]
    samples: list[float] = []
    for outcome in outcomes:
        samples.extend(float(value) for value in outcome.response_latencies_s)
    return samples


def _sustainability_clause(
    counts: dict[str, Any], total: int, pass_policy: PassPolicy
) -> dict[str, Any]:
    policy = pass_policy.sustainability_policy
    if policy == SUSTAINABILITY_STEADY_INTERVAL:
        return _clause(
            "sustainability",
            "unevaluable",
            None,
            pass_policy.sustainability_min,
            STEADY_INTERVAL_DEFINITION
            + " "
            + STEADY_INTERVAL_UNAVAILABLE_REASON,
        )
    if policy != SUSTAINABILITY_COMPLETION_FRACTION:
        return _clause(
            "sustainability",
            "unevaluable",
            None,
            pass_policy.sustainability_min,
            f"unknown sustainability policy {policy}",
        )
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
        COMPLETION_FRACTION_DEFINITION,
    )


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
        "definition": GOODPUT_DEFINITION,
        "n_total": len(outcomes),
        "cell_wall_span_s": wall_span_s,
    }
    if pass_policy.sustainability_policy == SUSTAINABILITY_STEADY_INTERVAL:
        return {
            **base,
            "availability": "unavailable",
            "reason": STEADY_INTERVAL_UNAVAILABLE_REASON,
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
    if pass_policy.latency_metric == LATENCY_METRIC_TTFS:
        if outcome.ttfs_s is None or outcome.ttfs_s > pass_policy.latency_slo_s:
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
