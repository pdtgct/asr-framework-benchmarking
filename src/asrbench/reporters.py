"""Reporters are projections of receipts.

A number in a report is read from the receipt. Nothing here re-measures a
session. The frontier is the primary panel: one point per lookahead, WER
against pooled TTFS p50, Pareto edge marked. Dominated points are a
finding. Cross-cadence transcript agreement is not computed.
"""

from __future__ import annotations

from typing import Any

from asrbench.config import ConfigError, config_from_mapping, validate
from asrbench.definitions import (
    FINALIZATION_CLOCK,
    FRONTIER_DEFINITION,
    NATIVE_LATENCY_CLOCK,
    SUSTAINED_STREAMS_DEFINITION,
    VLLM_FORMAT_NOTE,
)
from asrbench.fingerprint import capacity_group_key, frontier_group_key


class NotHomogeneousError(ValueError):
    """A pooled projection requires one cadence and one lookahead."""


def console_report(receipt: dict[str, Any]) -> str:
    """Render a numeric summary of one receipt.

    Args:
        receipt: A receipt object.

    Returns:
        Text ending in a newline. A failed conjunction names its clauses
        before the latencies.
    """
    summary = receipt["summary"]
    pooled = summary["pooled"]
    ttfs = pooled["ttfs_s"]
    response = pooled["response_latency_s"]
    wer = pooled["wer"]
    rtfx = pooled["streaming_rtfx"]
    counts = pooled["counts"]
    lookahead = receipt["lookahead"]
    lines = [
        f"schema {receipt['schema_version']}",
        f"comparison_key {receipt['comparison_key']}",
        "conjunction "
        + ("passed" if summary["conjunction_passed"] else "FAILED"),
        "failed " + ",".join(summary["failed_clauses"]),
        "unevaluable " + ",".join(summary["unevaluable_clauses"]),
        (
            f"ttfs_s p50={ttfs['p50']} p99={ttfs['p99']} "
            f"n_with_measure={ttfs['n_with_measure']} "
            f"n_total={ttfs['n_total']} "
            f"clock={ttfs['clock']} native={ttfs['clock_matches_native']}"
        ),
        (
            f"response_latency_s p50={response['p50']} p99={response['p99']} "
            f"n_samples={response['n_samples']} "
            f"n_with_measure={response['n_with_measure']} "
            f"n_total={response['n_total']} clock={response['clock']}"
        ),
        (
            f"wer policy={wer['policy']} micro={wer['micro_wer']} "
            f"S={wer['substitutions']} D={wer['deletions']} "
            f"I={wer['insertions']} "
            f"missing={wer['deletions_from_missing_transcript']} "
            f"n_failed={wer['n_failed']} "
            f"n_with_measure={wer['n_with_measure']} n_total={wer['n_total']}"
        ),
        (
            f"streaming_rtfx value={rtfx['value']} validity={rtfx['validity']} "
            f"successful_audio_s={rtfx['successful_audio_s']} "
            f"attempted_audio_s={rtfx['attempted_audio_s']}"
        ),
        (
            "outcomes "
            f"attempted={counts['attempted']} admitted={counts['admitted']} "
            f"completed={counts['completed']} "
            f"productive={counts['productive']} "
            f"errored={counts['errored']} timed_out={counts['timed_out']} "
            f"no_first_partial={counts['no_first_partial']}"
        ),
        (
            f"lookahead_frames {lookahead['frames']} "
            f"label_ms {lookahead['millisecond_label']} "
            f"source {lookahead['frames_source']}"
        ),
    ]
    return "\n".join(lines) + "\n"


def vllm_report(receipt: dict[str, Any]) -> dict[str, Any]:
    """Project a receipt into a vLLM-shaped summary object.

    TTFS and response latency stay under those names. This projection does
    not emit a TTFT field.

    Args:
        receipt: A receipt object.

    Returns:
        A JSON-ish object.
    """
    pooled = receipt["summary"]["pooled"]
    counts = pooled["counts"]
    ttfs = pooled["ttfs_s"]
    response = pooled["response_latency_s"]
    rtfx = pooled["streaming_rtfx"]
    duration = sum(trial["cell_wall_span_s"] for trial in receipt["trials"])
    return {
        "format": "vllm-serve-benchmark-shaped",
        "format_note": VLLM_FORMAT_NOTE,
        "schema_version": receipt["schema_version"],
        "duration_s": duration,
        "num_sessions": counts["attempted"],
        "completed_sessions": counts["completed"],
        "failed_sessions": counts["errored"] + counts["timed_out"],
        "percentile_method": ttfs["percentile_method"],
        "ttfs_s": _pick(
            ttfs,
            (
                "mean",
                "p50",
                "p90",
                "p95",
                "p99",
                "n_with_measure",
                "n_total",
                "n_samples",
                "clock",
                "definition",
                "clock_matches_native",
            ),
        ),
        "response_latency_s": _pick(
            response,
            (
                "mean",
                "p50",
                "p90",
                "p95",
                "p99",
                "n_with_measure",
                "n_total",
                "n_samples",
                "clock",
                "definition",
                "clock_matches_native",
            ),
        ),
        "streaming_rtfx": _pick(
            rtfx,
            (
                "value",
                "validity",
                "definition",
                "clock",
                "successful_audio_s",
                "attempted_audio_s",
            ),
        ),
    }


def vendor_tsv(receipt: dict[str, Any]) -> str:
    """Project one homogeneous receipt into a vendor-comparable TSV row.

    Pooled columns describe every measured session. ``across_mean``
    columns are the mean of the per-trial scalars, which is what a
    three-trial vendor sheet averages.

    Args:
        receipt: A receipt object.

    Returns:
        A header row, a comment quoting the TTFS clock, and one data row.

    Raises:
        NotHomogeneousError: The population mixes cadence or lookahead.
    """
    _require_homogeneous(receipt)
    pooled = receipt["summary"]["pooled"]
    across = receipt["summary"]["across_trials"]
    workload = receipt["resolved_config"]["workload"]
    counts = pooled["counts"]
    if counts["attempted"]:
        admission: float | None = counts["admitted"] / counts["attempted"]
    else:
        admission = None
    columns: tuple[tuple[str, object], ...] = (
        ("model", receipt["identity"]["target"]["model"]),
        ("lookahead_frames", workload["lookahead_frames"]),
        ("lookahead_ms_label", workload["lookahead_ms_label"]),
        ("lanes", workload["lanes"]),
        ("chunk_ms", workload["chunk_ms"]),
        ("cadence_ms", workload["cadence_ms"]),
        ("ttfs_p50_s", pooled["ttfs_s"]["p50"]),
        ("ttfs_p99_s", pooled["ttfs_s"]["p99"]),
        ("ttfs_p50_across_mean_s", across["ttfs_p50_s"]["mean"]),
        ("ttfs_p99_across_mean_s", across["ttfs_p99_s"]["mean"]),
        ("ttfs_n_with_measure", pooled["ttfs_s"]["n_with_measure"]),
        ("ttfs_n_total", pooled["ttfs_s"]["n_total"]),
        ("ttfs_clock", pooled["ttfs_s"]["clock"]),
        ("response_latency_p50_s", pooled["response_latency_s"]["p50"]),
        ("response_latency_p99_s", pooled["response_latency_s"]["p99"]),
        (
            "response_latency_p50_across_mean_s",
            across["response_latency_p50_s"]["mean"],
        ),
        (
            "response_latency_n_samples",
            pooled["response_latency_s"]["n_samples"],
        ),
        (
            "response_latency_n_with_measure",
            pooled["response_latency_s"]["n_with_measure"],
        ),
        ("response_latency_n_total", pooled["response_latency_s"]["n_total"]),
        ("response_latency_clock", pooled["response_latency_s"]["clock"]),
        ("finalization_p50_s", pooled["finalization_s"]["p50"]),
        ("wer", pooled["wer"]["micro_wer"]),
        ("wer_across_mean", across["micro_wer"]["mean"]),
        ("wer_policy", pooled["wer"]["policy"]),
        ("wer_n_with_measure", pooled["wer"]["n_with_measure"]),
        ("wer_n_total", pooled["wer"]["n_total"]),
        ("substitutions", pooled["wer"]["substitutions"]),
        ("deletions", pooled["wer"]["deletions"]),
        (
            "deletions_from_missing_transcript",
            pooled["wer"]["deletions_from_missing_transcript"],
        ),
        ("n_failed", pooled["wer"]["n_failed"]),
        ("insertions", pooled["wer"]["insertions"]),
        ("streaming_rtfx", pooled["streaming_rtfx"]["value"]),
        ("streaming_rtfx_across_mean", across["streaming_rtfx"]["mean"]),
        ("streaming_rtfx_validity", pooled["streaming_rtfx"]["validity"]),
        ("attempted_audio_s", pooled["streaming_rtfx"]["attempted_audio_s"]),
        ("successful_audio_s", pooled["streaming_rtfx"]["successful_audio_s"]),
        ("admission_fraction", admission),
        ("conjunction_passed", receipt["summary"]["conjunction_passed"]),
        ("failed_clauses", ",".join(receipt["summary"]["failed_clauses"])),
    )
    comment = (
        "# ttfs_clock "
        f"{pooled['ttfs_s']['clock']} definition quoted on the receipt"
    )
    header = "\t".join(name for name, _value in columns)
    row = "\t".join(_cell(value) for _name, value in columns)
    return comment + "\n" + header + "\n" + row + "\n"


def perf_table_tsv(receipt: dict[str, Any]) -> str:
    """Project one homogeneous receipt into NVIDIA's perf-table column order.

    The columns are a three-trial mean of per-trial finalization stats
    and streaming RTFX, in the published order: stream count, mean, p50,
    p90, p95, p99, throughput. Values are seconds and asrbench's
    streaming RTFX. A trial that lacks the measure stays in ``n_trials``
    and out of that column's mean. The comment quotes the TTFS clock so
    the row cannot be read as that clock or as a vendor millisecond
    latency.

    Args:
        receipt: A receipt object.

    Returns:
        A comment, a header, and one data row.

    Raises:
        NotHomogeneousError: The population mixes cadence or lookahead.
    """
    _require_homogeneous(receipt)
    trials = receipt["trials"]
    workload = receipt["resolved_config"]["workload"]
    pooled = receipt["summary"]["pooled"]["finalization_s"]
    final_mean = _trial_mean(trials, "finalization_s", "mean")
    final_p50 = _trial_mean(trials, "finalization_s", "p50")
    final_p90 = _trial_mean(trials, "finalization_s", "p90")
    final_p95 = _trial_mean(trials, "finalization_s", "p95")
    final_p99 = _trial_mean(trials, "finalization_s", "p99")
    rtfx = _trial_mean(trials, "streaming_rtfx", "value")
    columns: tuple[tuple[str, object], ...] = (
        ("num_streams", workload["lanes"]),
        ("finalization_mean_s", final_mean[0]),
        ("finalization_p50_s", final_p50[0]),
        ("finalization_p90_s", final_p90[0]),
        ("finalization_p95_s", final_p95[0]),
        ("finalization_p99_s", final_p99[0]),
        ("streaming_rtfx", rtfx[0]),
    )
    comment = (
        "# columns are three-trial means of per-trial finalization_s on "
        f"{FINALIZATION_CLOCK} and of streaming_rtfx. Not TTFS. Not a "
        "vendor millisecond latency. "
        f"ttfs_clock {NATIVE_LATENCY_CLOCK} "
        f"n_trials={len(trials)} "
        f"n_trials_with_finalization={final_p50[1]} "
        f"n_trials_with_streaming_rtfx={rtfx[1]} "
        f"finalization_n_with_measure={pooled['n_with_measure']} "
        f"finalization_n_total={pooled['n_total']}"
    )
    header = "\t".join(name for name, _value in columns)
    row = "\t".join(_cell(value) for _name, value in columns)
    return comment + "\n" + header + "\n" + row + "\n"


def _trial_mean(
    trials: list[dict[str, Any]], section: str, field: str
) -> tuple[float | None, int]:
    """Return the mean of one per-trial scalar and how many trials had it."""
    values: list[float] = []
    for trial in trials:
        value = trial["reduction"][section][field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        values.append(float(value))
    if not values:
        return None, 0
    return sum(values) / len(values), len(values)


def frontier_report(
    receipts: list[dict[str, Any]],
    sources: list[str],
) -> dict[str, Any]:
    """Project receipts onto latency/accuracy frontiers.

    Receipts that differ in anything except lookahead frame count and
    cadence land in different series. Within a series, a point is
    dominated when another
    point is at least as accurate and at least as fast, and strictly
    better on one of those. That is a finding, not a pass condition.

    Args:
        receipts: Receipt objects.
        sources: Display name for each receipt, same order.

    Returns:
        A JSON-ish frontier object.

    Raises:
        NotHomogeneousError: A receipt mixes cadence or lookahead.
    """
    for receipt in receipts:
        _require_homogeneous(receipt)
    order, grouped = _group(
        receipts, sources, frontier_group_key, _frontier_point
    )
    series = []
    for key in order:
        points = grouped[key]
        _mark_frontier(points)
        series.append({"group_key": key, "points": points})
    return {
        "definition": FRONTIER_DEFINITION,
        "latency_source": "pooled.ttfs_s.p50",
        "accuracy_source": "pooled.wer.micro_wer",
        "series": series,
    }


def frontier_text(report: dict[str, Any]) -> str:
    """Render a frontier report as aligned text.

    Args:
        report: The object returned by :func:`frontier_report`.

    Returns:
        Text ending in a newline.
    """
    lines = [
        "frontier",
        f"latency {report['latency_source']}",
        f"accuracy {report['accuracy_source']}",
    ]
    for series in report["series"]:
        lines.append(f"series {series['group_key']}")
        lines.append(
            "lookahead_frames\tcadence_ms\tlabel_ms\twer\t"
            "latency_s\tedge\tfindings\tsource"
        )
        for point in series["points"]:
            edge = "edge" if point["on_pareto_edge"] else "off"
            findings = ",".join(point["findings"]) or "-"
            lines.append(
                "\t".join(
                    [
                        _cell(point["lookahead_frames"]),
                        _cell(point.get("cadence_ms")),
                        _cell(point["lookahead_ms_label"]),
                        _cell(point["wer"]),
                        _cell(point["latency_s"]),
                        edge,
                        findings,
                        str(point["source"]),
                    ]
                )
            )
    return "\n".join(lines) + "\n"


def sustained_streams_report(
    receipts: list[dict[str, Any]],
    sources: list[str],
) -> dict[str, Any]:
    """Largest passing concurrency within each capacity group.

    Lane count is removed from the group key. Lookahead is not. A group
    whose lower concurrency failed while a higher one passed is reported
    as non-monotonic rather than smoothed.

    Args:
        receipts: Receipt objects.
        sources: Display name per receipt.

    Returns:
        A JSON-ish projection. ``value`` is null when no cell passed.

    Raises:
        NotHomogeneousError: A receipt mixes cadence or lookahead.
    """
    for receipt in receipts:
        _require_homogeneous(receipt)
    order, grouped = _group(
        receipts, sources, capacity_group_key, _capacity_cell
    )
    series = []
    for key in order:
        cells = grouped[key]
        passing = [
            cell["lanes"] for cell in cells if cell["conjunction_passed"]
        ]
        value = max(passing) if passing else None
        findings: list[str] = []
        if value is not None and any(
            not cell["conjunction_passed"] and cell["lanes"] < value
            for cell in cells
        ):
            findings.append("non_monotonic")
        series.append(
            {
                "group_key": key,
                "value": value,
                "cells": cells,
                "findings": findings,
            }
        )
    return {
        "definition": SUSTAINED_STREAMS_DEFINITION,
        "availability": "measured",
        "series": series,
    }


def _group(
    receipts: list[dict[str, Any]],
    sources: list[str],
    key_fn: Any,
    cell_fn: Any,
) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for receipt, source in zip(receipts, sources, strict=True):
        config, software, hardware = _identity(receipt)
        key = key_fn(config, software, hardware)
        if key not in grouped:
            order.append(key)
            grouped[key] = []
        grouped[key].append(cell_fn(receipt, source))
    return order, grouped


def _frontier_point(receipt: dict[str, Any], source: str) -> dict[str, Any]:
    pooled = receipt["summary"]["pooled"]
    return {
        "source": source,
        "lookahead_frames": receipt["lookahead"]["frames"],
        "lookahead_ms_label": receipt["lookahead"]["millisecond_label"],
        "cadence_ms": receipt["resolved_config"]["workload"]["cadence_ms"],
        "wer": pooled["wer"]["micro_wer"],
        "wer_policy": pooled["wer"]["policy"],
        "latency_s": pooled["ttfs_s"]["p50"],
        "latency_metric": "ttfs",
        "latency_percentile": 50,
        "latency_clock": pooled["ttfs_s"]["clock"],
        "n_with_measure_latency": pooled["ttfs_s"]["n_with_measure"],
        "n_total_latency": pooled["ttfs_s"]["n_total"],
        "n_with_measure_wer": pooled["wer"]["n_with_measure"],
        "n_total_wer": pooled["wer"]["n_total"],
        "findings": [],
    }


def _capacity_cell(receipt: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "source": source,
        "lanes": receipt["resolved_config"]["workload"]["lanes"],
        "lookahead_frames": receipt["lookahead"]["frames"],
        "conjunction_passed": receipt["summary"]["conjunction_passed"],
    }


def _pick(block: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: block[key] for key in keys}


def _identity(
    receipt: dict[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    payload = receipt.get("resolved_config")
    if not isinstance(payload, dict):
        raise ConfigError("receipt has no resolved_config object")
    config = validate(config_from_mapping(payload, complete=True), pin=False)
    return (
        config,
        receipt["identity"]["software"],
        receipt["identity"]["hardware"],
    )


def _mark_frontier(points: list[dict[str, Any]]) -> None:
    counts: dict[tuple[Any, Any], int] = {}
    for point in points:
        key = (point["lookahead_frames"], point.get("cadence_ms"))
        counts[key] = counts.get(key, 0) + 1
    eligible = [
        point
        for point in points
        if point["wer"] is not None and point["latency_s"] is not None
    ]
    for point in points:
        findings: list[str] = []
        point_key = (point["lookahead_frames"], point.get("cadence_ms"))
        if counts[point_key] > 1:
            findings.append("duplicate_lookahead")
        if point not in eligible:
            point["on_pareto_edge"] = False
            point["dominated"] = False
            point["dominated_by"] = []
            findings.append("missing_measure")
            point["findings"] = findings
            continue
        dominators = [
            other["lookahead_frames"]
            for other in eligible
            if other is not point
            and other["latency_s"] <= point["latency_s"]
            and other["wer"] <= point["wer"]
            and (
                other["latency_s"] < point["latency_s"]
                or other["wer"] < point["wer"]
            )
        ]
        point["dominated"] = bool(dominators)
        point["on_pareto_edge"] = not dominators
        point["dominated_by"] = dominators
        if dominators:
            findings.append("dominated")
        point["findings"] = findings


def _require_homogeneous(receipt: dict[str, Any]) -> None:
    population = receipt["resolved_config"]["workload"]["population"]
    frames = {entry["lookahead_frames"] for entry in population}
    cadences = {entry["cadence_ms"] for entry in population}
    if len(frames) != 1 or len(cadences) != 1:
        raise NotHomogeneousError("one cadence and one lookahead are required")


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
