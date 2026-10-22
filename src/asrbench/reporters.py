"""Reporters are projections of receipts.

A number in a report is read from the receipt. Nothing here re-measures a
session. The frontier is the primary panel: one point per lookahead, WER
against pooled TTFS p50, Pareto edge marked. Dominated points are a
finding. Cross-cadence transcript agreement is not computed.
"""

from __future__ import annotations

from typing import Any

from asrbench.config import config_from_mapping, finalize
from asrbench.definitions import (
    FRONTIER_DEFINITION,
    SUSTAINED_STREAMS_DEFINITION,
    VLLM_FORMAT_NOTE,
)
from asrbench.fingerprint import capacity_group_key, frontier_group_key


class NotHomogeneousError(ValueError):
    """Vendor TSV is only defined for a homogeneous population."""


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
        "ttfs_s": _latency_block(ttfs),
        "response_latency_s": _latency_block(response),
        "streaming_rtfx": {
            "value": rtfx["value"],
            "validity": rtfx["validity"],
            "definition": rtfx["definition"],
            "clock": rtfx["clock"],
            "successful_audio_s": rtfx["successful_audio_s"],
            "attempted_audio_s": rtfx["attempted_audio_s"],
        },
    }


def vendor_tsv(receipt: dict[str, Any]) -> str:
    """Project one homogeneous receipt into a vendor-comparable TSV row.

    Args:
        receipt: A receipt object.

    Returns:
        A header row, a comment quoting the TTFS clock, and one data row.

    Raises:
        NotHomogeneousError: The population mixes cadence or lookahead.
    """
    _require_homogeneous(receipt)
    workload = receipt["resolved_config"]["workload"]
    pooled = receipt["summary"]["pooled"]
    ttfs = pooled["ttfs_s"]
    response = pooled["response_latency_s"]
    final = pooled["finalization_s"]
    wer = pooled["wer"]
    rtfx = pooled["streaming_rtfx"]
    counts = pooled["counts"]
    admission = (
        counts["admitted"] / counts["attempted"]
        if counts["attempted"]
        else None
    )
    header = [
        "model",
        "lookahead_frames",
        "lookahead_ms_label",
        "lanes",
        "chunk_ms",
        "cadence_ms",
        "ttfs_p50_s",
        "ttfs_p99_s",
        "ttfs_n_with_measure",
        "ttfs_n_total",
        "ttfs_clock",
        "response_latency_p50_s",
        "response_latency_p99_s",
        "response_latency_n_samples",
        "response_latency_n_with_measure",
        "response_latency_n_total",
        "response_latency_clock",
        "finalization_p50_s",
        "wer",
        "wer_policy",
        "wer_n_with_measure",
        "wer_n_total",
        "substitutions",
        "deletions",
        "insertions",
        "streaming_rtfx",
        "streaming_rtfx_validity",
        "attempted_audio_s",
        "successful_audio_s",
        "admission_fraction",
        "conjunction_passed",
        "failed_clauses",
    ]
    row = [
        receipt["identity"]["target"]["model"],
        workload["lookahead_frames"],
        workload["lookahead_ms_label"],
        workload["lanes"],
        workload["chunk_ms"],
        workload["cadence_ms"],
        ttfs["p50"],
        ttfs["p99"],
        ttfs["n_with_measure"],
        ttfs["n_total"],
        ttfs["clock"],
        response["p50"],
        response["p99"],
        response["n_samples"],
        response["n_with_measure"],
        response["n_total"],
        response["clock"],
        final["p50"],
        wer["micro_wer"],
        wer["policy"],
        wer["n_with_measure"],
        wer["n_total"],
        wer["substitutions"],
        wer["deletions"],
        wer["insertions"],
        rtfx["value"],
        rtfx["validity"],
        rtfx["attempted_audio_s"],
        rtfx["successful_audio_s"],
        admission,
        receipt["summary"]["conjunction_passed"],
        ",".join(receipt["summary"]["failed_clauses"]),
    ]
    comment = f"# ttfs_clock {ttfs['clock']} definition quoted on the receipt"
    return (
        comment
        + "\n"
        + "\t".join(header)
        + "\n"
        + "\t".join(_cell(value) for value in row)
        + "\n"
    )


def frontier_report(
    receipts: list[dict[str, Any]],
    sources: list[str],
) -> dict[str, Any]:
    """Project receipts onto latency/accuracy frontiers.

    Receipts that differ in anything except lookahead frame count land in
    different series. Within a series, a point is dominated when another
    point is at least as accurate and at least as fast, and strictly
    better on one of those. That is a finding, not a pass condition.

    Args:
        receipts: Receipt objects.
        sources: Display name for each receipt, same order.

    Returns:
        A JSON-ish frontier object.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for receipt, source in zip(receipts, sources, strict=True):
        config, software, hardware = _identity(receipt)
        key = frontier_group_key(config, software, hardware)
        pooled = receipt["summary"]["pooled"]
        point = {
            "source": source,
            "lookahead_frames": receipt["lookahead"]["frames"],
            "lookahead_ms_label": receipt["lookahead"]["millisecond_label"],
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
        if key not in grouped:
            order.append(key)
            grouped[key] = []
        grouped[key].append(point)
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
            "lookahead_frames\tlabel_ms\twer\tlatency_s\tedge\tfindings\tsource"
        )
        for point in series["points"]:
            edge = "edge" if point["on_pareto_edge"] else "off"
            findings = ",".join(point["findings"]) or "-"
            lines.append(
                "\t".join(
                    [
                        _cell(point["lookahead_frames"]),
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
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for receipt, source in zip(receipts, sources, strict=True):
        config, software, hardware = _identity(receipt)
        key = capacity_group_key(config, software, hardware)
        cell = {
            "source": source,
            "lanes": receipt["resolved_config"]["workload"]["lanes"],
            "lookahead_frames": receipt["lookahead"]["frames"],
            "conjunction_passed": receipt["summary"]["conjunction_passed"],
        }
        if key not in grouped:
            order.append(key)
            grouped[key] = []
        grouped[key].append(cell)
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


def _identity(
    receipt: dict[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    config = finalize(config_from_mapping(receipt["resolved_config"]))
    software = receipt["identity"]["software"]
    hardware = receipt["identity"]["hardware"]
    return config, software, hardware


def _mark_frontier(points: list[dict[str, Any]]) -> None:
    counts: dict[Any, int] = {}
    for point in points:
        counts[point["lookahead_frames"]] = (
            counts.get(point["lookahead_frames"], 0) + 1
        )
    eligible = [
        point
        for point in points
        if point["wer"] is not None and point["latency_s"] is not None
    ]
    for point in points:
        findings: list[str] = []
        if counts[point["lookahead_frames"]] > 1:
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
        raise NotHomogeneousError(
            "vendor TSV requires one cadence and one lookahead"
        )


def _latency_block(distribution: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean": distribution["mean"],
        "p50": distribution["p50"],
        "p90": distribution["p90"],
        "p95": distribution["p95"],
        "p99": distribution["p99"],
        "n_with_measure": distribution["n_with_measure"],
        "n_total": distribution["n_total"],
        "n_samples": distribution["n_samples"],
        "clock": distribution["clock"],
        "definition": distribution["definition"],
        "clock_matches_native": distribution["clock_matches_native"],
    }


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
