"""Reporters read the receipt. The frontier is the primary panel."""

import asyncio
import copy
from pathlib import Path
from typing import Any

import pytest

from asrbench.config import resolve
from asrbench.identity import (
    FixedProbe,
    GitState,
    HardwareRecord,
    collect_hardware,
)
from asrbench.reporters import (
    NotHomogeneousError,
    console_report,
    frontier_report,
    frontier_text,
    sustained_streams_report,
    vendor_tsv,
    vllm_report,
)
from asrbench.runner import AdvancingClock, run_benchmark


def _receipt(tmp_path: Path) -> dict[str, Any]:
    config = resolve(env={}, overrides={("run", "output_path"): "receipt.json"})
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
    receipt: dict[str, Any] = asyncio.run(
        run_benchmark(
            config,
            clock=AdvancingClock(1.0),
            hardware=hardware,
            git_state=GitState(commit="unknown", dirty=False, available=False),
            package_version="0.0.1",
            python_version="3.12.0",
            run_id="report",
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
            materialize_dir=tmp_path,
            env={},
        )
    )
    return receipt


def _point(
    base: dict[str, Any],
    frames: int,
    wer: float | None,
    latency: float | None,
    *,
    gpu_name: str | None = None,
) -> dict[str, Any]:
    receipt = copy.deepcopy(base)
    receipt["lookahead"]["frames"] = frames
    receipt["lookahead"]["millisecond_label"] = float(frames * 80)
    workload = receipt["resolved_config"]["workload"]
    workload["lookahead_frames"] = frames
    workload["lookahead_ms_label"] = float(frames * 80)
    for entry in workload["population"]:
        entry["lookahead_frames"] = frames
        entry["lookahead_ms_label"] = float(frames * 80)
    receipt["summary"]["pooled"]["wer"]["micro_wer"] = wer
    receipt["summary"]["pooled"]["ttfs_s"]["p50"] = latency
    if gpu_name is not None:
        receipt["identity"]["hardware"]["gpu_name"] = gpu_name
    return receipt


def test_console_and_vllm_are_deterministic_projections(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    assert console_report(receipt) == console_report(receipt)
    report = vllm_report(receipt)
    assert "ttfs_s" in report
    assert "response_latency_s" in report
    assert "ttft" not in report
    assert "ttft" not in report["ttfs_s"]
    assert report["ttfs_s"]["clock"] != report["response_latency_s"]["clock"]
    text = vendor_tsv(receipt)
    assert text.startswith("# ttfs_clock ")
    assert "\tttfs_p50_s\t" in text
    assert text.count("\n") == 3


def test_vendor_tsv_refuses_a_mixed_population(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    workload = receipt["resolved_config"]["workload"]
    workload["population"] = [
        {"lanes": 1, "cadence_ms": 160, "lookahead_frames": 2},
        {"lanes": 1, "cadence_ms": 320, "lookahead_frames": 4},
    ]
    with pytest.raises(NotHomogeneousError):
        vendor_tsv(receipt)


def test_frontier_marks_the_pareto_edge_and_keeps_gpus_apart(
    tmp_path: Path,
) -> None:
    base = _receipt(tmp_path)
    points = [
        _point(base, 4, 0.2, 0.5),
        _point(base, 8, 0.1, 0.4),
        _point(base, 16, 0.05, 0.8),
        _point(base, 32, 0.15, 0.9),
        _point(base, 2, None, 0.1),
        _point(base, 8, 0.1, 0.4, gpu_name="H100"),
    ]
    report = frontier_report(
        points, [f"p{index}" for index in range(len(points))]
    )
    assert report["latency_source"] == "pooled.ttfs_s.p50"
    assert report["accuracy_source"] == "pooled.wer.micro_wer"
    assert len(report["series"]) == 2
    series = report["series"][0]
    by_frames = {point["lookahead_frames"]: point for point in series["points"]}
    assert by_frames[8]["on_pareto_edge"] is True
    assert by_frames[16]["on_pareto_edge"] is True
    assert by_frames[4]["dominated"] is True
    assert 8 in by_frames[4]["dominated_by"]
    assert by_frames[32]["dominated"] is True
    assert "dominated" in by_frames[32]["findings"]
    assert by_frames[2]["on_pareto_edge"] is False
    assert "missing_measure" in by_frames[2]["findings"]
    assert 2 not in by_frames[4]["dominated_by"]
    other = report["series"][1]["points"][0]
    assert other["lookahead_frames"] == 8
    assert report["series"][0]["group_key"] != report["series"][1]["group_key"]
    rendered = frontier_text(report)
    assert "frontier" in rendered
    assert "agreement" not in rendered


def test_sustained_streams_reports_a_non_monotonic_sweep(
    tmp_path: Path,
) -> None:
    base = _receipt(tmp_path)

    def lanes(count: int, passed: bool) -> dict[str, Any]:
        receipt = copy.deepcopy(base)
        workload = receipt["resolved_config"]["workload"]
        workload["lanes"] = count
        workload["population"][0]["lanes"] = count
        receipt["summary"]["conjunction_passed"] = passed
        return receipt

    report = sustained_streams_report(
        [lanes(1, False), lanes(4, True), lanes(2, True)],
        ["one", "four", "two"],
    )
    assert len(report["series"]) == 1
    series = report["series"][0]
    assert series["value"] == 4
    assert series["findings"] == ["non_monotonic"]
