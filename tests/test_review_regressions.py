"""Regressions for the step-2 review: comparison split and a golden receipt."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

from asrbench.compare import compare_receipts
from asrbench.config import resolve
from asrbench.identity import (
    FixedProbe,
    GitState,
    GpuRow,
    HardwareRecord,
    collect_hardware,
)
from asrbench.runner import AdvancingClock, run_benchmark

_FIXTURE = Path(__file__).parent / "fixtures" / "golden-receipt.json"
_CLIP = Path("golden-clip")


def _hardware() -> HardwareRecord:
    return collect_hardware(
        FixedProbe(None),
        cpu_model="injected-cpu",
        logical_cpu_count=2,
        architecture="x86_64",
        system="Linux",
        hostname="injected-host",
        collected_at="2026-01-01T00:00:00Z",
    )


def _run(
    tmp_path: Path,
    *,
    run_id: str = "review",
    overrides: dict[tuple[str, ...], Any] | None = None,
    materialize_dir: Path | None = None,
) -> dict[str, Any]:
    chosen = dict(overrides or {})
    chosen.setdefault(("run", "output_path"), "receipt.json")
    config = resolve(env={}, overrides=chosen)
    receipt: dict[str, Any] = asyncio.run(
        run_benchmark(
            config,
            clock=AdvancingClock(1.0),
            hardware=_hardware(),
            git_state=GitState(commit="unknown", dirty=False, available=False),
            package_version="0.0.1",
            python_version="3.12.0",
            run_id=run_id,
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
            materialize_dir=materialize_dir or tmp_path,
            env={},
        )
    )
    return receipt


def test_golden_receipt_matches_the_fixture() -> None:
    try:
        receipt = _run(
            Path("."),
            run_id="golden",
            overrides={
                (
                    "run",
                    "pass_policy",
                    "sustainability_policy",
                ): "completion_fraction",
            },
            materialize_dir=_CLIP,
        )
        text = json.dumps(receipt, indent=2, allow_nan=False) + "\n"
        assert _FIXTURE.read_text(encoding="utf-8") == text
        assert "golden-clip/" in text
        assert "/workspace" not in text
    finally:
        shutil.rmtree(_CLIP, ignore_errors=True)


def test_a_language_change_names_the_field(tmp_path: Path) -> None:
    left = _run(tmp_path)
    right = json.loads(json.dumps(left))
    right["resolved_config"]["workload"]["language"] = "fr"
    result = compare_receipts(left, right)
    assert result.status == "refused"
    assert any(item.field.endswith("language") for item in result.differences)


def test_driver_and_git_commit_warn_and_do_not_refuse(tmp_path: Path) -> None:
    left = _run(tmp_path)
    driver = json.loads(json.dumps(left))
    driver["probe"]["hardware"]["driver_version"] = "999.99"
    driver_result = compare_receipts(left, driver)
    assert driver_result.status == "comparable"
    assert any(
        item.field == "driver_version" for item in driver_result.warnings
    )
    commit = json.loads(json.dumps(left))
    commit["probe"]["software"]["git_commit"] = "abc123"
    commit_result = compare_receipts(left, commit)
    assert commit_result.status == "comparable"
    assert any(item.field == "git_commit" for item in commit_result.warnings)


def test_server_gpu_and_server_version_are_hard(tmp_path: Path) -> None:
    left = _run(
        tmp_path,
        overrides={
            ("target", "server_gpu_name"): "A40",
            ("target", "server_gpu_count"): 1,
            ("target", "server_gpu_memory_mib"): 46068,
            ("target", "server_version"): "1.2.3",
            ("target", "server_driver_version"): "550.54.15",
        },
    )
    assert left["identity"]["hardware"]["gpu_name"] == "A40"
    source = left["identity"]["hardware"]["gpu_identity_source"]
    assert source == "server_declared"
    assert left["identity"]["target"]["server_version"] == "1.2.3"
    assert left["identity"]["warn"]["driver_version"] == "550.54.15"
    driver_source = left["identity"]["warn"]["driver_version_source"]
    assert driver_source == "server_declared"
    same = _run(
        tmp_path,
        run_id="again",
        overrides={
            ("target", "server_gpu_name"): "A40",
            ("target", "server_gpu_count"): 1,
            ("target", "server_gpu_memory_mib"): 46068,
            ("target", "server_version"): "1.2.3",
            ("target", "server_driver_version"): "550.54.15",
        },
    )
    assert compare_receipts(left, same).status == "comparable"
    other = _run(
        tmp_path,
        run_id="other",
        overrides={
            ("target", "server_gpu_name"): "H100",
            ("target", "server_gpu_count"): 1,
            ("target", "server_gpu_memory_mib"): 81559,
            ("target", "server_version"): "9.9.9",
            ("target", "server_driver_version"): "550.54.15",
        },
    )
    refused = compare_receipts(left, other)
    assert refused.status == "refused"
    fields = {item.field for item in refused.differences}
    assert "gpu_name" in fields
    assert "server_version" in fields


def test_a_tampered_comparison_key_is_stored_versus_recomputed(
    tmp_path: Path,
) -> None:
    left = _run(tmp_path)
    right = json.loads(json.dumps(left))
    right["comparison_key"] = "sha256:" + ("ab" * 32)
    result = compare_receipts(left, right)
    assert result.status == "refused"
    assert any(
        item.component == "comparison_key"
        and item.field == "stored_vs_recomputed"
        for item in result.differences
    )


def test_distinct_client_probes_refuse_on_gpu_name(tmp_path: Path) -> None:
    def probed(rows: tuple[GpuRow, ...], run_id: str) -> dict[str, Any]:
        hardware = collect_hardware(
            FixedProbe(rows),
            cpu_model="injected-cpu",
            logical_cpu_count=2,
            architecture="x86_64",
            system="Linux",
            hostname="injected-host",
            collected_at="2026-01-01T00:00:00Z",
        )
        config = resolve(
            env={}, overrides={("run", "output_path"): "receipt.json"}
        )
        receipt: dict[str, Any] = asyncio.run(
            run_benchmark(
                config,
                clock=AdvancingClock(1.0),
                hardware=hardware,
                git_state=GitState(
                    commit="unknown", dirty=False, available=False
                ),
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

    left = probed((GpuRow("A40", 46068, "550.54.15", "GPU-a"),), "a40")
    right = probed((GpuRow("H100", 81559, "550.54.15", "GPU-b"),), "h100")
    result = compare_receipts(left, right)
    assert result.status == "refused"
    assert any(item.field == "gpu_name" for item in result.differences)
    assert left["identity"]["hardware"]["gpu_identity_source"] == (
        "client_probe"
    )


def test_schema_version_tamper_refuses(tmp_path: Path) -> None:
    left = _run(tmp_path)
    right = json.loads(json.dumps(left))
    right["schema_version"] = "asrbench-receipt/0"
    result = compare_receipts(left, right)
    assert result.status == "refused"
    assert any("schema_version" in item.field for item in result.differences)
