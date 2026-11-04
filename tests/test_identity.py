"""Hardware and git helpers are injectable and never raise into a run."""

import subprocess

from asrbench.config import resolve
from asrbench.fingerprint import identity_blocks
from asrbench.identity import (
    NVIDIA_SMI_ARGV,
    FixedProbe,
    GpuRow,
    NvidiaSmiProbe,
    collect_hardware,
    git_commit,
    package_repo,
    parse_nvidia_smi_csv,
)


def test_nvidia_smi_csv_parses_and_skips_short_lines() -> None:
    text = (
        "NVIDIA A40, 46068, 550.54.15, GPU-aaa\n"
        "not-enough-columns\n"
        "NVIDIA H100, 81559, 550.54.15, GPU-bbb\n"
    )
    rows = parse_nvidia_smi_csv(text)
    assert rows[0] == GpuRow("NVIDIA A40", 46068, "550.54.15", "GPU-aaa")
    assert len(rows) == 2


def test_missing_nvidia_smi_is_absent_not_fatal() -> None:
    def missing(
        argv: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        assert argv == list(NVIDIA_SMI_ARGV)
        del check, capture_output, text, timeout
        raise FileNotFoundError("nvidia-smi")

    probe = NvidiaSmiProbe(missing)
    assert probe.query() is None
    record = collect_hardware(
        probe,
        cpu_model="injected-cpu",
        logical_cpu_count=2,
        architecture="x86_64",
        system="Linux",
        hostname="injected-host",
        collected_at="2026-01-01T00:00:00Z",
    )
    assert record.gpu_name is None
    assert record.gpu_count is None
    assert record.probe_status == "absent"
    assert record.hostname == "injected-host"
    identity = identity_blocks(
        resolve(env={}),
        {
            "gpu_name": record.gpu_name,
            "gpu_count": record.gpu_count,
            "gpu_memory_mib": record.gpu_memory_mib,
            "driver_version": record.driver_version,
            "system": record.system,
            "architecture": record.architecture,
            "cpu_model": record.cpu_model,
            "logical_cpu_count": record.logical_cpu_count,
            "hostname": record.hostname,
            "probe_status": record.probe_status,
        },
        {
            "adapter": "mock",
            "schema_version": "asrbench-receipt/2",
            "measurement_semantics_version": "1",
            "package_version": "0",
            "python_version": "3.12.0",
            "git_commit": "unknown",
            "git_dirty": False,
        },
    )["hardware"]
    assert "hostname" not in identity
    assert "gpu_uuids" not in identity
    assert "driver_version" not in identity
    assert identity["gpu_identity_source"] == "client_probe"


def test_distinct_gpu_names_are_sorted_into_one_identity_field() -> None:
    rows = (
        GpuRow("H100", 80000, "550", "GPU-b"),
        GpuRow("A40", 46000, "550", "GPU-a"),
    )
    record = collect_hardware(
        FixedProbe(rows),
        cpu_model="cpu",
        logical_cpu_count=4,
        architecture="x86_64",
        system="Linux",
        hostname="host-should-not-be-identity",
        collected_at="2026-01-01T00:00:00Z",
    )
    assert record.gpu_name == "A40+H100"
    assert record.gpu_memory_mib == 46000
    assert record.gpu_uuids == ("GPU-a", "GPU-b")
    assert record.probe_status == "ok"


def test_git_commit_uses_the_injected_runner() -> None:
    seen: list[str | None] = []

    def runner(
        argv: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: float,
        cwd: str | None,
    ) -> subprocess.CompletedProcess[str]:
        del check, capture_output, text, timeout
        seen.append(cwd)
        if argv == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(argv, 0, stdout="abc123\n")
        return subprocess.CompletedProcess(argv, 0, stdout=" M file\n")

    state = git_commit(runner=runner)
    assert state.commit == "abc123"
    assert state.dirty is True
    assert state.available is True
    repo = package_repo()
    assert repo is not None
    assert seen == [str(repo), str(repo)]


def test_git_failure_is_unknown() -> None:
    def boom(
        argv: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: float,
        cwd: str | None,
    ) -> subprocess.CompletedProcess[str]:
        del argv, check, capture_output, text, timeout, cwd
        raise OSError("git is absent")

    state = git_commit(runner=boom)
    assert state.commit == "unknown"
    assert state.available is False
