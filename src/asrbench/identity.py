"""Hardware identity and the one git-commit helper.

Both are best-effort and stdlib-only. A missing GPU probe or a missing
git checkout yields nulls and ``unknown``, and never raises into a run.
Tests inject records instead of calling the probes.
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from asrbench.clock import utc_timestamp
from asrbench.definitions import HARDWARE_PROBE_NVIDIA_SMI

NVIDIA_SMI_ARGV: tuple[str, ...] = (
    "nvidia-smi",
    "--query-gpu=name,memory.total,driver_version,uuid",
    "--format=csv,noheader,nounits",
)


@dataclass(frozen=True)
class GpuRow:
    """One GPU as nvidia-smi printed it."""

    name: str
    memory_mib: int | None
    driver_version: str
    uuid: str


@dataclass(frozen=True)
class HardwareRecord:
    """Identity fields plus the probe residue excluded from the key."""

    system: str
    architecture: str
    cpu_model: str | None
    logical_cpu_count: int | None
    gpu_name: str | None
    gpu_count: int | None
    gpu_memory_mib: int | None
    driver_version: str | None
    hostname: str
    gpu_uuids: tuple[str, ...]
    collected_at: str
    probe_name: str
    probe_status: str


@dataclass(frozen=True)
class GitState:
    """HEAD commit, and the dirty bit that the key ignores."""

    commit: str
    dirty: bool
    available: bool


Runner = Callable[..., subprocess.CompletedProcess[str]]
GitRunner = Callable[..., subprocess.CompletedProcess[str]]


def parse_nvidia_smi_csv(text: str) -> tuple[GpuRow, ...]:
    """Parse ``nvidia-smi --format=csv,noheader,nounits`` stdout.

    Args:
        text: Probe stdout. Blank lines are ignored.

    Returns:
        One row per non-blank line that has four columns. A line that
        does not is skipped rather than failing the run.
    """
    rows: list[GpuRow] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        memory: int | None
        try:
            memory = int(parts[1])
        except ValueError:
            memory = None
        rows.append(
            GpuRow(
                name=parts[0],
                memory_mib=memory,
                driver_version=parts[2],
                uuid=parts[3],
            )
        )
    return tuple(rows)


def _join_unique(values: Sequence[str]) -> str | None:
    unique = sorted({value for value in values if value})
    if not unique:
        return None
    return "+".join(unique)


class NvidiaSmiProbe:
    """GPU probe that shells out to nvidia-smi.

    ``runner`` is injectable so tests never execute the binary.
    """

    name = HARDWARE_PROBE_NVIDIA_SMI

    def __init__(self, runner: Runner | None = None) -> None:
        """Bind the process runner.

        Args:
            runner: A ``subprocess.run``-shaped callable. The default runs
                nvidia-smi.
        """
        self._runner = runner if runner is not None else subprocess.run

    def query(self) -> tuple[GpuRow, ...] | None:
        """Return GPU rows, or ``None`` when the probe cannot be used.

        Returns:
            Rows, or ``None`` on a missing binary, timeout, or non-zero
            status. ``None`` means absent; an empty tuple means the
            command succeeded and saw no GPU.
        """
        try:
            completed = self._runner(
                list(NVIDIA_SMI_ARGV),
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if completed.returncode != 0:
            return None
        return parse_nvidia_smi_csv(completed.stdout)


class FixedProbe:
    """A probe that returns a prepared result. Tests and the schema use it."""

    name = "fixed"

    def __init__(self, rows: tuple[GpuRow, ...] | None) -> None:
        """Store the result ``query`` will return.

        Args:
            rows: GPU rows, or ``None`` to look like a missing probe.
        """
        self._rows = rows

    def query(self) -> tuple[GpuRow, ...] | None:
        """Return the prepared rows.

        Returns:
            The rows passed to the constructor.
        """
        return self._rows


def package_repo() -> Path | None:
    """Return the git checkout that contains this package.

    Walking from this file avoids ``cwd``. A caller who runs asrbench
    from another repository must not have that repository's HEAD
    recorded as this package's commit.

    Returns:
        The directory with ``pyproject.toml`` naming ``asrbench`` and a
        ``.git`` entry, or ``None`` when this install is not a checkout.
    """
    start = Path(__file__).resolve().parent
    for candidate in (start, *start.parents):
        marker = candidate / "pyproject.toml"
        if not marker.is_file() or not (candidate / ".git").exists():
            continue
        text = marker.read_text(encoding="utf-8", errors="replace")
        if 'name = "asrbench"' in text:
            return candidate
    return None


def read_cpu_model() -> str | None:
    """Read a CPU model string without failing the run.

    Returns:
        The first ``model name`` in ``/proc/cpuinfo``, otherwise
        ``platform.processor`` when that is non-empty, otherwise ``None``.
    """
    try:
        text = Path("/proc/cpuinfo").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        text = ""
    for line in text.splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip() or None
    processor = platform.processor().strip()
    return processor or None


def _gpus_to_identity(
    rows: tuple[GpuRow, ...] | None,
) -> tuple[
    str | None, int | None, int | None, str | None, tuple[str, ...], str
]:
    if rows is None:
        return None, None, None, None, (), "absent"
    if not rows:
        return None, 0, None, None, (), "ok"
    names = [row.name for row in rows]
    drivers = [row.driver_version for row in rows]
    memories = [row.memory_mib for row in rows if row.memory_mib is not None]
    uuids = tuple(sorted(row.uuid for row in rows if row.uuid))
    memory = min(memories) if memories else None
    return (
        _join_unique(names),
        len(rows),
        memory,
        _join_unique(drivers),
        uuids,
        "ok",
    )


def collect_hardware(
    probe: NvidiaSmiProbe | FixedProbe | None = None,
    *,
    cpu_model: str | None = None,
    logical_cpu_count: int | None = None,
    architecture: str | None = None,
    system: str | None = None,
    hostname: str | None = None,
    collected_at: str | None = None,
    read_cpu: Callable[[], str | None] | None = None,
) -> HardwareRecord:
    """Collect hardware identity. Never raises.

    Args:
        probe: GPU probe. The default is nvidia-smi. Pass a
            :class:`FixedProbe` in tests.
        cpu_model: Injected CPU model. When ``None``, the machine is read.
        logical_cpu_count: Injected count. When ``None``, ``os.cpu_count``.
        architecture: Injected architecture. When ``None``,
            ``platform.machine``.
        system: Injected system name. When ``None``, ``platform.system``.
        hostname: Injected hostname, recorded on the probe only.
        collected_at: Injected timestamp.
        read_cpu: Replacement for :func:`read_cpu_model`.

    Returns:
        A record whose identity fields are safe to hash and whose probe
        fields are not.
    """
    chosen = probe if probe is not None else NvidiaSmiProbe()
    status_override: str | None = None
    try:
        rows = chosen.query()
    except Exception:  # noqa: BLE001 - a probe bug must not fail the run
        rows = None
        status_override = "error"
    gpu_name, gpu_count, memory, driver, uuids, status = _gpus_to_identity(rows)
    if status_override is not None:
        status = status_override
    if cpu_model is None:
        reader = read_cpu if read_cpu is not None else read_cpu_model
        try:
            cpu_model = reader()
        except OSError:
            cpu_model = None
    if logical_cpu_count is None:
        logical_cpu_count = os.cpu_count()
    return HardwareRecord(
        system=system if system is not None else platform.system(),
        architecture=architecture
        if architecture is not None
        else platform.machine(),
        cpu_model=cpu_model,
        logical_cpu_count=logical_cpu_count,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        gpu_memory_mib=memory,
        driver_version=driver,
        hostname=hostname if hostname is not None else socket.gethostname(),
        gpu_uuids=uuids,
        collected_at=collected_at
        if collected_at is not None
        else utc_timestamp(),
        probe_name=chosen.name,
        probe_status=status,
    )


def git_commit(
    cwd: Path | None = None,
    runner: GitRunner | None = None,
) -> GitState:
    """Return the git commit of asrbench's own repository, or ``unknown``.

    This is the only git-commit helper in the package. When ``cwd`` is
    omitted the helper uses :func:`package_repo`, never the process
    working directory. A missing checkout is ``unknown`` rather than
    some other project's HEAD.

    Args:
        cwd: Repository directory. ``None`` selects this package's
            checkout.
        runner: Replacement for ``subprocess.run``. Tests pass one so the
            helper can be exercised without a repository. ``cwd`` is
            still the package repo unless the caller passes one.

    Returns:
        Commit, dirty bit, and whether git answered. Failures become
        ``commit='unknown'`` and ``available=False``.
    """
    repo = cwd if cwd is not None else package_repo()
    if repo is None and runner is None:
        return GitState(commit="unknown", dirty=False, available=False)

    def invoke(argv: list[str]) -> subprocess.CompletedProcess[str]:
        chosen = runner if runner is not None else subprocess.run
        return chosen(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
            cwd=str(repo) if repo is not None else None,
        )

    try:
        head = invoke(["git", "rev-parse", "HEAD"])
        status = invoke(["git", "status", "--porcelain"])
    except (OSError, subprocess.TimeoutExpired):
        return GitState(commit="unknown", dirty=False, available=False)
    if head.returncode != 0:
        return GitState(commit="unknown", dirty=False, available=False)
    commit = head.stdout.strip() or "unknown"
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else False
    return GitState(commit=commit, dirty=dirty, available=True)
