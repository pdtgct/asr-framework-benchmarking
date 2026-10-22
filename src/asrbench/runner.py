"""Closed-loop lanes, warmup exclusion, and trials.

The runner owns concurrency and the measured span. It does not score
outcomes. A request rate is refused rather than run as closed-loop lanes:
that mode is a later step, and a silent substitution would report the
wrong experiment.

A second transport plugs in through :func:`load_adapter`. An entitled
lane this process cannot open must come back as an outcome with
``terminal_status`` ``lane_unavailable`` and ``configured`` false, not as
an exception that fails the process.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from asrbench.adapter import SessionAdmissionBarrier, TransportAdapter
from asrbench.audio import BYTES_PER_SAMPLE, SAMPLE_RATE, read_pcm16
from asrbench.clips import materialize
from asrbench.clock import utc_timestamp
from asrbench.config import ConfigError
from asrbench.identity import (
    GitState,
    HardwareRecord,
    collect_hardware,
    git_commit,
)
from asrbench.outcome import (
    FINALIZATION_CLOCK,
    NATIVE_LATENCY_CLOCK,
    RESPONSE_LATENCY_CLOCK,
    SessionOutcome,
    SessionRequest,
)
from asrbench.receipt import build_receipt
from asrbench.spec import ResolvedConfig, TargetSpec


class MonotonicClock(Protocol):
    """A clock the runner uses only to bound a measured cell."""

    def monotonic(self) -> float:
        """Return seconds from an arbitrary origin."""


class SystemClock:
    """``time.monotonic``."""

    def monotonic(self) -> float:
        """Return the process monotonic clock.

        Returns:
            Seconds.
        """
        return time.monotonic()


class AdvancingClock:
    """Test clock. Each measured trial is exactly ``step`` seconds.

    The runner calls :meth:`monotonic` once before the measured iterations
    of a trial and once after. The second call of each pair advances.
    """

    def __init__(self, step: float, start: float = 0.0) -> None:
        """Store the span each trial should report.

        Args:
            step: Seconds added on every even call.
            start: Reading returned by the first call.
        """
        self.step = step
        self._time = start
        self._calls = 0

    def monotonic(self) -> float:
        """Return the scripted reading.

        Returns:
            The current scripted time. Even-numbered calls advance it by
            :attr:`step` first.
        """
        self._calls += 1
        if self._calls % 2 == 0:
            self._time += self.step
        return self._time


class CohortBarrier:
    """Release every lane of one iteration together.

    ``parties`` waiters must arrive before any of them returns.
    """

    def __init__(self, parties: int) -> None:
        """Create a barrier for one iteration.

        Args:
            parties: How many lanes must arrive.
        """
        self.parties = parties
        self._arrived = 0
        self._lock = asyncio.Lock()
        self._event = asyncio.Event()

    async def wait(self, label: str) -> dict[str, str]:
        """Wait until the whole iteration has arrived.

        Args:
            label: Session label, echoed so the outcome records who waited.

        Returns:
            A deterministic marker. No wall timestamp.
        """
        async with self._lock:
            self._arrived += 1
            if self._arrived >= self.parties:
                self._event.set()
        await self._event.wait()
        return {"cohort_label": label}


def load_adapter(target: TargetSpec) -> TransportAdapter:
    """Construct the adapter named by ``target.transport``.

    Args:
        target: The resolved target. Mock options are read from
            ``transport_options`` and are not a secret.

    Returns:
        An adapter the runner can call.

    Raises:
        ConfigError: The transport is unknown, or it is the realtime
            WebSocket adapter, which needs a caller-owned client session
            and is not opened by this function.
    """
    if target.transport == "mock":
        from asrbench.adapters.mock import MockAdapter

        options = target.transport_options
        transcript = options.get("transcript", "")
        if not isinstance(transcript, str):
            raise ConfigError("mock transcript must be a string")
        every = options.get("interim_every_chunks", 2)
        if isinstance(every, bool) or not isinstance(every, int) or every < 1:
            raise ConfigError(
                "mock interim_every_chunks must be a positive int"
            )
        return MockAdapter(transcript, every)
    if target.transport == "realtime-ws":
        raise ConfigError(
            "realtime-ws needs a caller-owned client session; "
            "pass the adapter to run_benchmark instead of asking the CLI "
            "to open it"
        )
    raise ConfigError(
        f"unknown transport {target.transport!r}. A further protocol "
        "plugs in by handling its transport name in load_adapter. An "
        "entitled lane this process cannot open must return an outcome "
        "with terminal_status 'lane_unavailable' and configured False, "
        "not raise out of the run."
    )


async def run_benchmark(
    config: ResolvedConfig,
    *,
    adapter: TransportAdapter | None = None,
    clock: MonotonicClock | None = None,
    hardware: HardwareRecord | None = None,
    git_state: GitState | None = None,
    package_version: str | None = None,
    python_version: str | None = None,
    run_id: str | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    materialize_dir: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run closed-loop lanes and return a receipt.

    Args:
        config: Resolved config.
        adapter: Transport. The default is :func:`load_adapter`.
        clock: Span clock. The default is the process monotonic clock.
        hardware: Injected identity. The default probes the machine.
        git_state: Injected commit. The default asks git.
        package_version: Injected version. The default is the package's.
        python_version: Injected interpreter version.
        run_id: Injected id. The default is a random UUID hex.
        started_at: Injected start marker.
        finished_at: Injected finish marker. When omitted, the marker is
            taken after the trials.
        materialize_dir: Where the clip WAV is written.
        env: Environment used only to record whether the auth variable
            was set.

    Returns:
        The receipt dict.

    Raises:
        ConfigError: The workload asks for a request rate. That is not
            this runner.
    """
    if config.workload.rate_rps is not None:
        raise ConfigError(
            "rate_rps is offered-load mode and is not implemented; "
            "it is not run as closed-loop lanes"
        )
    chosen = adapter if adapter is not None else load_adapter(config.target)
    chosen_clock = clock if clock is not None else SystemClock()
    directory = materialize_dir
    if directory is None:
        directory = Path(tempfile.mkdtemp(prefix="asrbench-"))
    wav_path = materialize(config.workload.clip, directory)
    pcm = read_pcm16(wav_path)
    audio_s = len(pcm) / BYTES_PER_SAMPLE / SAMPLE_RATE
    start_marker = started_at if started_at is not None else utc_timestamp()
    trials: list[dict[str, Any]] = []
    for trial_index in range(config.workload.trials):
        warmup_outcomes: list[SessionOutcome] = []
        for index in range(config.workload.warmup):
            warmup_outcomes.extend(
                await _wave(
                    config,
                    chosen,
                    trial_index=trial_index,
                    phase="warmup",
                    iteration=index,
                    audio_s=audio_s,
                    wav_path=wav_path,
                )
            )
        span_start = chosen_clock.monotonic()
        measured: list[SessionOutcome] = []
        for index in range(config.workload.iterations):
            measured.extend(
                await _wave(
                    config,
                    chosen,
                    trial_index=trial_index,
                    phase="measured",
                    iteration=index,
                    audio_s=audio_s,
                    wav_path=wav_path,
                )
            )
        span_end = chosen_clock.monotonic()
        trials.append(
            {
                "trial_index": trial_index,
                "wall_span_s": span_end - span_start,
                "outcomes": measured,
                "warmup_outcomes": warmup_outcomes,
            }
        )
    finish_marker = finished_at if finished_at is not None else utc_timestamp()
    from asrbench import __version__
    from asrbench.config import secret_records

    version = package_version if package_version is not None else __version__
    py_version = (
        python_version if python_version is not None else _python_version()
    )
    return build_receipt(
        config,
        trials,
        secrets=secret_records(config, env or {}),
        hardware=hardware if hardware is not None else collect_hardware(),
        git_state=git_state if git_state is not None else git_commit(),
        package_version=version,
        python_version=py_version,
        adapter_name=chosen.name,
        run_id=run_id if run_id is not None else uuid.uuid4().hex,
        started_at=start_marker,
        finished_at=finish_marker,
        materialized_path=str(wav_path),
    )


async def _wave(
    config: ResolvedConfig,
    adapter: TransportAdapter,
    *,
    trial_index: int,
    phase: str,
    iteration: int,
    audio_s: float,
    wav_path: Path,
) -> list[SessionOutcome]:
    lanes = config.workload.lanes
    barrier: SessionAdmissionBarrier = CohortBarrier(lanes)

    async def one(lane: int) -> SessionOutcome:
        kind = "w" if phase == "warmup" else "i"
        request = SessionRequest(
            endpoint_url=config.target.endpoint_url,
            model=config.target.model,
            clip_path=wav_path,
            label=f"t{trial_index}-lane{lane}-{kind}{iteration}",
            chunk_ms=config.workload.chunk_ms,
            cadence_ms=config.workload.cadence_ms,
            timeout_s=config.workload.timeout_s,
            mode="full",
            pace=config.workload.pacing,
        )
        try:
            return await asyncio.wait_for(
                adapter.run(request, admission_barrier=barrier),
                timeout=config.workload.timeout_s,
            )
        except (TimeoutError, asyncio.TimeoutError):
            return _failed(request, audio_s, timed_out=True, error=None)
        except Exception as error:  # noqa: BLE001 - a lane must stay data
            return _failed(request, audio_s, timed_out=False, error=error)

    results: Sequence[SessionOutcome] = await asyncio.gather(
        *(one(lane) for lane in range(lanes))
    )
    return list(results)


def _failed(
    request: SessionRequest,
    audio_s: float,
    *,
    timed_out: bool,
    error: BaseException | None,
) -> SessionOutcome:
    outcome = SessionOutcome(
        label=request.label,
        mode=request.mode,
        clip_name=request.clip_path.name,
        chunk_ms=request.chunk_ms,
        audio_duration_s=audio_s,
        latency_clock=NATIVE_LATENCY_CLOCK,
        response_latency_clock=RESPONSE_LATENCY_CLOCK,
        final_latency_clock=FINALIZATION_CLOCK,
        timed_out=timed_out,
        interim_texts=None,
        terminal_status="timed_out" if timed_out else "errored",
    )
    if timed_out:
        outcome.protocol_extras["timed_out"] = True
    if error is not None:
        outcome.protocol_extras["exception"] = (
            f"{type(error).__name__}: {error}"
        )
    return outcome


def _python_version() -> str:
    import platform

    return platform.python_version()
