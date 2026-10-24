"""Closed-loop lanes, warmup exclusion, and trials.

The runner owns concurrency and the measured span. It does not score
outcomes. A request rate is refused rather than run as closed-loop lanes:
that mode is a later step, and a silent substitution would report the
wrong experiment.

Each lane runs at its population entry's cadence and lookahead. A mixed
population is offered as declared: the top-level cadence is null, and no
lane falls through to the server's default. A single-entry population is
the same cadence on every lane.

Measured iterations are not lockstep waves. Lanes share one barrier
before the first measured session, then each lane loops its own
iterations. The adapter owns each ``timeout_s``. The runner's wait is
the hang guard in
:data:`asrbench.definitions.RUNNER_HANG_GUARD_DEFINITION`.

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
from pathlib import Path
from typing import Any, Protocol

from asrbench.adapter import TransportAdapter
from asrbench.audio import BYTES_PER_SAMPLE, SAMPLE_RATE, read_pcm16
from asrbench.clips import materialize
from asrbench.clock import utc_timestamp
from asrbench.config import ConfigError
from asrbench.definitions import RUNNER_HANG_GRACE_S
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
from asrbench.spec import (
    PopulationEntry,
    ResolvedConfig,
    TargetSpec,
    WorkloadSpec,
)


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
    """Release the lanes that reach the first measured session.

    A lane that leaves before :meth:`wait` must call :meth:`abandon` so
    the others are not held until the hang guard.
    """

    def __init__(self, parties: int, *, timeout_s: float | None = None) -> None:
        """Create a barrier for one trial's measured start.

        Args:
            parties: How many lanes are in the trial.
            timeout_s: Bound on the wait. ``None`` waits until release.
                The runner passes one session timeout, the budget of the
                slowest handshake.
        """
        self.parties = parties
        self.timeout_s = timeout_s
        self._arrived = 0
        self._labels: set[str] = set()
        self._lock = asyncio.Lock()
        self._event = asyncio.Event()

    async def wait(self, label: str) -> dict[str, str]:
        """Wait until every lane has arrived or abandoned.

        Args:
            label: Session label, echoed so the outcome records who waited.

        Returns:
            A deterministic marker. No wall timestamp.
        """
        async with self._lock:
            if label not in self._labels:
                self._labels.add(label)
                self._arrived += 1
            if self._arrived >= self.parties:
                self._event.set()
        if self.timeout_s is None:
            await self._event.wait()
        else:
            await asyncio.wait_for(self._event.wait(), self.timeout_s)
        return {"cohort_label": label}

    async def abandon(self, label: str) -> None:
        """Count a lane that will not arrive.

        Args:
            label: The same label :meth:`wait` would have used. A second
                call, or a call after ``wait``, does nothing.
        """
        async with self._lock:
            if label in self._labels:
                return
            self._labels.add(label)
            self.parties -= 1
            if self.parties < 1 or self._arrived >= self.parties:
                self._event.set()


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


def hang_guard_s(timeout_s: float, audio_s: float, *, pacing: bool) -> float:
    """Return the runner's hang budget for one session.

    realtime-ws may spend ``timeout_s`` before the barrier and
    ``timeout_s`` again after the audio. Peers may also sit at the
    barrier for one ``timeout_s``. Paced sends add the clip duration.

    Args:
        timeout_s: The budget the adapter itself owns, used twice, plus
            once for the barrier.
        audio_s: Clip duration.
        pacing: When true, real-time send time is added.

    Returns:
        Seconds. The adapter may return a partial outcome before this.
    """
    paced = audio_s if pacing else 0.0
    return 2 * timeout_s + timeout_s + paced + RUNNER_HANG_GRACE_S


def lane_entries(
    population: tuple[PopulationEntry, ...],
) -> list[PopulationEntry]:
    """Expand a population into one entry per lane, in declaration order.

    Args:
        population: Population entries. Each is repeated ``lanes`` times.

    Returns:
        One entry per lane. A mixed cadence stays per lane.
    """
    assigned: list[PopulationEntry] = []
    for entry in population:
        assigned.extend([entry] * entry.lanes)
    return assigned


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
        git_state: Injected commit. The default asks git for this package.
        package_version: Injected version. The default is the package's.
        python_version: Injected interpreter version.
        run_id: Injected id. The default is a random UUID hex.
        started_at: Injected start marker.
        finished_at: Injected finish marker. When omitted, the marker is
            taken after the trials.
        materialize_dir: Where the clip WAV is written. The receipt records
            that path, so the directory is left in place.
        env: Environment used only to record whether the auth variable
            was set.

    Returns:
        The receipt dict.

    Raises:
        ConfigError: The workload asks for a request rate, or the
            population does not cover ``lanes``.
    """
    if config.workload.rate_rps is not None:
        raise ConfigError(
            "rate_rps is offered-load mode and is not implemented; "
            "it is not run as closed-loop lanes"
        )
    assigned = lane_entries(config.workload.population)
    if len(assigned) != config.workload.lanes:
        raise ConfigError("population lane counts must sum to workload.lanes")
    chosen = adapter if adapter is not None else load_adapter(config.target)
    _refuse_unsent_options(chosen, config.workload)
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
        warmup_groups = await _phase(
            config,
            chosen,
            assigned,
            trial_index=trial_index,
            phase="warmup",
            count=config.workload.warmup,
            audio_s=audio_s,
            wav_path=wav_path,
            barrier=None,
        )
        barrier = CohortBarrier(
            len(assigned), timeout_s=config.workload.timeout_s
        )
        span_start = chosen_clock.monotonic()
        measured_groups = await _phase(
            config,
            chosen,
            assigned,
            trial_index=trial_index,
            phase="measured",
            count=config.workload.iterations,
            audio_s=audio_s,
            wav_path=wav_path,
            barrier=barrier,
        )
        span_end = chosen_clock.monotonic()
        warmup = [item for group in warmup_groups for item in group]
        measured = [item for group in measured_groups for item in group]
        trials.append(
            {
                "trial_index": trial_index,
                "wall_span_s": span_end - span_start,
                "outcomes": measured,
                "warmup_outcomes": warmup,
            }
        )
    finish_marker = finished_at if finished_at is not None else utc_timestamp()
    from asrbench import __version__
    from asrbench.config import secret_records

    version = package_version if package_version is not None else __version__
    fallback = _python_version()
    py_version = fallback if python_version is None else python_version
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


_OPTION_NAMES = (
    "language",
    "punctuation",
    "diarization",
    "interim_results",
    "transcript_policy",
    "mode",
)
_OPTION_DEFAULTS = WorkloadSpec()


def _refuse_unsent_options(
    adapter: TransportAdapter, workload: WorkloadSpec
) -> None:
    """Refuse non-default workload options the adapter does not send.

    Args:
        adapter: The transport. Adapters that omit
            ``supported_workload_options`` are not checked.
        workload: Offered work.

    Raises:
        ConfigError: A non-default option would be fingerprinted and
            never sent.
    """
    supported = getattr(adapter, "supported_workload_options", None)
    if supported is None:
        return
    refused = [
        name
        for name in _OPTION_NAMES
        if name not in supported
        and getattr(workload, name) != getattr(_OPTION_DEFAULTS, name)
    ]
    if refused:
        raise ConfigError(
            f"{adapter.name} does not send {', '.join(refused)}; "
            "refusing a non-default value that would be fingerprinted "
            "and not offered"
        )


async def _phase(
    config: ResolvedConfig,
    adapter: TransportAdapter,
    assigned: list[PopulationEntry],
    *,
    trial_index: int,
    phase: str,
    count: int,
    audio_s: float,
    wav_path: Path,
    barrier: CohortBarrier | None,
) -> list[list[SessionOutcome]]:
    groups: list[list[SessionOutcome]] = list(
        await asyncio.gather(
            *(
                _lane_sessions(
                    config,
                    adapter,
                    entry,
                    trial_index=trial_index,
                    lane=lane,
                    phase=phase,
                    count=count,
                    audio_s=audio_s,
                    wav_path=wav_path,
                    barrier=barrier,
                )
                for lane, entry in enumerate(assigned)
            )
        )
    )
    return groups


async def _lane_sessions(
    config: ResolvedConfig,
    adapter: TransportAdapter,
    entry: PopulationEntry,
    *,
    trial_index: int,
    lane: int,
    phase: str,
    count: int,
    audio_s: float,
    wav_path: Path,
    barrier: CohortBarrier | None,
) -> list[SessionOutcome]:
    outcomes: list[SessionOutcome] = []
    for index in range(count):
        outcomes.append(
            await _session(
                config,
                adapter,
                entry,
                trial_index=trial_index,
                lane=lane,
                phase=phase,
                iteration=index,
                audio_s=audio_s,
                wav_path=wav_path,
                barrier=barrier if index == 0 else None,
            )
        )
    return outcomes


async def _session(
    config: ResolvedConfig,
    adapter: TransportAdapter,
    entry: PopulationEntry,
    *,
    trial_index: int,
    lane: int,
    phase: str,
    iteration: int,
    audio_s: float,
    wav_path: Path,
    barrier: CohortBarrier | None,
) -> SessionOutcome:
    kind = "w" if phase == "warmup" else "i"
    request = SessionRequest(
        endpoint_url=config.target.endpoint_url,
        model=config.target.model,
        clip_path=wav_path,
        label=f"t{trial_index}-lane{lane}-{kind}{iteration}",
        chunk_ms=config.workload.chunk_ms,
        cadence_ms=entry.cadence_ms,
        timeout_s=config.workload.timeout_s,
        mode="full",
        pace=config.workload.pacing,
        language=config.workload.language,
        punctuation=config.workload.punctuation,
        diarization=config.workload.diarization,
        interim_results=config.workload.interim_results,
        transcript_policy=config.workload.transcript_policy,
        lookahead_frames=entry.lookahead_frames,
        workload_mode=config.workload.mode,
    )
    guard = hang_guard_s(
        config.workload.timeout_s, audio_s, pacing=config.workload.pacing
    )
    try:
        return await asyncio.wait_for(
            adapter.run(request, admission_barrier=barrier),
            timeout=guard,
        )
    except (TimeoutError, asyncio.TimeoutError):  # noqa: UP041
        return _failed(request, audio_s, timed_out=True, error=None)
    except Exception as error:  # noqa: BLE001 - a lane must stay data
        return _failed(request, audio_s, timed_out=False, error=error)
    finally:
        if barrier is not None:
            await barrier.abandon(request.label)


def _failed(
    request: SessionRequest,
    audio_s: float,
    *,
    timed_out: bool,
    error: BaseException | None,
) -> SessionOutcome:
    detail = None if error is None else f"{type(error).__name__}: {error}"
    return SessionOutcome(
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
        error_detail=detail,
    )


def _python_version() -> str:
    import platform

    return platform.python_version()
