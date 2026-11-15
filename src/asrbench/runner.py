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
import warnings
from pathlib import Path
from typing import Any, Protocol

from asrbench.adapter import TransportAdapter
from asrbench.audio import BYTES_PER_SAMPLE, SAMPLE_RATE, read_pcm16
from asrbench.clips import materialize, pcm16_wav
from asrbench.clock import utc_timestamp
from asrbench.config import ConfigError
from asrbench.definitions import (
    AUDIO_CONDITION_NATIVE,
    AUDIO_CONDITION_TELEPHONY_8K,
    RUNNER_HANG_GRACE_S,
)
from asrbench.fingerprint import sent_sample_rate
from asrbench.identity import (
    GitState,
    HardwareRecord,
    collect_hardware,
    git_commit,
)
from asrbench.manifest import (
    Manifest,
    ManifestError,
    assign_clip,
    load_manifest,
    materialize_manifest,
)
from asrbench.outcome import (
    FINALIZATION_CLOCK,
    NATIVE_LATENCY_CLOCK,
    RESPONSE_LATENCY_CLOCK,
    SessionOutcome,
    SessionRequest,
)
from asrbench.receipt import build_receipt
from asrbench.resample import (
    NARROWBAND_RATE,
    decimate_16k_to_8k,
    interpolate_8k_to_16k,
)
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
        ConfigError: The transport is unknown, the realtime WebSocket
            adapter was asked to open its own client session, or a
            transport extra is not installed.
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
        scripts = options.get("transcripts", {})
        if not isinstance(scripts, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in scripts.items()
        ):
            raise ConfigError("mock transcripts must map clip names to text")
        return MockAdapter(transcript, every, scripts)
    if target.transport == "realtime-ws":
        raise ConfigError(
            "realtime-ws needs a caller-owned client session; "
            "pass the adapter to run_benchmark instead of asking the CLI "
            "to open it"
        )
    if target.transport == "riva-grpc":
        from asrbench.adapters.riva_grpc import RivaGrpcAdapter

        return RivaGrpcAdapter.from_target(target)
    if target.transport == "nim-realtime":
        from asrbench.adapters.nim_realtime import NimRealtimeAdapter

        return NimRealtimeAdapter.from_target(target)
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
) -> list[tuple[int, PopulationEntry]]:
    """Expand a population into one entry per lane, in declaration order.

    Args:
        population: Population entries. Each is repeated ``lanes`` times.

    Returns:
        ``(entry_index, entry)`` per lane. A mixed cadence stays per lane.
        ``entry_index`` is the index in ``population``, shared by every
        lane of that entry.
    """
    assigned: list[tuple[int, PopulationEntry]] = []
    for index, entry in enumerate(population):
        assigned.extend((index, entry) for _ in range(entry.lanes))
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
    corpus = _load_corpus(config)
    if corpus is None:
        wav_path = materialize(config.workload.clip, directory)
        sources = [wav_path]
        materialized = str(wav_path)
    else:
        try:
            written = materialize_manifest(corpus, directory)
        except ManifestError as error:
            raise ConfigError(str(error)) from error
        sources = [written[clip.id] for clip in corpus.clips]
        materialized = str(directory)
    rate = sent_sample_rate(config.workload)
    _refuse_unsent_rate(chosen, rate)
    plays = [
        _condition(path, config.workload.audio_condition) for path in sources
    ]
    start_marker = started_at if started_at is not None else utc_timestamp()
    trials: list[dict[str, Any]] = []
    try:
        for trial_index in range(config.workload.trials):
            warmup_groups = await _phase(
                config,
                chosen,
                assigned,
                trial_index=trial_index,
                phase="warmup",
                count=config.workload.warmup,
                plays=plays,
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
                plays=plays,
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
    finally:
        await _close_adapter(chosen)
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
        materialized_path=materialized,
        corpus=corpus,
    )


def _condition(path: Path, condition: str) -> tuple[Path, float, int]:
    """Rewrite one materialized 16 kHz clip for the audio condition.

    The file keeps its name, so a corpus session still maps to its
    clip's reference. Duration is the source duration either way.
    """
    pcm = read_pcm16(path)
    seconds = len(pcm) / BYTES_PER_SAMPLE / SAMPLE_RATE
    if condition == AUDIO_CONDITION_NATIVE:
        return path, seconds, SAMPLE_RATE
    narrow = decimate_16k_to_8k(pcm)
    if condition == AUDIO_CONDITION_TELEPHONY_8K:
        path.write_bytes(pcm16_wav(narrow, NARROWBAND_RATE))
        return (
            path,
            len(narrow) / BYTES_PER_SAMPLE / NARROWBAND_RATE,
            (NARROWBAND_RATE),
        )
    wide = interpolate_8k_to_16k(narrow)
    path.write_bytes(pcm16_wav(wide, SAMPLE_RATE))
    return path, len(wide) / BYTES_PER_SAMPLE / SAMPLE_RATE, SAMPLE_RATE


def _refuse_unsent_rate(adapter: TransportAdapter, rate: int) -> None:
    """Refuse a sent rate the adapter cannot declare to its server.

    Adapters that omit ``supported_sample_rates`` send 16 kHz only.

    Raises:
        ConfigError: The adapter would mislabel the audio.
    """
    supported = getattr(adapter, "supported_sample_rates", (SAMPLE_RATE,))
    if rate not in supported:
        raise ConfigError(
            f"{adapter.name} sends only {sorted(supported)} Hz; use "
            "audio_condition telephony_8k_in_16k to offer 8 kHz content "
            "at 16 kHz"
        )


def _load_corpus(config: ResolvedConfig) -> Manifest | None:
    """Load the manifest a resolved workload names, re-checking its digest.

    Raises:
        ConfigError: The file moved or changed since resolution.
    """
    spec = config.workload.manifest
    if spec.digest is None:
        return None
    if spec.path is None:
        raise ConfigError("manifest workload has no local path to run")
    try:
        loaded = load_manifest(Path(spec.path))
    except ManifestError as error:
        raise ConfigError(str(error)) from error
    if loaded.digest != spec.digest:
        raise ConfigError(
            f"manifest {spec.path} digest {loaded.digest} != resolved "
            f"{spec.digest}"
        )
    return loaded


_OPTION_NAMES = (
    "language",
    "punctuation",
    "diarization",
    "interim_results",
    "transcript_policy",
    "mode",
    "enable_word_time_offsets",
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
    assigned: list[tuple[int, PopulationEntry]],
    *,
    trial_index: int,
    phase: str,
    count: int,
    plays: list[tuple[Path, float, int]],
    barrier: CohortBarrier | None,
) -> list[list[SessionOutcome]]:
    groups: list[list[SessionOutcome]] = list(
        await asyncio.gather(
            *(
                _lane_sessions(
                    config,
                    adapter,
                    entry,
                    population_index=index,
                    trial_index=trial_index,
                    lane=lane,
                    phase=phase,
                    count=count,
                    plays=plays,
                    lanes=len(assigned),
                    barrier=barrier,
                )
                for lane, (index, entry) in enumerate(assigned)
            )
        )
    )
    return groups


async def _lane_sessions(
    config: ResolvedConfig,
    adapter: TransportAdapter,
    entry: PopulationEntry,
    *,
    population_index: int,
    trial_index: int,
    lane: int,
    phase: str,
    count: int,
    plays: list[tuple[Path, float, int]],
    lanes: int,
    barrier: CohortBarrier | None,
) -> list[SessionOutcome]:
    outcomes: list[SessionOutcome] = []
    for index in range(count):
        wav_path, audio_s, rate = plays[
            assign_clip(
                lane=lane, iteration=index, lanes=lanes, clip_count=len(plays)
            )
        ]
        outcomes.append(
            await _session(
                config,
                adapter,
                entry,
                population_index=population_index,
                trial_index=trial_index,
                lane=lane,
                phase=phase,
                iteration=index,
                audio_s=audio_s,
                wav_path=wav_path,
                sample_rate=rate,
                barrier=barrier if index == 0 else None,
            )
        )
    return outcomes


async def _session(
    config: ResolvedConfig,
    adapter: TransportAdapter,
    entry: PopulationEntry,
    *,
    population_index: int,
    trial_index: int,
    lane: int,
    phase: str,
    iteration: int,
    audio_s: float,
    wav_path: Path,
    sample_rate: int,
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
        lifecycle="full",
        pace=config.workload.pacing,
        language=config.workload.language,
        punctuation=config.workload.punctuation,
        diarization=config.workload.diarization,
        interim_results=config.workload.interim_results,
        enable_word_time_offsets=config.workload.enable_word_time_offsets,
        transcript_policy=config.workload.transcript_policy,
        lookahead_frames=entry.lookahead_frames,
        workload_mode=config.workload.mode,
        sample_rate_hertz=sample_rate,
    )
    guard = hang_guard_s(
        config.workload.timeout_s, audio_s, pacing=config.workload.pacing
    )
    try:
        outcome = await asyncio.wait_for(
            adapter.run(request, admission_barrier=barrier),
            timeout=guard,
        )
    except (TimeoutError, asyncio.TimeoutError):  # noqa: UP041
        outcome = _failed(request, audio_s, timed_out=True, error=None)
    except Exception as error:  # noqa: BLE001 - a lane must stay data
        outcome = _failed(request, audio_s, timed_out=False, error=error)
    finally:
        if barrier is not None:
            await barrier.abandon(request.label)
    _record_entry(outcome, population_index, entry)
    return outcome


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
        lifecycle=request.lifecycle,
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


def _record_entry(
    outcome: SessionOutcome, population_index: int, entry: PopulationEntry
) -> None:
    """Stamp the population entry onto an outcome the runner produced.

    Adapters do not know which population entry a lane belongs to. The
    runner does, and every outcome — including a timeout the runner
    synthesized — carries that entry so a later reduction can refuse to
    pool a mixed population.
    """
    outcome.entry_index = population_index
    outcome.cadence_ms = entry.cadence_ms
    outcome.lookahead_frames = entry.lookahead_frames


def _python_version() -> str:
    import platform

    return platform.python_version()


async def _close_adapter(adapter: TransportAdapter) -> None:
    """Await the adapter's optional ``aclose``; never lose a receipt to it."""
    closer = getattr(adapter, "aclose", None)
    if not callable(closer):
        return
    try:
        await closer()
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"adapter aclose failed: {exc!r}", stacklevel=2)
