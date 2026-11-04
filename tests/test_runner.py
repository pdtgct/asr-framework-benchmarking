"""Mock lanes, warmup exclusion, timeouts, and the transport seam."""

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest

from asrbench.adapter import SessionAdmissionBarrier, SessionProgressCallback
from asrbench.config import ConfigError, resolve
from asrbench.identity import FixedProbe, GitState, collect_hardware
from asrbench.outcome import SessionOutcome, SessionRequest
from asrbench.runner import AdvancingClock, load_adapter, run_benchmark
from asrbench.spec import ResolvedConfig

_PRESET = Path("presets/mock-streaming.json")


def _hardware() -> object:
    return collect_hardware(
        FixedProbe(None),
        cpu_model="injected-cpu",
        logical_cpu_count=2,
        architecture="x86_64",
        system="Linux",
        hostname="injected-host",
        collected_at="2026-01-01T00:00:00Z",
    )


def _config() -> ResolvedConfig:
    config = resolve(
        env={},
        overrides={
            ("workload", "warmup"): 1,
            ("workload", "iterations"): 2,
            ("workload", "trials"): 2,
            ("workload", "lanes"): 1,
            ("run", "output_path"): "receipt.json",
            (
                "run",
                "pass_policy",
                "sustainability_policy",
            ): "completion_fraction",
        },
    )
    return config


def test_mock_run_excludes_warmup_and_uses_simulated_clocks(
    tmp_path: Path,
) -> None:
    from asrbench.identity import HardwareRecord

    hardware = _hardware()
    assert isinstance(hardware, HardwareRecord)
    receipt = asyncio.run(
        run_benchmark(
            _config(),
            clock=AdvancingClock(1.0),
            hardware=hardware,
            git_state=GitState(commit="unknown", dirty=True, available=False),
            package_version="0.0.1",
            python_version="3.12.0",
            run_id="runner-test",
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
            materialize_dir=tmp_path,
            env={},
        )
    )
    assert receipt["run_id"] == "runner-test"
    assert len(receipt["trials"]) == 2
    trial = receipt["trials"][0]
    assert trial["warmup_excluded"] == 1
    assert len(trial["outcomes"]) == 2
    assert len(trial["warmup_outcomes"]) == 1
    assert trial["cell_wall_span_s"] == 1.0
    assert receipt["summary"]["pooled"]["counts"]["attempted"] == 4
    assert receipt["summary"]["pooled"]["counts"]["warmup_excluded"] == 2
    outcome = trial["outcomes"][0]
    assert outcome["ttfs_s"] == 0.25
    assert outcome["latency_clock"].startswith("mock/simulated_")
    assert (
        receipt["summary"]["pooled"]["ttfs_s"]["clock_matches_native"] is False
    )
    rtfx = receipt["summary"]["pooled"]["streaming_rtfx"]
    assert rtfx["successful_audio_s"] == pytest.approx(6.4)
    assert rtfx["cell_wall_span_s"] == pytest.approx(2.0)
    assert rtfx["value"] == pytest.approx(3.2)
    assert rtfx["validity"] == "simulated"
    assert receipt["summary"]["conjunction_passed"] is True
    assert receipt["lookahead"]["server_reported_frames"] is None
    assert receipt["lookahead"]["frames_source"] == "client_declared"
    assert "super-secret" not in str(receipt)
    markers = outcome["protocol_extras"]["markers"]
    assert "cohort_label" in markers
    assert "time" not in markers


def test_server_reported_policy_does_not_invent_a_server_frame_count(
    tmp_path: Path,
) -> None:
    from asrbench.identity import HardwareRecord

    config = resolve(
        env={
            "ASRBENCH_LOOKAHEAD_IDENTITY_POLICY": "server_reported_when_present"
        },
        overrides={("run", "output_path"): "receipt.json"},
    )
    hardware = _hardware()
    assert isinstance(hardware, HardwareRecord)
    receipt = asyncio.run(
        run_benchmark(
            config,
            clock=AdvancingClock(1.0),
            hardware=hardware,
            git_state=GitState(commit="unknown", dirty=False, available=False),
            package_version="0.0.1",
            python_version="3.12.0",
            run_id="policy",
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
            materialize_dir=tmp_path,
            env={},
        )
    )
    assert receipt["lookahead"]["policy"] == "server_reported_when_present"
    assert receipt["lookahead"]["frames_source"] == "client_declared"
    assert receipt["lookahead"]["server_reported_frames"] is None
    assert receipt["lookahead"]["frames"] == 4


def test_a_stuck_lane_is_a_timeout_in_the_denominator(tmp_path: Path) -> None:
    from asrbench.identity import HardwareRecord

    class Stuck:
        name = "stuck"

        async def run(
            self,
            request: SessionRequest,
            *,
            admission_barrier: SessionAdmissionBarrier | None = None,
            progress_callback: SessionProgressCallback | None = None,
        ) -> SessionOutcome:
            del request, admission_barrier, progress_callback
            await asyncio.sleep(30)
            raise AssertionError("the runner should have timed the lane out")

    config = resolve(
        env={},
        overrides={
            ("workload", "timeout_s"): 0.05,
            ("workload", "pacing"): False,
            ("workload", "lanes"): 1,
            ("workload", "iterations"): 1,
            ("workload", "trials"): 1,
            ("run", "output_path"): "receipt.json",
        },
    )
    hardware = _hardware()
    assert isinstance(hardware, HardwareRecord)
    receipt = asyncio.run(
        run_benchmark(
            config,
            adapter=Stuck(),
            clock=AdvancingClock(1.0),
            hardware=hardware,
            git_state=GitState(commit="unknown", dirty=False, available=False),
            package_version="0.0.1",
            python_version="3.12.0",
            run_id="timeout",
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
            materialize_dir=tmp_path,
            env={},
        )
    )
    counts = receipt["summary"]["pooled"]["counts"]
    assert counts["attempted"] == 1
    assert counts["timed_out"] == 1
    assert counts["completed"] == 0
    assert receipt["trials"][0]["outcomes"][0]["ttfs_s"] is None


def test_a_raising_lane_is_an_errored_outcome(tmp_path: Path) -> None:
    from asrbench.identity import HardwareRecord

    class Boom:
        name = "boom"

        async def run(
            self,
            request: SessionRequest,
            *,
            admission_barrier: SessionAdmissionBarrier | None = None,
            progress_callback: SessionProgressCallback | None = None,
        ) -> SessionOutcome:
            del request, admission_barrier, progress_callback
            raise RuntimeError("adapter failed")

    config = resolve(
        env={},
        overrides={("run", "output_path"): "receipt.json"},
    )
    hardware = _hardware()
    assert isinstance(hardware, HardwareRecord)
    receipt = asyncio.run(
        run_benchmark(
            config,
            adapter=Boom(),
            clock=AdvancingClock(1.0),
            hardware=hardware,
            git_state=GitState(commit="unknown", dirty=False, available=False),
            package_version="0.0.1",
            python_version="3.12.0",
            run_id="boom",
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
            materialize_dir=tmp_path,
            env={},
        )
    )
    counts = receipt["summary"]["pooled"]["counts"]
    assert counts["errored"] == 1
    assert counts["attempted"] == 1
    text = receipt["trials"][0]["outcomes"][0]["error_detail"]
    assert "RuntimeError" in text
    extras = receipt["trials"][0]["outcomes"][0]["protocol_extras"]
    assert "exception" not in extras


def test_rate_rps_is_refused_rather_than_run_as_lanes() -> None:
    config = resolve(env={"ASRBENCH_RATE_RPS": "3"})
    with pytest.raises(ConfigError, match="rate_rps"):
        asyncio.run(run_benchmark(config, env={}))


def test_unknown_transport_names_the_lane_unavailable_seam() -> None:
    config = resolve(env={}, overrides={("target", "transport"): "future"})
    with pytest.raises(ConfigError, match="lane_unavailable") as caught:
        load_adapter(config.target)
    assert "future" in str(caught.value)


def test_realtime_ws_is_not_opened_by_the_runner() -> None:
    config = resolve(env={}, overrides={("target", "transport"): "realtime-ws"})
    with pytest.raises(ConfigError, match="caller-owned"):
        load_adapter(config.target)


def _drive(
    config: ResolvedConfig,
    tmp_path: Path,
    run_id: str,
    adapter: object | None = None,
) -> dict[str, Any]:
    from typing import cast

    from asrbench.adapter import TransportAdapter
    from asrbench.identity import HardwareRecord

    hardware = _hardware()
    assert isinstance(hardware, HardwareRecord)
    receipt: dict[str, Any] = asyncio.run(
        run_benchmark(
            config,
            adapter=cast(TransportAdapter | None, adapter),
            clock=AdvancingClock(1.0),
            hardware=hardware,
            git_state=GitState(commit="unknown", dirty=False, available=False),
            package_version="0.0.1",
            python_version="3.12.0",
            run_id=run_id,
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
            materialize_dir=tmp_path,
            env={},
        )
    )
    return receipt


def _success(request: SessionRequest) -> SessionOutcome:
    return SessionOutcome(
        label=request.label,
        mode=request.mode,
        clip_name=request.clip_path.name,
        chunk_ms=request.chunk_ms,
        audio_duration_s=1.6,
        configured=True,
        connected=True,
        transcript="hello",
        ttfs_s=0.2,
        final_latency_s=0.05,
        latency_clock="client_monotonic/test",
        interim_texts=["hello"],
        terminal_status="completed",
    )


def test_each_lane_runs_at_its_population_cadence(tmp_path: Path) -> None:
    payload = json.loads(_PRESET.read_text(encoding="utf-8"))
    workload = payload["workload"]
    workload["lanes"] = 3
    workload["iterations"] = 1
    workload["warmup"] = 0
    workload["trials"] = 1
    workload["cadence_ms"] = None
    workload["lookahead_frames"] = None
    workload["lookahead_ms_label"] = None
    workload["population"] = [
        {
            "lanes": 1,
            "cadence_ms": 160,
            "lookahead_frames": 2,
            "lookahead_ms_label": 160,
        },
        {
            "lanes": 2,
            "cadence_ms": 320,
            "lookahead_frames": 4,
            "lookahead_ms_label": 320,
        },
    ]
    preset = tmp_path / "mixed.json"
    preset.write_text(json.dumps(payload), encoding="utf-8")
    config = resolve(
        preset_path=preset,
        env={},
        overrides={("run", "output_path"): "receipt.json"},
    )
    seen: list[tuple[int | None, int | None]] = []

    class Recording:
        name = "recording"

        async def run(
            self,
            request: SessionRequest,
            *,
            admission_barrier: SessionAdmissionBarrier | None = None,
            progress_callback: SessionProgressCallback | None = None,
        ) -> SessionOutcome:
            del progress_callback
            seen.append((request.cadence_ms, request.lookahead_frames))
            if admission_barrier is not None:
                await admission_barrier.wait(request.label)
            return _success(request)

    receipt = _drive(config, tmp_path, "mixed", Recording())
    assert seen == [(160, 2), (320, 4), (320, 4)]
    resolved = receipt["resolved_config"]
    assert isinstance(resolved, dict)
    assert resolved["workload"]["cadence_ms"] is None
    assert resolved["workload"]["lookahead_frames"] is None


def test_hang_guard_covers_two_timeouts_and_the_barrier() -> None:
    from asrbench.definitions import RUNNER_HANG_GRACE_S
    from asrbench.runner import hang_guard_s

    assert hang_guard_s(1.0, 1.6, pacing=False) == 3.0 + RUNNER_HANG_GRACE_S
    paced = hang_guard_s(1.0, 1.6, pacing=True)
    assert paced == 3.0 + 1.6 + RUNNER_HANG_GRACE_S


def test_barrier_wait_plus_later_work_keeps_the_outcome(tmp_path: Path) -> None:
    class SlowBarrier:
        name = "slow-barrier"

        async def run(
            self,
            request: SessionRequest,
            *,
            admission_barrier: SessionAdmissionBarrier | None = None,
            progress_callback: SessionProgressCallback | None = None,
        ) -> SessionOutcome:
            del progress_callback
            if admission_barrier is None:
                return _success(request)
            if "lane0" in request.label:
                await asyncio.sleep(0.9 * request.timeout_s)
                await admission_barrier.wait(request.label)
                return _success(request)
            await admission_barrier.wait(request.label)
            await asyncio.sleep(0.5 * request.timeout_s)
            outcome = _success(request)
            outcome.ttfs_s = 0.5
            outcome.interim_texts = ["kept"]
            return outcome

    config = resolve(
        env={},
        overrides={
            ("workload", "timeout_s"): 1.0,
            ("workload", "pacing"): False,
            ("workload", "lanes"): 2,
            ("workload", "iterations"): 1,
            ("workload", "trials"): 1,
            ("workload", "warmup"): 0,
            ("run", "output_path"): "receipt.json",
        },
    )
    receipt = _drive(config, tmp_path, "hang", SlowBarrier())
    outcomes = receipt["trials"][0]["outcomes"]
    assert isinstance(outcomes, list)
    waiting = next(item for item in outcomes if "lane1" in str(item["label"]))
    assert isinstance(waiting, dict)
    assert waiting["ttfs_s"] == 0.5
    assert waiting["timed_out"] is False
    assert waiting["interim_texts"] == ["kept"]


def test_unsupported_workload_option_is_refused_before_the_run(
    tmp_path: Path,
) -> None:
    class Closed:
        name = "realtime-ws"
        supported_workload_options: frozenset[str] = frozenset()

        async def run(
            self,
            request: SessionRequest,
            *,
            admission_barrier: SessionAdmissionBarrier | None = None,
            progress_callback: SessionProgressCallback | None = None,
        ) -> SessionOutcome:
            del request, admission_barrier, progress_callback
            raise AssertionError("unsupported option must be refused first")

    config = resolve(
        env={},
        overrides={
            ("workload", "language"): "fr",
            ("workload", "lanes"): 1,
            ("workload", "iterations"): 1,
            ("workload", "trials"): 1,
            ("run", "output_path"): "receipt.json",
        },
    )
    with pytest.raises(ConfigError, match="language"):
        _drive(config, tmp_path, "refused", Closed())


def test_adapter_timeout_does_not_replace_a_returned_outcome(
    tmp_path: Path,
) -> None:
    class SlowButDone:
        name = "slow-but-done"

        async def run(
            self,
            request: SessionRequest,
            *,
            admission_barrier: SessionAdmissionBarrier | None = None,
            progress_callback: SessionProgressCallback | None = None,
        ) -> SessionOutcome:
            del admission_barrier, progress_callback
            await asyncio.sleep(0.2)
            outcome = _success(request)
            outcome.ttfs_s = 0.2
            return outcome

    config = resolve(
        env={},
        overrides={
            ("workload", "timeout_s"): 0.05,
            ("workload", "pacing"): True,
            ("workload", "lanes"): 1,
            ("workload", "iterations"): 1,
            ("workload", "trials"): 1,
            ("run", "output_path"): "receipt.json",
        },
    )
    receipt = _drive(config, tmp_path, "partial", SlowButDone())
    outcome = receipt["trials"][0]["outcomes"][0]
    assert isinstance(outcome, dict)
    assert outcome["ttfs_s"] == 0.2
    assert outcome["timed_out"] is False


def test_a_lane_that_raises_before_the_barrier_does_not_stall_peers(
    tmp_path: Path,
) -> None:
    class Uneven:
        name = "uneven"

        async def run(
            self,
            request: SessionRequest,
            *,
            admission_barrier: SessionAdmissionBarrier | None = None,
            progress_callback: SessionProgressCallback | None = None,
        ) -> SessionOutcome:
            del progress_callback
            if "lane0" in request.label:
                raise RuntimeError("before wait")
            if admission_barrier is not None:
                await admission_barrier.wait(request.label)
            return _success(request)

    config = resolve(
        env={},
        overrides={
            ("workload", "lanes"): 3,
            ("workload", "iterations"): 1,
            ("workload", "trials"): 1,
            ("workload", "pacing"): False,
            ("workload", "timeout_s"): 2.0,
            ("run", "output_path"): "receipt.json",
        },
    )
    receipt = _drive(config, tmp_path, "barrier", Uneven())
    counts = receipt["summary"]["pooled"]["counts"]
    assert isinstance(counts, dict)
    assert counts["attempted"] == 3
    assert counts["errored"] == 1
    assert counts["timed_out"] == 0
    assert counts["completed"] == 2


def test_later_iterations_are_not_lockstep_waves(tmp_path: Path) -> None:
    events: list[tuple[str, str, float]] = []

    class ClosedLoop:
        name = "closed-loop"

        async def run(
            self,
            request: SessionRequest,
            *,
            admission_barrier: SessionAdmissionBarrier | None = None,
            progress_callback: SessionProgressCallback | None = None,
        ) -> SessionOutcome:
            del progress_callback
            if admission_barrier is not None:
                await admission_barrier.wait(request.label)
            events.append((request.label, "start", time.monotonic()))
            if "lane0" in request.label:
                await asyncio.sleep(0.15)
            events.append((request.label, "end", time.monotonic()))
            return _success(request)

    config = resolve(
        env={},
        overrides={
            ("workload", "lanes"): 2,
            ("workload", "iterations"): 2,
            ("workload", "trials"): 1,
            ("workload", "pacing"): False,
            ("workload", "timeout_s"): 2.0,
            ("run", "output_path"): "receipt.json",
        },
    )
    _drive(config, tmp_path, "closed", ClosedLoop())
    start = next(
        item
        for item in events
        if item[0] == "t0-lane1-i1" and item[1] == "start"
    )
    end = next(
        item for item in events if item[0] == "t0-lane0-i0" and item[1] == "end"
    )
    assert start[2] < end[2]
