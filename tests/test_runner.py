"""Mock lanes, warmup exclusion, timeouts, and the transport seam."""

import asyncio
from pathlib import Path

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
    assert rtfx["validity"] == "pacing_fidelity_unavailable"
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
    text = receipt["trials"][0]["outcomes"][0]["protocol_extras"]["exception"]
    assert "RuntimeError" in text


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
