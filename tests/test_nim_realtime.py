"""NIM transcription WebSocket against an in-process server. No secret."""

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("aiohttp")

from asrbench.adapters.nim_realtime import (  # noqa: E402
    NimRealtimeAdapter,
    _endpoint_urls,
)
from asrbench.clips import pcm16_wav, silence_pcm16  # noqa: E402
from asrbench.config import ConfigError  # noqa: E402
from asrbench.outcome import (  # noqa: E402
    FINALIZATION_CLOCK,
    NATIVE_LATENCY_CLOCK,
    SessionOutcome,
    SessionRequest,
)
from asrbench.runner import (  # noqa: E402
    CohortBarrier,
    _refuse_unsent_options,
    load_adapter,
)
from asrbench.spec import TargetSpec, WorkloadSpec  # noqa: E402
from tests.nim_realtime_fake import FakeNimRealtime, RunningNim  # noqa: E402

_MARKER = "client-side-marker"
_EPHEMERAL = "ephemeral-marker"


def _wav(tmp_path: Path, duration_s: float) -> Path:
    path = tmp_path / "clip.wav"
    path.write_bytes(pcm16_wav(silence_pcm16(duration_s)))
    return path


def _request(path: Path, **overrides: object) -> SessionRequest:
    values: dict[str, object] = {
        "endpoint_url": "127.0.0.1:9",
        "model": "fake-model",
        "clip_path": path,
        "label": "s",
        "chunk_ms": 100,
        "cadence_ms": 160,
        "timeout_s": 5.0,
        "lifecycle": "full",
        "pace": False,
        "language": "en-US",
        "punctuation": True,
        "diarization": False,
        "interim_results": True,
        "workload_mode": "streaming",
    }
    values.update(overrides)
    return SessionRequest(**values)  # type: ignore[arg-type]


def _adapter(
    endpoint: str,
    *,
    auth_env: str | None = None,
    channel_options: dict[str, int] | None = None,
) -> NimRealtimeAdapter:
    target = TargetSpec(
        endpoint_url=endpoint,
        transport="nim-realtime",
        model="fake-model",
        auth_env=auth_env,
        tls_verify=False,
        transport_options={
            "use_ssl": False,
            "max_alternatives": 1,
            "enable_word_time_offsets": False,
            "custom_configuration": {},
        },
        channel_options=channel_options or {},
    )
    adapter = load_adapter(target)
    assert isinstance(adapter, NimRealtimeAdapter)
    return adapter


async def _run(
    fake: FakeNimRealtime,
    request: SessionRequest,
    *,
    auth_env: str | None = None,
) -> tuple[FakeNimRealtime, SessionOutcome]:
    server = RunningNim(fake)
    await server.start()
    try:
        outcome = await _adapter(server.endpoint, auth_env=auth_env).run(
            request
        )
        return fake, outcome
    finally:
        await server.stop()


def test_endpoint_is_origin_plus_public_paths() -> None:
    mint, socket = _endpoint_urls("localhost:9000", False)
    assert mint == ("http://localhost:9000/v1/realtime/transcription_sessions")
    assert socket == "ws://localhost:9000/v1/realtime?intent=transcription"
    secure_mint, secure_socket = _endpoint_urls("localhost:9000", True)
    assert secure_mint.startswith("https://")
    assert secure_socket.startswith("wss://")
    assert "realtime-token" not in secure_socket
    with pytest.raises(ConfigError, match="disagrees"):
        _endpoint_urls("https://localhost:9000", False)
    with pytest.raises(ConfigError, match="credentials"):
        _endpoint_urls("http://user:pw@localhost:9000", False)


def test_unknown_options_and_grpc_keepalive_are_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown nim-realtime"):
        NimRealtimeAdapter.from_target(
            TargetSpec(endpoint_url="127.0.0.1:9", transport="nim-realtime")
        )
    with pytest.raises(ConfigError, match="keepalive"):
        _adapter("127.0.0.1:9", channel_options={"keepalive_time_ms": 1000})


def test_interim_results_off_is_refused_before_the_socket() -> None:
    adapter = _adapter("127.0.0.1:9")
    with pytest.raises(ConfigError, match="interim_results"):
        _refuse_unsent_options(adapter, WorkloadSpec(interim_results=False))


def test_offline_is_not_implemented(tmp_path: Path) -> None:
    outcome = asyncio.run(
        _adapter("127.0.0.1:1").run(
            _request(_wav(tmp_path, 0.02), workload_mode="offline")
        )
    )
    assert outcome.terminal_status == "errored"
    assert outcome.error_code == "UNSUPPORTED"
    assert outcome.configured is False
    assert outcome.ttfs_s is None


def test_token_name_is_stored_and_the_value_is_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASRBENCH_NIM_TOKEN", _MARKER)
    adapter = _adapter("127.0.0.1:9", auth_env="ASRBENCH_NIM_TOKEN")
    assert adapter._auth_env == "ASRBENCH_NIM_TOKEN"
    assert _MARKER not in repr(vars(adapter))


def test_order_pacing_and_session_shape(tmp_path: Path) -> None:
    path = _wav(tmp_path, 0.3)

    async def body() -> tuple[FakeNimRealtime, SessionOutcome]:
        return await _run(
            FakeNimRealtime(),
            _request(path, chunk_ms=100, pace=True, label="paced"),
        )

    fake, outcome = asyncio.run(body())
    assert outcome.terminal_status == "completed"
    assert outcome.configured is True
    assert outcome.transcript == "HELLO"
    assert outcome.latency_clock == NATIVE_LATENCY_CLOCK
    assert outcome.final_latency_clock == FINALIZATION_CLOCK
    assert outcome.ttfs_s is not None
    assert fake.events == [
        "transcription_session.update",
        "input_audio_buffer.append",
        "input_audio_buffer.commit",
        "input_audio_buffer.append",
        "input_audio_buffer.commit",
        "input_audio_buffer.append",
        "input_audio_buffer.commit",
        "input_audio_buffer.done",
    ]
    assert "session.update" not in fake.events
    assert "intent=transcription" in fake.query
    assert "realtime-token" not in fake.query
    assert len(fake.append_at) == 3
    gaps = [
        fake.append_at[index + 1] - fake.append_at[index]
        for index in range(len(fake.append_at) - 1)
    ]
    assert min(gaps) >= 0.05
    assert fake.session is not None
    assert fake.session["input_audio_format"] == "pcm16"
    transcription = fake.session["input_audio_transcription"]
    assert transcription["language"] == "en-US"
    assert transcription["model"] == "fake-model"
    recognition = fake.session["recognition_config"]
    assert recognition["enable_automatic_punctuation"] is True
    assert recognition["enable_word_time_offsets"] is False
    assert "interim_results" not in recognition
    dumped = json.dumps(outcome.protocol_extras)
    assert "client_secret" not in dumped


def test_a_final_transcript_is_not_ttfs(tmp_path: Path) -> None:
    async def body() -> SessionOutcome:
        _fake, outcome = await _run(
            FakeNimRealtime(mode="final-only", delta=""),
            _request(_wav(tmp_path, 0.02)),
        )
        return outcome

    outcome = asyncio.run(body())
    assert outcome.terminal_status == "completed"
    assert outcome.transcript == "HELLO"
    assert outcome.ttfs_s is None
    assert outcome.response_latencies_s == []
    assert outcome.final_latency_s is not None
    assert outcome.interim_count == 0


def test_non_last_completed_is_a_partial(tmp_path: Path) -> None:
    async def body() -> SessionOutcome:
        _fake, outcome = await _run(
            FakeNimRealtime(mode="partial"),
            _request(_wav(tmp_path, 0.02)),
        )
        return outcome

    outcome = asyncio.run(body())
    assert outcome.transcript == "HELLO"
    assert outcome.interim_texts == ["PAR"]
    assert outcome.ttfs_s is not None
    assert outcome.interim_count == 1


def test_empty_delta_adds_no_latency(tmp_path: Path) -> None:
    async def body() -> SessionOutcome:
        _fake, outcome = await _run(
            FakeNimRealtime(delta=""),
            _request(_wav(tmp_path, 0.02)),
        )
        return outcome

    outcome = asyncio.run(body())
    assert outcome.interim_texts == [""]
    assert outcome.interim_count == 1
    assert outcome.ttfs_s is None
    assert outcome.response_latencies_s == []
    assert outcome.transcript == "HELLO"


def test_barrier_releases_before_audio(tmp_path: Path) -> None:
    path = _wav(tmp_path, 0.02)
    seen: list[tuple[str, str]] = []

    def progress(label: str, event: str, fields: dict[str, object]) -> None:
        del fields
        if event in {"transcription_session.updated", "first_audio.sent"}:
            seen.append((label, event))

    async def body() -> None:
        fake = FakeNimRealtime()
        server = RunningNim(fake)
        await server.start()
        barrier = CohortBarrier(2, timeout_s=2)
        try:

            async def one(label: str) -> None:
                await _adapter(server.endpoint).run(
                    _request(path, label=label, chunk_ms=20),
                    admission_barrier=barrier,
                    progress_callback=progress,
                )

            await asyncio.gather(one("a"), one("b"))
        finally:
            await server.stop()

    asyncio.run(body())
    assert [event for _label, event in seen] == [
        "transcription_session.updated",
        "transcription_session.updated",
        "first_audio.sent",
        "first_audio.sent",
    ]


def test_cancellation_closes_the_socket(tmp_path: Path) -> None:
    async def body() -> FakeNimRealtime:
        fake = FakeNimRealtime(mode="block")
        server = RunningNim(fake)
        await server.start()
        try:
            task = asyncio.create_task(
                _adapter(server.endpoint).run(
                    _request(_wav(tmp_path, 0.2), timeout_s=2)
                )
            )
            await asyncio.wait_for(fake.ready.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            for _ in range(50):
                if fake.saw_cancelled:
                    break
                await asyncio.sleep(0.02)
            return fake
        finally:
            await server.stop()

    fake = asyncio.run(body())
    assert fake.saw_cancelled


def test_mint_failures_redact_the_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASRBENCH_NIM_TOKEN", _MARKER)
    path = _wav(tmp_path, 0.02)

    async def one(mode: str) -> SessionOutcome:
        _fake, outcome = await _run(
            FakeNimRealtime(mode=mode),
            _request(path),
            auth_env="ASRBENCH_NIM_TOKEN",
        )
        return outcome

    for mode in ("unauth", "denied", "echo-auth"):
        outcome = asyncio.run(one(mode))
        assert outcome.terminal_status == "lane_unavailable"
        assert outcome.configured is False
        assert outcome.ttfs_s is None
        detail = outcome.error_detail or ""
        dumped = json.dumps(outcome.protocol_extras)
        assert _MARKER not in detail
        assert _MARKER not in dumped
    echoed = asyncio.run(one("echo-auth"))
    assert "[redacted]" in (echoed.error_detail or "")


def test_ephemeral_token_stays_off_the_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASRBENCH_NIM_TOKEN", _MARKER)

    async def body() -> tuple[
        FakeNimRealtime, SessionOutcome, NimRealtimeAdapter
    ]:
        fake = FakeNimRealtime(ephemeral=_EPHEMERAL)
        server = RunningNim(fake)
        await server.start()
        adapter = _adapter(server.endpoint, auth_env="ASRBENCH_NIM_TOKEN")
        try:
            outcome = await adapter.run(_request(_wav(tmp_path, 0.02)))
            return fake, outcome, adapter
        finally:
            await server.stop()

    fake, outcome, adapter = asyncio.run(body())
    assert outcome.terminal_status == "completed"
    assert fake.saw_authorization is True
    assert _EPHEMERAL in fake.protocols
    assert _EPHEMERAL not in fake.query
    dumped = json.dumps(outcome.protocol_extras)
    blob = dumped + repr(vars(adapter)) + (outcome.error_detail or "")
    assert _EPHEMERAL not in blob
    assert _MARKER not in blob
    identity = outcome.protocol_extras["server_identity"]
    assert "client_secret" not in identity["mint"]


def test_closed_port_is_lane_unavailable(tmp_path: Path) -> None:
    outcome = asyncio.run(
        _adapter("127.0.0.1:1").run(
            _request(_wav(tmp_path, 0.02), timeout_s=0.4)
        )
    )
    assert outcome.terminal_status == "lane_unavailable"
    assert outcome.connected is False
    assert outcome.configured is False
    assert outcome.ttfs_s is None
    assert outcome.final_latency_s is None
    assert outcome.response_latencies_s == []


def test_protocol_error_is_not_a_latency(tmp_path: Path) -> None:
    async def body() -> SessionOutcome:
        _fake, outcome = await _run(
            FakeNimRealtime(mode="error"),
            _request(_wav(tmp_path, 0.02)),
        )
        return outcome

    outcome = asyncio.run(body())
    assert outcome.terminal_status == "errored"
    assert outcome.error_code == "invalid_event"
    assert outcome.ttfs_s is None


def test_silence_before_updated_is_lane_unavailable(tmp_path: Path) -> None:
    async def body() -> SessionOutcome:
        _fake, outcome = await _run(
            FakeNimRealtime(mode="silent"),
            _request(_wav(tmp_path, 0.02), timeout_s=0.4),
        )
        return outcome

    outcome = asyncio.run(body())
    assert outcome.terminal_status == "lane_unavailable"
    assert outcome.configured is False
    assert outcome.ttfs_s is None


def test_deadline_after_audio_is_a_timeout(tmp_path: Path) -> None:
    async def body() -> SessionOutcome:
        _fake, outcome = await _run(
            FakeNimRealtime(mode="hang"),
            _request(_wav(tmp_path, 0.02), timeout_s=0.3),
        )
        return outcome

    outcome = asyncio.run(body())
    assert outcome.timed_out is True
    assert outcome.terminal_status == "timed_out"
    assert outcome.ttfs_s is None
    assert outcome.transcript is None
