"""Riva gRPC against an in-process servicer. No network and no secret."""

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("grpc")
pytest.importorskip("riva.client.proto.riva_asr_pb2")
pytest.importorskip("riva.client.proto.riva_asr_pb2_grpc")

from asrbench.adapters.riva_grpc import RivaGrpcAdapter
from asrbench.clips import pcm16_wav, silence_pcm16
from asrbench.config import ConfigError
from asrbench.outcome import (
    FINALIZATION_CLOCK,
    NATIVE_LATENCY_CLOCK,
    RESPONSE_LATENCY_CLOCK,
    SessionOutcome,
    SessionRequest,
)
from asrbench.reporters import (
    NotHomogeneousError,
    frontier_report,
    perf_table_tsv,
)
from asrbench.runner import CohortBarrier, load_adapter
from asrbench.spec import TargetSpec
from tests.riva_fake import FakeRiva, RunningFake

_MARKER = "client-side-marker"


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
        "interim_results": False,
        "workload_mode": "streaming",
    }
    values.update(overrides)
    return SessionRequest(**values)  # type: ignore[arg-type]


def _adapter(
    endpoint: str,
    *,
    auth_env: str | None = None,
    function_id: str | None = None,
    channel_options: dict[str, int] | None = None,
    use_ssl: bool = False,
    tls_verify: bool = False,
    allow_insecure_auth: bool = False,
) -> RivaGrpcAdapter:
    options: dict[str, object] = {
        "use_ssl": use_ssl,
        "max_alternatives": 1,
        "enable_word_time_offsets": False,
        "custom_configuration": {},
    }
    if function_id is not None:
        options["function_id"] = function_id
    if allow_insecure_auth:
        options["allow_insecure_auth"] = True
    target = TargetSpec(
        endpoint_url=endpoint,
        transport="riva-grpc",
        model="fake-model",
        auth_env=auth_env,
        tls_verify=tls_verify,
        transport_options=options,
        channel_options=channel_options or {},
    )
    adapter = load_adapter(target)
    assert isinstance(adapter, RivaGrpcAdapter)
    return adapter


def test_mock_options_are_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown riva"):
        RivaGrpcAdapter.from_target(
            TargetSpec(endpoint_url="127.0.0.1:1", transport="riva-grpc")
        )


def test_channel_options_are_keepalive_and_message_size() -> None:
    adapter = _adapter(
        "127.0.0.1:1",
        channel_options={
            "keepalive_time_ms": 1000,
            "max_send_message_bytes": 1024,
        },
    )
    assert ("grpc.keepalive_time_ms", 1000) in adapter._channel_args
    assert ("grpc.max_send_message_length", 1024) in adapter._channel_args
    with pytest.raises(ConfigError, match="channel_options"):
        _adapter("127.0.0.1:1", channel_options={"nope": 1})


def test_tls_verify_selects_verifying_or_plaintext_channel() -> None:
    class _Grpc:
        def ssl_channel_credentials(self) -> str:
            return "verified-roots"

        class aio:
            @staticmethod
            def secure_channel(
                endpoint: str, creds: str, options: object = None
            ) -> tuple[object, ...]:
                return ("secure", endpoint, creds, options)

            @staticmethod
            def insecure_channel(
                endpoint: str, options: object = None
            ) -> tuple[object, ...]:
                return ("insecure", endpoint, options)

    secure = _adapter("example:443", use_ssl=True, tls_verify=True)
    opened = secure._open_channel(_Grpc())
    assert opened[0] == "secure"
    assert opened[2] == "verified-roots"
    plain = _adapter("127.0.0.1:1")
    opened = plain._open_channel(_Grpc())
    assert opened[0] == "insecure"


def test_mismatched_tls_verify_is_refused() -> None:
    with pytest.raises(ConfigError, match="tls_verify"):
        _adapter("example:443", use_ssl=True, tls_verify=False)
    with pytest.raises(ConfigError, match="tls_verify"):
        _adapter("127.0.0.1:1", use_ssl=False, tls_verify=True)


def test_plaintext_bearer_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASRBENCH_RIVA_TOKEN", _MARKER)
    with pytest.raises(ConfigError, match="plaintext") as caught:
        _adapter("127.0.0.1:1", auth_env="ASRBENCH_RIVA_TOKEN")
    assert _MARKER not in str(caught.value)
    allowed = _adapter(
        "127.0.0.1:1",
        auth_env="ASRBENCH_RIVA_TOKEN",
        allow_insecure_auth=True,
    )
    assert _MARKER not in repr(vars(allowed))
    monkeypatch.delenv("ASRBENCH_RIVA_TOKEN", raising=False)
    built = _adapter("127.0.0.1:1", auth_env="ASRBENCH_RIVA_TOKEN")
    monkeypatch.setenv("ASRBENCH_RIVA_TOKEN", _MARKER)
    with pytest.raises(ConfigError, match="plaintext") as caught:
        asyncio.run(built.run(_request(Path("missing.wav"), timeout_s=0.2)))
    assert _MARKER not in str(caught.value)


def test_tls_bearer_does_not_need_allow_insecure_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASRBENCH_RIVA_TOKEN", _MARKER)
    adapter = _adapter(
        "example:443",
        auth_env="ASRBENCH_RIVA_TOKEN",
        use_ssl=True,
        tls_verify=True,
    )
    assert adapter._use_ssl is True
    assert adapter._tls_verify is True
    assert adapter._allow_insecure_auth is False
    assert _MARKER not in repr(vars(adapter))


def test_offline_recognize_is_not_implemented(tmp_path: Path) -> None:
    adapter = _adapter("127.0.0.1:1")
    outcome = asyncio.run(
        adapter.run(_request(_wav(tmp_path, 0.02), workload_mode="offline"))
    )
    assert outcome.terminal_status == "errored"
    assert outcome.error_code == "UNSUPPORTED"
    assert outcome.configured is False
    assert outcome.ttfs_s is None


def test_token_is_not_stored_on_the_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASRBENCH_RIVA_TOKEN", _MARKER)
    adapter = _adapter(
        "127.0.0.1:1",
        auth_env="ASRBENCH_RIVA_TOKEN",
        allow_insecure_auth=True,
    )
    assert _MARKER not in repr(vars(adapter))
    assert adapter._auth_env == "ASRBENCH_RIVA_TOKEN"


def test_streaming_order_finals_and_pacing(tmp_path: Path) -> None:
    server = RunningFake(FakeRiva(mode="interim", transcript="HELLO"))
    try:
        adapter = _adapter(server.endpoint)
        outcome = asyncio.run(
            adapter.run(
                _request(
                    _wav(tmp_path, 0.3),
                    chunk_ms=100,
                    pace=True,
                    interim_results=True,
                    label="paced",
                )
            )
        )
    finally:
        server.stop()
    assert outcome.terminal_status == "completed"
    assert outcome.configured is True
    assert outcome.transcript == "HELLO"
    assert outcome.interim_count >= 1
    assert outcome.interim_texts == ["PARTIAL"]
    assert outcome.ttfs_s is not None
    assert outcome.ttfs_s >= 0
    assert outcome.final_latency_s is not None
    assert outcome.final_latency_s >= 0
    assert outcome.latency_clock == NATIVE_LATENCY_CLOCK
    assert outcome.final_latency_clock == FINALIZATION_CLOCK
    assert outcome.response_latencies_s
    sequence = server.servicer.streams[0]
    kinds = [kind for kind, _stamp in sequence]
    assert kinds[0] == "streaming_config"
    assert "audio_content" in kinds
    assert "streaming_config" not in kinds[1:]
    audio_stamps = [
        stamp for kind, stamp in sequence if kind == "audio_content"
    ]
    assert len(audio_stamps) >= 2
    gaps = [
        right - left
        for left, right in zip(audio_stamps, audio_stamps[1:], strict=False)
    ]
    assert all(gap >= 0.05 for gap in gaps)
    config = server.servicer.configs[0].config
    assert config.language_code == "en-US"
    assert config.enable_automatic_punctuation is True
    assert config.enable_word_time_offsets is False
    assert config.max_alternatives == 1
    assert server.servicer.configs[0].interim_results is True
    identity = outcome.protocol_extras["server_identity"]["models"]
    assert identity[0]["model_name"] == "fake-model"
    assert identity[0]["parameters"]["kind"] == "fake"


def test_finals_are_concatenated_and_empty_interims_have_no_latency(
    tmp_path: Path,
) -> None:
    two = RunningFake(FakeRiva(mode="two-finals"))
    try:
        outcome = asyncio.run(
            _adapter(two.endpoint).run(_request(_wav(tmp_path, 0.02)))
        )
    finally:
        two.stop()
    assert outcome.transcript == "ABC"
    assert outcome.ttfs_s is None
    assert outcome.response_latencies_s == []
    first_final = outcome.protocol_extras["first_final_s"]
    assert isinstance(first_final, float)
    assert outcome.final_latency_s is not None
    assert first_final < outcome.final_latency_s
    empty = RunningFake(FakeRiva(mode="empty-interim", transcript="HELLO"))
    try:
        outcome = asyncio.run(
            _adapter(empty.endpoint).run(
                _request(
                    _wav(tmp_path, 0.02),
                    interim_results=True,
                )
            )
        )
    finally:
        empty.stop()
    assert outcome.interim_count == 1
    assert outcome.interim_texts == [""]
    assert outcome.ttfs_s is None
    assert outcome.response_latencies_s == []
    assert outcome.transcript == "HELLO"


def test_one_response_with_several_results_is_one_interim(
    tmp_path: Path,
) -> None:
    server = RunningFake(FakeRiva(mode="multi-result"))
    try:
        outcome = asyncio.run(
            _adapter(server.endpoint).run(
                _request(_wav(tmp_path, 0.02), interim_results=True)
            )
        )
    finally:
        server.stop()
    assert outcome.terminal_status == "completed"
    assert outcome.interim_texts == ["STABLE"]
    assert outcome.interim_count == 1
    assert len(outcome.response_latencies_s) == 1
    assert outcome.ttfs_s is not None
    assert outcome.transcript == "ABC"
    first_final = outcome.protocol_extras["first_final_s"]
    assert isinstance(first_final, float)
    assert outcome.final_latency_s is not None
    assert first_final <= outcome.final_latency_s


def test_barrier_releases_before_audio(tmp_path: Path) -> None:
    server = RunningFake(FakeRiva())
    barrier = CohortBarrier(2, timeout_s=2)
    path = _wav(tmp_path, 0.02)
    seen: list[tuple[str, str, float]] = []
    lock = threading.Lock()

    def progress(label: str, event: str, fields: dict[str, object]) -> None:
        del fields
        if event in {"streaming_config.sent", "first_audio.sent"}:
            with lock:
                seen.append((label, event, time.monotonic()))

    async def one(label: str) -> None:
        await _adapter(server.endpoint).run(
            _request(path, label=label, chunk_ms=20),
            admission_barrier=barrier,
            progress_callback=progress,
        )

    async def both() -> None:
        await asyncio.gather(one("a"), one("b"))

    try:
        asyncio.run(both())
    finally:
        server.stop()
    assert len(server.servicer.streams) == 2
    for sequence in server.servicer.streams:
        assert sequence[0][0] == "streaming_config"
        assert any(kind == "audio_content" for kind, _stamp in sequence)
    config_at = [
        stamp
        for _label, event, stamp in seen
        if event == "streaming_config.sent"
    ]
    audio_at = [
        stamp for _label, event, stamp in seen if event == "first_audio.sent"
    ]
    assert len(config_at) == 2
    assert len(audio_at) == 2
    assert max(config_at) <= min(audio_at)


def test_cancellation_is_visible_to_the_servicer(tmp_path: Path) -> None:
    server = RunningFake(FakeRiva(mode="block"))

    async def cancel() -> None:
        adapter = _adapter(server.endpoint)
        task = asyncio.create_task(
            adapter.run(_request(_wav(tmp_path, 0.2), timeout_s=2))
        )
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=3)
        for _ in range(50):
            if server.servicer.saw_cancelled:
                break
            await asyncio.sleep(0.02)

    try:
        asyncio.run(cancel())
        assert server.servicer.saw_cancelled
    finally:
        server.stop()


def test_auth_and_permission_failures_are_lane_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASRBENCH_RIVA_TOKEN", _MARKER)
    path = _wav(tmp_path, 0.02)
    for mode in ("unauth", "denied"):
        server = RunningFake(FakeRiva(mode=mode))
        try:
            outcome = asyncio.run(
                _adapter(
                    server.endpoint,
                    auth_env="ASRBENCH_RIVA_TOKEN",
                    allow_insecure_auth=True,
                ).run(_request(path))
            )
        finally:
            server.stop()
        assert outcome.terminal_status == "lane_unavailable"
        assert outcome.configured is False
        assert outcome.ttfs_s is None
        assert outcome.final_latency_s is None
        assert outcome.response_latencies_s == []
        detail = outcome.error_detail or ""
        assert "SENTINEL-VALUE" not in detail
        assert _MARKER not in detail
        assert _MARKER not in json.dumps(outcome.protocol_extras)


def test_echoed_token_is_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASRBENCH_RIVA_TOKEN", _MARKER)
    server = RunningFake(FakeRiva(mode="echo-auth"))
    try:
        outcome = asyncio.run(
            _adapter(
                server.endpoint,
                auth_env="ASRBENCH_RIVA_TOKEN",
                allow_insecure_auth=True,
            ).run(_request(_wav(tmp_path, 0.02)))
        )
    finally:
        server.stop()
    assert outcome.terminal_status == "lane_unavailable"
    assert outcome.error_detail is not None
    assert _MARKER not in outcome.error_detail
    assert "[redacted]" in outcome.error_detail


def test_function_id_is_sent_and_not_a_stored_secret(tmp_path: Path) -> None:
    server = RunningFake(FakeRiva())
    try:
        outcome = asyncio.run(
            _adapter(server.endpoint, function_id="func-test").run(
                _request(_wav(tmp_path, 0.02))
            )
        )
    finally:
        server.stop()
    assert server.servicer.function_id == "func-test"
    assert outcome.terminal_status == "completed"
    assert "func-test" not in json.dumps(outcome.protocol_extras)


def test_closed_port_is_lane_unavailable(tmp_path: Path) -> None:
    outcome = asyncio.run(
        _adapter("127.0.0.1:1").run(
            _request(_wav(tmp_path, 0.02), timeout_s=0.4)
        )
    )
    assert outcome.terminal_status == "lane_unavailable"
    assert outcome.configured is False
    assert outcome.connected is False
    assert outcome.ttfs_s is None
    assert outcome.final_latency_s is None
    assert outcome.response_latencies_s == []


def test_invalid_argument_is_a_protocol_error(tmp_path: Path) -> None:
    server = RunningFake(FakeRiva(mode="invalid"))
    try:
        outcome = asyncio.run(
            _adapter(server.endpoint).run(_request(_wav(tmp_path, 0.02)))
        )
    finally:
        server.stop()
    assert outcome.terminal_status == "errored"
    assert outcome.error_code == "INVALID_ARGUMENT"
    assert outcome.ttfs_s is None


class _GrpcStatus(Exception):
    """Stand-in for an ``AioRpcError`` with a named status."""

    def __init__(self, name: str, detail: str = "gone") -> None:
        super().__init__(name)
        self._name = name
        self._detail = detail

    def code(self) -> object:
        return type("Code", (), {"name": self._name})()

    def details(self) -> str:
        return self._detail


def test_unavailable_after_audio_keeps_the_partial(tmp_path: Path) -> None:
    server = RunningFake(FakeRiva(mode="unavailable-after-audio"))
    try:
        outcome = asyncio.run(
            _adapter(server.endpoint).run(
                _request(
                    _wav(tmp_path, 0.02),
                    interim_results=True,
                )
            )
        )
    finally:
        server.stop()
    assert outcome.terminal_status == "errored"
    assert outcome.error_code == "UNAVAILABLE"
    assert outcome.interim_texts == ["PARTIAL"]
    assert outcome.ttfs_s is not None
    assert outcome.ttfs_s >= 0


def test_unavailable_class_follows_whether_audio_started() -> None:
    adapter = _adapter("127.0.0.1:1")
    request = _request(Path("clip.wav"))
    before = _blank_outcome()
    adapter._fail_stream(
        before,
        _GrpcStatus("UNAVAILABLE"),
        {"first": None, "latest": None},
        request,
        None,
    )
    assert before.terminal_status == "lane_unavailable"
    assert before.ttfs_s is None
    after = _blank_outcome()
    after.ttfs_s = 0.2
    after.interim_count = 1
    adapter._fail_stream(
        after,
        _GrpcStatus("UNAVAILABLE"),
        {"first": 1.0, "latest": 1.0},
        request,
        None,
    )
    assert after.terminal_status == "errored"
    assert after.error_code == "UNAVAILABLE"
    assert after.ttfs_s == 0.2
    deadline = _blank_outcome()
    adapter._fail_stream(
        deadline,
        _GrpcStatus("DEADLINE_EXCEEDED"),
        {"first": 1.0, "latest": 1.0},
        request,
        None,
    )
    assert deadline.terminal_status == "timed_out"
    assert deadline.timed_out is True
    denied = _blank_outcome()
    adapter._fail_stream(
        denied,
        _GrpcStatus("UNAUTHENTICATED"),
        {"first": 1.0, "latest": 1.0},
        request,
        None,
    )
    assert denied.terminal_status == "lane_unavailable"


def _blank_outcome() -> SessionOutcome:
    return SessionOutcome(
        label="s",
        lifecycle="full",
        clip_name="clip.wav",
        chunk_ms=100,
        audio_duration_s=0.02,
        latency_clock=NATIVE_LATENCY_CLOCK,
        response_latency_clock=RESPONSE_LATENCY_CLOCK,
        final_latency_clock=FINALIZATION_CLOCK,
    )


def test_deadline_after_audio_is_a_timeout(tmp_path: Path) -> None:
    server = RunningFake(FakeRiva(mode="hang"))
    try:
        outcome = asyncio.run(
            _adapter(server.endpoint).run(
                _request(_wav(tmp_path, 0.02), timeout_s=0.3)
            )
        )
    finally:
        server.stop()
    assert outcome.terminal_status == "timed_out"
    assert outcome.timed_out is True
    assert outcome.ttfs_s is None
    assert outcome.final_latency_s is None


def test_unimplemented_config_still_recognizes(tmp_path: Path) -> None:
    server = RunningFake(FakeRiva(mode="config-unimplemented"))
    try:
        outcome = asyncio.run(
            _adapter(server.endpoint).run(_request(_wav(tmp_path, 0.02)))
        )
    finally:
        server.stop()
    assert outcome.terminal_status == "completed"
    identity = outcome.protocol_extras["server_identity"]
    assert identity["config_status"] == "UNIMPLEMENTED"
    assert identity["models"] == []


def test_config_probe_failure_still_streams(tmp_path: Path) -> None:
    path = _wav(tmp_path, 0.02)
    cases = (
        ("config-unimplemented", "UNIMPLEMENTED"),
        ("config-not-found", "NOT_FOUND"),
        ("config-denied", "PERMISSION_DENIED"),
    )
    for mode, status in cases:
        server = RunningFake(FakeRiva(mode=mode, transcript="HELLO"))
        try:
            outcome = asyncio.run(_adapter(server.endpoint).run(_request(path)))
        finally:
            server.stop()
        assert outcome.terminal_status == "completed", mode
        assert outcome.transcript == "HELLO"
        assert outcome.error_code is None
        identity = outcome.protocol_extras["server_identity"]
        assert identity["config_status"] == status
        assert identity["models"] == []
        assert identity["config_detail"]


def test_perf_table_tsv_refuses_a_mixed_population(tmp_path: Path) -> None:
    server = RunningFake(FakeRiva(transcript="HELLO"))
    try:
        outcome = asyncio.run(
            _adapter(server.endpoint).run(_request(_wav(tmp_path, 0.02)))
        )
    finally:
        server.stop()
    assert outcome.transcript == "HELLO"
    receipt = {
        "resolved_config": {
            "workload": {
                "lanes": 1,
                "population": [
                    {"lanes": 1, "cadence_ms": 160, "lookahead_frames": 2},
                    {"lanes": 1, "cadence_ms": 320, "lookahead_frames": 4},
                ],
            }
        },
        "trials": [],
        "summary": {
            "pooled": {"finalization_s": {"n_with_measure": 0, "n_total": 0}}
        },
    }
    with pytest.raises(NotHomogeneousError):
        perf_table_tsv(receipt)
    with pytest.raises(NotHomogeneousError):
        frontier_report([receipt], ["mixed"])
