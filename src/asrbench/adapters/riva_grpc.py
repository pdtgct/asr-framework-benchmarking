"""Riva Speech Recognition streaming adapter.

Public protocol, from the ``nvidia.riva.asr.RivaSpeechRecognition`` service
in nvidia-riva/common and the streaming client in nvidia-riva/python-clients:
the first ``StreamingRecognizeRequest`` carries ``streaming_config`` and no
audio, and every later request carries ``audio_content`` only. Audio here is
mono PCM16 at 16 kHz, paced at one ``chunk_ms`` per chunk when the session
asks for real time.

``riva.client.Auth`` is not used. It keeps metadata, including a bearer
token, on the auth object. This adapter builds metadata for the RPC and
does not assign the token to itself, the request, or the outcome.

An unreachable server, or ``UNAUTHENTICATED`` / ``PERMISSION_DENIED``, is
a ``lane_unavailable`` outcome. It is not a crash and it does not invent a
latency. A deadline after audio has started is a timeout.

``GetRivaSpeechRecognitionConfig`` is a probe. Any failure, including a
timeout and any gRPC status, is recorded on ``server_identity`` and the
session still streams. A channel that never becomes ready does not.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from collections.abc import AsyncIterator, Mapping
from typing import Any

from asrbench.adapter import SessionAdmissionBarrier, SessionProgressCallback
from asrbench.audio import (
    BYTES_PER_SAMPLE,
    SAMPLE_RATE,
    chunk_pcm16,
    read_pcm16,
)
from asrbench.config import ConfigError
from asrbench.outcome import (
    FINALIZATION_CLOCK,
    NATIVE_LATENCY_CLOCK,
    RESPONSE_LATENCY_CLOCK,
    SessionOutcome,
    SessionRequest,
)
from asrbench.spec import TargetSpec

_LANE_CODES = frozenset({"UNAVAILABLE", "UNAUTHENTICATED", "PERMISSION_DENIED"})
_TRANSPORT_KEYS = frozenset(
    {
        "use_ssl",
        "function_id",
        "function_version_id",
        "max_alternatives",
        "enable_word_time_offsets",
        "custom_configuration",
        "max_speaker_count",
    }
)
_CHANNEL_FIELDS = {
    "keepalive_time_ms": "grpc.keepalive_time_ms",
    "keepalive_timeout_ms": "grpc.keepalive_timeout_ms",
    "max_send_message_bytes": "grpc.max_send_message_length",
    "max_receive_message_bytes": "grpc.max_receive_message_length",
}
_BEARER = re.compile(r"(?i)bearer\s+\S+")
_DEFAULT_MAX_SPEAKERS = 2


def _import_riva() -> tuple[Any, Any, Any, Any]:
    """Import grpc and the generated Riva stubs, or fail as configuration.

    Returns:
        ``grpc``, the ASR protobuf module, the gRPC module, and the audio
        protobuf module.

    Raises:
        ConfigError: The ``riva`` extra is not installed.
    """
    try:
        import grpc
        import grpc.aio  # noqa: F401 - loads the aio API
        from riva.client.proto import (
            riva_asr_pb2,
            riva_asr_pb2_grpc,
            riva_audio_pb2,
        )
    except ImportError as error:
        raise ConfigError(
            "riva-grpc requires the riva extra (nvidia-riva-client and grpcio)"
        ) from error
    return grpc, riva_asr_pb2, riva_asr_pb2_grpc, riva_audio_pb2


def _redact(text: str, auth_env: str | None) -> str:
    """Remove bearer material from text that may be stored on an outcome."""
    cleaned = _BEARER.sub("Bearer [redacted]", text)
    if not auth_env:
        return cleaned
    token = os.environ.get(auth_env) or ""
    if token:
        cleaned = cleaned.replace(token, "[redacted]")
    del token
    return cleaned


def _grpc_status(error: BaseException) -> tuple[str | None, str]:
    """Return a status name and details without stringifying the error."""
    code = getattr(error, "code", None)
    details = getattr(error, "details", None)
    name: str | None = None
    if callable(code):
        try:
            status = code()
        except Exception:  # noqa: BLE001 - status lookup is best-effort
            status = None
        found = getattr(status, "name", None)
        name = found if isinstance(found, str) else None
    text = ""
    if callable(details):
        try:
            raw = details()
        except Exception:  # noqa: BLE001 - details lookup is best-effort
            raw = ""
        text = raw if isinstance(raw, str) else ""
    return name, text


def _progress(
    callback: SessionProgressCallback | None,
    label: str,
    event: str,
    **fields: object,
) -> None:
    """Report one bounded lifecycle edge. Never a chunk, never metadata."""
    if callback is not None:
        callback(label, event, fields)


class RivaGrpcAdapter:
    """Drive one ``StreamingRecognize`` session over gRPC.

    Channel tuning is read from ``channel_options``. Everything else in
    ``transport_options`` is sent and is hard target identity. The auth
    field stored here is the environment variable name.
    """

    name = "riva-grpc"
    supported_workload_options = frozenset(
        {
            "language",
            "punctuation",
            "diarization",
            "interim_results",
        }
    )

    def __init__(
        self,
        endpoint: str,
        *,
        use_ssl: bool,
        function_id: str | None,
        function_version_id: str | None,
        max_alternatives: int,
        word_time_offsets: bool,
        custom_configuration: dict[str, str],
        max_speaker_count: int,
        auth_env: str | None,
        channel_args: tuple[tuple[str, int], ...],
    ) -> None:
        """Store endpoint and identity options. Not a token."""
        self._endpoint = endpoint
        self._use_ssl = use_ssl
        self._function_id = function_id
        self._function_version_id = function_version_id
        self._max_alternatives = max_alternatives
        self._word_offsets = word_time_offsets
        self._custom = dict(custom_configuration)
        self._max_speakers = max_speaker_count
        self._auth_env = auth_env
        self._channel_args = channel_args

    @classmethod
    def from_target(cls, target: TargetSpec) -> RivaGrpcAdapter:
        """Build an adapter from a resolved target.

        Args:
            target: Endpoint, model, auth variable name, and options.
                ``transport_options`` are identity. ``channel_options``
                are keepalive and maximum message size.

        Returns:
            An adapter that has not opened a channel.

        Raises:
            ConfigError: An option has the wrong type, or the riva extra
                is not installed.
        """
        options = dict(target.transport_options)
        unknown = sorted(set(options) - _TRANSPORT_KEYS)
        if unknown:
            raise ConfigError(
                "unknown riva transport_options: " + ", ".join(unknown)
            )
        use_ssl = _bool_option(options, "use_ssl", default=False)
        function_id = _optional_str(options, "function_id")
        function_version_id = _optional_str(options, "function_version_id")
        max_alternatives = _int_option(
            options, "max_alternatives", default=1, minimum=1
        )
        word_offsets = _bool_option(
            options, "enable_word_time_offsets", default=False
        )
        custom = _custom_configuration(options.get("custom_configuration"))
        max_speakers = _int_option(
            options,
            "max_speaker_count",
            default=_DEFAULT_MAX_SPEAKERS,
            minimum=1,
        )
        channel_args = _channel_args(target.channel_options)
        _import_riva()
        return cls(
            target.endpoint_url,
            use_ssl=use_ssl,
            function_id=function_id,
            function_version_id=function_version_id,
            max_alternatives=max_alternatives,
            word_time_offsets=word_offsets,
            custom_configuration=custom,
            max_speaker_count=max_speakers,
            auth_env=target.auth_env,
            channel_args=channel_args,
        )

    async def run(
        self,
        request: SessionRequest,
        *,
        admission_barrier: SessionAdmissionBarrier | None = None,
        progress_callback: SessionProgressCallback | None = None,
    ) -> SessionOutcome:
        """Stream one clip and return what the server was observed to do.

        Args:
            request: Offered work. ``workload_mode`` other than
                ``streaming`` is reported on the outcome; offline
                ``Recognize`` is not this adapter.
            admission_barrier: Released after ``streaming_config`` is
                sent and before the first audio chunk.
            progress_callback: Bounded lifecycle edges only.

        Returns:
            The session outcome. A lane that cannot be opened has
            ``terminal_status`` ``lane_unavailable``, ``configured``
            false, and no latency samples.
        """
        outcome = _new_outcome(request)
        if request.workload_mode != "streaming":
            outcome.error_code = "UNSUPPORTED"
            outcome.error_detail = (
                "riva-grpc implements StreamingRecognize only"
            )
            outcome.terminal_status = "errored"
            return outcome
        holder: dict[str, Any] = {}
        try:
            await self._drive(
                request,
                outcome,
                admission_barrier,
                progress_callback,
                holder,
            )
        except asyncio.CancelledError:
            call = holder.get("call")
            if call is not None:
                call.cancel()
            raise
        return outcome

    async def _drive(
        self,
        request: SessionRequest,
        outcome: SessionOutcome,
        admission_barrier: SessionAdmissionBarrier | None,
        progress_callback: SessionProgressCallback | None,
        holder: dict[str, Any],
    ) -> None:
        grpc, rasr, rasr_grpc, raudio = _import_riva()
        pcm = read_pcm16(request.clip_path)
        chunks = chunk_pcm16(pcm, request.chunk_ms)
        audio_s = len(pcm) / BYTES_PER_SAMPLE / SAMPLE_RATE
        outcome.audio_duration_s = audio_s
        outcome.protocol_extras["adapter"] = self.name
        outcome.protocol_extras["audio_chunks"] = len(chunks)
        channel = self._open_channel(grpc)
        try:
            await self._prepare(
                request,
                outcome,
                channel,
                rasr,
                rasr_grpc,
                progress_callback,
            )
            if outcome.terminal_status == "lane_unavailable":
                return
            if outcome.terminal_status == "errored":
                return
            await self._stream(
                request,
                outcome,
                channel,
                rasr,
                rasr_grpc,
                raudio,
                chunks,
                audio_s,
                admission_barrier,
                progress_callback,
                holder,
            )
        finally:
            await _close_channel(channel)

    def _open_channel(self, grpc: Any) -> Any:
        options = list(self._channel_args) or None
        if self._use_ssl:
            return grpc.aio.secure_channel(
                self._endpoint,
                grpc.ssl_channel_credentials(),
                options=options,
            )
        return grpc.aio.insecure_channel(self._endpoint, options=options)

    async def _prepare(
        self,
        request: SessionRequest,
        outcome: SessionOutcome,
        channel: Any,
        rasr: Any,
        rasr_grpc: Any,
        progress_callback: SessionProgressCallback | None,
    ) -> None:
        """Fetch recognition config inside one ``timeout_s`` budget."""
        deadline = time.monotonic() + request.timeout_s
        try:
            await asyncio.wait_for(
                channel.channel_ready(), timeout=_remaining(deadline)
            )
        except (TimeoutError, asyncio.TimeoutError):  # noqa: UP041
            _mark_lane_unavailable(
                outcome,
                "UNAVAILABLE",
                "channel was not ready",
                progress_callback,
                request.label,
            )
            return
        except Exception as error:  # noqa: BLE001 - the lane stays data
            self._fail_setup(outcome, error, request.label, progress_callback)
            return
        outcome.connected = True
        _progress(progress_callback, request.label, "channel.ready")
        stub = rasr_grpc.RivaSpeechRecognitionStub(channel)
        metadata = self._metadata() or None
        try:
            response = await stub.GetRivaSpeechRecognitionConfig(
                rasr.RivaSpeechRecognitionConfigRequest(
                    model_name=request.model
                ),
                timeout=_remaining(deadline),
                metadata=metadata,
            )
        except (TimeoutError, asyncio.TimeoutError):  # noqa: UP041
            _record_config_probe(
                outcome,
                "DEADLINE_EXCEEDED",
                "recognition config timed out",
                progress_callback,
                request.label,
            )
            return
        except Exception as error:  # noqa: BLE001 - the lane stays data
            name, text = _grpc_status(error)
            detail = _redact(text, self._auth_env) or type(error).__name__
            _record_config_probe(
                outcome,
                name or type(error).__name__,
                detail,
                progress_callback,
                request.label,
            )
            return
        outcome.protocol_extras["server_identity"] = _server_identity(response)
        _progress(
            progress_callback,
            request.label,
            "recognition_config.fetched",
            models=len(outcome.protocol_extras["server_identity"]["models"]),
        )

    async def _stream(
        self,
        request: SessionRequest,
        outcome: SessionOutcome,
        channel: Any,
        rasr: Any,
        rasr_grpc: Any,
        raudio: Any,
        chunks: list[bytes],
        audio_s: float,
        admission_barrier: SessionAdmissionBarrier | None,
        progress_callback: SessionProgressCallback | None,
        holder: dict[str, Any],
    ) -> None:
        marks: dict[str, float | None] = {"first": None, "latest": None}
        streaming = _streaming_config(self, request, rasr, raudio)
        outcome.protocol_extras["recognition_request"] = _recognition_echo(
            self, request
        )
        outcome.interim_texts = []
        metadata = self._metadata() or None
        paced = audio_s if request.pace else 0.0
        deadline_s = request.timeout_s + paced + request.timeout_s
        stub = rasr_grpc.RivaSpeechRecognitionStub(channel)

        async def outgoing() -> AsyncIterator[Any]:
            yield rasr.StreamingRecognizeRequest(streaming_config=streaming)
            _progress(
                progress_callback,
                request.label,
                "streaming_config.sent",
            )
            if admission_barrier is not None:
                _progress(progress_callback, request.label, "barrier.waiting")
                try:
                    released = await admission_barrier.wait(request.label)
                except (TimeoutError, asyncio.TimeoutError):  # noqa: UP041
                    outcome.timed_out = True
                    outcome.terminal_status = "timed_out"
                    outcome.error_code = "DEADLINE_EXCEEDED"
                    outcome.error_detail = "TimeoutError: cohort barrier"
                    _progress(
                        progress_callback,
                        request.label,
                        "barrier.timeout",
                    )
                    return
                outcome.protocol_extras.setdefault("markers", {}).update(
                    released
                )
                _progress(progress_callback, request.label, "barrier.released")
            for index, chunk in enumerate(chunks):
                now = time.monotonic()
                if marks["first"] is None:
                    marks["first"] = now
                    _progress(
                        progress_callback,
                        request.label,
                        "first_audio.sent",
                        bytes=len(chunk),
                    )
                marks["latest"] = now
                yield rasr.StreamingRecognizeRequest(audio_content=chunk)
                if request.pace and index + 1 < len(chunks):
                    await asyncio.sleep(request.chunk_ms / 1000)

        call = stub.StreamingRecognize(
            outgoing(), metadata=metadata, timeout=deadline_s
        )
        holder["call"] = call
        try:
            async for response in call:
                outcome.connected = True
                outcome.configured = True
                _consume(response, outcome, marks, progress_callback, request)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - the lane stays data
            if outcome.timed_out:
                return
            self._fail_stream(outcome, error, marks, request, progress_callback)
            return
        if outcome.timed_out:
            return
        outcome.configured = True
        outcome.connected = True
        if outcome.transcript is not None and not outcome.error_code:
            outcome.terminal_status = "completed"
        _progress(
            progress_callback,
            request.label,
            "session.complete",
            has_text=bool(outcome.transcript),
        )

    def _metadata(self) -> list[tuple[str, str]]:
        """Build RPC metadata. The token is not stored on this adapter."""
        pairs: list[tuple[str, str]] = []
        if self._function_id:
            pairs.append(("function-id", self._function_id))
        if self._function_version_id:
            pairs.append(("function-version-id", self._function_version_id))
        env_name = self._auth_env
        if env_name:
            token = os.environ.get(env_name) or ""
            if token:
                pairs.append(("authorization", "Bearer " + token))
            del token
        return pairs

    def _fail_setup(
        self,
        outcome: SessionOutcome,
        error: BaseException,
        label: str,
        progress_callback: SessionProgressCallback | None,
    ) -> None:
        name, text = _grpc_status(error)
        detail = _redact(text, self._auth_env) or type(error).__name__
        # Setup has not sent audio. Unreachable, auth, and a deadline are
        # an unopened lane. Any other named status is a protocol error.
        if name in _LANE_CODES or name == "DEADLINE_EXCEEDED":
            _mark_lane_unavailable(
                outcome, name, detail, progress_callback, label
            )
            return
        if name is None and isinstance(error, OSError):
            _mark_lane_unavailable(
                outcome, "UNAVAILABLE", detail, progress_callback, label
            )
            return
        _mark_protocol_error(outcome, name or type(error).__name__, detail)

    def _fail_stream(
        self,
        outcome: SessionOutcome,
        error: BaseException,
        marks: dict[str, float | None],
        request: SessionRequest,
        progress_callback: SessionProgressCallback | None,
    ) -> None:
        name, text = _grpc_status(error)
        detail = _redact(text, self._auth_env) or type(error).__name__
        audio_started = isinstance(marks["first"], float)
        timed_out = name == "DEADLINE_EXCEEDED" or isinstance(
            error, (TimeoutError, asyncio.TimeoutError)
        )
        if timed_out and audio_started:
            outcome.timed_out = True
            outcome.terminal_status = "timed_out"
            outcome.error_code = name or "DEADLINE_EXCEEDED"
            outcome.error_detail = detail
            return
        if (
            name in _LANE_CODES
            or timed_out
            or (name is None and isinstance(error, OSError))
        ):
            _mark_lane_unavailable(
                outcome,
                name or "UNAVAILABLE",
                detail,
                progress_callback,
                request.label,
            )
            return
        _mark_protocol_error(outcome, name or type(error).__name__, detail)


def _new_outcome(request: SessionRequest) -> SessionOutcome:
    return SessionOutcome(
        label=request.label,
        lifecycle=request.lifecycle,
        clip_name=request.clip_path.name,
        chunk_ms=request.chunk_ms,
        audio_duration_s=0.0,
        latency_clock=NATIVE_LATENCY_CLOCK,
        response_latency_clock=RESPONSE_LATENCY_CLOCK,
        final_latency_clock=FINALIZATION_CLOCK,
    )


def _consume(
    response: Any,
    outcome: SessionOutcome,
    marks: dict[str, float | None],
    progress_callback: SessionProgressCallback | None,
    request: SessionRequest,
) -> None:
    """Map one streaming response onto the outcome clocks.

    Final transcripts are concatenated as the server sent them, with no
    extra space. TTFS and response latency count non-empty interims only.
    A final that arrives before any audio does not invent a latency.
    """
    parts: list[str] = []
    if outcome.transcript is not None:
        parts.append(outcome.transcript)
    for result in response.results:
        text = ""
        if result.alternatives:
            text = result.alternatives[0].transcript
        if result.is_final:
            parts.append(text)
            outcome.transcript = "".join(parts)
            latest = marks["latest"]
            if isinstance(latest, float):
                latency = time.monotonic() - latest
                if latency >= 0:
                    outcome.final_latency_s = latency
                    outcome.final_latency_clock = FINALIZATION_CLOCK
            _progress(progress_callback, request.label, "final.received")
            continue
        if outcome.interim_texts is None:
            outcome.interim_texts = []
        outcome.interim_texts.append(text)
        outcome.interim_count += 1
        if not text:
            continue
        now = time.monotonic()
        first = marks["first"]
        latest = marks["latest"]
        if isinstance(first, float) and outcome.ttfs_s is None:
            outcome.ttfs_s = now - first
        if isinstance(latest, float):
            outcome.response_latencies_s.append(now - latest)


def _record_config_probe(
    outcome: SessionOutcome,
    code: str,
    detail: str,
    progress_callback: SessionProgressCallback | None,
    label: str,
) -> None:
    """Remember a failed config probe and leave the session open.

    The probe is not admission. A server that rejects it, or does not
    implement it, can still answer ``StreamingRecognize``. The status is
    identity. It is not ``terminal_status``.

    Args:
        outcome: Session being filled in.
        code: gRPC status name, or ``DEADLINE_EXCEEDED`` on timeout.
        detail: Redacted status text.
        progress_callback: Bounded lifecycle edge.
        label: Session label.
    """
    outcome.protocol_extras["server_identity"] = {
        "models": [],
        "config_status": code,
        "config_detail": detail,
    }
    _progress(
        progress_callback,
        label,
        "recognition_config.probe_failed",
        code=code,
    )


def _mark_lane_unavailable(
    outcome: SessionOutcome,
    code: str,
    detail: str,
    progress_callback: SessionProgressCallback | None,
    label: str,
) -> None:
    """Record an unopened lane and drop any latency that was not earned."""
    outcome.terminal_status = "lane_unavailable"
    outcome.configured = False
    outcome.ttfs_s = None
    outcome.final_latency_s = None
    outcome.response_latencies_s = []
    outcome.timed_out = False
    outcome.error_code = code
    outcome.error_detail = detail
    outcome.protocol_errors.append({"code": code, "detail": detail})
    _progress(progress_callback, label, "lane_unavailable", code=code)


def _mark_protocol_error(
    outcome: SessionOutcome, code: str, detail: str
) -> None:
    outcome.terminal_status = "errored"
    outcome.error_code = code
    outcome.error_detail = detail
    outcome.protocol_errors.append({"code": code, "detail": detail})


def _server_identity(response: Any) -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    for item in response.model_config:
        parameters = {
            str(key): str(value) for key, value in item.parameters.items()
        }
        models.append({"model_name": item.model_name, "parameters": parameters})
    return {"models": models}


def _streaming_config(
    adapter: RivaGrpcAdapter,
    request: SessionRequest,
    rasr: Any,
    raudio: Any,
) -> Any:
    config = rasr.RecognitionConfig(
        encoding=raudio.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=SAMPLE_RATE,
        language_code=request.language or "",
        max_alternatives=adapter._max_alternatives,
        audio_channel_count=1,
        enable_word_time_offsets=adapter._word_offsets,
        enable_automatic_punctuation=request.punctuation,
        model=request.model,
    )
    for key, value in adapter._custom.items():
        config.custom_configuration[key] = value
    if request.diarization:
        config.diarization_config.enable_speaker_diarization = True
        config.diarization_config.max_speaker_count = adapter._max_speakers
    return rasr.StreamingRecognitionConfig(
        config=config,
        interim_results=request.interim_results,
    )


def _recognition_echo(
    adapter: RivaGrpcAdapter, request: SessionRequest
) -> dict[str, Any]:
    return {
        "language_code": request.language,
        "model": request.model,
        "sample_rate_hertz": SAMPLE_RATE,
        "encoding": "LINEAR_PCM",
        "audio_channel_count": 1,
        "max_alternatives": adapter._max_alternatives,
        "enable_automatic_punctuation": request.punctuation,
        "enable_word_time_offsets": adapter._word_offsets,
        "interim_results": request.interim_results,
        "diarization": request.diarization,
        "custom_configuration": dict(adapter._custom),
    }


async def _close_channel(channel: Any) -> None:
    try:
        await channel.close()
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - closing is cleanup
        return


def _remaining(deadline: float) -> float:
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError
    return left


def _bool_option(
    options: Mapping[str, Any], name: str, *, default: bool
) -> bool:
    if name not in options:
        return default
    value = options[name]
    if not isinstance(value, bool):
        raise ConfigError(f"riva transport_options.{name} must be a bool")
    return value


def _optional_str(options: Mapping[str, Any], name: str) -> str | None:
    if name not in options or options[name] is None:
        return None
    value = options[name]
    if not isinstance(value, str) or value == "":
        raise ConfigError(
            f"riva transport_options.{name} must be a non-empty string or null"
        )
    return value


def _int_option(
    options: Mapping[str, Any],
    name: str,
    *,
    default: int,
    minimum: int,
) -> int:
    if name not in options:
        return default
    value = options[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(
            f"riva transport_options.{name} must be an int >= {minimum}"
        )
    return value


def _custom_configuration(value: object) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(
            "riva transport_options.custom_configuration must be an object"
        )
    built: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise ConfigError(
                "riva custom_configuration keys and values must be strings"
            )
        built[key] = item
    return built


def _channel_args(
    options: Mapping[str, Any],
) -> tuple[tuple[str, int], ...]:
    unknown = sorted(set(options) - set(_CHANNEL_FIELDS))
    if unknown:
        raise ConfigError("unknown channel_options: " + ", ".join(unknown))
    built: list[tuple[str, int]] = []
    for key, grpc_key in _CHANNEL_FIELDS.items():
        if key not in options:
            continue
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigError(
                f"channel_options.{key} must be a non-negative int"
            )
        built.append((grpc_key, value))
    return tuple(built)
