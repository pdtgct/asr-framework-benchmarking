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

``target.tls_verify`` selects the channel. ``use_ssl`` with verification
uses ``grpc.ssl_channel_credentials``, which always checks the peer.
grpcio exposes no public way to leave TLS on and skip that check, so
``use_ssl`` true and ``tls_verify`` false is refused rather than verified
while the receipt says otherwise. Plaintext is both flags false. A bearer
token is not put on that channel unless ``allow_insecure_auth`` is set.
Loopback is not an implicit allow.

An unreachable server, or ``UNAUTHENTICATED`` / ``PERMISSION_DENIED``, is
a ``lane_unavailable`` outcome. It is not a crash and it does not invent a
latency. A deadline after audio has started is a timeout. ``UNAVAILABLE``
after audio has started is a session error: the lane was open and the
server failed. The same status before any audio is ``lane_unavailable``.

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
        "allow_insecure_auth",
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
    ``transport_options`` is sent. Those options are warn-only identity.
    Decode features that must match, including word time offsets, are
    workload fields. The auth field stored here is the environment
    variable name.
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
        tls_verify: bool,
        allow_insecure_auth: bool,
    ) -> None:
        """Store endpoint and identity options. Not a token."""
        self._endpoint = endpoint
        self._use_ssl = use_ssl
        self._tls_verify = tls_verify
        self._allow_insecure_auth = allow_insecure_auth
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
                ``transport_options`` are warn-only. ``channel_options``
                are keepalive and maximum message size.

        Returns:
            An adapter that has not opened a channel.

        Raises:
            ConfigError: An option has the wrong type, TLS flags
                disagree, a bearer would cross plaintext without
                ``allow_insecure_auth``, or the riva extra is not
                installed.
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
        allow_insecure_auth = _bool_option(
            options, "allow_insecure_auth", default=False
        )
        _require_tls_mode(use_ssl=use_ssl, tls_verify=target.tls_verify)
        channel_args = _channel_args(target.channel_options)
        _import_riva()
        adapter = cls(
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
            tls_verify=target.tls_verify,
            allow_insecure_auth=allow_insecure_auth,
        )
        adapter._refuse_plaintext_bearer()
        return adapter

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
        self._refuse_plaintext_bearer()
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
        _require_tls_mode(use_ssl=self._use_ssl, tls_verify=self._tls_verify)
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
        marks: dict[str, float | None] = {
            "first": None,
            "latest": None,
            "audio_sent_s": 0.0,
        }
        outcome.chunk_send_times_s = []
        outcome.response_receive_times_s = []
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
                    outcome.first_audio_sent_s = now
                    _progress(
                        progress_callback,
                        request.label,
                        "first_audio.sent",
                        bytes=len(chunk),
                    )
                marks["latest"] = now
                outcome.last_audio_sent_s = now
                if outcome.chunk_send_times_s is not None:
                    outcome.chunk_send_times_s.append(now)
                sent = float(marks["audio_sent_s"] or 0.0)
                sent += len(chunk) / BYTES_PER_SAMPLE / SAMPLE_RATE
                marks["audio_sent_s"] = sent
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
                _drop_unset_frontier(outcome)
                return
            self._fail_stream(outcome, error, marks, request, progress_callback)
            _drop_unset_frontier(outcome)
            return
        _drop_unset_frontier(outcome)
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
        self._refuse_plaintext_bearer()
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

    def _refuse_plaintext_bearer(self) -> None:
        """Refuse to attach a bearer token to a plaintext channel.

        Raises:
            ConfigError: ``auth_env`` names a non-empty variable,
                ``use_ssl`` is false, and ``allow_insecure_auth`` is
                false. The message does not contain the token.
        """
        if self._use_ssl or self._allow_insecure_auth:
            return
        env_name = self._auth_env
        if not env_name:
            return
        token = os.environ.get(env_name) or ""
        try:
            if token:
                raise ConfigError(
                    "refusing to send a bearer token over plaintext; "
                    "set transport_options.allow_insecure_auth true "
                    "to allow it"
                )
        finally:
            del token

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
        # The lane accepted audio. A later UNAVAILABLE is the server
        # failing the session, and samples already taken stay.
        if audio_started and name == "UNAVAILABLE":
            _mark_protocol_error(outcome, name, detail)
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

    One ``StreamingRecognizeResponse`` is one interim, even when it
    carries several non-final results. Those transcripts are
    concatenated with no added separator, matching the public client's
    partial join. Final transcripts are concatenated the same way, with
    no extra space. ``final_latency_s`` stays the last final.
    ``protocol_extras["first_final_s"]`` is the first final and is set
    once. TTFS and response latency count a non-empty interim once per
    response. ``stability`` is not a latency. A result that arrives
    before any audio does not invent a latency.
    """
    now = time.monotonic()
    if outcome.response_receive_times_s is not None:
        outcome.response_receive_times_s.append(now)
    sent = marks.get("audio_sent_s")
    audio_sent = float(sent) if isinstance(sent, float) else 0.0
    parts: list[str] = []
    if outcome.transcript is not None:
        parts.append(outcome.transcript)
    interim_parts: list[str] = []
    saw_interim = False
    _note_response_frontier(outcome, response, now, audio_sent)
    for result in response.results:
        text = ""
        if result.alternatives:
            text = result.alternatives[0].transcript
        if result.is_final:
            parts.append(text)
            outcome.transcript = "".join(parts)
            outcome.final_received_s = now
            _note_words(outcome, result, now)
            latest = marks["latest"]
            if isinstance(latest, float):
                latency = time.monotonic() - latest
                if latency >= 0:
                    if "first_final_s" not in outcome.protocol_extras:
                        outcome.protocol_extras["first_final_s"] = latency
                    outcome.final_latency_s = latency
                    outcome.final_latency_clock = FINALIZATION_CLOCK
            _progress(progress_callback, request.label, "final.received")
            continue
        saw_interim = True
        interim_parts.append(text)
    if not saw_interim:
        return
    joined = "".join(interim_parts)
    if outcome.interim_texts is None:
        outcome.interim_texts = []
    outcome.interim_texts.append(joined)
    outcome.interim_count += 1
    if not joined:
        return
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
    outcome.first_audio_sent_s = None
    outcome.last_audio_sent_s = None
    outcome.final_received_s = None
    outcome.audio_progress = None
    outcome.word_timings = None
    outcome.chunk_send_times_s = None
    outcome.response_receive_times_s = None
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


def _note_response_frontier(
    outcome: SessionOutcome,
    response: Any,
    received_s: float,
    audio_sent_s: float,
) -> None:
    """Record one ``audio_processed`` sample for this response.

    Proto3 stores a missing float as 0.0, so a 0.0 is not a sample.
    Several results in one response share one receive time; the sample
    is the maximum strictly positive value among them. A later 0.0,
    including a missing interim after a real frontier, is ignored so it
    does not look like the frontier went backwards. A later positive
    value that is smaller is kept, and the reducer marks that decrease
    invalid.
    """
    positive: list[float] = []
    for result in response.results:
        descriptor = getattr(result, "DESCRIPTOR", None)
        fields = getattr(descriptor, "fields_by_name", None)
        if not fields or "audio_processed" not in fields:
            continue
        value = float(result.audio_processed)
        if value > 0.0:
            positive.append(value)
    if not positive:
        return
    if outcome.audio_progress is None:
        outcome.audio_progress = []
    outcome.audio_progress.append(
        {
            "audio_processed_s": max(positive),
            "received_s": received_s,
            "audio_sent_s": audio_sent_s,
        }
    )


def _note_words(
    outcome: SessionOutcome, result: Any, received_s: float
) -> None:
    """Record word ends. Proto ``end_time`` is milliseconds."""
    if not result.alternatives:
        return
    words = getattr(result.alternatives[0], "words", None)
    if not words:
        return
    if outcome.word_timings is None:
        outcome.word_timings = []
    for word in words:
        outcome.word_timings.append(
            {
                "end_time_s": float(word.end_time) / 1000.0,
                "final_received_s": received_s,
            }
        )


def _drop_unset_frontier(outcome: SessionOutcome) -> None:
    """Drop an all-zero frontier. Proto3 cannot tell unset from zero.

    ``_note_response_frontier`` already skips 0.0. This remains for a
    series that is still all zeros.
    """
    progress = outcome.audio_progress
    if not progress:
        outcome.audio_progress = None
        return
    if all(item["audio_processed_s"] == 0.0 for item in progress):
        outcome.audio_progress = None


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


def _require_tls_mode(*, use_ssl: bool, tls_verify: bool) -> None:
    """Refuse a TLS combination this process cannot perform honestly.

    ``grpc.ssl_channel_credentials`` always verifies the peer. Leaving
    TLS on while ``tls_verify`` is false would either skip a check the
    library will not skip, or verify and record that it did not. Neither
    is done. Plaintext with ``tls_verify`` true would record a check
    that did not happen.

    Args:
        use_ssl: Open a TLS channel.
        tls_verify: Require peer-certificate verification.

    Raises:
        ConfigError: The two flags disagree.
    """
    if use_ssl == tls_verify:
        return
    if use_ssl:
        raise ConfigError(
            "tls_verify is false and use_ssl is true; grpcio verifies "
            "every SSL peer, so set tls_verify true or use_ssl false"
        )
    raise ConfigError(
        "tls_verify is true and use_ssl is false; plaintext has no "
        "certificate to verify"
    )


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
