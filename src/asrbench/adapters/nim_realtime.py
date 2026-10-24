"""NIM transcription WebSocket adapter.

Public protocol, from the NVIDIA NIM speech realtime ASR page and the
``RealtimeClientASR`` flow in nvidia-riva/python-clients: mint
``POST /v1/realtime/transcription_sessions``, then connect to
``/v1/realtime?intent=transcription``. An ephemeral ``client_secret``
travels on ``Sec-WebSocket-Protocol`` as ``realtime`` plus
``realtime-token.<value>``, never in the URL. The client then sends
``transcription_session.update``, ``input_audio_buffer.append`` and
``input_audio_buffer.commit`` for each PCM16 chunk, and
``input_audio_buffer.done`` after the last chunk.

This is not the vLLM realtime dialect in ``realtime_ws``. That dialect
sends ``session.update`` and reads ``transcription.delta``. Relabeling
one as the other would measure the wrong server.

``aiohttp`` is imported only inside this adapter. The bearer and the
ephemeral token are read for the handshake and are not stored on the
adapter, the request, or the outcome. An unopened socket, or HTTP 401 /
403 from the mint, is ``lane_unavailable``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

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

_TRANSPORT_KEYS = frozenset(
    {
        "use_ssl",
        "max_alternatives",
        "enable_word_time_offsets",
        "enable_profanity_filter",
        "enable_verbatim_transcripts",
        "custom_configuration",
        "max_speaker_count",
    }
)
_CHANNEL_FIELDS = frozenset(
    {"max_send_message_bytes", "max_receive_message_bytes"}
)
_BEARER = re.compile(r"(?i)bearer\s+\S+")
_MINT_PATH = "/v1/realtime/transcription_sessions"
_SOCKET_PATH = "/v1/realtime?intent=transcription"
_DEFAULT_MAX_SPEAKERS = 2
_DETAIL_LIMIT = 240


def _import_aiohttp() -> Any:
    """Import aiohttp, or fail as configuration.

    Returns:
        The aiohttp module.

    Raises:
        ConfigError: The websocket extra is not installed.
    """
    try:
        import aiohttp
    except ImportError as error:
        raise ConfigError(
            "nim-realtime requires the websocket extra (aiohttp)"
        ) from error
    return aiohttp


def _redact(
    text: str, auth_env: str | None, ephemeral: str | None = None
) -> str:
    """Remove bearer material and the ephemeral token from stored text."""
    cleaned = _BEARER.sub("Bearer [redacted]", text)
    secrets: list[str] = []
    if auth_env:
        env_value = os.environ.get(auth_env) or ""
        if env_value:
            secrets.append(env_value)
        del env_value
    if ephemeral:
        secrets.append(ephemeral)
    for secret in secrets:
        cleaned = cleaned.replace(secret, "[redacted]")
    del secrets
    return cleaned


def _endpoint_urls(endpoint: str, use_ssl: bool) -> tuple[str, str]:
    """Return the mint URL and the websocket URL for one endpoint.

    The endpoint is ``host:port`` or an origin with no path, query, or
    userinfo. The adapter adds the public paths. A scheme must agree
    with ``use_ssl``.

    Raises:
        ConfigError: The endpoint is not that shape.
    """
    raw = endpoint.strip()
    if "://" in raw:
        parsed = urlsplit(raw)
        if parsed.username or parsed.password:
            raise ConfigError(
                "nim-realtime endpoint must not carry credentials"
            )
        if parsed.scheme in {"https", "wss"}:
            secure = True
        elif parsed.scheme in {"http", "ws"}:
            secure = False
        else:
            raise ConfigError(
                "nim-realtime endpoint scheme must be http, https, ws, or wss"
            )
        if secure != use_ssl:
            raise ConfigError(
                "nim-realtime endpoint scheme disagrees with use_ssl"
            )
        if parsed.path not in ("", "/"):
            raise ConfigError(
                "nim-realtime endpoint is an origin; the adapter adds "
                "/v1/realtime?intent=transcription"
            )
        if parsed.query or parsed.fragment:
            raise ConfigError(
                "nim-realtime endpoint must not carry a query or fragment"
            )
        netloc = parsed.netloc
    else:
        host, separator, port = raw.rpartition(":")
        if not host or separator != ":" or not port.isdigit():
            raise ConfigError("nim-realtime endpoint must be host:port")
        secure = use_ssl
        netloc = f"{host}:{port}"
    http = "https" if secure else "http"
    socket = "wss" if secure else "ws"
    return (
        f"{http}://{netloc}{_MINT_PATH}",
        f"{socket}://{netloc}{_SOCKET_PATH}",
    )


def _progress(
    callback: SessionProgressCallback | None,
    label: str,
    event: str,
    **fields: object,
) -> None:
    """Report one bounded lifecycle edge. Never a chunk, never a token."""
    if callback is not None:
        callback(label, event, fields)


class NimRealtimeAdapter:
    """Drive one NIM transcription session over a WebSocket.

    Channel message sizes are read from ``channel_options``. Everything
    in ``transport_options`` is sent and is hard target identity. The
    auth field stored here is the environment variable name for the
    mint request. gRPC keepalive is not a field of this protocol.
    """

    name = "nim-realtime"
    supported_workload_options = frozenset(
        {"language", "punctuation", "diarization"}
    )

    def __init__(
        self,
        endpoint: str,
        *,
        use_ssl: bool,
        max_alternatives: int,
        word_time_offsets: bool,
        profanity_filter: bool,
        verbatim_transcripts: bool,
        custom_configuration: dict[str, str],
        max_speaker_count: int,
        auth_env: str | None,
        max_msg_size: int | None,
    ) -> None:
        """Store endpoint and identity options. Not a token."""
        self._endpoint = endpoint
        self._use_ssl = use_ssl
        self._max_alternatives = max_alternatives
        self._word_offsets = word_time_offsets
        self._profanity = profanity_filter
        self._verbatim = verbatim_transcripts
        self._custom = dict(custom_configuration)
        self._max_speakers = max_speaker_count
        self._auth_env = auth_env
        self._max_msg_size = max_msg_size

    @classmethod
    def from_target(cls, target: TargetSpec) -> NimRealtimeAdapter:
        """Build an adapter from a resolved target.

        Args:
            target: Endpoint, model, auth variable name, and options.

        Returns:
            An adapter that has not opened a socket.

        Raises:
            ConfigError: An option has the wrong type, or aiohttp is
                not installed.
        """
        options = dict(target.transport_options)
        unknown = sorted(set(options) - _TRANSPORT_KEYS)
        if unknown:
            raise ConfigError(
                "unknown nim-realtime transport_options: " + ", ".join(unknown)
            )
        _endpoint_urls(target.endpoint_url, _bool_option(options, "use_ssl"))
        _import_aiohttp()
        return cls(
            target.endpoint_url,
            use_ssl=_bool_option(options, "use_ssl"),
            max_alternatives=_int_option(
                options, "max_alternatives", default=1
            ),
            word_time_offsets=_bool_option(
                options, "enable_word_time_offsets", default=False
            ),
            profanity_filter=_bool_option(
                options, "enable_profanity_filter", default=False
            ),
            verbatim_transcripts=_bool_option(
                options, "enable_verbatim_transcripts", default=False
            ),
            custom_configuration=_custom_configuration(
                options.get("custom_configuration", {})
            ),
            max_speaker_count=_int_option(
                options, "max_speaker_count", default=_DEFAULT_MAX_SPEAKERS
            ),
            auth_env=target.auth_env,
            max_msg_size=_message_size(target.channel_options),
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
                ``streaming`` is reported on the outcome. This adapter
                does not speak offline Recognize.
            admission_barrier: Released after
                ``transcription_session.updated`` and before the first
                audio chunk.
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
                "nim-realtime implements the transcription WebSocket only"
            )
            outcome.terminal_status = "errored"
            return outcome
        holder: dict[str, Any] = {}
        try:
            await self._drive(
                request, outcome, admission_barrier, progress_callback, holder
            )
        except asyncio.CancelledError:
            socket = holder.get("ws")
            if socket is not None:
                await socket.close()
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
        aiohttp = _import_aiohttp()
        pcm = read_pcm16(request.clip_path)
        chunks = chunk_pcm16(pcm, request.chunk_ms)
        outcome.audio_duration_s = len(pcm) / BYTES_PER_SAMPLE / SAMPLE_RATE
        outcome.protocol_extras["adapter"] = self.name
        outcome.protocol_extras["audio_chunks"] = len(chunks)
        outcome.protocol_extras["recognition_request"] = _recognition_echo(
            self, request
        )
        mint_url, ws_url = _endpoint_urls(self._endpoint, self._use_ssl)
        # Mint, connect, and transcription_session.updated share one
        # timeout_s. The final wait is a second timeout_s. The cohort
        # barrier is the runner's third. That fits the hang guard.
        setup_deadline = time.monotonic() + request.timeout_s
        timeout = aiohttp.ClientTimeout(
            total=None, sock_connect=request.timeout_s
        )
        session = aiohttp.ClientSession(timeout=timeout)
        ephemeral = ""
        cancelled = False
        try:
            ephemeral = await self._mint(
                aiohttp,
                session,
                mint_url,
                request,
                outcome,
                progress_callback,
                setup_deadline,
            )
            if outcome.terminal_status:
                return
            protocols = ["realtime"]
            if ephemeral:
                protocols.append("realtime-token." + ephemeral)
            try:
                socket = await self._connect(
                    session,
                    ws_url,
                    protocols,
                    request,
                    outcome,
                    progress_callback,
                    ephemeral,
                    setup_deadline,
                )
            finally:
                del protocols
            if socket is None:
                return
            holder["ws"] = socket
            try:
                await self._speak(
                    socket,
                    chunks,
                    request,
                    outcome,
                    admission_barrier,
                    progress_callback,
                    ephemeral,
                    setup_deadline,
                )
            except asyncio.CancelledError:
                cancelled = True
                raise
            finally:
                await _cleanup(_close_socket(socket), cancelled=cancelled)
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            del ephemeral
            await _cleanup(_close_session(session), cancelled=cancelled)

    async def _mint(
        self,
        aiohttp: Any,
        session: Any,
        mint_url: str,
        request: SessionRequest,
        outcome: SessionOutcome,
        progress_callback: SessionProgressCallback | None,
        setup_deadline: float,
    ) -> str:
        """POST the session. Return an ephemeral token, or ``""``."""
        headers = self._mint_headers()
        try:
            try:
                mint_timeout = _remaining(setup_deadline)
            except TimeoutError:
                _mark_lane_unavailable(
                    outcome,
                    "UNAVAILABLE",
                    "transcription session mint timed out",
                    progress_callback,
                    request.label,
                )
                return ""
            try:
                response_cm = session.post(
                    mint_url,
                    json={},
                    headers=headers,
                    allow_redirects=False,
                    timeout=aiohttp.ClientTimeout(total=mint_timeout),
                )
                async with response_cm as response:
                    body = await response.text()
                    status = int(response.status)
            except asyncio.CancelledError:
                raise
            except (TimeoutError, asyncio.TimeoutError):  # noqa: UP041
                _mark_lane_unavailable(
                    outcome,
                    "UNAVAILABLE",
                    "transcription session mint timed out",
                    progress_callback,
                    request.label,
                )
                return ""
            except OSError as error:
                _mark_lane_unavailable(
                    outcome,
                    "UNAVAILABLE",
                    _redact(str(error), self._auth_env),
                    progress_callback,
                    request.label,
                )
                return ""
        finally:
            del headers
        if status != 200:
            self._fail_mint(
                outcome, status, body, request.label, progress_callback
            )
            return ""
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            _mark_protocol_error(
                outcome, "INVALID_RESPONSE", "mint body was not JSON"
            )
            return ""
        if not isinstance(payload, dict):
            _mark_protocol_error(
                outcome, "INVALID_RESPONSE", "mint body was not an object"
            )
            return ""
        token = _ephemeral_token(payload)
        outcome.protocol_extras["server_identity"] = {
            "mint": _strip_secrets(payload)
        }
        _progress(progress_callback, request.label, "session.minted")
        return token

    def _fail_mint(
        self,
        outcome: SessionOutcome,
        status: int,
        body: str,
        label: str,
        progress_callback: SessionProgressCallback | None,
    ) -> None:
        detail = (
            _redact(body, self._auth_env)[:_DETAIL_LIMIT] or f"HTTP {status}"
        )
        if status == 401:
            _mark_lane_unavailable(
                outcome, "UNAUTHENTICATED", detail, progress_callback, label
            )
            return
        if status == 403:
            _mark_lane_unavailable(
                outcome, "PERMISSION_DENIED", detail, progress_callback, label
            )
            return
        if status >= 500 or status in {301, 302, 303, 307, 308}:
            _mark_lane_unavailable(
                outcome, "UNAVAILABLE", detail, progress_callback, label
            )
            return
        _mark_protocol_error(outcome, f"HTTP_{status}", detail)

    async def _connect(
        self,
        session: Any,
        ws_url: str,
        protocols: list[str],
        request: SessionRequest,
        outcome: SessionOutcome,
        progress_callback: SessionProgressCallback | None,
        ephemeral: str,
        setup_deadline: float,
    ) -> Any | None:
        kwargs: dict[str, Any] = {"protocols": protocols}
        if self._max_msg_size is not None:
            kwargs["max_msg_size"] = self._max_msg_size
        try:
            socket = await asyncio.wait_for(
                session.ws_connect(ws_url, **kwargs),
                timeout=_remaining(setup_deadline),
            )
        except asyncio.CancelledError:
            raise
        except (TimeoutError, asyncio.TimeoutError):  # noqa: UP041
            _mark_lane_unavailable(
                outcome,
                "UNAVAILABLE",
                "websocket connect timed out",
                progress_callback,
                request.label,
            )
            return None
        except Exception as error:  # noqa: BLE001 - the lane stays data
            status = getattr(error, "status", None)
            text = _redact(str(error), self._auth_env, ephemeral or None)
            if status in {401, 403}:
                code = (
                    "UNAUTHENTICATED" if status == 401 else "PERMISSION_DENIED"
                )
                _mark_lane_unavailable(
                    outcome,
                    code,
                    text[:_DETAIL_LIMIT],
                    progress_callback,
                    request.label,
                )
                return None
            _mark_lane_unavailable(
                outcome,
                "UNAVAILABLE",
                text[:_DETAIL_LIMIT] or type(error).__name__,
                progress_callback,
                request.label,
            )
            return None
        outcome.connected = True
        outcome.interim_texts = []
        _progress(progress_callback, request.label, "websocket.connected")
        return socket

    async def _speak(
        self,
        socket: Any,
        chunks: list[bytes],
        request: SessionRequest,
        outcome: SessionOutcome,
        admission_barrier: SessionAdmissionBarrier | None,
        progress_callback: SessionProgressCallback | None,
        ephemeral: str,
        setup_deadline: float,
    ) -> None:
        marks: dict[str, float | None] = {"first": None, "latest": None}
        updated = asyncio.Event()
        finished = asyncio.Event()

        async def receive() -> None:
            try:
                async for message in socket:
                    if message.type is not _import_aiohttp().WSMsgType.TEXT:
                        continue
                    self._on_message(
                        message.data,
                        outcome,
                        marks,
                        updated,
                        finished,
                        progress_callback,
                        request,
                        ephemeral,
                    )
                    if finished.is_set():
                        return
            except asyncio.CancelledError:
                raise
            finally:
                updated.set()
                finished.set()

        reader = asyncio.create_task(receive())
        cancelled = False
        try:
            await _send(
                socket,
                {
                    "type": "transcription_session.update",
                    "session": _session_body(self, request),
                },
            )
            _progress(
                progress_callback,
                request.label,
                "transcription_session.update.sent",
            )
            try:
                updated_in_time = await _wait_event(
                    updated, _remaining(setup_deadline)
                )
            except TimeoutError:
                updated_in_time = False
            if not updated_in_time:
                _mark_lane_unavailable(
                    outcome,
                    "UNAVAILABLE",
                    "transcription_session.updated was not received",
                    progress_callback,
                    request.label,
                )
                return
            if outcome.terminal_status or not outcome.configured:
                if not outcome.terminal_status:
                    _mark_lane_unavailable(
                        outcome,
                        "UNAVAILABLE",
                        "socket closed before transcription_session.updated",
                        progress_callback,
                        request.label,
                    )
                return
            if not await _pass_barrier(
                request, outcome, admission_barrier, progress_callback
            ):
                return
            await self._send_audio(
                socket, chunks, request, marks, finished, progress_callback
            )
            if finished.is_set() or outcome.terminal_status:
                return
            await _send(socket, {"type": "input_audio_buffer.done"})
            _progress(
                progress_callback, request.label, "input_audio_buffer.done.sent"
            )
            if not await _wait_event(finished, request.timeout_s):
                outcome.timed_out = True
                outcome.terminal_status = "timed_out"
                outcome.error_code = "DEADLINE_EXCEEDED"
                outcome.error_detail = (
                    "TimeoutError: is_last_result was not received"
                )
                return
            self._finish(outcome, marks, progress_callback, request)
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception as error:  # noqa: BLE001 - the lane stays data
            if outcome.terminal_status:
                return
            detail = _redact(str(error), self._auth_env, ephemeral or None)
            if isinstance(marks["first"], float):
                _mark_protocol_error(
                    outcome, type(error).__name__, detail[:_DETAIL_LIMIT]
                )
                return
            _mark_lane_unavailable(
                outcome,
                "UNAVAILABLE",
                detail[:_DETAIL_LIMIT] or type(error).__name__,
                progress_callback,
                request.label,
            )
        finally:
            reader.cancel()
            await _cleanup(
                asyncio.gather(reader, return_exceptions=True),
                cancelled=cancelled,
            )

    def _on_message(
        self,
        raw: str,
        outcome: SessionOutcome,
        marks: dict[str, float | None],
        updated: asyncio.Event,
        finished: asyncio.Event,
        progress_callback: SessionProgressCallback | None,
        request: SessionRequest,
        ephemeral: str,
    ) -> None:
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            _mark_protocol_error(
                outcome, "INVALID_RESPONSE", "event was not JSON"
            )
            finished.set()
            updated.set()
            return
        if not isinstance(event, dict):
            _mark_protocol_error(
                outcome, "INVALID_RESPONSE", "event was not an object"
            )
            finished.set()
            updated.set()
            return
        kind = event.get("type")
        if kind == "transcription_session.updated":
            session = event.get("session")
            identity = outcome.protocol_extras.setdefault("server_identity", {})
            if isinstance(identity, dict):
                identity["session"] = _strip_secrets(session)
            outcome.configured = True
            updated.set()
            _progress(
                progress_callback,
                request.label,
                "transcription_session.updated",
            )
            return
        if kind == "conversation.item.input_audio_transcription.delta":
            delta = event.get("delta")
            _note_partial(
                outcome, marks, delta if isinstance(delta, str) else ""
            )
            return
        if kind == "conversation.item.input_audio_transcription.completed":
            text = event.get("transcript")
            if not isinstance(text, str):
                _mark_protocol_error(
                    outcome,
                    "INVALID_RESPONSE",
                    "completed transcript was not a string",
                )
                finished.set()
                return
            last = event.get("is_last_result") is True
            if not last:
                _note_partial(outcome, marks, text)
                return
            outcome.transcript = text
            latest = marks["latest"]
            if isinstance(latest, float):
                latency = time.monotonic() - latest
                if latency >= 0:
                    outcome.final_latency_s = latency
            _progress(progress_callback, request.label, "final.received")
            finished.set()
            return
        if kind in {
            "conversation.item.input_audio_transcription.failed",
            "error",
        }:
            code, detail = _error_parts(event)
            detail = _redact(detail, self._auth_env, ephemeral or None)
            _mark_protocol_error(outcome, code, detail[:_DETAIL_LIMIT])
            finished.set()
            updated.set()
            return
        if kind in {
            "input_audio_buffer.committed",
            "input_audio_buffer.cleared",
        }:
            return
        _mark_protocol_error(
            outcome,
            "UNEXPECTED_EVENT",
            _redact(str(kind), self._auth_env, ephemeral or None)[
                :_DETAIL_LIMIT
            ],
        )
        finished.set()
        updated.set()

    async def _send_audio(
        self,
        socket: Any,
        chunks: list[bytes],
        request: SessionRequest,
        marks: dict[str, float | None],
        finished: asyncio.Event,
        progress_callback: SessionProgressCallback | None,
    ) -> None:
        for index, chunk in enumerate(chunks):
            if finished.is_set():
                return
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
            encoded = base64.b64encode(chunk).decode("ascii")
            await _send(
                socket,
                {"type": "input_audio_buffer.append", "audio": encoded},
            )
            del encoded
            await _send(socket, {"type": "input_audio_buffer.commit"})
            if request.pace and index + 1 < len(chunks):
                await asyncio.sleep(request.chunk_ms / 1000)

    def _finish(
        self,
        outcome: SessionOutcome,
        marks: dict[str, float | None],
        progress_callback: SessionProgressCallback | None,
        request: SessionRequest,
    ) -> None:
        if outcome.timed_out or outcome.terminal_status in {
            "lane_unavailable",
            "errored",
            "timed_out",
        }:
            return
        if outcome.error_code:
            outcome.terminal_status = "errored"
            return
        if outcome.transcript is not None:
            outcome.terminal_status = "completed"
            _progress(
                progress_callback,
                request.label,
                "session.complete",
                has_text=bool(outcome.transcript),
            )
            return
        if isinstance(marks["first"], float):
            _mark_protocol_error(
                outcome,
                "INCOMPLETE",
                "socket closed before is_last_result",
            )
            return
        _mark_lane_unavailable(
            outcome,
            "UNAVAILABLE",
            "socket closed before audio",
            progress_callback,
            request.label,
        )

    def _mint_headers(self) -> dict[str, str]:
        """Build mint headers. The token is not stored on this adapter."""
        headers = {"Accept": "application/json"}
        env_name = self._auth_env
        if not env_name:
            return headers
        token = os.environ.get(env_name) or ""
        if token:
            headers["Authorization"] = "Bearer " + token
        del token
        return headers


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


def _session_body(
    adapter: NimRealtimeAdapter, request: SessionRequest
) -> dict[str, Any]:
    """Build the documented ``transcription_session.update`` session object."""
    recognition: dict[str, Any] = {
        "max_alternatives": adapter._max_alternatives,
        "enable_automatic_punctuation": request.punctuation,
        "enable_word_time_offsets": adapter._word_offsets,
        "enable_profanity_filter": adapter._profanity,
        "enable_verbatim_transcripts": adapter._verbatim,
    }
    if adapter._custom:
        recognition["custom_configuration"] = ",".join(
            f"{key}:{value}" for key, value in sorted(adapter._custom.items())
        )
    body: dict[str, Any] = {
        "modalities": ["text"],
        "input_audio_format": "pcm16",
        "input_audio_transcription": {
            "language": request.language,
            "model": request.model,
            "prompt": "",
        },
        "input_audio_params": {
            "sample_rate_hz": SAMPLE_RATE,
            "num_channels": 1,
        },
        "recognition_config": recognition,
    }
    if request.diarization:
        body["speaker_diarization"] = {
            "enable_speaker_diarization": True,
            "max_speaker_count": adapter._max_speakers,
        }
    return body


def _recognition_echo(
    adapter: NimRealtimeAdapter, request: SessionRequest
) -> dict[str, Any]:
    return {
        "language": request.language,
        "model": request.model,
        "input_audio_format": "pcm16",
        "sample_rate_hz": SAMPLE_RATE,
        "max_alternatives": adapter._max_alternatives,
        "enable_automatic_punctuation": request.punctuation,
        "enable_word_time_offsets": adapter._word_offsets,
        "enable_profanity_filter": adapter._profanity,
        "enable_verbatim_transcripts": adapter._verbatim,
        "diarization": request.diarization,
        "custom_configuration": dict(adapter._custom),
    }


def _note_partial(
    outcome: SessionOutcome, marks: dict[str, float | None], text: str
) -> None:
    """Record one partial. Empty text has no latency."""
    if outcome.interim_texts is None:
        outcome.interim_texts = []
    outcome.interim_texts.append(text)
    outcome.interim_count += 1
    if not text:
        return
    now = time.monotonic()
    first = marks["first"]
    latest = marks["latest"]
    if isinstance(first, float) and outcome.ttfs_s is None:
        outcome.ttfs_s = now - first
    if isinstance(latest, float):
        outcome.response_latencies_s.append(now - latest)


def _error_parts(event: Mapping[str, Any]) -> tuple[str, str]:
    error = event.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        message = error.get("message")
        name = code if isinstance(code, str) and code else "ERROR"
        detail = message if isinstance(message, str) else ""
        return name, detail
    return "ERROR", ""


def _ephemeral_token(payload: Mapping[str, Any]) -> str:
    """Return the minted client secret, or ``""`` when none was minted."""
    secret = payload.get("client_secret")
    if not isinstance(secret, dict):
        return ""
    value = secret.get("value")
    if not isinstance(value, str) or value == "":
        return ""
    return value


def _strip_secrets(value: object) -> object:
    """Drop ``client_secret`` before anything is stored on the outcome."""
    if isinstance(value, dict):
        return {
            str(key): _strip_secrets(item)
            for key, item in value.items()
            if key != "client_secret"
        }
    if isinstance(value, list):
        return [_strip_secrets(item) for item in value]
    return value


async def _pass_barrier(
    request: SessionRequest,
    outcome: SessionOutcome,
    admission_barrier: SessionAdmissionBarrier | None,
    progress_callback: SessionProgressCallback | None,
) -> bool:
    """Wait for the cohort. Return false when the barrier timed out."""
    if admission_barrier is None:
        return True
    _progress(progress_callback, request.label, "barrier.waiting")
    try:
        released = await admission_barrier.wait(request.label)
    except (TimeoutError, asyncio.TimeoutError):  # noqa: UP041
        outcome.timed_out = True
        outcome.terminal_status = "timed_out"
        outcome.error_code = "DEADLINE_EXCEEDED"
        outcome.error_detail = "TimeoutError: cohort barrier"
        _progress(progress_callback, request.label, "barrier.timeout")
        return False
    outcome.protocol_extras.setdefault("markers", {}).update(released)
    _progress(progress_callback, request.label, "barrier.released")
    return True


async def _send(socket: Any, payload: Mapping[str, Any]) -> None:
    await socket.send_str(json.dumps(payload))


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


async def _wait_event(event: asyncio.Event, timeout: float) -> bool:
    """Wait until ``event`` is set or ``timeout`` elapses.

    Returns true when the event is set. ``asyncio.wait_for`` on Python
    3.10 drops a cancellation when the inner future is already done, so
    a cancelled session would continue and report a timeout.
    """
    if timeout <= 0 and not event.is_set():
        return False
    if event.is_set():
        await asyncio.sleep(0)
        return True
    waiter = asyncio.create_task(event.wait())
    try:
        await asyncio.wait({waiter}, timeout=timeout)
    except asyncio.CancelledError:
        waiter.cancel()
        raise
    if waiter.done() and not waiter.cancelled():
        return True
    waiter.cancel()
    return False


async def _cleanup(action: Any, *, cancelled: bool) -> None:
    """Await cleanup, then restore a cancellation 3.10 would drop.

    An ``await`` in ``finally`` on Python 3.10 completes and discards a
    pending ``CancelledError``. Raising again keeps the runner's hang
    guard and the caller's ``except CancelledError`` honest.
    """
    try:
        await action
    finally:
        if cancelled:
            raise asyncio.CancelledError


async def _close_socket(socket: Any) -> None:
    try:
        await socket.close()
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - closing is cleanup
        return


async def _close_session(session: Any) -> None:
    try:
        await session.close()
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - closing is cleanup
        return


def _remaining(deadline: float) -> float:
    """Return the seconds left before a setup deadline.

    Raises:
        TimeoutError: The deadline has passed.
    """
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError
    return left


def _bool_option(
    options: Mapping[str, Any], name: str, *, default: bool = False
) -> bool:
    if name not in options:
        return default
    value = options[name]
    if not isinstance(value, bool):
        raise ConfigError(
            f"nim-realtime transport_options.{name} must be a bool"
        )
    return value


def _int_option(options: Mapping[str, Any], name: str, *, default: int) -> int:
    if name not in options:
        return default
    value = options[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(
            f"nim-realtime transport_options.{name} must be an int >= 1"
        )
    return value


def _custom_configuration(value: object) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError("nim-realtime custom_configuration must be an object")
    built: dict[str, str] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not isinstance(item, str)
            or not key
            or any(mark in key or mark in item for mark in ",:")
        ):
            raise ConfigError(
                "nim-realtime custom_configuration entries must be strings "
                "without comma or colon"
            )
        built[key] = item
    return built


def _message_size(options: Mapping[str, Any]) -> int | None:
    """Return the receive limit. gRPC keepalive is not accepted here."""
    unknown = sorted(set(options) - _CHANNEL_FIELDS)
    if unknown:
        raise ConfigError(
            "unknown nim-realtime channel_options: "
            + ", ".join(unknown)
            + ". keepalive is a gRPC channel arg and is not sent"
        )
    receive = options.get("max_receive_message_bytes")
    send = options.get("max_send_message_bytes")
    chosen = receive if receive is not None else send
    if chosen is None:
        return None
    if isinstance(chosen, bool) or not isinstance(chosen, int) or chosen < 1:
        raise ConfigError("nim-realtime message size must be a positive int")
    return chosen
