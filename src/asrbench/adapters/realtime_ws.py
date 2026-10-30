"""Adapter for the OpenAI-shaped realtime transcription WebSocket route.

The dialect, as vLLM core implements it:

    C->S  session.update            {model}          -> validates model
    C->S  input_audio_buffer.commit {final: false}   -> STARTS generation
    C->S  input_audio_buffer.append {audio: b64}     -> paced PCM16 @16k
    C->S  input_audio_buffer.commit {final: true}    -> closes audio
    S->C  session.created / session.updated / transcription.delta
          transcription.done {text, usage} / error {error, code}

The non-final commit is what starts generation. Omitting it is not a
degraded run but a hang: the session accepts audio and never produces a
transcript.

``lifecycle`` selects a lifecycle probe rather than a workload. ``full`` is the
measurement path; ``disconnect`` and ``abort`` exercise server cleanup, and
``no-start-commit`` and ``late-start`` reproduce specific client mistakes on
purpose, so a hang can be attributed rather than guessed at. Only ``full``
and ``late-start`` are expected to yield a transcript.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any

import aiohttp

from asrbench.adapter import SessionAdmissionBarrier, SessionProgressCallback
from asrbench.audio import (
    BYTES_PER_SAMPLE,
    SAMPLE_RATE,
    chunk_pcm16,
    read_pcm16,
)
from asrbench.clock import utc_timestamp
from asrbench.outcome import (
    FINALIZATION_CLOCK,
    NATIVE_LATENCY_CLOCK,
    RESPONSE_LATENCY_CLOCK,
    SessionOutcome,
    SessionRequest,
)


def _mirror_failure(outcome: SessionOutcome) -> None:
    """Copy timeout and exception residue onto the typed outcome fields.

    Reducers read ``timed_out`` and ``error_detail``. The extras stay as
    protocol residue.
    """
    extras = outcome.protocol_extras
    if extras.get("timed_out") is True:
        outcome.timed_out = True
    detail = extras.get("exception")
    if isinstance(detail, str):
        outcome.error_detail = detail


#: Modes that are expected to yield a final transcript.
TRANSCRIBING_MODES = frozenset({"full", "late-start"})


def _progress(
    callback: SessionProgressCallback | None,
    label: str,
    event: str,
    **fields: object,
) -> None:
    """Report one bounded protocol transition when diagnostics are enabled."""
    if callback is not None:
        callback(label, event, fields)


class RealtimeWebSocketAdapter:
    """Drives one ``/v1/realtime`` session over an aiohttp WebSocket."""

    name = "realtime-ws"
    #: Workload fields this dialect puts on ``session.update``. Cadence
    #: is sent separately. Word time offsets are not a field of this
    #: dialect. The runner refuses a non-default value outside this set
    #: before the session starts.
    supported_workload_options: frozenset[str] = frozenset()

    def __init__(self, session: aiohttp.ClientSession) -> None:
        """Bind the adapter to a client session owned by the caller."""
        self._session = session

    async def run(
        self,
        request: SessionRequest,
        *,
        admission_barrier: SessionAdmissionBarrier | None = None,
        progress_callback: SessionProgressCallback | None = None,
    ) -> SessionOutcome:
        """Drive one realtime session and return its observed result."""
        pcm = read_pcm16(request.clip_path)
        frames = chunk_pcm16(pcm, request.chunk_ms)
        url = request.endpoint_url
        label = request.label

        outcome = SessionOutcome(
            label=label,
            lifecycle=request.lifecycle,
            clip_name=request.clip_path.name,
            chunk_ms=request.chunk_ms,
            audio_duration_s=len(pcm) / BYTES_PER_SAMPLE / SAMPLE_RATE,
            latency_clock=NATIVE_LATENCY_CLOCK,
            response_latency_clock=RESPONSE_LATENCY_CLOCK,
        )
        extras = outcome.protocol_extras

        first_append_at: float | None = None
        latest_append_at: float | None = None
        final_commit_at: float | None = None
        done = asyncio.Event()
        updated = asyncio.Event()
        handshake_complete = asyncio.Event()

        try:
            _progress(progress_callback, label, "websocket.connecting", url=url)
            async with self._session.ws_connect(url, max_msg_size=0) as ws:
                _progress(progress_callback, label, "websocket.connected")

                async def receive() -> None:
                    try:
                        async for message in ws:
                            if message.type is not aiohttp.WSMsgType.TEXT:
                                continue
                            event = json.loads(message.data)
                            kind = event.get("type")
                            if kind == "session.created":
                                outcome.connected = True
                                _progress(
                                    progress_callback,
                                    label,
                                    "session_created.received",
                                )
                            elif kind == "session.updated":
                                outcome.configured = True
                                extras.setdefault("markers", {})[
                                    "session_updated_at"
                                ] = utc_timestamp()
                                updated.set()
                                handshake_complete.set()
                                _progress(
                                    progress_callback,
                                    label,
                                    "session_updated.received",
                                )
                            elif kind == "transcription.delta":
                                outcome.interim_count += 1
                                arrival = time.monotonic()
                                delta = event.get("delta", "")
                                if (
                                    delta
                                    and outcome.ttfs_s is None
                                    and first_append_at is not None
                                ):
                                    outcome.ttfs_s = arrival - first_append_at
                                if delta and latest_append_at is not None:
                                    outcome.response_latencies_s.append(
                                        arrival - latest_append_at
                                    )
                            elif kind == "transcription.done":
                                outcome.transcript = event.get("text", "")
                                extras["usage"] = event.get("usage")
                                received = time.monotonic()
                                outcome.final_received_s = received
                                if outcome.last_audio_sent_s is not None:
                                    outcome.final_latency_s = (
                                        received - outcome.last_audio_sent_s
                                    )
                                    outcome.final_latency_clock = (
                                        FINALIZATION_CLOCK
                                    )
                                elif final_commit_at is not None:
                                    outcome.final_latency_s = (
                                        received - final_commit_at
                                    )
                                    outcome.final_latency_clock = (
                                        FINALIZATION_CLOCK
                                    )
                                done.set()
                                _progress(
                                    progress_callback,
                                    label,
                                    "transcription_done.received",
                                    has_text=bool(outcome.transcript),
                                )
                                return
                            elif kind == "error":
                                outcome.protocol_errors.append(event)
                                _progress(
                                    progress_callback,
                                    label,
                                    "protocol_error.received",
                                    code=event.get("code"),
                                )
                                if not updated.is_set():
                                    handshake_complete.set()
                    finally:
                        if (
                            not updated.is_set()
                            and not outcome.protocol_errors
                            and extras.get("exception") is None
                        ):
                            extras["exception"] = (
                                "WebSocket closed before session.updated"
                            )
                        handshake_complete.set()
                        done.set()

                reader = asyncio.create_task(receive())

                async def send(payload: dict[str, Any]) -> None:
                    await ws.send_str(json.dumps(payload))

                async def append_all() -> bool:
                    """Stream every frame; True if the stream ran to the end."""
                    nonlocal first_append_at, latest_append_at
                    cutoff = len(frames) // 2
                    for index, frame in enumerate(frames, start=1):
                        if reader.done():
                            return False
                        if first_append_at is None:
                            # The client-clock TTFS starts when transmission of
                            # the first audio sample begins, before send can
                            # yield.
                            first_append_at = time.monotonic()
                            outcome.first_audio_sent_s = first_append_at
                            extras.setdefault("markers", {})[
                                "first_audio_at"
                            ] = utc_timestamp()
                        latest_append_at = time.monotonic()
                        outcome.last_audio_sent_s = latest_append_at
                        await send(
                            {
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(frame).decode(
                                    "ascii"
                                ),
                            }
                        )
                        if index == 1:
                            _progress(
                                progress_callback,
                                label,
                                "first_audio_append.sent",
                                bytes=len(frame),
                            )
                        if (
                            request.lifecycle == "disconnect"
                            and index >= cutoff
                        ):
                            extras["disconnected_after_frames"] = index
                            await ws.close()
                            return False
                        if request.pace:
                            await asyncio.sleep(request.chunk_ms / 1000)
                    return True

                update: dict[str, Any] = {
                    "type": "session.update",
                    "model": request.model,
                }
                if request.cadence_ms is not None:
                    update["cadence"] = f"{request.cadence_ms}ms"
                await send(update)
                _progress(
                    progress_callback,
                    label,
                    "session_update.sent",
                    cadence_ms=request.cadence_ms,
                )
                try:
                    await asyncio.wait_for(
                        handshake_complete.wait(), timeout=request.timeout_s
                    )
                except (TimeoutError, asyncio.TimeoutError):  # noqa: UP041
                    extras["timed_out"] = True
                    extras["exception"] = (
                        "TimeoutError: session.updated was not received"
                    )
                    _progress(
                        progress_callback,
                        label,
                        "session_update.timeout",
                        timeout_s=request.timeout_s,
                    )
                if not updated.is_set():
                    reader.cancel()
                    await asyncio.gather(reader, return_exceptions=True)
                    _mirror_failure(outcome)
                    return outcome
                if admission_barrier is not None:
                    _progress(progress_callback, label, "barrier.waiting")
                    try:
                        released = await admission_barrier.wait(label)
                    except (TimeoutError, asyncio.TimeoutError):  # noqa: UP041
                        extras["timed_out"] = True
                        extras["exception"] = "TimeoutError: cohort barrier"
                        _progress(
                            progress_callback,
                            label,
                            "barrier.timeout",
                            timeout_s=request.timeout_s,
                        )
                        _mirror_failure(outcome)
                        return outcome
                    extras.setdefault("markers", {}).update(released)
                    _progress(progress_callback, label, "barrier.released")

                if request.lifecycle in ("full", "disconnect", "abort"):
                    # The non-final commit is what starts generation.
                    await send(
                        {"type": "input_audio_buffer.commit", "final": False}
                    )
                    _progress(progress_callback, label, "start_commit.sent")
                    streamed = await append_all()
                elif request.lifecycle == "late-start":
                    streamed = await append_all()
                    await send(
                        {"type": "input_audio_buffer.commit", "final": False}
                    )
                    _progress(progress_callback, label, "start_commit.sent")
                elif request.lifecycle == "no-start-commit":
                    streamed = await append_all()
                else:
                    raise SystemExit(f"unknown lifecycle {request.lifecycle}")

                if streamed:
                    final_commit_at = time.monotonic()
                    await send(
                        {"type": "input_audio_buffer.commit", "final": True}
                    )
                    _progress(progress_callback, label, "final_commit.sent")
                    if request.lifecycle == "abort":
                        # Close while generation is still in flight.
                        await asyncio.sleep(0.05)
                        extras["aborted"] = True
                        await ws.close()
                    else:
                        try:
                            await asyncio.wait_for(
                                done.wait(), timeout=request.timeout_s
                            )
                        except (  # noqa: UP041
                            TimeoutError,
                            asyncio.TimeoutError,
                        ):
                            extras["timed_out"] = True
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
                _progress(
                    progress_callback,
                    label,
                    "session.complete",
                    has_text=bool(outcome.transcript),
                    error_count=len(outcome.protocol_errors),
                )
        except Exception as error:  # noqa: BLE001 - reported, never raised
            extras["exception"] = f"{type(error).__name__}: {error}"
            _progress(
                progress_callback,
                label,
                "session.exception",
                error_type=type(error).__name__,
            )

        _mirror_failure(outcome)
        return outcome
