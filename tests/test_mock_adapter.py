"""The adapter interface must be implementable off any one transport.

A protocol interface with exactly one implementation has not been shown to
be an interface. This suite implements a second one — over no transport at
all — and drives it through the same shapes, which is what the framework
promises: a benchmark that runs with no server, no GPU, and no transport
library, so the core is exercisable by anyone.
"""

import asyncio
import wave
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest

from asrbench.adapter import (
    SessionAdmissionBarrier,
    SessionProgressCallback,
    TransportAdapter,
)
from asrbench.audio import (
    BYTES_PER_SAMPLE,
    SAMPLE_RATE,
    chunk_pcm16,
    read_pcm16,
)
from asrbench.outcome import SessionOutcome, SessionRequest


class ReplayAdapter:
    """Replays a scripted transcript against the clip's real chunking."""

    name = "replay"

    def __init__(self, transcript: str, interim_every: int = 2) -> None:
        self.transcript = transcript
        self.interim_every = interim_every
        self.seen: list[str] = []

    async def run(
        self,
        request: SessionRequest,
        *,
        admission_barrier: SessionAdmissionBarrier | None = None,
        progress_callback: SessionProgressCallback | None = None,
    ) -> SessionOutcome:
        """Drive one session without touching a network transport."""
        pcm = read_pcm16(request.clip_path)
        chunks = chunk_pcm16(pcm, request.chunk_ms)
        outcome = SessionOutcome(
            label=request.label,
            lifecycle=request.lifecycle,
            clip_name=request.clip_path.name,
            chunk_ms=request.chunk_ms,
            audio_duration_s=len(pcm) / BYTES_PER_SAMPLE / SAMPLE_RATE,
            latency_clock="replay/none",
            response_latency_clock="replay/none",
        )
        outcome.connected = True
        outcome.configured = True
        if admission_barrier is not None:
            outcome.protocol_extras.setdefault("markers", {}).update(
                await admission_barrier.wait(request.label)
            )
        for index, _chunk in enumerate(chunks, start=1):
            self.seen.append(request.label)
            if index % self.interim_every == 0:
                outcome.interim_count += 1
                outcome.response_latencies_s.append(0.001 * index)
                if outcome.ttfs_s is None:
                    outcome.ttfs_s = 0.001 * index
        outcome.transcript = self.transcript
        outcome.final_latency_s = 0.002
        return outcome


def _write_wav(path: Path, seconds: float) -> Path:
    """Write a silent mono PCM16 clip of the requested duration."""
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(BYTES_PER_SAMPLE)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(b"\x00\x00" * int(SAMPLE_RATE * seconds))
    return path


def _request(clip: Path, **overrides: Any) -> SessionRequest:
    """Build a request for the replay transport, with named overrides."""
    request = SessionRequest(
        endpoint_url="replay://local",
        model="mock-asr",
        clip_path=clip,
        label="lane0",
        chunk_ms=160,
        cadence_ms=320,
        timeout_s=5.0,
        lifecycle="full",
        pace=False,
    )
    return replace(request, **overrides) if overrides else request


def test_a_non_transport_adapter_satisfies_the_interface(
    tmp_path: Path,
) -> None:
    """The protocol is implementable without any wire format."""
    adapter: TransportAdapter = ReplayAdapter("hello world")
    clip = _write_wav(tmp_path / "clip.wav", 1.6)

    outcome = asyncio.run(adapter.run(_request(clip)))

    assert adapter.name == "replay"
    assert outcome.transcript == "hello world"
    assert outcome.configured is True
    assert outcome.interim_count == 5


def test_the_outcome_is_self_identifying(tmp_path: Path) -> None:
    """A reducer sees only outcomes, so each must carry its own identity."""
    clip = _write_wav(tmp_path / "clip.wav", 1.6)

    outcome = asyncio.run(ReplayAdapter("hello").run(_request(clip)))

    assert outcome.label == "lane0"
    assert outcome.clip_name == "clip.wav"
    assert outcome.chunk_ms == 160
    assert outcome.audio_duration_s == 1.6
    assert outcome.response_latencies_s == [0.002, 0.004, 0.006, 0.008, 0.010]


def test_the_clock_travels_with_every_latency(tmp_path: Path) -> None:
    """Two receipts measured differently must not be poolable by accident."""
    clip = _write_wav(tmp_path / "clip.wav", 0.32)

    outcome = asyncio.run(ReplayAdapter("x").run(_request(clip)))

    assert outcome.latency_clock == "replay/none"
    assert outcome.response_latency_clock == "replay/none"


def test_chunking_is_shared_not_reimplemented_per_adapter(
    tmp_path: Path,
) -> None:
    """Two adapters must offer identical work for identical requests."""
    clip = _write_wav(tmp_path / "clip.wav", 1.0)
    pcm = read_pcm16(clip)

    chunks = chunk_pcm16(pcm, 160)

    assert len(chunks) == 7
    assert all(len(chunk) == 160 * 16 * 2 for chunk in chunks[:-1])
    # The tail is short rather than padded: padding changes scored audio.
    assert len(chunks[-1]) == 1000 * 16 * 2 - 6 * 160 * 16 * 2
    assert b"".join(chunks) == pcm


def test_the_adapter_awaits_the_admission_barrier(tmp_path: Path) -> None:
    """Concurrent cells depend on every lane being released together."""
    clip = _write_wav(tmp_path / "clip.wav", 0.32)
    released: list[str] = []

    class RecordingBarrier:
        async def wait(self, label: str) -> dict[str, str]:
            released.append(label)
            return {"go_observed_at": "2026-08-22T00:00:00Z"}

    outcome = asyncio.run(
        ReplayAdapter("x").run(
            _request(clip, label="lane7"), admission_barrier=RecordingBarrier()
        )
    )

    assert released == ["lane7"]
    assert outcome.protocol_extras["markers"] == {
        "go_observed_at": "2026-08-22T00:00:00Z"
    }


def test_the_request_is_frozen(tmp_path: Path) -> None:
    """Offered work cannot drift mid-run, or the receipt would be a lie."""
    request = _request(_write_wav(tmp_path / "clip.wav", 0.16))

    for field_name in ("chunk_ms", "cadence_ms", "lifecycle", "pace"):
        with pytest.raises(FrozenInstanceError):
            setattr(request, field_name, 999)
