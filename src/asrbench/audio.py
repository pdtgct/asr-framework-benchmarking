"""Source-clip reading, shared by every adapter.

Clip handling is transport-independent: a WebSocket adapter and a gRPC
adapter chunk the same PCM the same way, so it belongs here rather than in
either one.
"""

from __future__ import annotations

import wave
from pathlib import Path

SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2


def clip_seconds(path: Path) -> float:
    """Return one clip's duration in seconds."""
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def read_pcm16(path: Path) -> bytes:
    """Read one mono 16 kHz PCM16 WAV and return its frame bytes."""
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != BYTES_PER_SAMPLE:
            raise SystemExit(f"{path}: need PCM16")
        if handle.getnchannels() != 1:
            raise SystemExit(f"{path}: need mono")
        if handle.getframerate() != SAMPLE_RATE:
            raise SystemExit(
                f"{path}: need {SAMPLE_RATE} Hz, got {handle.getframerate()}"
            )
        return handle.readframes(handle.getnframes())


def chunk_pcm16(pcm: bytes, chunk_ms: int) -> list[bytes]:
    """Split PCM into transmission pieces of ``chunk_ms`` each.

    The final piece is short whenever the clip does not divide evenly, which
    is the common case: it is transmitted as-is rather than padded, because
    padding would change the audio the server scores.
    """
    frame_bytes = int(SAMPLE_RATE * chunk_ms / 1000) * BYTES_PER_SAMPLE
    return [pcm[i : i + frame_bytes] for i in range(0, len(pcm), frame_bytes)]
