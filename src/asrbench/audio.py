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


#: Rates a clip may be sent at: wideband, and narrowband telephony.
SUPPORTED_SAMPLE_RATES = (8_000, 16_000)


def read_pcm16(path: Path, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Read one mono PCM16 WAV at ``sample_rate`` and return its frames."""
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != BYTES_PER_SAMPLE:
            raise SystemExit(f"{path}: need PCM16")
        if handle.getnchannels() != 1:
            raise SystemExit(f"{path}: need mono")
        if handle.getframerate() != sample_rate:
            raise SystemExit(
                f"{path}: need {sample_rate} Hz, got {handle.getframerate()}"
            )
        return handle.readframes(handle.getnframes())


def chunk_pcm16(
    pcm: bytes, chunk_ms: int, sample_rate: int = SAMPLE_RATE
) -> list[bytes]:
    """Split PCM into transmission pieces of ``chunk_ms`` each.

    The final piece is short whenever the clip does not divide evenly, which
    is the common case: it is transmitted as-is rather than padded, because
    padding would change the audio the server scores.
    """
    frame_bytes = int(sample_rate * chunk_ms / 1000) * BYTES_PER_SAMPLE
    return [pcm[i : i + frame_bytes] for i in range(0, len(pcm), frame_bytes)]
