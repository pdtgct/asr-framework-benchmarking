"""Deterministic clip bytes for the mock path.

The digest is of the PCM16 payload, not of a WAV container, so a
container-header change cannot move a workload fingerprint. No audio is
committed; silence is regenerated from the recipe.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path

from asrbench.audio import BYTES_PER_SAMPLE, SAMPLE_RATE, read_pcm16
from asrbench.canonical import sha256_bytes
from asrbench.spec import ClipSpec


class ClipError(ValueError):
    """The clip recipe does not produce the audio it names."""


def silence_pcm16(duration_s: float, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Return mono PCM16 silence of ``duration_s``.

    Frame count is ``round(duration_s * sample_rate)`` so a duration that
    is not an exact binary fraction still lands on a frame.

    Args:
        duration_s: Requested duration, in seconds.
        sample_rate: Hertz. Only :data:`asrbench.audio.SAMPLE_RATE` is
            accepted.

    Returns:
        Interleaved little-endian PCM16 bytes. Mono, so that is just the
        samples.

    Raises:
        ClipError: The rate is not 16 kHz or the duration is not positive.
    """
    if sample_rate != SAMPLE_RATE:
        raise ClipError(f"clips are {SAMPLE_RATE} Hz, got {sample_rate}")
    if duration_s <= 0:
        raise ClipError("clip duration must be positive")
    frames = int(round(duration_s * sample_rate))
    if frames < 1:
        raise ClipError("clip duration rounds to no frames")
    return b"\x00\x00" * frames


def pcm16_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap PCM16 mono bytes in a WAV container the adapters can read.

    Args:
        pcm: Payload from :func:`silence_pcm16` or an equivalent reader.
        sample_rate: Hertz.

    Returns:
        A WAV file as bytes.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(BYTES_PER_SAMPLE)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def clip_pcm(clip: ClipSpec) -> bytes:
    """Return the PCM16 payload a clip recipe names.

    Args:
        clip: The clip section of a workload.

    Returns:
        PCM16 bytes.

    Raises:
        ClipError: The generator is unknown or its inputs are missing.
    """
    if clip.generator == "pcm16-silence":
        if clip.duration_s is None:
            raise ClipError("pcm16-silence requires duration_s")
        return silence_pcm16(clip.duration_s, clip.sample_rate)
    if clip.generator == "file":
        if not clip.path:
            raise ClipError("file clips require path")
        try:
            return read_pcm16(Path(clip.path))
        except OSError as error:
            raise ClipError(f"{clip.path}: {error}") from error
    raise ClipError(f"unknown clip generator {clip.generator!r}")


def pin_digest(clip: ClipSpec) -> ClipSpec:
    """Fill ``digest`` from the payload, refusing a wrong pin.

    Args:
        clip: A clip that may or may not already name a digest.

    Returns:
        The same clip with ``digest`` set to the payload hash.

    Raises:
        ClipError: A declared digest does not match the payload.
    """
    digest = sha256_bytes(clip_pcm(clip))
    if clip.digest is not None and clip.digest != digest:
        raise ClipError(
            f"clip digest mismatch for {clip.id}: payload {digest} "
            f"!= pinned {clip.digest}"
        )
    if clip.digest == digest:
        return clip
    return ClipSpec(
        id=clip.id,
        generator=clip.generator,
        duration_s=clip.duration_s,
        sample_rate=clip.sample_rate,
        digest=digest,
        path=clip.path,
        licence=clip.licence,
        attribution=clip.attribution,
        note=clip.note,
    )


def materialize(clip: ClipSpec, dest_dir: Path) -> Path:
    """Write the clip's WAV into ``dest_dir`` and return the path.

    The workload fingerprint does not include this path.

    Args:
        clip: A clip whose digest already matches the payload.
        dest_dir: Directory created if needed.

    Returns:
        The WAV path.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{clip.id}.wav"
    path.write_bytes(pcm16_wav(clip_pcm(clip), clip.sample_rate))
    return path
