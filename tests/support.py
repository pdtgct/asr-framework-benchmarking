"""Shared constructors for the default-tier tests."""

from __future__ import annotations

from asrbench.outcome import (
    FINALIZATION_CLOCK,
    NATIVE_LATENCY_CLOCK,
    RESPONSE_LATENCY_CLOCK,
    SessionOutcome,
)


class _Unset:
    """Marks an argument the caller did not pass."""


_UNSET = _Unset()


def session(
    *,
    label: str = "s",
    transcript: str | None = "hello",
    ttfs_s: float | None = 0.2,
    final_latency_s: float | None = 0.05,
    response_latencies_s: list[float] | None = None,
    latency_clock: str = NATIVE_LATENCY_CLOCK,
    response_latency_clock: str = RESPONSE_LATENCY_CLOCK,
    final_latency_clock: str = FINALIZATION_CLOCK,
    configured: bool = True,
    interim_texts: list[str] | None | _Unset = _UNSET,
    timed_out: bool = False,
    error_code: str | None = None,
    terminal_status: str = "",
    protocol_errors: list[dict[str, object]] | None = None,
    protocol_extras: dict[str, object] | None = None,
    audio_duration_s: float = 1.6,
) -> SessionOutcome:
    """Build one outcome. Defaults are a successful one-word session."""
    texts: list[str] | None
    if isinstance(interim_texts, _Unset):
        texts = ["hello"] if transcript == "hello" else None
    else:
        texts = interim_texts
    return SessionOutcome(
        label=label,
        mode="full",
        clip_name="clip.wav",
        chunk_ms=160,
        audio_duration_s=audio_duration_s,
        configured=configured,
        connected=True,
        transcript=transcript,
        ttfs_s=ttfs_s,
        final_latency_s=final_latency_s,
        response_latencies_s=(
            [0.2]
            if response_latencies_s is None
            else list(response_latencies_s)
        ),
        latency_clock=latency_clock,
        response_latency_clock=response_latency_clock,
        final_latency_clock=final_latency_clock,
        interim_texts=texts,
        timed_out=timed_out,
        error_code=error_code,
        terminal_status=terminal_status,
        protocol_errors=list(protocol_errors or []),
        protocol_extras=dict(protocol_extras or {}),
    )
