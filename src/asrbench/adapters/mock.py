"""Deterministic adapter for the default test tier.

Intervals are simulated. They are labeled with ``mock/simulated_*`` clocks
so a receipt cannot present them as client-monotonic observations. The
adapter does not sleep: a real-time sleep would make the mock path too
slow to be the default tier, and a near-zero wall span is not pacing.
"""

from __future__ import annotations

from asrbench.adapter import SessionAdmissionBarrier, SessionProgressCallback
from asrbench.audio import (
    BYTES_PER_SAMPLE,
    SAMPLE_RATE,
    chunk_pcm16,
    read_pcm16,
)
from asrbench.outcome import SessionOutcome, SessionRequest

#: Simulated stand-in for :data:`asrbench.outcome.NATIVE_LATENCY_CLOCK`.
MOCK_TTFS_CLOCK = "mock/simulated_first_audio_to_first_nonempty_partial"
#: Simulated stand-in for :data:`asrbench.outcome.RESPONSE_LATENCY_CLOCK`.
MOCK_RESPONSE_CLOCK = "mock/simulated_latest_audio_to_nonempty_partial"
#: Simulated stand-in for :data:`asrbench.outcome.FINALIZATION_CLOCK`.
MOCK_FINAL_CLOCK = "mock/simulated_last_audio_to_final"

# 1/8 second. Exact in binary floating point, so receipts do not wobble.
_STEP_S = 0.125
_FINAL_S = 0.0625


class MockAdapter:
    """Replay a scripted transcript over the clip's real chunking."""

    name = "mock"
    supported_workload_options = frozenset(
        {
            "language",
            "punctuation",
            "diarization",
            "interim_results",
            "transcript_policy",
            "mode",
        }
    )

    def __init__(self, transcript: str, interim_every_chunks: int = 2) -> None:
        """Store the script.

        Args:
            transcript: Final text every successful session returns.
            interim_every_chunks: Emit one more word of that text every
                this many chunks. Must be at least 1.

        Raises:
            ValueError: ``interim_every_chunks`` is less than 1.
        """
        if interim_every_chunks < 1:
            raise ValueError("interim_every_chunks must be >= 1")
        self._transcript = transcript
        self._every = interim_every_chunks

    async def run(
        self,
        request: SessionRequest,
        *,
        admission_barrier: SessionAdmissionBarrier | None = None,
        progress_callback: SessionProgressCallback | None = None,
    ) -> SessionOutcome:
        """Drive one session against no network.

        Args:
            request: Offered work for this session.
            admission_barrier: Released before any interim is produced.
            progress_callback: Ignored. The mock has no lifecycle to trace.

        Returns:
            A successful outcome whose latencies are simulated.
        """
        del progress_callback
        pcm = read_pcm16(request.clip_path)
        chunks = chunk_pcm16(pcm, request.chunk_ms)
        outcome = SessionOutcome(
            label=request.label,
            lifecycle=request.lifecycle,
            clip_name=request.clip_path.name,
            chunk_ms=request.chunk_ms,
            audio_duration_s=len(pcm) / BYTES_PER_SAMPLE / SAMPLE_RATE,
            latency_clock=MOCK_TTFS_CLOCK,
            response_latency_clock=MOCK_RESPONSE_CLOCK,
            final_latency_clock=MOCK_FINAL_CLOCK,
            connected=True,
            configured=True,
            transcript=self._transcript,
            final_latency_s=_FINAL_S,
            terminal_status="completed",
        )
        outcome.protocol_extras["adapter"] = self.name
        if admission_barrier is not None:
            markers = await admission_barrier.wait(request.label)
            outcome.protocol_extras.setdefault("markers", {}).update(markers)
        words = self._transcript.split()
        emitted = 0
        texts: list[str] = []
        for index, _chunk in enumerate(chunks, start=1):
            if index % self._every != 0:
                continue
            emitted = min(len(words), emitted + 1)
            if emitted == 0:
                continue
            text = " ".join(words[:emitted])
            texts.append(text)
            latency = _STEP_S * index
            outcome.response_latencies_s.append(latency)
            outcome.interim_count += 1
            if outcome.ttfs_s is None:
                outcome.ttfs_s = latency
        outcome.interim_texts = texts
        return outcome
