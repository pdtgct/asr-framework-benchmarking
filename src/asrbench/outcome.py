"""The request and outcome shapes every transport adapter speaks.

These mirror vLLM's ``RequestFuncInput`` / ``RequestFuncOutput`` pair
(``vllm/benchmarks/lib/endpoint_request_func.py``): one frozen description
of the work offered, one mutable record of what was observed. Adapters own
the protocol; everything downstream of an adapter sees only these two.

Every duration here is **seconds**, and every latency is client-observed on
a monotonic clock. The clock a value was taken on travels with the outcome
in :attr:`SessionOutcome.latency_clock` and
:attr:`SessionOutcome.response_latency_clock` rather than being implied by
the field name, so two receipts measured differently cannot be pooled by
accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Interval from the first audio sample transmitted to the first non-empty
#: interim response. The name is emitted into every artifact.
NATIVE_LATENCY_CLOCK = (
    "client_monotonic/first_audio_append_to_nonempty_transcription_delta"
)
#: Interval from the most recent audio transmitted to each non-empty interim
#: response. Distinct from the above, and not interchangeable with a
#: server-reported response latency.
RESPONSE_LATENCY_CLOCK = (
    "client_monotonic/latest_audio_append_to_nonempty_transcription_delta"
)


@dataclass(frozen=True)
class SessionRequest:
    """One streaming session's offered work, complete and ambient-free.

    Every field an adapter needs is present here; an adapter that reaches
    for process state, environment, or a parsed CLI namespace has escaped
    the contract and makes the run unreproducible from its receipt.
    """

    #: Fully-qualified transport endpoint, e.g. ``ws://host:port/v1/realtime``.
    endpoint_url: str
    #: Model name offered to the server at session configuration.
    model: str
    #: Source clip; mono PCM16 at :data:`asrbench.audio.SAMPLE_RATE`.
    clip_path: Path
    #: Identifies this session among its concurrent peers.
    label: str
    #: Audio carried per transmitted piece.
    chunk_ms: int
    #: Requested server emission cadence; ``None`` leaves the server default.
    cadence_ms: int | None
    #: Bound on both session configuration and transcript completion.
    timeout_s: float
    #: Lifecycle probe selector, not a workload knob. See the module note in
    #: ``asrbench.adapters.realtime_ws``.
    mode: str
    #: Transmit in real time (one ``chunk_ms`` of audio per ``chunk_ms``)
    #: rather than as fast as the transport accepts.
    pace: bool


@dataclass
class SessionOutcome:
    """What one streaming session was observed to do.

    Mutable because an adapter fills it in as events arrive. Anything the
    adapter observed that this shape does not name goes in
    :attr:`protocol_extras` rather than being dropped — a protocol residue
    that is not represented here cannot be recovered later.
    """

    #: Echoed from the request so an outcome is self-identifying.
    label: str
    mode: str
    clip_name: str
    chunk_ms: int
    #: Duration of the source clip, not of the session.
    audio_duration_s: float

    #: The server acknowledged that the session exists.
    connected: bool = False
    #: The server accepted the session configuration. For protocols that
    #: gate admission on configuration, this is the admission boundary.
    configured: bool = False
    #: Every interim response observed, including empty ones. Empty interims
    #: are excluded from :attr:`response_latencies_s` but counted here.
    interim_count: int = 0
    #: The final transcript; ``None`` when none was ever received, which is
    #: distinct from a received-but-empty transcript.
    transcript: str | None = None
    #: Protocol-level errors reported by the server, verbatim.
    protocol_errors: list[dict[str, Any]] = field(default_factory=list)

    #: Time to first speech, on :data:`NATIVE_LATENCY_CLOCK`.
    ttfs_s: float | None = None
    #: Final commit to completed transcript.
    final_latency_s: float | None = None
    #: Per-interim latency, on :data:`RESPONSE_LATENCY_CLOCK`.
    response_latencies_s: list[float] = field(default_factory=list)
    latency_clock: str = ""
    response_latency_clock: str = ""

    #: Protocol-specific residue, in observation order.
    protocol_extras: dict[str, Any] = field(default_factory=dict)
