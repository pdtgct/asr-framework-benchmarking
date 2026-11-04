"""A target-neutral streaming ASR benchmark.

The framework compares containerized ASR servers through their *public*
protocols. Nothing here knows which server is under test: a
:class:`~asrbench.adapter.TransportAdapter` turns a
:class:`~asrbench.outcome.SessionRequest` into a
:class:`~asrbench.outcome.SessionOutcome`, and everything downstream reads
only the outcome.

The package depends on the standard library plus its declared transport
dependencies, and on nothing else — no engine, no server implementation, no
surrounding repository. That boundary is enforced by a test, because it is
what makes results reproducible by someone who has only this package.
"""

from asrbench.adapter import (
    SessionAdmissionBarrier,
    SessionProgressCallback,
    TransportAdapter,
)
from asrbench.definitions import SCHEMA_VERSION
from asrbench.outcome import (
    FINALIZATION_CLOCK,
    NATIVE_LATENCY_CLOCK,
    RESPONSE_LATENCY_CLOCK,
    SessionOutcome,
    SessionRequest,
)

__all__ = [
    "FINALIZATION_CLOCK",
    "NATIVE_LATENCY_CLOCK",
    "RESPONSE_LATENCY_CLOCK",
    "SCHEMA_VERSION",
    "SessionAdmissionBarrier",
    "SessionOutcome",
    "SessionProgressCallback",
    "SessionRequest",
    "TransportAdapter",
    "__version__",
]

__version__ = "0.0.1"
