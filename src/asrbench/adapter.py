"""The interface a transport adapter implements.

One protocol, one adapter. An adapter's whole job is to turn a
:class:`~asrbench.outcome.SessionRequest` into a
:class:`~asrbench.outcome.SessionOutcome` over some wire format; it decides
nothing about whether the result is good, which cells to run, or how to
aggregate. That separation is what lets one runner compare servers that
share no protocol.

Adapters report failure through the outcome, not by raising: a session that
errored, timed out, or was refused is data, and a runner that had to catch
exceptions to collect it would lose the observations made before the throw.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from asrbench.outcome import SessionOutcome, SessionRequest

#: Called with ``(label, event, fields)`` at each protocol transition.
#: Diagnostics only — a benchmark result never depends on it.
SessionProgressCallback = Callable[[str, str, dict[str, object]], None]


class SessionAdmissionBarrier(Protocol):
    """A barrier between admission acknowledgement and first audio.

    Concurrent cells need every lane admitted before any lane sends, or the
    early lanes measure a load the later ones never joined.
    """

    async def wait(self, label: str) -> dict[str, str]:
        """Wait until the configured session cohort may send audio.

        Returns:
            Timestamp markers describing when the cohort was released.
        """


class TransportAdapter(Protocol):
    """Drives one streaming session over one protocol."""

    #: Stable identifier recorded in receipts, e.g. ``"realtime-ws"``.
    name: str

    async def run(
        self,
        request: SessionRequest,
        *,
        admission_barrier: SessionAdmissionBarrier | None = None,
        progress_callback: SessionProgressCallback | None = None,
    ) -> SessionOutcome:
        """Drive one session to completion and report what was observed."""
