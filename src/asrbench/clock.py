"""Wall-clock markers.

Latencies are measured on a monotonic clock; these timestamps exist only to
correlate a client-side observation with a server-side log or metric scrape.
They are markers, never durations — subtracting two of them across machines
compares two different clocks.
"""

from __future__ import annotations

import datetime


def utc_timestamp() -> str:
    """Return an RFC 3339 UTC marker timestamp."""
    # `datetime.UTC` is 3.11+; this package supports 3.10.
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
