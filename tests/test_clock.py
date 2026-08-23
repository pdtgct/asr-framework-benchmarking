"""Marker timestamps must be produced on every supported interpreter.

`utc_timestamp` is called only for correlation, so nothing in the test suite
exercised it — which is how a 3.11-only spelling of UTC reached the package
while declaring 3.10 support. This calls it.
"""

import datetime

from asrbench.clock import utc_timestamp


def test_the_marker_is_rfc_3339_utc() -> None:
    stamp = utc_timestamp()

    assert stamp.endswith("Z")
    parsed = datetime.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == datetime.timedelta(0)


def test_the_marker_is_produced_on_this_interpreter() -> None:
    """The failure mode was an AttributeError at call time, not at import."""
    assert utc_timestamp() != ""
