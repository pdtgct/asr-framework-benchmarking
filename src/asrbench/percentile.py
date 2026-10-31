"""The one percentile helper in the package.

Three named methods live here. Reducers call :func:`percentile` and record
which method produced the number. A second implementation elsewhere would
put two answers next to one definition.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from asrbench.definitions import (
    PERCENTILE_METHOD_FLOOR_INDEX,
    PERCENTILE_METHOD_NEAREST_RANK,
    PERCENTILE_METHOD_NUMPY_LINEAR,
)

PERCENTILE_METHODS = (
    PERCENTILE_METHOD_NUMPY_LINEAR,
    PERCENTILE_METHOD_NEAREST_RANK,
    PERCENTILE_METHOD_FLOOR_INDEX,
)


class PercentileError(ValueError):
    """The percentile request itself is not a real query."""


def percentile(
    samples: Sequence[float],
    q: float,
    method: str = PERCENTILE_METHOD_NUMPY_LINEAR,
) -> float | None:
    """Return the ``q`` percentile of ``samples`` under ``method``.

    ``q`` is on the closed interval from 0 to 100. An empty sample returns
    ``None``. It does not return zero: zero is a measurement, and an empty
    sample is the absence of one.

    Args:
        samples: Observed values. They are copied and sorted; the argument
            is not modified.
        q: Percentile in ``[0, 100]``.
        method: ``numpy_linear``, ``nearest_rank``, or ``floor_index``.

    Returns:
        The percentile, or ``None`` when ``samples`` is empty.

    Raises:
        PercentileError: ``q`` is outside ``[0, 100]``, a sample is not
            finite, or ``method`` is not a named policy.
    """
    if method not in PERCENTILE_METHODS:
        raise PercentileError(f"unknown percentile method {method!r}")
    if (
        isinstance(q, bool)
        or not isinstance(q, (int, float))
        or not math.isfinite(q)
        or q < 0.0
        or q > 100.0
    ):
        raise PercentileError("percentile q must be in [0, 100]")
    if not samples:
        return None
    ordered: list[float] = []
    for sample in samples:
        if (
            isinstance(sample, bool)
            or not isinstance(sample, (int, float))
            or not math.isfinite(sample)
        ):
            raise PercentileError("percentile samples must be finite")
        ordered.append(float(sample))
    ordered.sort()
    if method == PERCENTILE_METHOD_NUMPY_LINEAR:
        return _numpy_linear(ordered, q)
    if method == PERCENTILE_METHOD_FLOOR_INDEX:
        return _floor_index(ordered, q)
    return _nearest_rank(ordered, q)


def _numpy_linear(ordered: list[float], q: float) -> float:
    """Hyndman-Fan type 7, NumPy's default ``method='linear'``.

    Args:
        ordered: Sorted samples, length at least 1.
        q: Percentile in ``[0, 100]``.

    Returns:
        The interpolated order statistic.
    """
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (q / 100.0)
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _nearest_rank(ordered: list[float], q: float) -> float:
    """Nearest-rank percentile, 1-based and clamped.

    Args:
        ordered: Sorted samples, length at least 1.
        q: Percentile in ``[0, 100]``.

    Returns:
        The selected order statistic, with no interpolation.
    """
    rank = max(1, math.ceil(q / 100.0 * len(ordered)))
    rank = min(rank, len(ordered))
    return ordered[rank - 1]


def _floor_index(ordered: list[float], q: float) -> float:
    """Riva's ``sorted[floor(p * n / 100)]``, clamped into range.

    Args:
        ordered: Sorted samples, length at least 1.
        q: Percentile in ``[0, 100]``.

    Returns:
        The selected order statistic, with no interpolation.
    """
    index = math.floor(q / 100.0 * len(ordered))
    index = min(max(index, 0), len(ordered) - 1)
    return ordered[index]
