"""The one percentile helper, including the empty-sample rule."""

import math

import pytest

from asrbench.percentile import PercentileError, percentile


def test_empty_is_absent_not_zero() -> None:
    assert percentile([], 50) is None
    assert percentile([], 0) is None
    assert percentile([], 100) is None


def test_single_sample_is_every_percentile() -> None:
    assert percentile([5.0], 0) == 5.0
    assert percentile([5.0], 50) == 5.0
    assert percentile([5.0], 100) == 5.0


def test_two_samples_linear_interpolates_and_nearest_rank_does_not() -> None:
    assert percentile([10.0, 20.0], 50) == 15.0
    assert percentile([10.0, 20.0], 50, method="nearest_rank") == 10.0


def test_ties_stay_ties() -> None:
    assert percentile([5.0, 5.0, 5.0], 90) == 5.0


def test_endpoints_are_the_order_statistics() -> None:
    samples = [1.0, 2.0, 3.0, 4.0]
    assert percentile(samples, 0) == 1.0
    assert percentile(samples, 100) == 4.0


def test_numpy_linear_matches_hyndman_fan_type_7() -> None:
    samples = [1.0, 2.0, 3.0, 4.0]
    assert math.isclose(percentile(samples, 90) or 0.0, 3.7)
    assert math.isclose(percentile(samples, 99) or 0.0, 3.97)


def test_input_is_not_mutated() -> None:
    samples = [3.0, 1.0, 2.0]
    assert percentile(samples, 50) == 2.0
    assert samples == [3.0, 1.0, 2.0]


def test_bad_queries_are_errors() -> None:
    with pytest.raises(PercentileError):
        percentile([1.0], -0.1)
    with pytest.raises(PercentileError):
        percentile([1.0], 100.1)
    with pytest.raises(PercentileError):
        percentile([1.0], 50, method="excel")
    with pytest.raises(PercentileError):
        percentile([1.0], float("nan"))
    with pytest.raises(PercentileError):
        percentile([float("nan")], 50)
    with pytest.raises(PercentileError):
        percentile([float("inf")], 50)


def test_numpy_linear_matches_statistics_quantiles() -> None:
    import random
    import statistics

    samples = [random.Random(0).random() for _ in range(40)]
    cuts = statistics.quantiles(samples, n=100, method="inclusive")
    for q in range(1, 100):
        assert percentile(samples, q) == pytest.approx(cuts[q - 1])
