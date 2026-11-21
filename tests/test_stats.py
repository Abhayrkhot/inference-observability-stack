import math
import random

import pytest

from inferobs.analysis.stats import bucket_counts, histogram_quantile, quantile, summarize


def test_quantile_matches_linear_interpolation():
    data = [10, 20, 30, 40, 50]
    assert quantile(data, 0.5) == 30
    assert quantile(data, 0.9) == pytest.approx(46.0)
    assert quantile(data, 1.0) == 50
    assert math.isnan(quantile([], 0.5))


def test_bucket_counts_le_is_inclusive():
    assert bucket_counts([0.1, 0.2, 0.25, 9.0], [0.1, 0.25, 1.0]) == [1, 3, 3, 4]


def test_histogram_quantile_interpolates_like_promql():
    bounds = [0.1, 0.2, 0.4]
    cum = [10, 30, 40, 40]  # 10 in (0,0.1], 20 in (0.1,0.2], 10 in (0.2,0.4]
    # rank 20 of 40 falls halfway through the second bucket
    assert histogram_quantile(0.5, bounds, cum) == pytest.approx(0.15)
    # first bucket interpolates from 0
    assert histogram_quantile(0.125, bounds, cum) == pytest.approx(0.05)
    # rank in +Inf bucket returns the highest finite bound
    assert histogram_quantile(0.99, [0.1, 0.2], [10, 20, 40]) == 0.2
    assert math.isnan(histogram_quantile(0.5, bounds, [0, 0, 0, 0]))


def test_histogram_quantile_error_shrinks_with_finer_buckets():
    rng = random.Random(1)
    vals = [rng.lognormvariate(-4.5, 0.4) for _ in range(20000)]  # ~11 ms median
    exact = quantile(vals, 0.99)
    coarse = [0.01, 0.025, 0.05, 0.1]
    fine = [0.002 * 1.2**i for i in range(40)]
    e_coarse = abs(histogram_quantile(0.99, coarse, bucket_counts(vals, coarse)) - exact) / exact
    e_fine = abs(histogram_quantile(0.99, fine, bucket_counts(vals, fine)) - exact) / exact
    assert e_fine < 0.05 < e_coarse


def test_summarize_keys():
    s = summarize([1.0, 2.0, 3.0])
    assert set(s) == {"p50", "p90", "p95", "p99", "mean", "n"}
    assert s["n"] == 3 and s["mean"] == 2.0
