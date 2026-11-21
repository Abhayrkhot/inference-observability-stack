"""Exact quantiles and Prometheus-style histogram quantile estimation."""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable, Sequence


def quantile(values: Sequence[float], q: float) -> float:
    """Exact quantile with linear interpolation between order statistics (numpy's default method)."""
    if not values:
        return math.nan
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(s[lo])
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def bucket_counts(values: Iterable[float], bounds: Sequence[float]) -> list[int]:
    """Cumulative counts per upper bound, plus +Inf, as Prometheus exposes them (le is inclusive)."""
    counts = [0] * (len(bounds) + 1)
    for v in values:
        counts[bisect.bisect_left(bounds, v)] += 1
    out, acc = [], 0
    for c in counts:
        acc += c
        out.append(acc)
    return out


def histogram_quantile(q: float, bounds: Sequence[float], cumulative: Sequence[float]) -> float:
    """Reimplementation of PromQL histogram_quantile() for classic histograms.

    ``cumulative`` has one entry per bound plus a final +Inf entry. Mirrors Prometheus: linear
    interpolation inside the bucket, lower edge 0 for the first bucket (when its bound is positive),
    and the highest finite bound returned when the rank falls in the +Inf bucket.
    """
    total = cumulative[-1]
    if total == 0:
        return math.nan
    rank = q * total
    i = bisect.bisect_left(cumulative, rank)
    if i >= len(bounds):  # +Inf bucket
        return float(bounds[-1])
    upper = bounds[i]
    lower = bounds[i - 1] if i > 0 else 0.0
    prev = cumulative[i - 1] if i > 0 else 0.0
    count = cumulative[i] - prev
    if i == 0 and upper <= 0:
        return float(upper)
    if count == 0:
        return float(upper)
    return lower + (upper - lower) * ((rank - prev) / count)


def summarize(values: Sequence[float], qs: Sequence[float] = (0.5, 0.9, 0.95, 0.99)) -> dict[str, float]:
    out = {f"p{round(q * 100):d}": quantile(values, q) for q in qs}
    out["mean"] = sum(values) / len(values) if values else math.nan
    out["n"] = len(values)
    return out
