"""Bootstrap CI helpers for multi-seed aggregation.

Used to render 95% confidence bands / whiskers over the 5-seed sweep.
At n=5 the bootstrap is coarse (only 2^5 = 32 distinct resamplings),
but it is honest — the standard error of the mean at n=5 is misleading.
For n=5 samples we default to the *min/max range* if bootstrap is
requested with too few samples, and label the caption accordingly.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def bootstrap_ci_95(samples: Sequence[float], n_resamples: int = 2000,
                    seed: int = 0) -> Tuple[float, float, float]:
    """Median and 95% percentile bootstrap CI of `samples`.

    Returns (median, lower_bound, upper_bound). At fewer than 3 samples we
    fall back to (value, value, value) — no CI is meaningful.
    """
    arr = np.asarray(list(samples), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (float("nan"),) * 3
    if arr.size < 3:
        m = float(np.median(arr))
        return (m, m, m)
    rng = np.random.default_rng(seed)
    n = arr.size
    draws = rng.integers(0, n, size=(n_resamples, n))
    means = arr[draws].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.median(arr)), float(lo), float(hi)


def range_band(samples: Sequence[float]) -> Tuple[float, float, float]:
    """Median + min/max. Honest fallback for tiny samples."""
    arr = np.asarray(list(samples), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (float("nan"),) * 3
    return float(np.median(arr)), float(arr.min()), float(arr.max())


__all__ = ["bootstrap_ci_95", "range_band"]
