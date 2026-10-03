"""Understandable numerical features extracted from request windows."""

from __future__ import annotations

import numpy as np


FEATURE_NAMES = [
    "mean_position",
    "std_position",
    "request_range",
    "min_position",
    "max_position",
    "mean_abs_step",
    "median_abs_step",
    "increasing_ratio",
    "decreasing_ratio",
    "direction_change_ratio",
    "head_to_mean_distance",
]


def extract_features(requests: list[int], initial_head: int) -> dict[str, float]:
    """Return workload-only features, excluding scheduler performance values."""

    if not requests:
        return {name: 0.0 for name in FEATURE_NAMES}

    values = np.asarray(requests, dtype=float)
    diffs = np.diff(values)
    abs_diffs = np.abs(diffs)

    if len(diffs) == 0:
        increasing_ratio = 0.0
        decreasing_ratio = 0.0
        direction_change_ratio = 0.0
        mean_abs_step = 0.0
        median_abs_step = 0.0
    else:
        increasing_ratio = float(np.mean(diffs > 0))
        decreasing_ratio = float(np.mean(diffs < 0))
        signs = np.sign(diffs)
        non_zero_signs = signs[signs != 0]
        if len(non_zero_signs) <= 1:
            direction_change_ratio = 0.0
        else:
            direction_change_ratio = float(np.mean(non_zero_signs[1:] != non_zero_signs[:-1]))
        mean_abs_step = float(np.mean(abs_diffs))
        median_abs_step = float(np.median(abs_diffs))

    mean_position = float(np.mean(values))
    return {
        "mean_position": mean_position,
        "std_position": float(np.std(values, ddof=0)),
        "request_range": float(np.max(values) - np.min(values)),
        "min_position": float(np.min(values)),
        "max_position": float(np.max(values)),
        "mean_abs_step": mean_abs_step,
        "median_abs_step": median_abs_step,
        "increasing_ratio": increasing_ratio,
        "decreasing_ratio": decreasing_ratio,
        "direction_change_ratio": direction_change_ratio,
        "head_to_mean_distance": abs(float(initial_head) - mean_position),
    }
