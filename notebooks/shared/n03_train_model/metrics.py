"""
Module for metrics and benchmarks related to location distributions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd  # type: ignore

logger = logging.getLogger(__name__)


def worst_k_decile(y_true: pd.Series, y_pred: pd.Series, k: float = 0.1) -> dict:
    """
    Compute metrics for the worst-k fraction of samples by absolute error.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.
        k: Fraction of samples to consider (default = 0.1).

    Returns:
        Dictionary with mean and max absolute error for the worst-k samples.
    """
    err = (y_true - y_pred).abs()
    n = max(1, int(len(err) * k))
    top = err.nlargest(n)
    return {
        "worst_k": float(k),
        "worst_k_mean_abs_err": float(top.mean()),
        "worst_k_max_abs_err": float(top.max()),
        "worst_k_count": int(n),
    }


def compute_location_drift(
    df: pd.DataFrame,
    target_weights: Dict[str, float],
    tolerance: float,
) -> Dict[str, Any]:
    """
    Compare empirical distribution of 'location' with target weights.

    Args:
        df: DataFrame containing 'location'.
        target_weights: Mapping location → target weight (fractions summing to ~1).
        tolerance: Absolute difference threshold to flag drift.

    Returns:
        Dictionary {location: {target_weight, empirical_weight, difference, drifted, ratio}}.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"`df` must be a pandas DataFrame, got {type(df).__name__}.")

    if "location" not in df.columns:
        raise ValueError("Column 'location' is required for drift computation.")

    empirical = df["location"].value_counts(normalize=True).to_dict()
    drift_report: Dict[str, Any] = {}

    for loc, target in target_weights.items():
        emp = float(empirical.get(loc, 0.0))
        diff = emp - target
        drifted = abs(diff) > tolerance
        ratio = emp / target if target > 0 else float("inf")

        drift_report[loc] = {
            "target_weight": float(target),
            "empirical_weight": emp,
            "difference": diff,
            "drifted": drifted,
            "ratio": ratio,
        }

    drifted_locs = [loc for loc, vals in drift_report.items() if vals["drifted"]]
    if drifted_locs:
        logger.warning("Detected location drift for: %s", drifted_locs)
    else:
        logger.info("No significant location drift detected.")

    return drift_report


def location_benchmark(
    df: pd.DataFrame,
    target_weights: Dict[str, float],
    tolerance: float = 0.05,
) -> pd.DataFrame:
    """
    Generate a benchmark DataFrame comparing empirical vs. target location distributions.

    Args:
        df: DataFrame containing 'location'.
        target_weights: Mapping location → target weight.
        tolerance: Absolute difference threshold to flag drift.

    Returns:
        DataFrame indexed by location with columns:
        [target_weight, empirical_weight, difference, drifted, ratio].
    """
    drift_report = compute_location_drift(df, target_weights, tolerance)

    return pd.DataFrame.from_records(
        [{"location": k, **v} for k, v in drift_report.items()],
        columns=["location", "target_weight", "empirical_weight", "difference", "drifted", "ratio"],
    ).set_index("location")
