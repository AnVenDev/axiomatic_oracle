"""
Modulo per metriche e benchmark relativi alle distribuzioni di location.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd     # type: ignore

logger = logging.getLogger(__name__)


def compute_location_drift(
    df: pd.DataFrame,
    target_weights: Dict[str, float],
    tolerance: float
) -> Dict[str, Any]:
    """
    Confronta la distribuzione empirica di 'location' con pesi target attesi.

    Args:
        df: DataFrame contenente 'location'.
        target_weights: Mappatura location → peso atteso (fractions ~1).
        tolerance: Soglia di differenza assoluta per segnalare drift.

    Returns:
        Dict {location: {target_weight, empirical_weight, difference, drifted, ratio}}.
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
    tolerance: float = 0.05
) -> pd.DataFrame:
    """
    Genera un DataFrame di benchmark tra distribuzione empirica e target di location.

    Args:
        df: DataFrame contenente 'location'.
        target_weights: Mappatura location → peso atteso.
        tolerance: Soglia di differenza assoluta per segnalare drift.

    Returns:
        DataFrame indicizzato per location con colonne:
        [target_weight, empirical_weight, difference, drifted, ratio].
    """
    drift_report = compute_location_drift(df, target_weights, tolerance)

    return pd.DataFrame.from_records(
        [{"location": k, **v} for k, v in drift_report.items()],
        columns=["location", "target_weight", "empirical_weight", "difference", "drifted", "ratio"]
    ).set_index("location")