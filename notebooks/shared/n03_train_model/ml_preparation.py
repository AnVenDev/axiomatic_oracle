from __future__ import annotations

"""
ML Preparation:
- Robust target–feature correlation analysis (excluding leaky features)
- Configurable categorical encoding strategies
- Input validation and empty-DataFrame handling
- Structured logging with details about analyzed features
"""

import logging
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.model_selection import train_test_split, GroupKFold  # type: ignore

from notebooks.shared.common.constants import (
    VALUATION_K,
    PRICE_PER_SQM,
    PRICE_PER_SQM_VS_REGION_AVG,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MLPreparationAnalyzer",
]

# Features that would leak target information and must be excluded from analysis
ML_LEAKY_FEATURES: Set[str] = {
    PRICE_PER_SQM,
    PRICE_PER_SQM_VS_REGION_AVG,
    "_viz_price_per_sqm",
}

# Thresholds that choose the encoding strategy per categorical column
ENCODING_STRATEGY_THRESHOLDS = {
    "one_hot_max": 10,        # one-hot if unique categories <= 10
    "target_encoding_min": 50 # target-encode if unique categories >= 50
}


def train_test_split_grouped(X, y, groups, test_size: float = 0.2, random_state: int = 42):
    """
    Group-aware split using GroupKFold. Returns indices for a single fold.
    Notes:
        - The first fold is returned (deterministic with sorted group labels).
        - Intended for quick/easy grouped splitting, not exhaustive CV.
    """
    gkf = GroupKFold(n_splits=int(1 / test_size))
    groups = pd.Series(groups).astype(str)
    # Selecting the first yielded split (deterministic order from GroupKFold)
    for train_idx, test_idx in gkf.split(X, y, groups):
        return train_idx, test_idx
    # Fallback (should not happen)
    return np.array([], dtype=int), np.array([], dtype=int)


def stable_feature_order(df: pd.DataFrame) -> list[str]:
    """
    Returns columns ordered with numeric features first (alphabetically),
    followed by the remaining features (alphabetically).
    """
    num = sorted(df.select_dtypes(include=[np.number]).columns)
    oth = sorted([c for c in df.columns if c not in num])
    return num + oth


def stratify_by_quantiles(y: pd.Series, q: int = 10) -> pd.Series:
    """Return quantile labels for stratification in a regression setting."""
    y_numeric = pd.to_numeric(y, errors="coerce")
    ranks = y_numeric.rank(method="first").to_numpy()
    bins = np.ceil(ranks / (len(y_numeric) / q)).astype(int)
    bins[~np.isfinite(y_numeric)] = -1  # NaN/inf go to a special stratum
    return pd.Series(bins, index=y.index)


def train_val_test_split_regression(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
):
    """
    Train/val/test split with stratification by target deciles (for regression).

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    strat = stratify_by_quantiles(y, q=10)
    X_tmp, X_test, y_tmp, y_test, s_tmp, _ = train_test_split(
        X, y, strat, test_size=test_size, random_state=random_state, stratify=strat
    )
    # Validation from the temporary split; replace special stratum -1 by the mode
    strat_tmp = s_tmp.replace(-1, s_tmp.mode().iat[0])
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_size, random_state=random_state, stratify=strat_tmp
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


class MLPreparationAnalyzer:
    """
    Dataset analysis & preparation utilities for ML.

    Attributes:
        target_column: Name of the target column.
        leaky_features: Set of features to exclude to prevent leakage.
    """

    def __init__(
        self,
        target_column: str = VALUATION_K,
        leaky_features: Optional[Set[str]] = None,
    ) -> None:
        self.target_column = target_column
        self.leaky_features = leaky_features or ML_LEAKY_FEATURES

    def analyze_target_correlations(
        self,
        df: pd.DataFrame,
        top_n: int = 10,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Compute correlations between numeric features and the target, excluding leaky ones.

        Args:
            df: Input DataFrame.
            top_n: Number of top-correlated features to return.

        Returns:
            correlations_df: DataFrame with feature and correlation value.
            analysis_meta: Dict with analysis details/status.
        """
        if df.empty:
            logger.warning("[MLPrep] Empty DataFrame")
            return pd.DataFrame(), {"status": "empty_dataframe"}

        if self.target_column not in df.columns:
            logger.warning("[MLPrep] Target '%s' not present", self.target_column)
            return pd.DataFrame(), {"status": "target_missing"}

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        clean_cols = [c for c in numeric_cols if c not in self.leaky_features]

        if self.target_column not in clean_cols:
            logger.warning("[MLPrep] Target '%s' is not numeric or was excluded", self.target_column)
            return pd.DataFrame(), {"status": "target_not_numeric"}

        correlations = df[clean_cols].corr()[self.target_column].drop(self.target_column)
        top_corr = correlations.abs().sort_values(ascending=False).head(top_n)

        result_df = pd.DataFrame(
            {
                "feature": top_corr.index,
                "correlation": correlations[top_corr.index],
            }
        )

        logger.info(
            "[MLPrep] Correlations for '%s' analyzed — Top %d features: %s",
            self.target_column,
            top_n,
            list(top_corr.index),
        )

        return result_df, {
            "status": "success",
            "top_n": top_n,
            "excluded_features": list(self.leaky_features),
            "analyzed_features": clean_cols,
        }

    def apply_encoding_strategies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply categorical encodings based on configurable thresholds.

        Strategy:
          - One-hot if unique categories <= one_hot_max.
          - Target-encode if unique categories >= target_encoding_min.
          - Leave unchanged otherwise.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with encodings applied.
        """
        if df.empty:
            logger.warning("[MLPrep] Empty DataFrame in apply_encoding_strategies")
            return df

        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            unique_count = df[col].nunique()
            if unique_count <= ENCODING_STRATEGY_THRESHOLDS["one_hot_max"]:
                df = pd.get_dummies(df, columns=[col], drop_first=True)
                logger.debug("[MLPrep] One-hot encoding applied to %s", col)
            elif unique_count >= ENCODING_STRATEGY_THRESHOLDS["target_encoding_min"]:
                target_means = df.groupby(col)[self.target_column].mean()
                df[col] = df[col].map(target_means)
                logger.debug("[MLPrep] Target encoding applied to %s", col)
            else:
                logger.debug(
                    "[MLPrep] Column %s left unchanged (unique_count=%d)",
                    col,
                    unique_count,
                )

        return df
