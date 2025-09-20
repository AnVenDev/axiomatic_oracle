# notebooks/shared/common/eda_core.py
from __future__ import annotations
from os import PathLike

"""
EDA Core (I/O-free)
-------------------
Pure logic for exploratory data analysis:
- descriptive analytics and plotting (no disk writes)
- outlier detection (IQR / Z-score) and aggregation helpers
- anomaly detection (Isolation Forest + robust z-scores) with reporting
- temporal analysis (freshness) and statistical diagnostics
- feature importance and ablation study (no saving)

All file I/O (CSV/Parquet/PNG/JSON) should be handled in `eda_reports.py`.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Mapping

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from datetime import datetime, timezone

# ML / Stats
from sklearn.preprocessing import StandardScaler, OrdinalEncoder  # type: ignore
from sklearn.ensemble import RandomForestRegressor, IsolationForest  # type: ignore
from sklearn.model_selection import cross_validate  # type: ignore
from sklearn.inspection import permutation_importance  # type: ignore
import statsmodels.api as sm  # type: ignore

# Optional SciPy (skew/kurtosis/normaltest/chi2)
try:
    from scipy import stats  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    stats = None  # type: ignore

# Plotting (seaborn optional)
import matplotlib.pyplot as plt  # type: ignore
try:  # pragma: no cover - optional dependency
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover
    sns = None  # type: ignore

from notebooks.shared.common.sanity_checks import validate_dataset
from notebooks.shared.common.constants import (
    ASSET_ID,
    LOCATION,
    REGION,
    URBAN_TYPE,
    VALUATION_K,
    SIZE_M2,
    LAG_HOURS,
    ENERGY_CLASS,
    LUXURY_SCORE,
    PRICE_PER_SQM,
    PRICE_PER_SQM_VS_REGION_AVG,
    DECADE_BUILT,
    BUILDING_AGE_YEARS,
    ASSET_TYPE,
    CONDITION_SCORE,
    RISK_SCORE,
    ENV_SCORE,
    LAST_VERIFIED_TS,
    PREDICTION_TS,
    ENERGY_CLASSES,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Classes
    "DescriptiveAnalyzer",
    "OutlierDetector",
    "AnomalyDetector",
    "TemporalAnalyzer",
    "StatisticalTester",
    "FeatureRecommender",          # kept in __all__ for backward-compat if referenced elsewhere
    "FeatureImportanceAnalyzer",
    # Utils
    "load_and_validate_dataset",
    "ensure_temporal_columns",
    "plot_correlation_heatmap",
]

# ------------------------------ Constants -----------------------------------

DEFAULT_CONTAMINATION: float = 0.05
DEFAULT_STRONG_Z_THRESHOLD: float = 2.5
DEFAULT_SEVERITY_PERCENTILE: float = 90.0
MAD_CONSTANT: float = 1.4826  # for robust z-score

# Features that cause target leakage
LEAKY_FEATURES: Set[str] = {
    PRICE_PER_SQM,
    "_viz_price_per_sqm",
    PRICE_PER_SQM_VS_REGION_AVG,
}

# Default anomaly candidate features
DEFAULT_CANDIDATE_FEATURES: List[str] = [
    SIZE_M2, LUXURY_SCORE, ENV_SCORE,
    "condition_minus_risk", CONDITION_SCORE, RISK_SCORE,
]

_DEFAULT_THRESHOLDS = [30, 60, 90]  # days

MIN_SAMPLES_NORMALITY: int = 8
NORMALITY_ALPHA: float = 0.05
CHI2_ALPHA: float = 0.05
HIGH_CORR_THRESHOLD: float = 0.85
HIGH_VIF_THRESHOLD: float = 10.0

DEFAULT_NORMALITY_FEATURES: List[str] = [
    VALUATION_K,
    SIZE_M2,
    "condition_minus_risk",
    LAG_HOURS,
]

DEFAULT_NUMERIC_FEATURES: List[str] = [
    PRICE_PER_SQM,
    PRICE_PER_SQM_VS_REGION_AVG,
    VALUATION_K,
    SIZE_M2,
    "condition_minus_risk",
    DECADE_BUILT,
    BUILDING_AGE_YEARS,
    "location_premium",
]

ID_LIKE_FEATURES: Set[str] = {"id", ASSET_ID, "index"}

CATEGORICAL_FEATURES: List[str] = [
    LOCATION, ENERGY_CLASS, URBAN_TYPE, REGION,
    "age_category", "luxury_category", "value_segment"
]

TARGET_DERIVED_FEATURES: Set[str] = {
    "value_segment", "value_segment_encoded",
    "valuation_k_log", "price_per_sqm_capped"
}

PROXY_FEATURES: Set[str] = {
    "price_vs_region_avg", SIZE_M2, PRICE_PER_SQM,
    "rooms_per_sqm", "age_years", "bathrooms_per_room",
    LUXURY_SCORE, ENV_SCORE, "efficiency_score"
}

# Domain constraints (for IQR clamping). Can be extended by caller.
DEFAULT_DOMAIN_CONSTRAINTS = {
    "non_negative": set(),   # e.g., {"price_per_sqm", "valuation_k"}
    "unit_interval": set(),  # e.g., {"env_score"}
}

# --------------------- Dataset utils / validation ---------------------------

def load_and_validate_dataset(
    data_path: "PathLike[str] | str",
    asset_type: str = "property",
    validate: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load a dataset and optionally run schema validation.

    Returns:
        (df, validation_report)
    """
    from pathlib import Path

    path = Path(data_path)
    if not path.exists():
        logger.error("[EXPLORE] Dataset file not found", extra={"path": str(path)})
        return pd.DataFrame(), {}

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    report: Dict[str, Any] = {}
    if validate:
        report = validate_dataset(df, asset_type=asset_type, raise_on_failure=False)
        if not report.get("overall_passed", True):
            logger.warning(
                "[EXPLORE] Schema validation issues",
                extra={"missing": report.get("schema", {}).get("missing")}
            )
    return df, report


def ensure_temporal_columns(
    df: pd.DataFrame,
    reference_time: Optional[datetime] = None
) -> pd.DataFrame:
    """Ensure the DataFrame has consistent temporal columns."""
    reference_time = reference_time or datetime.now(timezone.utc)

    for col in (LAST_VERIFIED_TS, PREDICTION_TS):
        if col not in df:
            if col == PREDICTION_TS:
                df[col] = reference_time
                logger.info("[EXPLORE] %s added with reference_time", PREDICTION_TS)
            else:
                df[col] = pd.NaT
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    df[LAG_HOURS] = (df[PREDICTION_TS] - df[LAST_VERIFIED_TS]).dt.total_seconds().div(3600)
    return df


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: List[str],
    ax: Optional[plt.Axes] = None,
    cmap: str = "vlag",
    annot: bool = True,
    center: float = 0.0
) -> plt.Figure:
    """Create a correlation heatmap for the given columns (no saving)."""
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        created_fig = True
    else:
        fig = ax.figure

    corr = df[columns].corr().loc[columns, columns]
    if sns is not None:
        sns.heatmap(
            corr, annot=annot, fmt=".2f",
            cmap=cmap, center=center,
            square=True, cbar_kws={"shrink": 0.8},
            ax=ax
        )
    else:
        im = ax.imshow(corr.values, cmap=cmap)
        ax.set_xticks(range(len(columns)))
        ax.set_yticks(range(len(columns)))
        ax.set_xticklabels(columns, rotation=45, ha="right")
        ax.set_yticklabels(columns)
        fig.colorbar(im, ax=ax, shrink=0.8)

    ax.set_title("Correlation Matrix")
    if created_fig:
        fig.tight_layout()
    return fig

# --------------------- Descriptive / Visualizations -------------------------

class DescriptiveAnalyzer:
    """Descriptive analytics and distribution/relationship plots (no I/O)."""

    def box_plot_by_category(
        self,
        df: pd.DataFrame,
        category_col: str,
        value_col: str,
        ax: Optional[plt.Axes] = None,
        order: Optional[List[str]] = None,
        showfliers: bool = False,
        annotate_means: bool = True,
    ) -> plt.Figure:
        """Boxplot of value_col split by category_col (no save)."""
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
            created_fig = True
        else:
            fig = ax.figure

        if category_col not in df or value_col not in df:
            logger.warning(
                "[DESCR] Missing columns for boxplot",
                extra={"category_col": category_col, "value_col": value_col}
            )
            ax.text(0.5, 0.5, "Data missing", ha="center", va="center")
            return fig

        if order is None:
            order = sorted(df[category_col].dropna().unique().tolist())

        if sns is not None:
            sns.boxplot(
                data=df, x=category_col, y=value_col,
                order=order, showfliers=showfliers, ax=ax
            )
        else:
            groups = [df[df[category_col] == cat][value_col].dropna().values for cat in order]
            ax.boxplot(groups, labels=order, showfliers=showfliers)

        ax.set_xlabel(category_col.replace("_", " ").title())
        ax.set_ylabel(value_col.replace("_", " ").title())
        ax.set_title(f"{value_col.replace('_', ' ').title()} by {category_col.replace('_', ' ').title()}")

        if annotate_means:
            means = df.groupby(category_col, observed=True)[value_col].mean()
            for i, cat in enumerate(order):
                if cat in means:
                    ax.text(i, means[cat], f"{means[cat]:.2f}",
                            ha="center", va="bottom", fontsize=8, color="black")

        if created_fig:
            fig.tight_layout()
        return fig

    def plot_iqr_outliers(
        self,
        series: pd.Series,
        ax: Optional[plt.Axes] = None,
        whisker_coef: float = 1.5,
        title: Optional[str] = None
    ) -> plt.Figure:
        """Highlight IQR-based outliers in a boxplot (no save)."""
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 3))
            created_fig = True
        else:
            fig = ax.figure

        clean = series.dropna()
        if clean.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title or series.name)
            return fig

        q1, q3 = np.percentile(clean, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - whisker_coef * iqr, q3 + whisker_coef * iqr
        outliers = clean[(clean < lower) | (clean > upper)]

        if sns is not None:
            sns.boxplot(x=clean, ax=ax, orient="h", showfliers=False)
        else:
            ax.boxplot(clean, vert=False, showfliers=False)

        ax.scatter(outliers, np.zeros_like(outliers), alpha=0.6, label="outliers")
        ax.set_yticks([])
        ax.set_xlabel(series.name)
        ax.set_title(title or f"IQR Outliers: {series.name}")
        ax.legend()

        if created_fig:
            fig.tight_layout()
        return fig

    def create_relationship_plots(
        self,
        df: pd.DataFrame,
        figsize: Tuple[int, int] = (12, 10)
    ) -> plt.Figure:
        """Create 4 panels: Condition vs Risk, Size vs Valuation, Boxplot for Energy/Condition, and a score heatmap."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Scatter Condition vs Risk
        ax = axes[0, 0]
        if CONDITION_SCORE in df and RISK_SCORE in df:
            if sns is not None:
                sns.scatterplot(
                    data=df, x=CONDITION_SCORE, y=RISK_SCORE,
                    hue=ENERGY_CLASS if ENERGY_CLASS in df else None,
                    alpha=0.6, ax=ax
                )
            else:
                ax.scatter(df[CONDITION_SCORE], df[RISK_SCORE], alpha=0.6)
            ax.set_title("Condition vs Risk Score")
        else:
            logger.warning("[DESCR] Missing data for Condition vs Risk scatter")
            ax.text(0.5, 0.5, "Data missing", ha="center", va="center")

        # Size vs Valuation
        ax = axes[0, 1]
        if SIZE_M2 in df and VALUATION_K in df:
            if sns is not None:
                sns.regplot(x=SIZE_M2, y=VALUATION_K, data=df, scatter_kws={"alpha": 0.4}, ax=ax)
            else:
                ax.scatter(df[SIZE_M2], df[VALUATION_K], alpha=0.4)
            ax.set_title("Size vs Valuation")
        else:
            logger.warning("[DESCR] Missing data for Size vs Valuation")
            ax.text(0.5, 0.5, "Data missing", ha="center", va="center")

        # Boxplot Condition by Energy Class
        ax = axes[1, 0]
        if CONDITION_SCORE in df and ENERGY_CLASS in df:
            self.box_plot_by_category(df, ENERGY_CLASS, CONDITION_SCORE, ax, order=list(ENERGY_CLASSES))
        else:
            logger.warning("[DESCR] Missing data for Energy/Condition boxplot")
            ax.text(0.5, 0.5, "No energy/condition data", ha="center", va="center")

        # Score correlations heatmap
        ax = axes[1, 1]
        score_cols = [c for c in [CONDITION_SCORE, RISK_SCORE, LUXURY_SCORE, ENV_SCORE] if c in df]
        if len(score_cols) >= 2:
            plot_correlation_heatmap(df, score_cols, ax=ax)
        else:
            logger.warning("[DESCR] Not enough score columns for heatmap")
            ax.text(0.5, 0.5, "Not enough score columns", ha="center", va="center")

        fig.tight_layout()
        return fig

# -------------------------------- Outliers ----------------------------------

class OutlierDetector:
    """
    Outlier detection on a DataFrame (I/O-free).

    Parameters
    ----------
    method : str
        'iqr' or 'zscore'.
    iqr_multiplier : float
        IQR whisker multiplier.
    z_threshold : float
        Absolute threshold for Z-score.
    domain_constraints : Mapping[str, Set[str]]
        Constraints (non_negative, unit_interval) to clamp bounds.
    """

    def __init__(
        self,
        method: str = "iqr",
        iqr_multiplier: float = 1.5,
        z_threshold: float = 3.0,
        domain_constraints: Optional[Mapping[str, Set[str]]] = None,
    ) -> None:
        self.method = method.lower()
        self.iqr_multiplier = float(iqr_multiplier)
        self.z_threshold = float(z_threshold)
        self.domain_constraints = {
            "non_negative": set(domain_constraints.get("non_negative", set())) if domain_constraints else set(),
            "unit_interval": set(domain_constraints.get("unit_interval", set())) if domain_constraints else set(),
        }

    # ---------------- Public API ----------------

    def detect_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Detect outliers for the specified columns (no saves).

        Returns
        -------
        dict
            Per-column results with stats and metadata.
        """
        if df.empty:
            logger.warning("Empty DataFrame: cannot detect outliers.")
            return {}

        if columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            columns = [col for col in DEFAULT_NUMERIC_FEATURES if col in numeric_cols]

        methods_map = {"iqr": self._detect_outliers_iqr, "zscore": self._detect_outliers_zscore}
        if self.method not in methods_map:
            raise ValueError(f"Unknown method: {self.method}")

        return methods_map[self.method](df, columns)

    def combine_outlier_results(
        self,
        df: pd.DataFrame,
        outliers_summary: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Combine outlier hits across columns into a single DataFrame.
        """
        outlier_records: List[Dict[str, Any]] = []

        for col, info in outliers_summary.items():
            for idx in info.get("indices", []):
                outlier_records.append({
                    "index": idx,
                    "outlier_source": col,
                    "outlier_count": info["count"],
                    "outlier_pct": info["percentage"]
                })

        if not outlier_records:
            logger.info("No outliers found in the analyzed columns.")
            return pd.DataFrame()

        combined_df = pd.DataFrame(outlier_records)

        aggregated = (
            combined_df.groupby("index", observed=True)
            .agg({"outlier_source": list, "outlier_count": "first"})
            .reset_index()
        )
        aggregated["n_outlier_sources"] = aggregated["outlier_source"].apply(len)

        combined_full = aggregated.merge(
            df.reset_index(),
            on="index",
            how="left"
        ).sort_values("n_outlier_sources", ascending=False)

        return combined_full

    def get_outlier_summary_stats(
        self,
        df: pd.DataFrame,
        outliers_summary: Dict[str, Any],
        combined_outliers: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compute summary metrics about detected outliers."""
        total_rows = len(df)
        unique_records = (
            len(combined_outliers["index"].unique())
            if not combined_outliers.empty else 0
        )

        source_distribution = {}
        if not combined_outliers.empty:
            source_distribution = (
                combined_outliers["n_outlier_sources"]
                .value_counts()
                .sort_index()
                .to_dict()
            )

        top_outliers = []
        if not combined_outliers.empty:
            for _, row in combined_outliers.head(5).iterrows():
                top_outliers.append({
                    "index": int(row["index"]),
                    "n_sources": int(row["n_outlier_sources"]),
                    "sources": row["outlier_source"]
                })

        return {
            "total_records": total_rows,
            "unique_outlier_records": unique_records,
            "outlier_percentage": round((unique_records / total_rows * 100) if total_rows else 0.0, 2),
            "by_column": {
                col: {"count": info["count"], "percentage": info["percentage"]}
                for col, info in outliers_summary.items()
            },
            "source_distribution": source_distribution,
            "top_multi_source_outliers": top_outliers
        }

    # ---------------- Private methods ----------------

    def _detect_outliers_iqr(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        for col in columns:
            if col not in df.columns:
                logger.warning("Column '%s' not found, skipping.", col)
                continue

            series = df[col].dropna()
            if series.empty:
                logger.warning("No data for '%s', skipping.", col)
                continue

            lower, upper = self._compute_clamped_iqr_bounds(col, series)
            mask = (df[col] < lower) | (df[col] > upper)
            outliers = df.loc[mask].copy()

            results[col] = {
                "method": "IQR",
                "multiplier": self.iqr_multiplier,
                "count": int(len(outliers)),
                "percentage": round(len(outliers) / len(df) * 100, 2),
                "bounds": {"lower": float(round(lower, 4)), "upper": float(round(upper, 4))},
                "indices": outliers.index.tolist(),
                "statistics": self._get_basic_stats(series)
            }

            logger.info(
                "[OUTLIER][IQR] %s: %d (%.2f%%) within bounds [%.2f, %.2f]",
                col, len(outliers), results[col]["percentage"], lower, upper
            )

        return results

    def _detect_outliers_zscore(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        for col in columns:
            if col not in df.columns:
                continue

            series = df[col].dropna()
            if series.empty:
                continue

            mean = series.mean()
            std = series.std()
            if std == 0 or np.isnan(std):
                logger.warning("Std=0 for '%s', skipping.", col)
                continue

            z_scores = np.abs((df[col] - mean) / std)
            mask = z_scores > self.z_threshold
            outliers = df.loc[mask].copy()

            results[col] = {
                "method": "Z-score",
                "threshold": self.z_threshold,
                "count": int(len(outliers)),
                "percentage": round(len(outliers) / len(df) * 100, 2),
                "indices": outliers.index.tolist(),
                "statistics": {
                    "mean": float(mean),
                    "std": float(std),
                    "max_zscore": float(z_scores.max()) if not z_scores.empty else 0.0
                }
            }

        return results

    def _compute_clamped_iqr_bounds(self, col: str, series: pd.Series) -> Tuple[float, float]:
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - self.iqr_multiplier * IQR
        upper = Q3 + self.iqr_multiplier * IQR

        if col in self.domain_constraints["non_negative"]:
            lower = max(lower, 0.0)
        if col in self.domain_constraints["unit_interval"]:
            lower = max(lower, 0.0)
            upper = min(upper, 1.0)

        return float(lower), float(upper)

    @staticmethod
    def _get_basic_stats(series: pd.Series) -> Dict[str, float]:
        return {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "q1": float(series.quantile(0.25)),
            "q3": float(series.quantile(0.75)),
            "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
        }

# -------------------------------- Anomalies ---------------------------------

class AnomalyDetector:
    """Multi-level anomaly detection with Isolation Forest (no I/O)."""

    def __init__(
        self,
        contamination: float = DEFAULT_CONTAMINATION,
        strong_z_threshold: float = DEFAULT_STRONG_Z_THRESHOLD,
        severity_percentile: float = DEFAULT_SEVERITY_PERCENTILE,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        self.contamination = float(contamination)
        self.strong_z_threshold = float(strong_z_threshold)
        self.severity_percentile = float(severity_percentile)
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        feature_candidates: Optional[List[str]] = None,
        exclude_features: Optional[Set[str]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run anomaly detection + refinement, returning an enriched df and a report."""
        df = df.copy()
        report: Dict[str, Any] = {"status": "started"}

        # 1) Feature selection
        features = self._select_features(
            df=df,
            candidates=list(feature_candidates) if feature_candidates is not None else list(DEFAULT_CANDIDATE_FEATURES),
            exclude=set(exclude_features) if exclude_features is not None else set(LEAKY_FEATURES),
        )
        if not features:
            logger.warning("[ANOM] No features available for anomaly detection.")
            report["status"] = "no_features"
            return df, report
        report["features_used"] = features

        # 2) Complete rows only
        X_valid, valid_mask = self._prepare_data(df, features)
        if X_valid.empty:
            logger.warning("[ANOM] No complete rows for anomaly detection.")
            report["status"] = "no_complete_rows"
            return df, report
        report["n_valid_rows"] = int(len(X_valid))

        # 3) Robust z-scores
        z_scores_df = self._calculate_robust_zscores(X_valid)

        # 4) Isolation Forest
        anomaly_results = self._run_isolation_forest(X_valid)

        # 5) Assignment
        df = self._assign_anomaly_scores(
            df=df,
            valid_mask=valid_mask,
            anomaly_results=anomaly_results,
            z_scores_df=z_scores_df,
        )

        # 6) Severity + refine
        df = self._calculate_severity_and_refine(df, valid_mask)

        # 7) Report
        report.update(self._generate_report(df))
        report["status"] = "completed"
        logger.info(
            "[ANOM] Anomaly detection completed: raw=%s refined=%s",
            report.get("n_anomalies_raw"), report.get("n_anomalies_refined")
        )
        return df, report

    def visualize_anomalies(
        self,
        df: pd.DataFrame,
        feature_pairs: Optional[List[Tuple[str, str]]] = None,
        figsize: Tuple[int, int] = (15, 5),
    ) -> Optional[plt.Figure]:
        """Plot anomalies across feature pairs (no saving)."""
        # default pairs
        if feature_pairs is None:
            candidate_pairs: List[Tuple[str, str]] = []
            if SIZE_M2 in df.columns and VALUATION_K in df.columns:
                candidate_pairs.append((SIZE_M2, VALUATION_K))
            if "condition_minus_risk" in df.columns and LUXURY_SCORE in df.columns:
                candidate_pairs.append(("condition_minus_risk", LUXURY_SCORE))
            if ENV_SCORE in df.columns and VALUATION_K in df.columns:
                candidate_pairs.append((ENV_SCORE, VALUATION_K))
            feature_pairs = candidate_pairs

        valid_pairs = [(x, y) for (x, y) in feature_pairs if x in df.columns and y in df.columns]
        if not valid_pairs:
            logger.warning("[ANOM] No valid feature pairs for visualization.")
            return None

        n_plots = len(valid_pairs)
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        axes = [axes] if n_plots == 1 else list(axes)

        palette = {"normal": "tab:blue", "low": "tab:orange", "high": "tab:red"}
        for ax, (x_col, y_col) in zip(axes, valid_pairs):
            if "anomaly_category" in df.columns and sns is not None:
                sns.scatterplot(
                    data=df, x=x_col, y=y_col,
                    hue="anomaly_category", palette=palette,
                    alpha=0.6, s=40, ax=ax
                )
                ax.legend(title="Anomaly", loc="best")
            else:
                ax.scatter(df[x_col], df[y_col], alpha=0.6, s=40)

            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{x_col} vs {y_col}")

        fig.suptitle("Anomaly Detection Results", fontsize=14)
        fig.tight_layout()
        return fig

    def extract_anomaly_frames(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Extract helpful DataFrames: raw/refined (no saving).
        Returns:
            {"raw": df_raw, "refined": df_refined}
        """
        out: Dict[str, pd.DataFrame] = {}
        if "anomaly_flag" in df.columns:
            raw = df[df["anomaly_flag"] == True]
            if not raw.empty:
                out["raw"] = raw.copy()
        if "anomaly_refined" in df.columns:
            ref = df[df["anomaly_refined"] == True]
            if not ref.empty:
                out["refined"] = ref.copy()
        return out

    # ---------------- Private helpers --------------

    def _select_features(self, df: pd.DataFrame, candidates: List[str], exclude: Set[str]) -> List[str]:
        chosen: List[str] = []
        # add log of target as candidate feature (if allowed)
        if VALUATION_K in df.columns and VALUATION_K not in exclude:
            df["valuation_k_log"] = np.log1p(df[VALUATION_K])
            if "valuation_k_log" not in candidates:
                candidates = [*candidates, "valuation_k_log"]

        for feat in candidates:
            if feat in df.columns and feat not in exclude:
                chosen.append(feat)

        logger.info("[ANOM] Selected features (%d): %s", len(chosen), chosen)
        return chosen

    def _prepare_data(self, df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
        X_raw = df[features].copy()
        valid_mask = X_raw.notna().all(axis=1)
        X_valid = X_raw.loc[valid_mask].copy()
        logger.info("[ANOM] Valid rows: %d / %d", len(X_valid), len(df))
        return X_valid, valid_mask

    def _calculate_robust_zscores(self, X: pd.DataFrame) -> pd.DataFrame:
        def robust_z(series: pd.Series) -> pd.Series:
            median = series.median()
            mad = float(np.median(np.abs(series - median)))
            if mad == 0 or pd.isna(mad):
                return pd.Series(0.0, index=series.index)
            return (series - median) / (mad * MAD_CONSTANT)
        return X.apply(robust_z, axis=0)

    def _run_isolation_forest(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        iso = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        iso.fit(X_scaled)

        decision_scores = iso.decision_function(X_scaled)  # >0 more normal
        predictions = iso.predict(X_scaled)                 # -1 anomaly
        return {
            "decision_scores": decision_scores,
            "anomaly_scores": -decision_scores,   # higher = more anomalous
            "predictions": predictions == -1,     # bool
        }

    def _assign_anomaly_scores(
        self,
        df: pd.DataFrame,
        valid_mask: pd.Series,
        anomaly_results: Dict[str, np.ndarray],
        z_scores_df: pd.DataFrame,
    ) -> pd.DataFrame:
        df.loc[valid_mask, "anomaly_score_raw"] = anomaly_results["decision_scores"]
        df.loc[valid_mask, "anomaly_score"] = anomaly_results["anomaly_scores"]
        df.loc[valid_mask, "anomaly_flag"] = anomaly_results["predictions"]

        abs_z = z_scores_df.abs()
        top_contributors: List[str] = []
        top_z_scores: List[float] = []
        n_strong_contributors: List[int] = []
        z_details: List[Dict[str, float]] = []

        for idx in z_scores_df.index:
            row_abs = abs_z.loc[idx]
            top_feat = row_abs.idxmax()
            top_contributors.append(str(top_feat))
            top_z_scores.append(float(z_scores_df.loc[idx, top_feat]))
            n_strong_contributors.append(int((row_abs >= self.strong_z_threshold).sum()))
            z_details.append({str(c): float(z_scores_df.loc[idx, c]) for c in z_scores_df.columns})

        df.loc[valid_mask, "top_contributor"] = top_contributors
        df.loc[valid_mask, "top_z_score"] = top_z_scores
        df.loc[valid_mask, "n_strong_contributors"] = n_strong_contributors
        df.loc[valid_mask, "z_scores_detail"] = z_details
        return df

    def _calculate_severity_and_refine(self, df: pd.DataFrame, valid_mask: pd.Series) -> pd.DataFrame:
        df["severity_score"] = df["anomaly_score"] * (1 + np.minimum(df["n_strong_contributors"].fillna(0), 2))

        valid_sev = df.loc[valid_mask, "severity_score"].dropna()
        cutoff = float(np.percentile(valid_sev, self.severity_percentile)) if not valid_sev.empty else 0.0

        high_sev = df["severity_score"] >= cutoff
        multi_contrib = df["n_strong_contributors"] >= 2
        not_only_val = df.get("top_contributor") != "valuation_k_log"
        s = df.get("anomaly_flag")
        if s is None:
            base_anom = pd.Series(False, index=df.index, dtype="boolean")
        else:
            base_anom = pd.Series(s, copy=False)
            if base_anom.dtype == "object":
                base_anom = base_anom.astype("boolean")
            base_anom = base_anom.fillna(False)

        df["anomaly_refined"] = (high_sev & (multi_contrib | not_only_val) & base_anom)
        df["anomaly_category"] = np.where(
            df["anomaly_refined"], "high",
            np.where(df["anomaly_flag"], "low", "normal"),
        )
        return df

    def _generate_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        n_total = int(len(df))
        n_raw = int(df["anomaly_flag"].sum()) if "anomaly_flag" in df.columns else 0
        n_ref = int(df["anomaly_refined"].sum()) if "anomaly_refined" in df.columns else 0

        rpt: Dict[str, Any] = {
            "n_anomalies_raw": n_raw,
            "pct_anomalies_raw": float(round((n_raw / n_total * 100) if n_total else 0.0, 2)),
            "n_anomalies_refined": n_ref,
            "pct_anomalies_refined": float(round((n_ref / n_total * 100) if n_total else 0.0, 2)),
            "contamination": float(self.contamination),
            "strong_z_threshold": float(self.strong_z_threshold),
            "severity_percentile": float(self.severity_percentile),
        }

        if n_ref > 0 and "severity_score" in df.columns:
            top = df[df["anomaly_refined"]].nlargest(5, "severity_score")
            rpt["top_anomalies"] = [
                {
                    "asset_id": row.get("asset_id", "N/A"),
                    "severity_score": float(row["severity_score"]),
                    "top_contributor": row.get("top_contributor"),
                    "top_z_score": float(row.get("top_z_score", 0.0)),
                    "n_strong_contributors": int(row.get("n_strong_contributors", 0)),
                }
                for _, row in top.iterrows()
            ]

        if "top_contributor" in df.columns:
            counts = df[df.get("anomaly_refined", False) == True]["top_contributor"].value_counts()
            rpt["top_contributors"] = {str(k): int(v) for k, v in counts.to_dict().items()}

        return rpt

# -------------------------------- Temporal ----------------------------------

class TemporalAnalyzer:
    """Analyze data freshness and temporal trends (no I/O)."""

    def __init__(
        self,
        stale_thresholds: Optional[List[int]] = None,
        reference_time: Optional[datetime] = None
    ) -> None:
        self.stale_thresholds = stale_thresholds or _DEFAULT_THRESHOLDS
        self.reference_time = reference_time or datetime.now(timezone.utc)

    def analyze(
        self,
        df: pd.DataFrame,
        target: str = VALUATION_K
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run temporal analysis and return (df_enriched, report)."""
        df = df.copy()
        report: Dict[str, Any] = {"status": "started"}

        # Ensure timestamps
        if LAST_VERIFIED_TS not in df:
            df[LAST_VERIFIED_TS] = pd.NaT
            report["warning"] = "missing LAST_VERIFIED_TS"

        for col in (LAST_VERIFIED_TS, PREDICTION_TS):
            df[col] = pd.to_datetime(df.get(col, pd.NaT), utc=True, errors="coerce")

        # Compute age in days/hours
        age = (self.reference_time - df[LAST_VERIFIED_TS]).dt
        df["days_since_verification"] = age.days
        df["hours_since_verification"] = age.total_seconds().div(3600)

        # Staleness flags
        for th in self.stale_thresholds:
            df[f"is_stale_{th}d"] = df["days_since_verification"] > th

        # Stats
        stats = self._temporal_stats(df)
        report["temporal_stats"] = stats

        # Correlation + regression
        if target in df:
            report["correlation"] = self._correlate(df, target)
            report["regression"] = self._regress(df, target)
        else:
            report["warning"] = f"missing target '{target}'"

        report["status"] = "completed"
        logger.info("[TEMPORAL] Analysis completed", extra={"target": target, "stats": stats})
        return df, report

    def plot(self, df: pd.DataFrame, target: Optional[str] = None) -> plt.Figure:
        """Histogram of freshness and optional scatter of target vs time (no save)."""
        df = df.copy()
        n_axes = 2 if (target is not None and target in df) else 1
        fig, axes = plt.subplots(1, n_axes, figsize=(14, 5))
        axes = np.atleast_1d(axes)

        # Histogram
        if sns is not None:
            sns.histplot(df["days_since_verification"].dropna(), kde=True, ax=axes[0])
        else:
            axes[0].hist(df["days_since_verification"].dropna(), bins=30)
        for th in self.stale_thresholds:
            axes[0].axvline(th, linestyle="--", label=f">{th}d")
        axes[0].legend()

        # Scatter
        if n_axes == 2 and target in df:
            ax = axes[1]
            sc = ax.scatter(
                df[LAST_VERIFIED_TS], df[target],
                c=df["days_since_verification"], cmap="viridis", alpha=0.6
            )
            fig.colorbar(sc, ax=ax, label="days_since_verification")
            ax.set_xlabel("Last Verified TS")
            ax.set_ylabel(target)
            for label in ax.get_xticklabels():
                label.set_rotation(30)

        fig.tight_layout()
        return fig

    # Helpers -----------------------------------------------------------------

    def _temporal_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        days = df["days_since_verification"].dropna()
        if days.empty:
            return {}
        return {
            "count": int(len(days)),
            "mean": float(days.mean()),
            "median": float(days.median()),
            "pct_over_thresholds": {str(th): float((days > th).mean() * 100) for th in self.stale_thresholds},
        }

    def _correlate(self, df: pd.DataFrame, target: str) -> Dict[str, Any]:
        mask = df["days_since_verification"].notna() & df[target].notna()
        if not mask.any():
            return {}
        subset = df.loc[mask, ["days_since_verification", target]]
        return {
            "pearson": float(subset.corr().iloc[0, 1]),
            "spearman": float(subset.corr(method="spearman").iloc[0, 1]),
        }

    def _regress(self, df: pd.DataFrame, target: str) -> Dict[str, Any]:
        mask = df["days_since_verification"].notna() & df[target].notna()
        if mask.sum() < 10:
            return {"status": "insufficient_data"}
        X = sm.add_constant(df.loc[mask, "days_since_verification"])
        y = df.loc[mask, target]
        model = sm.OLS(y, X).fit(cov_type="HC3")
        coef = float(model.params.get("days_since_verification", 0.0))
        return {
            "coef": coef,
            "p_value": float(model.pvalues.get("days_since_verification", 1.0)),
            "r2": float(model.rsquared),
        }

# ------------------------------ Statistics ----------------------------------

class StatisticalTester:
    """Run diagnostic statistical tests on a dataset (no I/O)."""

    def __init__(self, normality_alpha: float = NORMALITY_ALPHA, chi2_alpha: float = CHI2_ALPHA):
        self.normality_alpha = float(normality_alpha)
        self.chi2_alpha = float(chi2_alpha)

    def run_comprehensive_tests(
        self,
        df: pd.DataFrame,
        normality_features: Optional[List[str]] = None,
        categorical_pairs: Optional[List[Tuple[str, str]]] = None,
        *,
        max_numeric_for_stats: int = 20,
    ) -> Dict[str, Any]:
        """Full suite: normality, categorical dependencies, and distribution stats."""
        norm = self.test_normality(df, normality_features)
        deps = self.test_categorical_dependencies(df, categorical_pairs)
        dist = self.compute_distribution_statistics(df, max_numeric_for_stats=max_numeric_for_stats)

        results: Dict[str, Any] = {
            "normality": norm,
            "categorical_dependencies": deps,
            "distribution_stats": dist,
            "summary": self._summarize_test_results({"normality": norm, "categorical_dependencies": deps}),
        }
        return results


    def test_normality(self, df: pd.DataFrame, features: Optional[List[str]] = None) -> Dict[str, Any]:
        """D'Agostino K² normality test on the specified features."""
        feats = list(features) if features is not None else list(DEFAULT_NORMALITY_FEATURES)
        results: Dict[str, Any] = {}

        def _safe_moments(series: pd.Series) -> tuple[float, float]:
            """
            Return (skewness, kurtosis) robustly:
            - cast to float ignoring non-numerics
            - avoid RuntimeWarnings for near-constant series (variance ~0)
            - if SciPy is unavailable, return (0.0, 0.0)
            """
            vals = pd.to_numeric(series, errors="coerce").astype(float)
            vals = vals[np.isfinite(vals)]
            if vals.size < 3 or stats is None:
                return 0.0, 0.0
            span = float(np.nanmax(vals) - np.nanmin(vals)) if vals.size else 0.0
            var = float(np.nanvar(vals))
            if not np.isfinite(var) or var < 1e-12 or span < 1e-12:
                return 0.0, 0.0
            return (
                float(stats.skew(vals, nan_policy="omit", bias=True)),
                float(stats.kurtosis(vals, nan_policy="omit", bias=True)),
            )
    
        for col in feats:
            if col not in df.columns:
                results[col] = {"status": "missing"}
                logger.info("Column '%s' missing for normality test", col)
                continue
            
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            n = int(series.size)
    
            if n < MIN_SAMPLES_NORMALITY:
                results[col] = {"status": "insufficient_data", "n_samples": n, "min_required": MIN_SAMPLES_NORMALITY}
                logger.info("Skipping normality test for %s (only %d samples)", col, n)
                continue
            
            if stats is None:
                results[col] = {"status": "skipped_no_scipy", "n_samples": n}
                logger.warning("SciPy not available: skipping normality test for %s", col)
                continue
            
            try:
                stat, p_value = stats.normaltest(series)
                is_normal = bool(p_value > self.normality_alpha)
                skewness, kurtosis = _safe_moments(series)
    
                results[col] = {
                    "status": "tested",
                    "statistic": float(stat) if np.isfinite(stat) else float("nan"),
                    "p_value": float(p_value) if np.isfinite(p_value) else float("nan"),
                    "is_normal": is_normal,
                    "skewness": float(skewness),
                    "kurtosis": float(kurtosis),
                    "n_samples": n,
                    "interpretation": self._interpret_normality(is_normal, float(skewness), float(kurtosis)),
                }
    
                logger.info(
                    "%s: p=%.4f → %s (skew=%.2f, kurt=%.2f)",
                    col, p_value, "Normal ✅" if is_normal else "Non-normal ❌", skewness, kurtosis,
                )
            except Exception as e:
                results[col] = {"status": "error", "error": str(e)}
                logger.error("Normality test failed for %s: %s", col, e)
    
        return results

    def _interpret_normality(self, is_normal: bool, skewness: float, kurtosis: float) -> str:
        if is_normal:
            return "Approximately normal distribution"
        notes: List[str] = []
        if abs(skewness) > 1:
            notes.append("right-skewed" if skewness > 0 else "left-skewed")
        elif abs(skewness) > 0.5:
            notes.append("moderately skewed")
        if abs(kurtosis) > 1:
            notes.append("heavy-tailed" if kurtosis > 0 else "light-tailed")
        return "Non-normal: " + ", ".join(notes) if notes else "Non-normal distribution"

    def test_categorical_dependencies(
        self,
        df: pd.DataFrame,
        pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        """χ² tests on variable pairs (auto-bins numeric variables)."""
        test_pairs = list(pairs) if pairs is not None else [(ENERGY_CLASS, LUXURY_SCORE)]
        results: Dict[str, Any] = {}

        for var1, var2 in test_pairs:
            key = f"{var1}_vs_{var2}"

            if var1 not in df.columns or var2 not in df.columns:
                missing = [v for v in (var1, var2) if v not in df.columns]
                results[key] = {"status": "missing_columns", "missing": missing}
                continue

            try:
                if stats is None:
                    results[key] = {"status": "skipped_no_scipy"}
                    logger.warning("SciPy not available: skipping chi² for %s", key)
                    continue

                data = df[[var1, var2]].copy().dropna(how="any")

                # If var2 is numeric, discretize it into 3 quantiles
                var2_cat = var2
                if pd.api.types.is_numeric_dtype(data[var2]):
                    data[f"{var2}_cat"] = pd.qcut(data[var2], q=3, labels=["low", "medium", "high"], duplicates="drop")
                    var2_cat = f"{var2}_cat"

                contingency = pd.crosstab(data[var1], data[var2_cat])
                if contingency.size == 0 or contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    results[key] = {"status": "empty_contingency"}
                    continue

                chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
                is_independent = bool(p_value > self.chi2_alpha)

                n = contingency.to_numpy().sum()
                min_dim = min(contingency.shape) - 1
                cramers_v = float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 and n > 0 else 0.0

                results[key] = {
                    "status": "tested",
                    "chi2": float(chi2),
                    "p_value": float(p_value),
                    "dof": int(dof),
                    "is_independent": is_independent,
                    "cramers_v": cramers_v,
                    "contingency_shape": tuple(contingency.shape),
                    "interpretation": self._interpret_chi2(is_independent, cramers_v),
                }

                logger.info(
                    "Chi² (%s vs %s): p=%.4f → %s (Cramér's V=%.3f)",
                    var1, var2, p_value, "Independent ✅" if is_independent else "Dependent ❌", cramers_v,
                )
            except Exception as e:
                results[key] = {"status": "error", "error": str(e)}
                logger.error("Chi² test failed for %s: %s", key, e)

        return results

    def _interpret_chi2(self, is_independent: bool, cramers_v: float) -> str:
        if is_independent:
            return "Variables are statistically independent"
        if cramers_v < 0.1:
            effect = "negligible"
        elif cramers_v < 0.3:
            effect = "small"
        elif cramers_v < 0.5:
            effect = "medium"
        else:
            effect = "large"
        return f"Variables are dependent with {effect} effect size"

    def compute_distribution_statistics(
        self,
        df: pd.DataFrame,
        *,
        max_numeric_for_stats: int = 20,
    ) -> Dict[str, Any]:
        """Distribution statistics for numeric variables (limiting the number of columns)."""

        def _safe_moments(series: pd.Series) -> tuple[float, float]:
            """
            Return (skewness, kurtosis) robustly:
            - cast to float ignoring non-numerics
            - avoid RuntimeWarnings for near-constant series (variance ~0)
            - if SciPy is unavailable, return (0.0, 0.0)
            """
            vals = pd.to_numeric(series, errors="coerce").astype(float)
            vals = vals[np.isfinite(vals)]
            if vals.size < 3 or stats is None:
                return 0.0, 0.0
            span = float(np.nanmax(vals) - np.nanmin(vals)) if vals.size else 0.0
            var = float(np.nanvar(vals))
            if not np.isfinite(var) or var < 1e-12 or span < 1e-12:
                return 0.0, 0.0
            return (
                float(stats.skew(vals, nan_policy="omit", bias=True)),
                float(stats.kurtosis(vals, nan_policy="omit", bias=True)),
            )
    
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return {}

        cols = numeric_cols[: max(1, int(max_numeric_for_stats))]
        stats_dict: Dict[str, Any] = {}

        for col in cols:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty:
                continue

            mean = float(series.mean())
            std = float(series.std())
            entry: Dict[str, Any] = {
                "mean": mean,
                "median": float(series.median()),
                "std": std,
                "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
                "cv": float(std / mean) if mean != 0 else np.inf,
                "count": int(series.size),
            }

            skewness, kurtosis = _safe_moments(series)
            entry["skewness"] = float(skewness)
            entry["kurtosis"] = float(kurtosis)

            stats_dict[col] = entry

        return stats_dict

    def _summarize_test_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize normality and categorical dependency findings."""
        summary: Dict[str, Any] = {}

        # Normality
        normality = results.get("normality", {}) or {}
        tested = [k for k, v in normality.items() if v.get("status") == "tested"]
        normal = [k for k, v in normality.items() if v.get("is_normal") is True]
        summary["normality"] = {
            "n_tested": int(len(tested)),
            "n_normal": int(len(normal)),
            "pct_normal": float((len(normal) / len(tested) * 100) if tested else 0.0),
            "non_normal_features": [k for k in tested if k not in normal],
        }

        # Categorical dependencies
        deps = results.get("categorical_dependencies", {}) or {}
        tested_pairs = [k for k, v in deps.items() if v.get("status") == "tested"]
        dependent = [k for k, v in deps.items() if v.get("is_independent") is False]
        summary["dependencies"] = {
            "n_tested": int(len(tested_pairs)),
            "n_dependent": int(len(dependent)),
            "dependent_pairs": dependent,
        }

        return summary

# ------------------------- Feature Importance / Ablation ---------------------

class FeatureImportanceAnalyzer:
    """Feature importance and ablation study using Random Forest (no I/O)."""

    def __init__(
        self,
        target_column: str = VALUATION_K,
        n_estimators: int = 100,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.target_column = target_column
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model: Optional[RandomForestRegressor] = None
        self.feature_names: Optional[List[str]] = None

    def prepare_features(
        self,
        df: pd.DataFrame,
        exclude_features: Optional[Set[str]] = None,
        include_proxies: bool = False
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Prepare features excluding leakage and target-derived; encode categoricals."""
        if self.target_column not in df.columns:
            raise ValueError(f"Target '{self.target_column}' not found in DataFrame")

        y = df[self.target_column].copy()

        exclude: Set[str] = {
            self.target_column, *LEAKY_FEATURES,
            *ID_LIKE_FEATURES, *TARGET_DERIVED_FEATURES
        }
        if exclude_features:
            exclude.update(exclude_features)
        if not include_proxies:
            exclude.update(PROXY_FEATURES)

        df_features = df.drop(columns=[c for c in exclude if c in df.columns], errors="ignore").copy()
        encoded_features = self._encode_categorical_features(df_features)
        df_features = pd.concat([df_features, encoded_features], axis=1)

        feature_cols = [
            col for col in df_features.columns
            if col not in exclude and pd.api.types.is_numeric_dtype(df_features[col])
        ] + [col for col in df_features.columns if col.endswith("_encoded")]

        feature_cols = list(dict.fromkeys(feature_cols))  # dedup preserving order

        if not feature_cols and include_proxies:
            logger.warning("Empty feature set; force-adding selected proxy features")
            for col in [SIZE_M2, LUXURY_SCORE, ENV_SCORE]:
                if col in df.columns and col not in exclude:
                    feature_cols.append(col)

        if not feature_cols:
            raise ValueError("No valid features available for analysis")

        logger.info("Selected %d features for importance analysis", len(feature_cols))

        X = df_features[feature_cols].fillna(0)
        self.feature_names = feature_cols

        return X, y, feature_cols

    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features with OrdinalEncoder."""
        encoded_dfs: List[pd.DataFrame] = []

        for col in CATEGORICAL_FEATURES:
            if col not in df.columns:
                continue
            try:
                encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                values = df[[col]].astype(str)
                encoded = encoder.fit_transform(values)
                encoded_dfs.append(pd.DataFrame(encoded, columns=[f"{col}_encoded"], index=df.index))
                logger.debug("Encoded %s with %d categories", col, len(encoder.categories_[0]))
            except Exception as e:
                logger.warning("Encoding failed for '%s': %s", col, e)

        return pd.concat(encoded_dfs, axis=1) if encoded_dfs else pd.DataFrame(index=df.index)

    def calculate_importances(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        calculate_permutation: bool = True,
        n_repeats: int = 10
    ) -> Dict[str, pd.DataFrame]:
        """Compute feature importance (RF built-in and optional permutation)."""
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        self.model.fit(X, y)

        builtin_imp = pd.DataFrame({
            "feature": X.columns,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)

        results: Dict[str, pd.DataFrame] = {"builtin": builtin_imp}

        if calculate_permutation:
            logger.info("Computing permutation importances...")
            perm_imp = permutation_importance(
                self.model, X, y,
                n_repeats=n_repeats,
                random_state=self.random_state,
                scoring="r2",
                n_jobs=self.n_jobs
            )
            results["permutation"] = pd.DataFrame({
                "feature": X.columns,
                "importance": perm_imp.importances_mean,
                "std": perm_imp.importances_std
            }).sort_values("importance", ascending=False)

        return results

    def perform_ablation_study(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features_to_ablate: List[str],
        cv_folds: int = 5
    ) -> pd.DataFrame:
        """Ablation study by removing specific features (returns a results DataFrame)."""
        base_features = X.columns.tolist()
        settings = {"full": base_features}

        for feat in features_to_ablate:
            if feat in base_features:
                settings[f"no_{feat}"] = [f for f in base_features if f != feat]

        if len(features_to_ablate) > 1:
            settings["no_all"] = [f for f in base_features if f not in features_to_ablate]

        results: List[Dict[str, Union[str, float, int]]] = []

        for name, features in settings.items():
            if not features:
                continue

            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            cv_results = cross_validate(
                model, X[features], y,
                cv=cv_folds,
                scoring=["r2", "neg_mean_absolute_error"],
                return_train_score=False,
                n_jobs=self.n_jobs
            )
            results.append({
                "setting": name,
                "n_features": len(features),
                "mean_r2": float(np.mean(cv_results["test_r2"])),
                "std_r2": float(np.std(cv_results["test_r2"])),
                "mean_mae": float(-np.mean(cv_results["test_neg_mean_absolute_error"])),
                "std_mae": float(np.std(cv_results["test_neg_mean_absolute_error"])),
            })

        results_df = pd.DataFrame(results).set_index("setting")
        if "full" in results_df.index:
            full_r2 = float(results_df.loc["full", "mean_r2"])
            results_df["r2_drop"] = full_r2 - results_df["mean_r2"]
            results_df["r2_drop_pct"] = (results_df["r2_drop"] / full_r2 * 100) if full_r2 != 0 else np.inf

        return results_df.sort_values("mean_r2", ascending=False)
