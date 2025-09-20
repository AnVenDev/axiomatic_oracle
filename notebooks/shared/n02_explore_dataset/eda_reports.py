# notebooks/shared/n02_explore_dataset/eda_reports.py
'''
EDA Orchestration + I/O layer
-----------------------------
- Invokes the *pure* classes in `shared.nb02.eda_core` (no I/O there)
- Exports CSV/Parquet/JSON/PNG
- Produces a manifest with metadata and artifact paths

Typical usage (e.g., in notebook 02):
    runner = EDAReportRunner(output_dir="outputs/analysis", meta={"schema_version": "1.0.0"})
    manifest = runner.run_full_eda(df)

Or granular invocations (only outliers, only importance, etc.).
'''

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from os import path
from typing import Any, Dict, List, Optional, Tuple, Mapping

from pathlib import Path
from notebooks.shared.common.utils import canonical_json_dumps
import numpy as np                  # type: ignore
import pandas as pd                 # type: ignore

import matplotlib                   # type: ignore
import matplotlib.pyplot as plt     # type: ignore

# Import the I/O-free core
from notebooks.shared.n02_explore_dataset.eda_core import (
    DescriptiveAnalyzer,
    OutlierDetector,
    AnomalyDetector,
    TemporalAnalyzer,
    StatisticalTester,
    FeatureImportanceAnalyzer,
    plot_correlation_heatmap,
)

from notebooks.shared.common.constants import (
    VALUATION_K,
    SIZE_M2,
    ENERGY_CLASS,
    LOCATION,
)

logger = logging.getLogger(__name__)

# ------------------------------ I/O Utilities -------------------------------

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _safe_to_parquet(df: pd.DataFrame, path: Path) -> Optional[Path]:
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception as e:
        logger.warning("Parquet not available (%s). Skipping parquet for %s.", e, path.name)
        return None

def save_dataframe(
    df: pd.DataFrame,
    basepath: Path,
    name: str,
    *,
    save_csv: bool = True,
    save_parquet: bool = True,
) -> Dict[str, str]:
    """Save a DataFrame to CSV and/or Parquet. Returns mapping {format: path}."""
    out: Dict[str, str] = {}
    _ensure_dir(basepath)
    if df is None or df.empty:
        return out

    if save_csv:
        p_csv = basepath / f"{name}.csv"
        df.to_csv(p_csv, index=False)
        out["csv"] = str(p_csv)
    if save_parquet:
        p_parq = basepath / f"{name}.parquet"
        saved = _safe_to_parquet(df, p_parq)
        if saved is not None:
            out["parquet"] = str(saved)
    return out

def save_json(obj: Any, basepath: Path, name: str, *, indent: int = 2) -> str:
    _ensure_dir(basepath)
    p = basepath / f"{name}.json"
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    return str(p)

def save_fig(fig: plt.Figure, basepath: Path, name: str, *, dpi: int = 144, transparent: bool = False) -> str:
    _ensure_dir(basepath)
    p = basepath / f"{name}.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight", transparent=transparent)
    return str(p)

def collect_library_versions() -> Dict[str, str]:
    """Collect versions of main libraries (best-effort)."""
    versions: Dict[str, str] = {}
    try:
        import numpy  # type: ignore
        versions["numpy"] = numpy.__version__
    except Exception:
        pass
    try:
        import pandas  # type: ignore
        versions["pandas"] = pandas.__version__
    except Exception:
        pass
    try:
        import sklearn  # type: ignore
        versions["scikit_learn"] = sklearn.__version__  # type: ignore
    except Exception:
        pass
    try:
        import statsmodels  # type: ignore
        versions["statsmodels"] = statsmodels.__version__  # type: ignore
    except Exception:
        pass
    try:
        import scipy  # type: ignore
        versions["scipy"] = scipy.__version__
    except Exception:
        pass
    try:
        versions["matplotlib"] = matplotlib.__version__
    except Exception:
        pass
    return versions

# ---------------------------- Insights Analyzer -----------------------------

DEFAULT_TOP_N = 5

class InsightsAnalyzer:
    """
    Generate high-level insights from an asset dataset (no I/O).
    Intended to be used by the runner and then saved to JSON.
    """

    def __init__(self, top_n: int = DEFAULT_TOP_N) -> None:
        if not isinstance(top_n, int) or top_n <= 0:
            raise ValueError("top_n must be a positive integer.")
        self.top_n = top_n

    def generate_value_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            logger.warning("Empty dataset: cannot generate insights.")
            return {}

        insights: Dict[str, Any] = {
            "top_assets": {},
            "worst_assets": {},
            "location_analysis": {},
            "correlations": {},
            "summary": {},
        }

        try:
            if VALUATION_K in df.columns:
                insights["top_assets"] = self._get_top_assets(df)
                insights["correlations"] = self._analyze_correlations(df)
                insights["summary"] = self._generate_summary(df)

            insights["worst_assets"] = self._get_worst_assets(df)

            if LOCATION in df.columns and VALUATION_K in df.columns:
                insights["location_analysis"] = self._analyze_locations(df)

            logger.info("Insights generation completed successfully.")
            return insights

        except Exception as e:
            logger.exception(f"Error while generating insights: {e}")
            return insights

    # --- helpers

    def _get_top_assets(self, df: pd.DataFrame) -> Dict[str, Any]:
        from notebooks.shared.common.constants import ASSET_ID, PRICE_PER_SQM, CONDITION_SCORE
        cols = [ASSET_ID, VALUATION_K, SIZE_M2, ENERGY_CLASS, CONDITION_SCORE, "condition_minus_risk"]
        available = [c for c in cols if c in df.columns]

        if not available:
            logger.warning("No columns available to identify top_assets.")
            return {}

        top_val = df.nlargest(self.top_n, VALUATION_K)[available]
        result = {
            "by_valuation": {
                "data": top_val.to_dict("records"),
                "mean_valuation": float(top_val[VALUATION_K].mean()),
                "mean_size": float(top_val[SIZE_M2].mean()) if SIZE_M2 in top_val else None,
            }
        }

        if PRICE_PER_SQM in df.columns:
            top_eff = df.nsmallest(self.top_n, PRICE_PER_SQM)[[ASSET_ID, PRICE_PER_SQM, VALUATION_K]]
            result["by_efficiency"] = {
                "data": top_eff.to_dict("records"),
                "mean_price_per_sqm": float(top_eff[PRICE_PER_SQM].mean()),
            }

        logger.info("Identified top %d assets by valuation/efficiency.", self.top_n)
        return result

    def _get_worst_assets(self, df: pd.DataFrame) -> Dict[str, Any]:
        from notebooks.shared.common.constants import ASSET_ID, CONDITION_SCORE, LUXURY_SCORE
        col = CONDITION_SCORE if CONDITION_SCORE in df else (
            "condition_minus_risk" if "condition_minus_risk" in df else None
        )

        if not col:
            logger.warning("No condition metric available.")
            return {}

        available = [c for c in [ASSET_ID, VALUATION_K, ENERGY_CLASS, col, LUXURY_SCORE] if c in df.columns]
        worst = df.nsmallest(self.top_n, col)[available]

        logger.info("Identified worst %d assets by metric %s.", self.top_n, col)
        return {
            "by_condition": {
                "data": worst.to_dict("records"),
                "condition_metric": col,
                "mean_condition": float(worst[col].mean()),
                "total_valuation_at_risk": float(worst[VALUATION_K].sum()) if VALUATION_K in worst else None,
            }
        }

    def _analyze_locations(self, df: pd.DataFrame) -> Dict[str, Any]:
        grp = df.groupby(LOCATION, observed=True)[VALUATION_K]

        return {
            "valuation": {
                "mean": grp.mean().sort_values(ascending=False).to_dict(),
                "median": grp.median().sort_values(ascending=False).to_dict(),
                "total": grp.sum().sort_values(ascending=False).to_dict(),
                "count": grp.size().to_dict(),
            }
        }

    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        num_df = df.select_dtypes(include=[np.number])
        if VALUATION_K not in num_df:
            logger.warning("VALUATION_K not found among numeric columns for correlation.")
            return {}

        pearson_corr = num_df.corr(method="pearson")[VALUATION_K].drop(labels=[VALUATION_K], errors="ignore")
        abs_sorted = pearson_corr.abs().sort_values(ascending=False)

        return {
            "pearson": {
                "top_positive": pearson_corr.nlargest(5).to_dict(),
                "top_negative": pearson_corr.nsmallest(5).to_dict(),
                "top_absolute": abs_sorted.head(5).to_dict(),
            }
        }

    def _generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        total_val = float(df[VALUATION_K].sum())
        return {
            "portfolio": {
                "total_value": total_val,
                "mean_value": float(df[VALUATION_K].mean()),
                "value_concentration": float(
                    df.nlargest(max(1, int(len(df) * 0.1)), VALUATION_K)[VALUATION_K].sum() / total_val
                ) if total_val else None,
            }
        }

# -------------------------------- Exporter ----------------------------------

@dataclass
class ReportExporter:
    output_dir: Path
    meta: Dict[str, Any] = field(default_factory=dict)
    dpi: int = 144

    def __post_init__(self) -> None:
        _ensure_dir(self.output_dir)

    def export_df(self, name: str, df: pd.DataFrame) -> Dict[str, str]:
        return save_dataframe(df, self.output_dir, name)

    def export_fig(self, name: str, fig: plt.Figure, *, transparent: bool = False) -> str:
        return save_fig(fig, self.output_dir, name, dpi=self.dpi, transparent=transparent)

    def write_manifest(self, manifest: Dict[str, Any], name: str = "eda_manifest") -> str:
        final = {"meta": self._meta_block(), "artifacts": manifest}
        return save_json(final, self.output_dir, name)
    
    def export_json(self, name: str, payload: Mapping[str, Any]) -> str:
        obj = {"meta": self._meta_block(), "data": payload}
        out_path = Path(self.output_dir) / f"{name}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(canonical_json_dumps(obj), encoding="utf-8")
        return str(out_path)

    def _meta_block(self) -> Dict[str, Any]:
        base = {
            "library_versions": collect_library_versions(),
        }
        base.update(self.meta or {})
        return base

# --------------------------------- Runner -----------------------------------

class EDAReportRunner:
    """
    EDA orchestrator: calls the core and uses ReportExporter to save artifacts.
    """

    def __init__(self, output_dir: str | Path = "outputs/analysis", meta: Optional[Dict[str, Any]] = None, dpi: int = 144) -> None:
        self.exporter = ReportExporter(Path(output_dir), meta or {}, dpi=dpi)
        # Reusable core analyzers
        self.descr = DescriptiveAnalyzer()
        self.tester = StatisticalTester()
        self.outlier = OutlierDetector()
        self.anom = AnomalyDetector()
        self.temp = TemporalAnalyzer()
        self.fimp = FeatureImportanceAnalyzer()

    # ---------- Independent sections (each returns artifacts + paths) --------

    def export_descriptive(self, df: pd.DataFrame) -> Dict[str, Any]:
        artifacts: Dict[str, Any] = {}

        # Main plots
        fig_rel = self.descr.create_relationship_plots(df)
        artifacts["relationship_plots"] = self.exporter.export_fig("relationships", fig_rel)
        plt.close(fig_rel)

        # Correlation heatmap on a sensible subset
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if VALUATION_K in num_cols:
            cols = [c for c in num_cols if c != VALUATION_K][:10]  # avoid huge heatmaps
            if cols:
                fig_corr = plot_correlation_heatmap(df, [VALUATION_K, *cols[:9]])
                artifacts["correlation_heatmap"] = self.exporter.export_fig("correlation_heatmap", fig_corr)
                plt.close(fig_corr)

        return artifacts

    def export_outliers(self, df: pd.DataFrame, *, columns: Optional[List[str]] = None, method: str = "iqr") -> Dict[str, Any]:
        # Configure method
        self.outlier.method = method.lower()
        summary = self.outlier.detect_outliers(df, columns=columns)
        combined = self.outlier.combine_outlier_results(df, summary)
        stats = self.outlier.get_outlier_summary_stats(df, summary, combined)

        artifacts = {
            "summary_json": self.exporter.export_json("outliers_summary", summary),
            "stats_json": self.exporter.export_json("outliers_stats", stats),
            "combined": self.exporter.export_df("outliers_combined", combined),
        }
        return artifacts

    def export_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        df_enriched, report = self.anom.detect_anomalies(df)
        frames = self.anom.extract_anomaly_frames(df_enriched)

        artifacts: Dict[str, Any] = {
            "anomaly_report": self.exporter.export_json("anomaly_report", report),
            "dataset_enriched": self.exporter.export_df("dataset_with_anomalies", df_enriched),
        }

        for label, frame in frames.items():
            artifacts[f"anomalies_{label}"] = self.exporter.export_df(f"anomalies_{label}", frame)

        # Optional figure
        fig = self.anom.visualize_anomalies(df_enriched)
        if fig is not None:
            artifacts["anomalies_plot"] = self.exporter.export_fig("anomalies_plot", fig)
            plt.close(fig)

        return artifacts

    def export_temporal(self, df: pd.DataFrame, *, target: str = VALUATION_K) -> Dict[str, Any]:
        df_time, report = self.temp.analyze(df, target=target)
        artifacts: Dict[str, Any] = {
            "temporal_report": self.exporter.export_json("temporal_report", report),
            "dataset_temporal": self.exporter.export_df("dataset_temporal", df_time),
        }

        fig = self.temp.plot(df_time, target=target if target in df_time else None)
        artifacts["temporal_plot"] = self.exporter.export_fig("temporal_plot", fig)
        plt.close(fig)
        return artifacts

    def export_statistics(
        self,
        df: pd.DataFrame,
        normality_features: Optional[List[str]] = None,
        categorical_pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        feats = None
        if normality_features:
            feats_present = [c for c in normality_features if c in df.columns]
            missing = sorted(set(normality_features) - set(feats_present))
            if missing:
                logger.info("Normality: skipping missing columns: %s", missing)
            feats = feats_present if feats_present else None

        results = self.tester.run_comprehensive_tests(
            df,
            normality_features=feats,
            categorical_pairs=categorical_pairs,
        )
        return {"stats_suite": self.exporter.export_json("statistical_suite", results)}

    def export_feature_importance(
        self,
        df: pd.DataFrame,
        *,
        exclude_features: Optional[List[str]] = None,
        include_proxies: bool = False,
        calc_permutation: bool = True,
        ablation_features: Optional[List[str]] = None,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        X, y, feature_cols = self.fimp.prepare_features(
            df,
            exclude_features=set(exclude_features) if exclude_features else None,
            include_proxies=include_proxies,
        )

        imps = self.fimp.calculate_importances(X, y, calculate_permutation=calc_permutation)
        artifacts: Dict[str, Any] = {}

        for k, df_imp in imps.items():
            artifacts[f"{k}_importance"] = self.exporter.export_df(f"feature_importance_{k}", df_imp)

        if ablation_features:
            abl = self.fimp.perform_ablation_study(X, y, ablation_features, cv_folds=cv_folds)
            artifacts["ablation_results"] = self.exporter.export_df("ablation_study_results", abl)

        # JSON with metadata of used features
        artifacts["feature_set"] = self.exporter.export_json("feature_set", {"features": feature_cols})

        return artifacts

    def export_insights(self, df: pd.DataFrame, *, top_n: int = DEFAULT_TOP_N) -> Dict[str, Any]:
        ins = InsightsAnalyzer(top_n=top_n)
        insights = ins.generate_value_insights(df)
        return {"insights_json": self.exporter.export_json("insights", insights)}

    # --------------------------- Full Orchestration --------------------------

    def run_full_eda(
        self,
        df: pd.DataFrame,
        *,
        include_sections: Optional[List[str]] = None,
        normality_features: Optional[List[str]] = None,
        categorical_pairs: Optional[List[Tuple[str, str]]] = None,
        feature_importance: bool = True,
        anomaly_detection: bool = True,
        outlier_detection: bool = True,
        temporal_analysis: bool = True,
        insights: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full EDA flow (selectable sections) and write a manifest with all artifact paths.
        """
        sections = include_sections or []
        manifest: Dict[str, Any] = {}

        # Descriptive + plots
        if not sections or "descriptive" in sections:
            manifest["descriptive"] = self.export_descriptive(df)

        # Outliers
        if (not sections or "outliers" in sections) and outlier_detection:
            manifest["outliers"] = self.export_outliers(df)

        # Anomalies
        if (not sections or "anomalies" in sections) and anomaly_detection:
            manifest["anomalies"] = self.export_anomalies(df)

        # Temporal
        if (not sections or "temporal" in sections) and temporal_analysis:
            manifest["temporal"] = self.export_temporal(df)

        # Statistics
        if not sections or "statistics" in sections:
            manifest["statistics"] = self.export_statistics(df, normality_features, categorical_pairs)

        # Feature importance
        if (not sections or "feature_importance" in sections) and feature_importance:
            manifest["feature_importance"] = self.export_feature_importance(df)

        # Business-friendly insights
        if (not sections or "insights" in sections) and insights:
            manifest["insights"] = self.export_insights(df)

        # Final manifest
        manifest_path = self.exporter.write_manifest(manifest)

        # Aliases consistent with notebooks
        manifest["manifest_path"] = manifest_path
        manifest["_manifest_file"] = manifest_path  # back-compat

        logger.info("EDA manifest written: %s", manifest_path)
        return manifest
