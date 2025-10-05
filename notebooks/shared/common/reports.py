from __future__ import annotations
"""
Unified Report toolkit.

Capabilities
- ReportManager: load/merge/save JSON (NumPy-safe, canonical JSON when desired).
- DistributionAnalyzer: categorical distribution analysis + optional benchmarks.
- run_sanity_checks: orchestration for benchmarks, drift, incoherence, outliers, caps, decomposition.
- build_basic_stats: re-export of shared.quality implementation.

Design
- Zero I/O in analytics helpers (return dicts/DFs; persistence lives in ReportManager).
- Stable, JSON-serializable outputs for downstream services.
- Soft dependencies: lazy/optional imports to avoid tight coupling to notebook-only code.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from shared.common.constants import (
    LOCATION,
    REGION,
    URBAN_TYPE,
    ZONE,
    ENERGY_CLASS,
    ENERGY_CLASSES,
    YEAR_BUILT,
    LAST_VERIFIED_TS,
    ORIENTATION,
    VIEW,
    HEATING,
    PRICE_PER_SQM,
    VALUATION_K,
    Cols,
)
from shared.common.utils import NumpyJSONEncoder, canonical_json_dumps

from shared.common.quality import (
    build_basic_stats as _build_basic_stats,
    generate_base_quality_report,
    enrich_quality_report,
    apply_price_caps,
    decompose_price_per_sqm,
    flag_strongly_incoherent,
    summarize_valuation_distribution_with_incoherence,
    get_top_outliers,
)

# Optional: helpers from dataset/model notebooks (loaded lazily below)
# - shared.n03_train_model.metrics: location_benchmark, compute_location_drift
# - shared.common.sanity_checks: price_benchmark, critical_city_order_check
# - shared.n01_generate_dataset.asset_factory: normalize_pricing_input

logger = logging.getLogger(__name__)

__all__ = [
    "ReportManager",
    "DistributionAnalyzer",
    "build_basic_stats",
    "run_sanity_checks",
]

# -----------------------------------------------------------------------------
# Lazy imports to avoid hard coupling to notebook-only modules
# -----------------------------------------------------------------------------
def _lazy_metrics():
    try:
        from shared.n03_train_model.metrics import (  # type: ignore
            location_benchmark,
            compute_location_drift,
        )
        return location_benchmark, compute_location_drift
    except Exception:
        return None, None


def _lazy_sanity_checks():
    try:
        from shared.common.sanity_checks import (  # type: ignore
            price_benchmark,
            critical_city_order_check,
        )
        return price_benchmark, critical_city_order_check
    except Exception:
        return None, None


def _lazy_normalize_pricing_input():
    try:
        from shared.n01_generate_dataset.asset_factory import normalize_pricing_input  # type: ignore
        return normalize_pricing_input
    except Exception:
        return None


# =============================================================================
# ReportManager: load / merge / save JSON
# =============================================================================
class ReportManager:
    """Minimal persistence manager for JSON reports (atomic enough for notebook workflows)."""

    def __init__(self, log_dir: Union[str, Path] = "../logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def load(self, filename: str = "quality_report.json") -> Dict[str, Any]:
        path = self.log_dir / filename
        if not path.exists():
            logger.warning("[REPORT] Report not found", extra={"path": str(path)})
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("[REPORT] Load error", extra={"path": str(path), "error": str(e)})
            return {}

    def merge(self, base: Dict[str, Any], new_data: Dict[str, Any], prefix: Optional[str] = None) -> Dict[str, Any]:
        """Shallow merge with optional namespacing."""
        if prefix:
            base.setdefault(prefix, {}).update(new_data)
        else:
            base.update(new_data)
        return base

    def save(
        self,
        report: Dict[str, Any],
        filename: str,
        use_numpy_encoder: bool = True,
        canonical: bool = True,
    ) -> bool:
        """
        Persist a report to JSON.

        Args:
            report: JSON-serializable dict (NumPy types allowed if use_numpy_encoder=True).
            filename: destination filename under log_dir.
            use_numpy_encoder: when not canonical, use NumpyJSONEncoder for NumPy types.
            canonical: if True, write canonical JSON (sorted keys, compact separators, UTF-8).
        """
        path = self.log_dir / filename
        try:
            if canonical:
                text = canonical_json_dumps(report)
            else:
                text = json.dumps(
                    report,
                    ensure_ascii=False,
                    indent=2,
                    cls=NumpyJSONEncoder if use_numpy_encoder else None,  # type: ignore[arg-type]
                )
            with path.open("w", encoding="utf-8") as f:
                f.write(text)
            logger.info("[REPORT] Report saved", extra={"path": str(path)})
            return True
        except Exception as e:
            logger.error("[REPORT] Save error", extra={"path": str(path), "error": str(e)})
            return False


# =============================================================================
# DistributionAnalyzer: categorical distributions + optional benchmarks
# =============================================================================
class DistributionAnalyzer:
    """Lightweight analyzer for categorical distributions on a DataFrame snapshot."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def analyze_location(
        self,
        target_weights: Optional[Dict[str, float]] = None,
        tolerance: float = 0.05,
    ) -> Dict[str, Any]:
        if LOCATION not in self.df:
            logger.warning("[DIST] Missing column", extra={"column": LOCATION})
            return {}

        counts = self.df[LOCATION].value_counts(dropna=True)
        if counts.empty:
            return {"counts": {}, "percentages": {}, "total": int(len(self.df)), "summary": {"n_unique": 0}}

        pct = (counts / len(self.df) * 100).round(1)

        result: Dict[str, Any] = {
            "counts": counts.to_dict(),
            "percentages": pct.to_dict(),
            "total": int(len(self.df)),
            "summary": {
                "n_unique": int(counts.size),
                "top_location": counts.idxmax(),
                "top_pct": float(pct.iloc[0]),
            },
        }

        # Optional benchmark/drift vs target distribution
        if target_weights:
            lb, _ = _lazy_metrics()
            if lb:
                try:
                    tw_sum = sum(target_weights.values())
                    normalized = {k: (v / tw_sum if tw_sum else 0.0) for k, v in target_weights.items()}
                    bench_df = lb(self.df, target_weights=normalized, tolerance=tolerance)
                    result["benchmark"] = {
                        "data": bench_df.reset_index().to_dict(orient="records"),
                        "drifted": bench_df.index[bench_df["drifted"]].tolist(),
                    }
                except Exception as e:
                    logger.info("location_benchmark skipped: %s", e)

        logger.info("[DIST] Location distribution", extra={"counts": counts.to_dict()})
        return result

    def analyze_categorical(
        self,
        column: str,
        expected: Optional[List[str]] = None,
        order: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if column not in self.df:
            logger.warning("[DIST] Missing column", extra={"column": column})
            return {}

        vc = self.df[column].value_counts(dropna=False)
        if order:
            vc = vc.reindex(order).fillna(0).astype(int)
        pct = (vc / len(self.df) * 100).round(1)

        invalid: List[str] = []
        if expected:
            mask = self.df[column].notna() & ~self.df[column].isin(expected)
            invalid = list(pd.Series(self.df.loc[mask, column]).drop_duplicates())

        return {
            "counts": vc.to_dict(),
            "percentages": pct.to_dict(),
            "missing": int(self.df[column].isna().sum()),
            "invalid": invalid,
            "summary": {
                "n_unique": int(vc.size),
                "mode": (self.df[column].mode().iloc[0] if not vc.empty else None),
            },
        }

    def analyze_energy_class(self) -> Dict[str, Any]:
        return self.analyze_categorical(
            ENERGY_CLASS,
            expected=list(ENERGY_CLASSES),
            order=list(ENERGY_CLASSES),
        )

    def all_distributions(
        self,
        columns: Optional[List[str]] = None,
        target_weights: Optional[Dict[str, float]] = None,
        tolerance: float = 0.05,
    ) -> Dict[str, Any]:
        if columns is None:
            columns = [LOCATION, ENERGY_CLASS, ZONE, URBAN_TYPE, REGION]

        out: Dict[str, Any] = {}
        for col in columns:
            if col == LOCATION:
                out[col] = self.analyze_location(target_weights, tolerance)
            elif col == ENERGY_CLASS:
                out[col] = self.analyze_energy_class()
            else:
                out[col] = self.analyze_categorical(col)
        return out


# =============================================================================
# Orchestration: sanity/quality (re-uses shared.quality APIs)
# =============================================================================
def build_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Re-export for backward compatibility."""
    return _build_basic_stats(df)


def _normalize_weights(raw: Dict[str, float] | Any, locations: List[str]) -> Dict[str, float]:
    """Normalize weights over present locations; fall back to uniform."""
    if not locations:
        return {}
    if not isinstance(raw, dict):
        logger.warning("location_weights is not a dict; using uniform weights.")
        return {loc: 1.0 / len(locations) for loc in locations}

    total = float(sum(float(raw.get(loc, 0.0)) for loc in locations))
    if total <= 0.0 or not np.isfinite(total):
        logger.warning("location_weights sum ≤ 0 or non-finite; using uniform weights.")
        return {loc: 1.0 / len(locations) for loc in locations}

    normalized = {loc: float(raw.get(loc, 0.0)) / total for loc in locations}
    s = sum(normalized.values())
    if not np.isclose(s, 1.0):
        normalized = {k: v / s for k, v in normalized.items()}
    return normalized


def run_sanity_checks(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Execute benchmarks, drift checks, incoherence flags, outliers, price caps, and a sample decomposition.
    Returns: (report_dict, df_enriched)
    """
    df = df.copy()

    # --- normalize weights & optional domain enforcement --------------------
    raw_location_weights = config.get("location_weights", {}) or {}
    locations = df[LOCATION].dropna().astype(str).unique().tolist() if LOCATION in df.columns else []
    normalized_location_weights = _normalize_weights(raw_location_weights, locations)

    # Optional domain enforcement (if the notebook helper exists)
    try:
        from shared.n03_train_model.preprocessing import enforce_categorical_domains  # type: ignore
    except Exception:
        enforce_categorical_domains = None  # type: ignore
    if enforce_categorical_domains:
        try:
            try:
                df = enforce_categorical_domains(df, normalized_location_weights)  # new API
            except TypeError:
                df = enforce_categorical_domains(df)  # backward-compat
        except Exception as e:
            logger.warning("enforce_categorical_domains failed: %s", e)

    # --- sanity benchmarks: location distribution & price medians -----------
    sb: Dict[str, Any] = {}
    location_tol = float(config.get("expected_profile", {}).get("location_distribution_tolerance", 0.05))

    lb, _ = _lazy_metrics()
    if lb:
        try:
            bench_df = lb(df, target_weights=normalized_location_weights, tolerance=location_tol)
            sb["location_distribution"] = bench_df.reset_index().to_dict(orient="records")
            logger.info("[BENCH] Location benchmark computed.")
        except Exception as e:
            logger.error("location_benchmark failed: %s", e)
            sb["location_distribution"] = []
    else:
        sb["location_distribution"] = []

    price_benchmark, critical_city_order_check = _lazy_sanity_checks()
    city_med = None
    if price_benchmark:
        try:
            city_med, zone_med = price_benchmark(df)
            sb["city_price_medians"] = city_med.to_dict()
            if zone_med is not None:
                sb["zone_price_medians"] = pd.DataFrame(zone_med).fillna(0).to_dict()
            logger.info("[BENCH] Price benchmark computed.")
        except Exception as e:
            logger.error("price_benchmark failed: %s", e)

    # Ordering alerts (optional)
    if critical_city_order_check and city_med is not None:
        try:
            city_order_cfg = config.get("city_ordering", {}) or {}
            min_ratio = float(city_order_cfg.get("min_ratio", 1.05))
            min_abs_diff = float(city_order_cfg.get("min_abs_diff", 50.0))
            require_both = bool(city_order_cfg.get("require_both", False))

            top_city_alerts = critical_city_order_check(
                city_med, min_ratio=min_ratio, min_abs_diff=min_abs_diff, require_both=require_both
            )
            failed_alerts = [a for a in top_city_alerts if not a.get("passes", False)]
            sb["top_city_alerts"] = top_city_alerts
            sb["failed_city_ordering"] = failed_alerts
        except Exception as e:
            logger.error("critical_city_order_check failed: %s", e)
            sb["top_city_alerts"] = []
            sb["failed_city_ordering"] = []
    else:
        sb["top_city_alerts"] = []
        sb["failed_city_ordering"] = []

    # Drift (best-effort)
    _, compute_location_drift = _lazy_metrics()
    if compute_location_drift:
        try:
            drift_info = compute_location_drift(df, target_weights=normalized_location_weights, tolerance=location_tol)
            df.attrs["location_drift_report"] = drift_info
            sb["location_drift"] = drift_info
        except Exception as e:
            logger.error("compute_location_drift failed: %s", e)

    # --- quality summary + incoherence -------------------------------------
    incoh_cfg = config.get("incoherence", {}) or {}
    val_q = float(incoh_cfg.get("val_threshold_quantile", 0.95))
    conf_thresh = float(incoh_cfg.get("confidence_thresh", 0.6))
    w = incoh_cfg.get("weights", {"condition": 0.5, "luxury": 0.3, "env": 0.2}) or {}
    w_cond = float(w.get("condition", 0.5))
    w_lux = float(w.get("luxury", 0.3))
    w_env = float(w.get("env", 0.2))

    summary = summarize_valuation_distribution_with_incoherence(
        df,
        val_threshold_quantile=val_q,
        confidence_thresh=conf_thresh,
        w_condition=w_cond,
        w_lux=w_lux,
        w_env=w_env,
    )
    sb.setdefault("valuation_summary", {}).update(summary)

    strong_mask, confidence_series = flag_strongly_incoherent(
        df,
        val_threshold_quantile=val_q,
        confidence_thresh=conf_thresh,
        w_condition=w_cond,
        w_lux=w_lux,
        w_env=w_env,
    )
    df["confidence_score"] = confidence_series
    df["strongly_incoherent"] = strong_mask
    sb.setdefault("incoherence", {})["strongly_incoherent"] = {
        "count": int(strong_mask.sum()),
        "fraction": float(pd.Series(strong_mask).mean()) if len(df) else 0.0,
        "thresholds": {"val_threshold_quantile": val_q, "confidence_thresh": conf_thresh},
        "weights": {"condition": w_cond, "luxury": w_lux, "env": w_env},
    }

    # --- outliers -----------------------------------------------------------
    try:
        top_outliers_df = get_top_outliers(df, n=30)
        sb["top_outliers"] = top_outliers_df.head(10).to_dict(orient="records")
    except Exception as e:
        logger.error("get_top_outliers failed: %s", e)
        sb["top_outliers"] = []

    # --- price caps ---------------------------------------------------------
    try:
        price_caps_cfg = config.get("price_caps", {}) or {}
        max_multiplier = float(price_caps_cfg.get("max_multiplier", 3.0))
        df_capped = apply_price_caps(df, config.get("city_base_prices", {}) or {}, max_multiplier=max_multiplier)
        violations = df_capped[df_capped.get(Cols.PRICE_PER_SQM_CAPPED + "_violated", False)]
        sb.setdefault("price_caps", {})["violations_count"] = int(len(violations))
        example_cols = [LOCATION, ZONE, PRICE_PER_SQM, Cols.PRICE_PER_SQM_CAPPED]
        present_cols = [c for c in example_cols if c in violations.columns]
        sb["price_caps"]["example_violations"] = (
            violations[present_cols].head(10).to_dict(orient="records") if not violations.empty else []
        )
    except Exception as e:
        logger.error("apply_price_caps failed: %s", e)
        sb.setdefault("price_caps", {}).setdefault("violations_count", 0)
        sb["price_caps"].setdefault("example_violations", [])

    # --- decomposition example (if outliers present) ------------------------
    try:
        normalize_pricing_input = _lazy_normalize_pricing_input()
        if "top_outliers_df" in locals() and not top_outliers_df.empty and normalize_pricing_input:
            ex = top_outliers_df.iloc[0]
            interim = {
                "location": ex.get(LOCATION),
                "zone": ex.get(ZONE),
                "year_built": ex.get(YEAR_BUILT),
                "is_top_floor": False,
                "is_ground_floor": False,
                "energy_class": ex.get(ENERGY_CLASS, "C"),
                "state": ex.get("condition", "good"),
                "has_balcony": bool(ex.get("has_balcony", False)),
                "has_garden": bool(ex.get("has_garden", False)),
                "has_garage": bool(ex.get("garage", False)),
                # Prefer month from timestamp if available
                "month": (pd.to_datetime(ex.get(LAST_VERIFIED_TS)).month if ex.get(LAST_VERIFIED_TS) else None),
                "view": ex.get(VIEW, ""),
                "orientation": ex.get(ORIENTATION, ""),
                "heating": ex.get(HEATING, ""),
            }
            decomp = decompose_price_per_sqm(
                interim,
                normalize_pricing_input(config),  # type: ignore[arg-type]
                seasonality=config.get("seasonality", {}) or {},
                city_base_prices=config.get("city_base_prices", {}) or {},
            )
            sb["decomposition_example"] = decomp
    except Exception as e:
        logger.error("decomposition example failed: %s", e)

    # --- standard report (base + enriched) ---------------------------------
    try:
        base_report = generate_base_quality_report(df)
        full_report = enrich_quality_report(df, base_report, config)
    except Exception as e:
        logger.warning("standard report build failed: %s", e)
        full_report = {}

    # Always include fresh basic stats and duplicates
    try:
        full_report["basic_stats"] = _build_basic_stats(df)
    except Exception as e:
        logger.warning("basic_stats build failed: %s", e)
        full_report["basic_stats"] = {}

    try:
        dup_mask = df.duplicated()
        dup_count = int(dup_mask.sum())
        examples = df.loc[dup_mask].head(5).to_dict(orient="records") if dup_count else []
        full_report["duplicates"] = {"total_duplicates": dup_count, "examples": examples}
    except Exception as e:
        logger.warning("duplicates build failed: %s", e)
        full_report["duplicates"] = {"total_duplicates": 0, "examples": []}

    full_report.setdefault("sanity_benchmarks", {}).update(sb)
    return full_report, df
