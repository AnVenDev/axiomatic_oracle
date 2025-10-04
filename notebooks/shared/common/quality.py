from __future__ import annotations
"""
Quality & valuation utilities:
- Decompose price_per_sqm via pricing.explain_price (transparent multiplier breakdown)
- Summary statistics and incoherence metrics
- Top outliers extraction
- Price caps (city/zone aware)
- API-compatible reports: generate_base_quality_report / enrich_quality_report

Design notes:
- Zero I/O: functions return dicts/DataFrames; persistence is handled by callers.
- Optional integrations (benchmark/drift/decomposition-example) are imported lazily.
"""

import logging
from typing import Any, Dict, List, Mapping, Tuple, Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from shared.common.constants import (
    LOCATION, ZONE, ORIENTATION, VIEW, HEATING, VALUATION_K,
    CONDITION_SCORE, ENV_SCORE, LUXURY_SCORE, PRICE_PER_SQM, Cols,
)
from shared.common.pricing import (
    explain_price,
    normalize_priors,
    DEFAULT_BASE_PRICE_FALLBACK,
)
from shared.common.utils import normalize_location_weights as _normalize_weights

logger = logging.getLogger(__name__)

__all__ = [
    # core quality utilities
    "decompose_price_per_sqm",
    "summarize_valuation_distribution",
    "compute_confidence_score",
    "flag_strongly_incoherent",
    "summarize_valuation_distribution_with_incoherence",
    "get_top_outliers",
    "apply_price_caps",
    # report API (compat)
    "build_basic_stats",
    "generate_base_quality_report",
    "enrich_quality_report",
]


# -----------------------------------------------------------------------------
# Lazy optional metrics (avoid hard coupling)
# -----------------------------------------------------------------------------
def _lazy_metrics():
    try:
        from shared.n03_train_model.metrics import (  # type: ignore
            compute_location_drift,
            location_benchmark,
        )
        return compute_location_drift, location_benchmark
    except Exception:
        return None, None


# -----------------------------------------------------------------------------
# Weights / defaults
# -----------------------------------------------------------------------------
W_CONDITION_DEF: float = 0.5
W_LUXURY_DEF: float = 0.3
W_ENV_DEF: float = 0.2


# -----------------------------------------------------------------------------
# Decomposition using pricing.explain_price
# -----------------------------------------------------------------------------
def decompose_price_per_sqm(
    interim: Mapping[str, Any],
    pricing_normalized: Mapping[str, Any],
    seasonality: Mapping[int, float],
    city_base_prices: Mapping[str, Mapping[str, float]],
) -> Dict[str, Any]:
    """
    Reconstruct price_per_sqm composition using `common.pricing.explain_price`.

    Args:
        interim: minimal row-like mapping; supports legacy 'month' → coerced to Cols.LISTING_MONTH.
        pricing_normalized: priors (pre-normalized or raw; normalization is idempotent).
        seasonality: {month → multiplier}.
        city_base_prices: {city → {zone → base_price}}.

    Returns:
        dict with keys: base, final_no_noise, multipliers, composed_multiplier.
    """
    if not isinstance(interim, Mapping):
        raise TypeError("`interim` must be a Mapping")
    if not isinstance(pricing_normalized, Mapping):
        raise TypeError("`pricing_normalized` must be a Mapping")

    interim_std = dict(interim)
    # Legacy → canonical month field
    if "month" in interim_std and Cols.LISTING_MONTH not in interim_std:
        interim_std[Cols.LISTING_MONTH] = interim_std.pop("month")

    priors = normalize_priors(pricing_normalized)

    decomp = explain_price(
        row=pd.Series(interim_std),
        priors=priors,
        seasonality=seasonality,
        city_base_prices=city_base_prices,
    )

    logger.debug(
        "Decomposition city=%s zone=%s | base=%.2f final=%.2f",
        interim_std.get(LOCATION),
        interim_std.get(ZONE),
        decomp.get("base", float("nan")),
        decomp.get("final_no_noise", float("nan")),
    )
    return decomp


# -----------------------------------------------------------------------------
# Summaries & incoherence
# -----------------------------------------------------------------------------
def summarize_valuation_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Quantiles and descriptive stats for valuation_k.
    """
    if VALUATION_K not in df.columns:
        raise KeyError(f"{VALUATION_K} missing from dataframe for summary.")

    series = pd.to_numeric(df[VALUATION_K], errors="coerce")
    q = series.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "n_rows": int(series.shape[0]),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
        "quantiles": {str(k): float(v) for k, v in q.to_dict().items()},
        "skew": float(series.skew()),
        "kurtosis": float(series.kurt()),
    }


def compute_confidence_score(
    df: pd.DataFrame,
    w_condition: float = W_CONDITION_DEF,
    w_luxury: float = W_LUXURY_DEF,
    w_env: float = W_ENV_DEF,
) -> pd.Series:
    """
    Composite confidence score combining condition, luxury, env.
    Returns a float32 Series aligned to df.
    """
    cond = pd.to_numeric(df.get(CONDITION_SCORE, 0.0), errors="coerce").fillna(0.0)
    lux = pd.to_numeric(df.get(LUXURY_SCORE, 0.0), errors="coerce").fillna(0.0)
    env = pd.to_numeric(df.get(ENV_SCORE, 0.0), errors="coerce").fillna(0.0)
    return (w_condition * cond + w_luxury * lux + w_env * env).astype("float32")


def flag_strongly_incoherent(
    df: pd.DataFrame,
    val_threshold_quantile: float = 0.95,
    confidence_thresh: float = 0.6,
    w_condition: float = W_CONDITION_DEF,
    w_luxury: float = W_LUXURY_DEF,
    w_env: float = W_ENV_DEF,
) -> Tuple[pd.Series, pd.Series]:
    """
    Flag high-valuation assets (above quantile) with low composite confidence.

    Returns:
        (boolean mask Series, confidence Series)
    """
    if VALUATION_K not in df.columns:
        raise KeyError(f"{VALUATION_K} missing from dataframe for incoherence flagging.")

    series_val = pd.to_numeric(df[VALUATION_K], errors="coerce")
    threshold = float(series_val.quantile(val_threshold_quantile))
    confidence = compute_confidence_score(df, w_condition=w_condition, w_luxury=w_luxury, w_env=w_env)
    mask = (series_val > threshold) & (confidence < confidence_thresh)
    return mask.astype(bool), confidence


def summarize_valuation_distribution_with_incoherence(
    df: pd.DataFrame,
    val_threshold_quantile: float = 0.95,
    confidence_thresh: float = 0.6,
    w_condition: float = W_CONDITION_DEF,
    w_luxury: float = W_LUXURY_DEF,
    w_env: float = W_ENV_DEF,
) -> Dict[str, Any]:
    """
    Extend summary with strongly-incoherent counts and fractions.
    """
    summary = summarize_valuation_distribution(df)
    incoherent_mask, _ = flag_strongly_incoherent(
        df,
        val_threshold_quantile=val_threshold_quantile,
        confidence_thresh=confidence_thresh,
        w_condition=w_condition,
        w_luxury=w_luxury,
        w_env=w_env,
    )
    n = int(incoherent_mask.sum())
    summary["strongly_incoherent"] = {
        "count": n,
        "fraction": (n / len(df)) if len(df) else 0.0,
        "confidence_thresh": float(confidence_thresh),
        "val_threshold_quantile": float(val_threshold_quantile),
        "weights": {
            "condition": float(w_condition),
            "luxury": float(w_luxury),
            "env": float(w_env),
        },
    }
    return summary


# -----------------------------------------------------------------------------
# Outliers
# -----------------------------------------------------------------------------
def get_top_outliers(
    df: pd.DataFrame,
    n: int = 20,
    extra_fields: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Return the top-n rows with highest valuation_k, keeping only existing columns.
    Does not mutate the input frame.
    """
    if VALUATION_K not in df.columns or df.empty or n <= 0:
        return df.head(0).copy()

    if extra_fields is None:
        extra_fields = [
            LOCATION, ZONE, Cols.SIZE_M2, PRICE_PER_SQM, VALUATION_K,
            LUXURY_SCORE, ENV_SCORE, CONDITION_SCORE, Cols.RISK_SCORE,
            VIEW, ORIENTATION, Cols.YEAR_BUILT, HEATING, Cols.CONDITION,
        ]

    available = [c for c in extra_fields if c in df.columns]
    top = df.sort_values(VALUATION_K, ascending=False).head(int(n))
    logger.debug("Top outliers extracted: %d rows (n=%d)", len(top), n)
    return top.loc[:, available].copy()


# -----------------------------------------------------------------------------
# Price caps
# -----------------------------------------------------------------------------
def apply_price_caps(
    df: pd.DataFrame,
    city_base_prices: Mapping[str, Mapping[str, float]],
    max_multiplier: float = 3.0,
    price_col: str = PRICE_PER_SQM,
    capped_col: str = Cols.PRICE_PER_SQM_CAPPED,
) -> pd.DataFrame:
    """
    Cap price_per_sqm against (base price * max_multiplier) per (location, zone).

    Adds:
      - `capped_col`: capped value (float32)
      - `f"{capped_col}_violated"`: bool flag where price > cap

    Returns a copy of the DataFrame with derived columns.
    """
    if price_col not in df.columns:
        out = df.copy()
        out[capped_col] = np.nan
        out[f"{capped_col}_violated"] = False
        return out

    out = df.copy()

    # Base price per row via mapping (location, zone)
    loc = out.get(LOCATION, pd.Series("", index=out.index)).astype(str)
    zon = out.get(ZONE, pd.Series("", index=out.index)).astype(str)

    def _lookup_base(c: str, z: str) -> float:
        return float(city_base_prices.get(c, {}).get(z, DEFAULT_BASE_PRICE_FALLBACK))

    base_arr = np.fromiter((_lookup_base(c, z) for c, z in zip(loc, zon)), dtype=float, count=len(out))
    cap_arr = base_arr * float(max_multiplier)

    price = pd.to_numeric(out[price_col], errors="coerce").to_numpy(dtype=float)
    capped = np.fmin(price, cap_arr)

    out[capped_col] = capped.astype("float32")
    out[f"{capped_col}_violated"] = price > capped

    n_viol = int(out[f"{capped_col}_violated"].sum())
    logger.info(
        "Price caps: %d violations over %d rows (max_multiplier=%.2f, cap[min]=%.2f, cap[max]=%.2f).",
        n_viol, len(out), max_multiplier, float(cap_arr.min()), float(cap_arr.max()),
    )

    return out


# -----------------------------------------------------------------------------
# Report API (backward-compatible)
# -----------------------------------------------------------------------------
def build_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Basic stats (shape, dtypes, missing ratios, describe) + target summary (if present).
    """
    numeric = df.select_dtypes(include=[np.number])
    categorical = df.select_dtypes(exclude=[np.number])
    out: Dict[str, Any] = {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_pct": {c: float(pd.isna(df[c]).mean()) for c in df.columns},
        "describe_numeric": numeric.describe().to_dict() if not numeric.empty else {},
        "describe_categorical": categorical.describe().to_dict() if not categorical.empty else {},
    }
    if VALUATION_K in df.columns:
        series = pd.to_numeric(df[VALUATION_K], errors="coerce")
        out["target_summary"] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
        }
    else:
        out["target_summary"] = {}
    return out


def generate_base_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Minimal, stable, JSON-serializable report.
    """
    report: Dict[str, Any] = {
        "basic_stats": build_basic_stats(df),
    }

    # Append valuation summary if present
    if VALUATION_K in df.columns:
        try:
            report["valuation_summary"] = summarize_valuation_distribution(df)
        except Exception as e:
            logger.warning("Unable to compute valuation summary: %s", e)
            report["valuation_summary"] = {}

    # Duplicates example set (best-effort)
    try:
        dup_mask = df.duplicated()
        dup_count = int(dup_mask.sum())
        examples = df.loc[dup_mask].head(5).to_dict(orient="records") if dup_count else []
        report["duplicates"] = {"total_duplicates": dup_count, "examples": examples}
    except Exception as e:
        logger.warning("Unable to compute duplicates: %s", e)
        report["duplicates"] = {"total_duplicates": 0, "examples": []}

    return report


def enrich_quality_report(
    df: pd.DataFrame,
    base_report: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Enrich a base report with:
      - Incoherence metrics
      - Top outliers
      - Price caps (if city_base_prices provided)
      - Optional benchmarks/drift (lazy imports)

    Non-raising: optional blocks fail with warnings, returning partial data.
    """
    rpt = dict(base_report or generate_base_quality_report(df))
    cfg = config or {}

    # --- incoherence --------------------------------------------------------
    incoh_cfg = cfg.get("incoherence", {}) or {}
    val_q = float(incoh_cfg.get("val_threshold_quantile", 0.95))
    conf_thresh = float(incoh_cfg.get("confidence_thresh", 0.6))
    weights = incoh_cfg.get("weights", {"condition": W_CONDITION_DEF, "luxury": W_LUXURY_DEF, "env": W_ENV_DEF}) or {}
    w_cond = float(weights.get("condition", W_CONDITION_DEF))
    w_lux = float(weights.get("luxury", W_LUXURY_DEF))
    w_env = float(weights.get("env", W_ENV_DEF))

    try:
        summary = summarize_valuation_distribution_with_incoherence(
            df,
            val_threshold_quantile=val_q,
            confidence_thresh=conf_thresh,
            w_condition=w_cond,
            w_lux=w_lux,
            w_env=w_env,
        )
        rpt.setdefault("valuation_summary", {}).update(summary)

        strong_mask, _ = flag_strongly_incoherent(
            df,
            val_threshold_quantile=val_q,
            confidence_thresh=conf_thresh,
            w_condition=w_cond,
            w_lux=w_lux,
            w_env=w_env,
        )
        rpt.setdefault("incoherence", {})["strongly_incoherent"] = {
            "count": int(strong_mask.sum()),
            "fraction": float(pd.Series(strong_mask).mean()) if len(df) else 0.0,
            "thresholds": {"val_threshold_quantile": val_q, "confidence_thresh": conf_thresh},
            "weights": {"condition": w_cond, "luxury": w_lux, "env": w_env},
        }
    except Exception as e:
        logger.warning("Incoherence enrich failed: %s", e)

    # --- outliers -----------------------------------------------------------
    try:
        top_outliers_df = get_top_outliers(df, n=30)
        rpt["top_outliers"] = top_outliers_df.head(10).to_dict(orient="records")
    except Exception as e:
        logger.warning("Top outliers enrich failed: %s", e)
        rpt["top_outliers"] = []

    # --- price caps (optional) ----------------------------------------------
    try:
        price_caps_cfg = cfg.get("price_caps", {}) or {}
        max_multiplier = float(price_caps_cfg.get("max_multiplier", 3.0))
        city_base_prices = cfg.get("city_base_prices", {}) or {}
        if city_base_prices and PRICE_PER_SQM in df.columns:
            df_cap = apply_price_caps(df, city_base_prices, max_multiplier=max_multiplier)
            violations = df_cap[df_cap.get(Cols.PRICE_PER_SQM_CAPPED + "_violated", False)]
            example_cols = [LOCATION, ZONE, PRICE_PER_SQM, Cols.PRICE_PER_SQM_CAPPED]
            present_cols = [c for c in example_cols if c in violations.columns]
            rpt["price_caps"] = {
                "violations_count": int(len(violations)),
                "example_violations": (
                    violations[present_cols].head(10).to_dict(orient="records") if not violations.empty else []
                ),
            }
    except Exception as e:
        logger.warning("Price caps enrich failed: %s", e)

    # --- optional: location benchmark & drift (lazy) ------------------------
    try:
        target_weights = cfg.get("location_weights", {}) or {}
        if LOCATION in df.columns and target_weights:
            tol = float(cfg.get("expected_profile", {}).get("location_distribution_tolerance", 0.05))
            compute_location_drift, location_benchmark = _lazy_metrics()
            if compute_location_drift is None or location_benchmark is None:
                # Nothing to add; keep report consistent
                pass
            else:
                try:
                    bench_df = location_benchmark(df, target_weights=_normalize_weights(target_weights), tolerance=tol)
                    rpt.setdefault("sanity_benchmarks", {})["location_distribution"] = (
                        bench_df.reset_index().to_dict(orient="records")
                        if hasattr(bench_df, "reset_index")
                        else pd.DataFrame.from_dict(bench_df).reset_index().to_dict(orient="records")
                    )
                except Exception as e:
                    logger.info("location_benchmark skipped: %s", e)

                try:
                    drift = compute_location_drift(df, target_weights=_normalize_weights(target_weights), tolerance=tol)
                    rpt.setdefault("sanity_benchmarks", {})["location_drift"] = drift
                except Exception as e:
                    logger.info("compute_location_drift skipped: %s", e)
    except Exception:
        pass

    # --- optional: decomposition example ------------------------------------
    try:
        if "city_base_prices" in cfg:
            sample = df.sort_values(VALUATION_K, ascending=False).head(1)
            if not sample.empty:
                row = sample.iloc[0]
                interim = {
                    "location": row.get(LOCATION),
                    "zone": row.get(ZONE),
                    "year_built": row.get(Cols.YEAR_BUILT, None),
                    "is_top_floor": bool(row.get(Cols.IS_TOP_FLOOR, False)),
                    "is_ground_floor": bool(row.get(Cols.IS_GROUND_FLOOR, False)),
                    "energy_class": row.get(Cols.ENERGY_CLASS, "C"),
                    "state": row.get(Cols.CONDITION, "good"),
                    "has_balcony": bool(row.get(Cols.HAS_BALCONY, False)),
                    "has_garden": bool(row.get(Cols.HAS_GARDEN, False)),
                    "has_garage": bool(row.get(Cols.GARAGE, False)),
                    Cols.LISTING_MONTH: (
                        int(row.get(Cols.LISTING_MONTH)) if pd.notna(row.get(Cols.LISTING_MONTH)) else None
                    ),
                    "view": row.get(VIEW, ""),
                    "orientation": row.get(ORIENTATION, ""),
                    "heating": row.get(HEATING, ""),
                }
                decomp = decompose_price_per_sqm(
                    interim,
                    normalize_priors(cfg.get("pricing", {}) or cfg),
                    seasonality=cfg.get("seasonality", {}) or {},
                    city_base_prices=cfg.get("city_base_prices", {}) or {},
                )
                rpt.setdefault("sanity_benchmarks", {})["decomposition_example"] = decomp
    except Exception as e:
        logger.info("Decomposition example skipped: %s", e)

    return rpt
