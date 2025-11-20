from __future__ import annotations

"""
Lightweight sklearn-compatible transformers for serving-time preprocessing.

Components
- GeoCanonizer: canonicalizes minimal geo columns ({city, zone, region}) with safe defaults.
- PriorsGuard: fills/repairs 'city_zone_prior' and 'region_index_prior' with robust fallbacks.
- EnsureDerivedFeatures: recomputes ONLY missing simple derived features, idempotently.

Design goals
- Zero side-effects, clone-safe, no recursion.
- No dependency on training-time feature builders (no PropertyDerivedFeatures).
- Best-effort conversions; never raise for optional enrichments.
- Compatible with pandas DataFrame and array/dict-like inputs per sklearn contract.
"""

from typing import Any, Dict, Optional, Tuple, List
import logging
from datetime import datetime

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore

__all__ = ["GeoCanonizer", "PriorsGuard", "EnsureDerivedFeatures"]

logger = logging.getLogger(__name__)

# Robust constants import (modern first, legacy fallback)
try:  # pragma: no cover
    from shared.common.constants import EXPECTED_PRICE_PER_SQM_EUR_RANGE
except Exception:  # pragma: no cover
    from notebooks.shared.common.constants import EXPECTED_PRICE_PER_SQM_EUR_RANGE


# ----------------------------------------------------------------------------- 
# Helpers
# -----------------------------------------------------------------------------
def _to_dataframe(X: Any) -> pd.DataFrame:
    """Ensure we always work with a DataFrame (sklearn contract: accept array/dict-like)."""
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X)


def _lower_strip_safe(s: pd.Series) -> pd.Series:
    """Lower/strip strings while preserving NA semantics (avoid 'nan' literals)."""
    return s.astype("string").str.strip().str.lower().astype("object")


def _coerce_bool01(series: pd.Series) -> pd.Series:
    """Map truthy to {0,1} robustly, preserving NaNs."""
    try:
        return series.apply(lambda v: np.nan if pd.isna(v) else int(bool(v))).astype("float64")
    except Exception:
        return pd.to_numeric(series, errors="coerce").astype("float64")


# ----------------------------------------------------------------------------- 
# GeoCanonizer
# -----------------------------------------------------------------------------
class GeoCanonizer(BaseEstimator, TransformerMixin):
    """
    Canonicalize geography fields with safe defaults.

    Behavior
    - Create 'city' from 'location' if missing.
    - Ensure 'zone' exists (default: 'semi_center').
    - Ensure 'region' exists (np.nan).
    - Lowercase/trim {city, zone, region} without stringifying NaNs.
    """

    def fit(self, X: Any, y: Optional[pd.Series] = None) -> "GeoCanonizer":
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        df = _to_dataframe(X).copy()

        # Create missing geo columns
        if "city" not in df.columns and "location" in df.columns:
            df["city"] = df["location"]
        if "zone" not in df.columns:
            df["zone"] = "semi_center"
        if "region" not in df.columns:
            df["region"] = pd.Series(pd.NA, index=df.index, dtype="object")

        # Normalize safely
        for col in ("city", "zone", "region"):
            if col in df.columns:
                try:
                    df[col] = _lower_strip_safe(df[col])
                except Exception as e:
                    logger.debug("GeoCanonizer: skip normalize for %s: %s", col, e)

        return df


# ----------------------------------------------------------------------------- 
# PriorsGuard
# -----------------------------------------------------------------------------
class PriorsGuard(BaseEstimator, TransformerMixin):
    """
    Populate/repair prior features with robust fallbacks.

    Produces/repairs
    - 'city_zone_prior' (float64)
    - 'region_index_prior' (float64)

    Resolution order for city_zone_prior:
      1) keep existing numeric value if present
      2) city_base[city][zone] if available
      3) zone_medians[zone] if available
      4) global_cityzone_median

    Resolution for region_index_prior:
      map(normalized region) using `region_index`, otherwise NaN.
    """

    def __init__(
        self,
        city_base: Optional[Dict[str, Dict[str, float]]] = None,
        region_index: Optional[Dict[str, float]] = None,
        zone_medians: Optional[Dict[str, float]] = None,
        global_cityzone_median: float = 0.0,
        repair_out_of_range: bool = True,
        cz_range: Tuple[float, float] = EXPECTED_PRICE_PER_SQM_EUR_RANGE,
    ) -> None:
        # Normalize dictionaries once (keys lower/strip, values -> float)
        def _norm_nested(d: Optional[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
            out: Dict[str, Dict[str, float]] = {}
            for c, zones in (d or {}).items():
                c_norm = str(c).strip().lower()
                out[c_norm] = {str(z).strip().lower(): float(v) for z, v in (zones or {}).items()}
            return out

        def _norm_flat(d: Optional[Dict[str, float]]) -> Dict[str, float]:
            return {str(k).strip().lower(): float(v) for k, v in (d or {}).items()}

        self.city_base = _norm_nested(city_base)
        self.region_index = _norm_flat(region_index) or {"north": 1.05, "center": 1.00, "south": 0.92}
        self.zone_medians = _norm_flat(zone_medians)
        self.global_cityzone_median = float(global_cityzone_median)
        self.repair_out_of_range = bool(repair_out_of_range)
        self.cz_range: Tuple[float, float] = (float(cz_range[0]), float(cz_range[1]))

    def fit(self, X: Any, y: Optional[pd.Series] = None) -> "PriorsGuard":
        return self

    # ---- internals ---------------------------------------------------------
    def _lookup_city_zone_prior(self, city: Any, zone: Any) -> float:
        # Respect NaNs
        if pd.isna(city) or pd.isna(zone):
            key = zone if isinstance(zone, str) else str(zone)
            return float(self.zone_medians.get(str(key).strip().lower(), self.global_cityzone_median))
        c = str(city).strip().lower()
        z = str(zone).strip().lower()
        val = self.city_base.get(c, {}).get(z, np.nan)
        if pd.isna(val):
            return float(self.zone_medians.get(z, self.global_cityzone_median))
        return float(val)

    def _ensure_geo_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "city" not in out and "location" in out:
            out["city"] = out["location"]
        if "zone" not in out:
            out["zone"] = "semi_center"
        if "region" not in out:
            out["region"] = pd.Series(pd.NA, index=out.index, dtype="object")
        for c in ("city", "zone", "region"):
            try:
                out[c] = _lower_strip_safe(out[c])
            except Exception:
                pass
        return out

    # ---- transform ---------------------------------------------------------
    def transform(self, X: Any) -> pd.DataFrame:
        df = _to_dataframe(X).copy()
        df = self._ensure_geo_columns(df)

        # --- city_zone_prior ------------------------------------------------
        existing_cz = pd.to_numeric(df.get("city_zone_prior", np.nan), errors="coerce")

        if "city_zone_prior" not in df.columns:
            df["city_zone_prior"] = existing_cz  # initialize column

        need_cz = existing_cz.isna()
        if self.repair_out_of_range:
            lo, hi = self.cz_range
            bad = ~(existing_cz.between(lo, hi))  # False for NaN already
            if bad.any():
                need_cz = need_cz | bad
                df.loc[bad, "city_zone_prior"] = np.nan

        if need_cz.any():
            try:
                filled = np.fromiter(
                    (self._lookup_city_zone_prior(c, z) for c, z in zip(df.loc[need_cz, "city"], df.loc[need_cz, "zone"])),
                    dtype="float64",
                    count=int(need_cz.sum()),
                )
                df.loc[need_cz, "city_zone_prior"] = filled
            except Exception as e:
                logger.warning("PriorsGuard: city_zone_prior fill failed: %s", e)

        df["city_zone_prior"] = pd.to_numeric(df["city_zone_prior"], errors="coerce").astype("float64")

        # --- region_index_prior --------------------------------------------
        existing_rip = pd.to_numeric(df.get("region_index_prior", np.nan), errors="coerce")
        if "region_index_prior" not in df.columns:
            df["region_index_prior"] = existing_rip  # initialize column

        need_rip = existing_rip.isna()
        if need_rip.any():
            try:
                mapped = df.loc[need_rip, "region"].map(self.region_index)
                df.loc[need_rip, "region_index_prior"] = pd.to_numeric(mapped, errors="coerce")
            except Exception as e:
                logger.warning("PriorsGuard: region_index_prior fill failed: %s", e)

        df["region_index_prior"] = pd.to_numeric(df["region_index_prior"], errors="coerce").astype("float64")

        return df


# ----------------------------------------------------------------------------- 
# EnsureDerivedFeatures (SAFE, autonomous, idempotent)
# -----------------------------------------------------------------------------
class EnsureDerivedFeatures(BaseEstimator, TransformerMixin):
    """
    Serving-time guard that (re)computes ONLY the missing simple derived columns.
    Idempotent: if a column already exists, it is left untouched.

    NOTE: This implementation is **self-contained** and does NOT import or call
    PropertyDerivedFeatures to avoid recursion/loops in saved pipelines.

    Extra kwargs:
    - region_index: optional mapping used by some pipelines; stored on self for
      compatibility but not required by the core logic.
    - city_base: optional mapping (e.g. city priors); stored on self for
      compatibility but not required by the core logic.
    """

    def __init__(
        self,
        required_cols: Optional[List[str]] = None,
        zone_thresholds_km: Optional[Dict[str, float]] = None,  # optional helper for zone by distance
        region_index: Optional[Dict[str, float]] = None,
        city_base: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        # if None -> compute a standard minimal set
        self.required_cols = required_cols
        self.zone_thresholds_km = zone_thresholds_km or {"center": 1.5, "semi_center": 5.0}

        # extra, for compatibility with training/serving pipelines
        self.region_index = region_index or {}
        self.city_base = city_base or {}

    def fit(self, X: Any, y: Optional[pd.Series] = None) -> "EnsureDerivedFeatures":
        return self

    def _derive_minimal(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # ---- geo conveniences (do NOT lowercase here; GeoCanonizer already does)
        if "city" not in out.columns and "location" in out.columns:
            out["city"] = out["location"]

        if "zone" not in out.columns:
            # try distance-based if available, else semi_center
            if "distance_to_center_km" in out.columns:
                try:
                    th = self.zone_thresholds_km
                    d = pd.to_numeric(out["distance_to_center_km"], errors="coerce")
                    zone = pd.Series("periphery", index=out.index, dtype="object")
                    zone = np.where(d <= float(th.get("center", 1.5)), "center", zone)
                    zone = np.where(
                        (d > float(th.get("center", 1.5)))
                        & (d <= float(th.get("semi_center", 5.0))),
                        "semi_center",
                        zone,
                    )
                    out["zone"] = pd.Series(zone, index=out.index, dtype="object")
                except Exception:
                    out["zone"] = "semi_center"
            else:
                out["zone"] = "semi_center"

        if "region" not in out.columns:
            out["region"] = pd.Series(pd.NA, index=out.index, dtype="object")

        # ---- simple derived (no validators, no externals)
        # age_years
        if "age_years" not in out.columns:
            try:
                yb = pd.to_numeric(
                    out.get("year_built", pd.Series(np.nan, index=out.index)),
                    errors="coerce",
                )
                out["age_years"] = (datetime.utcnow().year - yb).clip(lower=0)
            except Exception:
                out["age_years"] = np.nan

        # luxury_score = mean(has_garden, has_balcony, garage/has_garage)
        if "luxury_score" not in out.columns:
            g = _coerce_bool01(out.get("has_garden", pd.Series(np.nan, index=out.index)))
            b = _coerce_bool01(out.get("has_balcony", pd.Series(np.nan, index=out.index)))
            # prefer canonical 'garage' if present; else 'has_garage'
            if "garage" in out.columns:
                ga = _coerce_bool01(out["garage"])
            else:
                ga = _coerce_bool01(
                    out.get("has_garage", pd.Series(np.nan, index=out.index))
                )
            out["luxury_score"] = (g.fillna(0) + b.fillna(0) + ga.fillna(0)) / 3.0

        # env_score = (aq/100) * (1 - noise/100), clipped [0,1]
        if "env_score" not in out.columns:
            try:
                aq = pd.to_numeric(
                    out.get("air_quality_index", pd.Series(0, index=out.index)),
                    errors="coerce",
                ).clip(lower=0)
                nz = pd.to_numeric(
                    out.get("noise_level", pd.Series(0, index=out.index)),
                    errors="coerce",
                ).clip(lower=0)
                out["env_score"] = np.clip(
                    (aq / 100.0) * (1.0 - nz / 100.0), 0.0, 1.0
                )
            except Exception:
                out["env_score"] = np.nan

        # is_top_floor
        if "is_top_floor" not in out.columns:
            try:
                fl = pd.to_numeric(
                    out.get("floor", pd.Series(np.nan, index=out.index)),
                    errors="coerce",
                )
                bf = pd.to_numeric(
                    out.get("building_floors", pd.Series(np.nan, index=out.index)),
                    errors="coerce",
                )
                out["is_top_floor"] = (fl == bf).astype("float64")
            except Exception:
                out["is_top_floor"] = np.nan

        # listing_month
        if "listing_month" not in out.columns:
            try:
                out["listing_month"] = int(datetime.utcnow().month)
            except Exception:
                out["listing_month"] = np.nan

        # normalize a few boolean-ish commonly used downstream
        for k in (
            "public_transport_nearby",
            "has_elevator",
            "has_garden",
            "has_balcony",
            "has_garage",
            "garage",
        ):
            if k in out.columns:
                out[k] = _coerce_bool01(out[k])

        return out

    def transform(self, X: Any) -> pd.DataFrame:
        df = _to_dataframe(X).copy()

        # compute minimal derived set once
        derived = self._derive_minimal(df)

        # Decide which columns to enforce
        if self.required_cols is None:
            # enforce only the new columns (idempotent add)
            cols_to_add = [c for c in derived.columns if c not in df.columns]
        else:
            cols_to_add = [
                c
                for c in self.required_cols
                if c in derived.columns and c not in df.columns
            ]

        for c in cols_to_add:
            try:
                df[c] = derived[c]
            except Exception as e:
                logger.debug("EnsureDerivedFeatures: failed to add %s: %s", c, e)

        return df