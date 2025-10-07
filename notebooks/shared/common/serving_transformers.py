from __future__ import annotations

from notebooks.shared.common.constants import EXPECTED_PRICE_PER_SQM_EUR_RANGE
"""
Lightweight sklearn-compatible transformers for serving-time preprocessing.

Components
- GeoCanonizer: canonicalizes minimal geo columns ({city, zone, region}) with safe defaults.
- PriorsGuard: fills/repairs 'city_zone_prior' and 'region_index_prior' with robust fallbacks.

Design
- Side-effect free, clone-safe: all mutable inputs are copied/normalized in __init__.
- Best-effort conversions; never raise for optional enrichments.
- Compatible with pandas DataFrame and array/dict-like inputs per sklearn contract.
"""

from typing import Any, Dict, Optional, Tuple
import logging

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore

__all__ = ["GeoCanonizer", "PriorsGuard", "EnsureDerivedFeatures"]

logger = logging.getLogger(__name__)

# Compute-once import to avoid circulars at module import time
try:  # pragma: no cover
    from shared.common.transformers import PropertyDerivedFeatures
except Exception:  # pragma: no cover
    PropertyDerivedFeatures = None  # type: ignore


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
            # cannot resolve city; try zone median then global
            key = zone if isinstance(zone, str) else str(zone)
            return float(self.zone_medians.get(str(key).strip().lower(), self.global_cityzone_median))
        c = str(city).strip().lower()
        z = str(zone).strip().lower()
        # city-specific price if present
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
# EnsureDerivedFeatures
# -----------------------------------------------------------------------------
class EnsureDerivedFeatures(BaseEstimator, TransformerMixin):
    """
    Serving-time guard that (re)computes **only** the missing derived columns
    using PropertyDerivedFeatures. Idempotente: se una colonna esiste, non la tocca.

    Use it when the saved pipeline does NOT embed feature engineering,
    or as a safety net if you cannot guarantee upstream derivations.
    """

    def __init__(
        self,
        city_base: Optional[Dict[str, Dict[str, float]]] = None,
        region_index: Optional[Dict[str, float]] = None,
        required_cols: Optional[list[str]] = None,
    ) -> None:
        self.city_base = city_base or {}
        self.region_index = region_index or {}
        # If None, we compute the full set produced by PropertyDerivedFeatures
        self.required_cols = required_cols

        if PropertyDerivedFeatures is None:
            logger.warning("EnsureDerivedFeatures: PropertyDerivedFeatures not available at import time.")

    def fit(self, X: Any, y: Optional[pd.Series] = None) -> "EnsureDerivedFeatures":
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        df = _to_dataframe(X).copy()

        if PropertyDerivedFeatures is None:
            # Best-effort: nothing to add
            return df

        # Compute full derived set once
        pdf = PropertyDerivedFeatures(city_base=self.city_base, region_index=self.region_index)
        derived = pdf.transform(df)

        # Decide which columns to enforce
        cols = self.required_cols or [c for c in derived.columns if c not in df.columns]

        # Add only missing ones (idempotent)
        for c in cols:
            if c not in df.columns and c in derived.columns:
                try:
                    df[c] = derived[c]
                except Exception as e:
                    logger.debug("EnsureDerivedFeatures: failed to add %s: %s", c, e)

        return df