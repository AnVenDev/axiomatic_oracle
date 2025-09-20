from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore

__all__ = ["GeoCanonizer", "PriorsGuard"]


def _to_dataframe(X: Any) -> pd.DataFrame:
    """Ensure we always work with a DataFrame."""
    if isinstance(X, pd.DataFrame):
        return X
    # Defensive: try to coerce dict/array-like; let pandas raise if impossible
    return pd.DataFrame(X)


def _lower_strip_safe(s: pd.Series) -> pd.Series:
    """Lower/strip without converting NaN to literal 'nan'."""
    # Use pandas StringDtype to preserve <NA>, then reinsert NaNs
    as_str = s.astype("string")
    lowered = as_str.str.strip().str.lower()
    # Convert back to object to play well with mixed downstream uses
    return lowered.astype("object")


class GeoCanonizer(BaseEstimator, TransformerMixin):
    """
    Canonicalize geography fields with safe defaults.

    - Create 'city' from 'location' if missing.
    - Ensure 'zone' exists (default: 'semi_center').
    - Ensure 'region' exists (left as NaN for downstream enrichment).
    - Lowercase and trim {city, zone, region} without stringifying NaNs.
    """

    def fit(self, X: Any, y: Optional[pd.Series] = None) -> "GeoCanonizer":
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        df = _to_dataframe(X).copy()

        # Add missing geo columns
        if "city" not in df.columns and "location" in df.columns:
            df["city"] = df["location"]
        if "zone" not in df.columns:
            df["zone"] = "semi_center"
        if "region" not in df.columns:
            df["region"] = pd.Series(pd.NA, index=df.index, dtype="object")

        # Normalize string content safely
        for col in ("city", "zone", "region"):
            if col in df.columns:
                df[col] = _lower_strip_safe(df[col])

        return df


class PriorsGuard(BaseEstimator, TransformerMixin):
    """
    Populate/repair prior features with robust fallbacks.

    Produces/repairs:
      - 'city_zone_prior'  (float64)
      - 'region_index_prior' (float64)

    Resolution order for city_zone_prior:
      1) explicit value if present and valid
      2) city_base[city][zone] if available
      3) zone_medians[zone] if available
      4) global_cityzone_median

    Resolution for region_index_prior:
      map(region) using `region_index`, otherwise NaN.
    """

    def __init__(
        self,
        city_base: Optional[Dict[str, Dict[str, float]]] = None,
        region_index: Optional[Dict[str, float]] = None,
        zone_medians: Optional[Dict[str, float]] = None,
        global_cityzone_median: float = 0.0,
    ) -> None:
        # Clone-safe: do not mutate input dicts
        self.city_base = dict(city_base or {})
        self.region_index = dict(region_index or {"north": 1.05, "center": 1.00, "south": 0.92})
        self.zone_medians = dict(zone_medians or {})
        self.global_cityzone_median = float(global_cityzone_median)

    def fit(self, X: Any, y: Optional[pd.Series] = None) -> "PriorsGuard":
        return self

    # ---- helpers ---------------------------------------------------------
    def _lookup_city_zone_prior(self, city: Any, zone: Any) -> float:
        # Respect NaNs
        if pd.isna(city) or pd.isna(zone):
            return float(self.zone_medians.get(zone, self.global_cityzone_median))
        c = str(city).strip().lower()
        z = str(zone).strip().lower()
        # City-specific mapping first
        by_city = self.city_base.get(c, {})
        val = by_city.get(z, np.nan)
        if pd.isna(val):
            # Fall back to zone median, then global
            return float(self.zone_medians.get(z, self.global_cityzone_median))
        return float(val)

    def _ensure_geo_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Light guard: PriorsGuard can be used standalone as well
        out = df.copy()
        if "city" not in out and "location" in out:
            out["city"] = out["location"]
        if "zone" not in out:
            out["zone"] = "semi_center"
        if "region" not in out:
            out["region"] = pd.Series(pd.NA, index=out.index, dtype="object")
        # Avoid 'nan' strings later on
        for c in ("city", "zone", "region"):
            out[c] = _lower_strip_safe(out[c])
        return out

    # ---- transform -------------------------------------------------------
    def transform(self, X: Any) -> pd.DataFrame:
        df = _to_dataframe(X).copy()
        df = self._ensure_geo_columns(df)

        # --- city_zone_prior ------------------------------------------------
        if "city_zone_prior" in df.columns:
            cz = pd.to_numeric(df["city_zone_prior"], errors="coerce")
        else:
            cz = pd.Series(np.nan, index=df.index, dtype="float64")

        need_cz = cz.isna()
        if need_cz.any():
            city = df["city"]
            zone = df["zone"]
            # Vectorized via list comprehension (fast enough for serving sizes)
            filled = np.fromiter(
                (self._lookup_city_zone_prior(c, z) for c, z in zip(city[need_cz], zone[need_cz])),
                dtype="float64",
                count=int(need_cz.sum()),
            )
            cz.loc[need_cz] = filled

        df["city_zone_prior"] = cz.astype("float64")

        # --- region_index_prior --------------------------------------------
        if "region_index_prior" in df.columns:
            rip = pd.to_numeric(df["region_index_prior"], errors="coerce")
        else:
            rip = pd.Series(np.nan, index=df.index, dtype="float64")

        need_rip = rip.isna()
        if need_rip.any():
            mapped = df.loc[need_rip, "region"].map(self.region_index)
            rip.loc[need_rip] = pd.to_numeric(mapped, errors="coerce")

        df["region_index_prior"] = rip.astype("float64")

        return df
