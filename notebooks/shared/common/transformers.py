from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore

__all__ = ["PropertyDerivedFeatures"]


def _to_dataframe(X: Any) -> pd.DataFrame:
    """Ensure input is a DataFrame (sklearn contract: accept array-like / dict-like)."""
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X)


def _lower_strip_safe(s: pd.Series) -> pd.Series:
    """Lower/strip strings while preserving NA (no 'nan' literals)."""
    return s.astype("string").str.strip().str.lower().astype("object")


class PropertyDerivedFeatures(BaseEstimator, TransformerMixin):
    """
    Feature engineering for Property assets (deterministic, side-effect free).

    Produces:
      - log_size_m2                     : log(1 + size_m2)
      - sqm_per_room                    : size_m2 / rooms
      - rooms_per_100sqm                : 100 * rooms / size_m2
      - baths_per_100sqm                : 100 * bathrooms / size_m2
      - elev_x_floor                    : has_elevator * max(floor - 1, 0)
      - no_elev_high_floor              : (1 - has_elevator) * 1[floor >= 3]
      - city_zone_prior (float32)       : lookup(city, zone) from city_base (lower/strip)
      - region_index_prior (float32)    : map(region) from region_index (lower/strip)

    Notes
    - Robust to missing columns; missing inputs yield NaNs for derived fields.
    - Output dtypes: engineered floats -> float32 for memory efficiency.
    - Input dictionaries are copied and normalized in __init__ (clone-safe).
    """

    def __init__(
        self,
        city_base: Optional[Dict[str, Dict[str, float]]] = None,
        region_index: Optional[Dict[str, float]] = None,
    ) -> None:
        # Normalize keys to lowercase for O(1) lookups during transform.
        self.city_base: Dict[str, Dict[str, float]] = {
            str(city).strip().lower(): {
                str(z).strip().lower(): float(v) for z, v in (zones or {}).items()
            }
            for city, zones in (city_base or {}).items()
        }
        self.region_index: Dict[str, float] = {
            str(k).strip().lower(): float(v) for k, v in (region_index or {}).items()
        }

    # sklearn API -------------------------------------------------------------

    def fit(self, X: Any, y: Optional[pd.Series] = None) -> "PropertyDerivedFeatures":
        # Stateless: nothing to fit.
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        df = _to_dataframe(X).copy()

        # ---- source columns (best-effort, tolerant to missing) --------------
        size = pd.to_numeric(df.get("size_m2", np.nan), errors="coerce")
        rooms = pd.to_numeric(df.get("rooms", np.nan), errors="coerce")
        baths = pd.to_numeric(df.get("bathrooms", np.nan), errors="coerce")
        floor = pd.to_numeric(df.get("floor", np.nan), errors="coerce")
        elev = pd.to_numeric(df.get("has_elevator", np.nan), errors="coerce")

        # Fill where appropriate for arithmetic (keep NaN semantics otherwise)
        floor_f = floor.fillna(0)
        elev_f = elev.fillna(0)

        # ---- derived numerics (safe divisions) ------------------------------
        df["log_size_m2"] = np.log1p(size)

        # size guards: avoid division by zero (use np.divide to preserve NaNs)
        df["sqm_per_room"] = np.divide(size, rooms, where=rooms.notna())
        df["rooms_per_100sqm"] = np.divide(100.0 * rooms, size, where=size.notna())
        df["baths_per_100sqm"] = np.divide(100.0 * baths, size, where=size.notna())

        # elevator/floor interactions
        df["elev_x_floor"] = elev_f * np.maximum(floor_f - 1, 0)
        df["no_elev_high_floor"] = ((1 - elev_f) * (floor_f >= 3).astype(int)).astype("float64")

        # ---- priors: city_zone_prior & region_index_prior -------------------
        # Normalize geo strings safely (preserve NaN)
        city = df.get("city")
        zone = df.get("zone")
        region = df.get("region")

        # city_zone_prior
        if city is not None and zone is not None:
            c_norm = _lower_strip_safe(city)
            z_norm = _lower_strip_safe(zone)
            # Vectorized via list comprehension (fast for serving sizes)
            cz_vals = np.fromiter(
                (
                    self.city_base.get(c, {}).get(z, np.nan)
                    if (pd.notna(c) and pd.notna(z))
                    else np.nan
                    for c, z in zip(c_norm, z_norm)
                ),
                dtype="float64",
                count=len(df),
            )
            df["city_zone_prior"] = cz_vals
        else:
            df["city_zone_prior"] = np.nan

        # region_index_prior
        if region is not None:
            r_norm = _lower_strip_safe(region)
            df["region_index_prior"] = pd.to_numeric(
                r_norm.map(self.region_index), errors="coerce"
            )
        else:
            df["region_index_prior"] = np.nan

        # ---- dtype compaction (float32 where appropriate) -------------------
        to_f32 = [
            "log_size_m2",
            "sqm_per_room",
            "baths_per_100sqm",
            "rooms_per_100sqm",
            "city_zone_prior",
            "region_index_prior",
            "elev_x_floor",
        ]
        for col in to_f32:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

        # Keep no_elev_high_floor as float32 as well for consistency
        df["no_elev_high_floor"] = pd.to_numeric(df["no_elev_high_floor"], errors="coerce").astype("float32")

        return df
