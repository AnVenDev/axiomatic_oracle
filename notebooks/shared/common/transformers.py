from __future__ import annotations

"""
Deterministic feature engineering for Property assets (train = serve).

Outputs (selected)
- log_size_m2            : log(1 + size_m2)
- sqm_per_room           : size_m2 / rooms
- rooms_per_100sqm       : 100 * rooms / size_m2
- baths_per_100sqm       : 100 * bathrooms / size_m2
- elev_x_floor           : has_elevator * max(floor - 1, 0)
- no_elev_high_floor     : (1 - has_elevator) * 1[floor >= 3]
- city_zone_prior        : lookup(city, zone) from provided city_base
- region_index_prior     : lookup(region) from provided region_index
- garage_vs_central      : garage / max(distance_to_center_km, eps)
- attic_vs_floors        : attic * building_floors

New high-impact interactions
- floor_ratio            : floor / max(building_floors - 1, 1)
- elev_needed_no_elev    : 1[building_floors >= MIN_FLOORS_FOR_ELEVATOR] * (1 - has_elevator)
- balcony_vs_floor       : has_balcony * max(floor, 0)
- garden_vs_urban        : has_garden * 1[urban_type == 'urban']
- parking_vs_central     : parking_spot / max(distance_to_center_km, eps)
- humidity_x_ground      : is_ground_floor * scaled(humidity_level)
- noise_x_lower          : (1 - floor_ratio) * scaled(noise_level)
- energy_rank            : A..G → 7..1
- energy_x_size          : energy_rank * log_size_m2
- pt_x_periphery         : public_transport_nearby * 1[zone == 'periphery']

Design
- Stateless and side-effect free (sklearn-compatible Transformer).
- Best-effort with missing inputs → derived columns become NaN.
- Float outputs compacted to float32 for memory efficiency.
"""

from typing import Any, Dict, Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore

# Canonical names / thresholds (single source of truth).
try:  # pragma: no cover
    from shared.common.constants import (
        # core numerics
        SIZE_M2, ROOMS, BATHROOMS, FLOOR, BUILDING_FLOORS, DISTANCE_TO_CENTER_KM,
        # booleans / categories
        HAS_ELEVATOR, HAS_GARDEN, HAS_BALCONY, GARAGE, PARKING_SPOT,
        IS_TOP_FLOOR, IS_GROUND_FLOOR, ATTIC, PUBLIC_TRANSPORT_NEARBY,
        URBAN_TYPE, ZONE, REGION, ENERGY_CLASS,
        # mappings & thresholds
        Mappings, Thresholds, Zone,
    )
except Exception:  # pragma: no cover
    from .constants import (
        SIZE_M2, ROOMS, BATHROOMS, FLOOR, BUILDING_FLOORS, DISTANCE_TO_CENTER_KM,
        HAS_ELEVATOR, HAS_GARDEN, HAS_BALCONY, GARAGE, PARKING_SPOT,
        IS_TOP_FLOOR, IS_GROUND_FLOOR, ATTIC, PUBLIC_TRANSPORT_NEARBY,
        URBAN_TYPE, ZONE, REGION, ENERGY_CLASS,
        Mappings, Thresholds, Zone,
    )

__all__ = ["PropertyDerivedFeatures"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _to_dataframe(X: Any) -> pd.DataFrame:
    """Ensure input is a DataFrame (sklearn contract: accept array-like / dict-like)."""
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X)


def _lower_strip_safe(s: pd.Series) -> pd.Series:
    """Lower/strip strings while preserving NA (no 'nan' literals)."""
    return s.astype("string").str.strip().str.lower().astype("object")


def _as_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _scale_01(x: pd.Series, lo: float, hi: float) -> pd.Series:
    """Clip to [lo, hi] then scale to [0,1]; preserves NaN."""
    v = _as_float(x)
    v = v.clip(lower=lo, upper=hi)
    return (v - lo) / max(hi - lo, 1e-9)


# --------------------------------------------------------------------------- #
# Transformer
# --------------------------------------------------------------------------- #

class PropertyDerivedFeatures(BaseEstimator, TransformerMixin):
    """
    Deterministic feature engineering for Property assets.

    Parameters
    ----------
    city_base : dict[str, dict[str, float]] | None
        Mapping city->zone->base price (used to compute `city_zone_prior`).
    region_index : dict[str, float] | None
        Mapping region->index (used to compute `region_index_prior`).
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
        size = _as_float(df.get(SIZE_M2, np.nan))
        rooms = _as_float(df.get(ROOMS, np.nan))
        baths = _as_float(df.get(BATHROOMS, np.nan))
        floor = _as_float(df.get(FLOOR, np.nan))
        floors_tot = _as_float(df.get(BUILDING_FLOORS, np.nan))

        elev = _as_float(df.get(HAS_ELEVATOR, np.nan))
        garden = _as_float(df.get(HAS_GARDEN, np.nan))
        balcony = _as_float(df.get(HAS_BALCONY, np.nan))
        garage = _as_float(df.get(GARAGE, np.nan))
        parking = _as_float(df.get(PARKING_SPOT, np.nan))
        attic = _as_float(df.get(ATTIC, np.nan))

        is_top = _as_float(df.get(IS_TOP_FLOOR, np.nan))
        is_ground = _as_float(df.get(IS_GROUND_FLOOR, np.nan))
        dist_center = _as_float(df.get(DISTANCE_TO_CENTER_KM, np.nan))
        pt_near = _as_float(df.get(PUBLIC_TRANSPORT_NEARBY, np.nan))

        humidity = _as_float(df.get("humidity_level", np.nan))
        noise = _as_float(df.get("noise_level", np.nan))
        energy = df.get(ENERGY_CLASS)
        zone = df.get(ZONE)
        urban_type = df.get(URBAN_TYPE)
        region = df.get(REGION)
        city = df.get("city")

        # Fill for arithmetic convenience (but preserve NaN semantics later)
        floor_f = floor.fillna(0)
        elev_f = elev.fillna(0)

        # ---- base numerics (safe divisions) ---------------------------------
        df["log_size_m2"] = np.log1p(size)

        df["sqm_per_room"] = np.divide(size, rooms, where=rooms.notna())
        df["rooms_per_100sqm"] = np.divide(100.0 * rooms, size, where=size.notna())
        df["baths_per_100sqm"] = np.divide(100.0 * baths, size, where=size.notna())

        # elevator/floor interactions (legacy core)
        df["elev_x_floor"] = elev_f * np.maximum(floor_f - 1, 0)
        df["no_elev_high_floor"] = ((1 - elev_f) * (floor_f >= 3).astype(int)).astype("float64")

        # ---- priors: city_zone_prior & region_index_prior -------------------
        if city is not None and zone is not None:
            c_norm = _lower_strip_safe(city)
            z_norm = _lower_strip_safe(zone)
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

        if region is not None:
            r_norm = _lower_strip_safe(region)
            df["region_index_prior"] = pd.to_numeric(r_norm.map(self.region_index), errors="coerce")
        else:
            df["region_index_prior"] = np.nan

        # ---- core conditional interactions ----------------------------------
        eps = float(getattr(Thresholds, "CENTRALITY_EPS_KM", 0.25))
        if GARAGE in df.columns and DISTANCE_TO_CENTER_KM in df.columns:
            dist_safe = np.maximum(dist_center, eps)
            df["garage_vs_central"] = np.divide(garage, dist_safe, where=dist_safe.astype("float64") > 0)
        else:
            df["garage_vs_central"] = np.nan

        if ATTIC in df.columns and BUILDING_FLOORS in df.columns:
            df["attic_vs_floors"] = np.multiply(attic.fillna(0), floors_tot.fillna(0))
        else:
            df["attic_vs_floors"] = np.nan

        # ---- NEW: high-impact interpretable interactions --------------------
        # 1) floor_ratio
        denom = np.maximum(floors_tot - 1.0, 1.0)
        df["floor_ratio"] = np.divide(floor_f, denom, where=denom > 0)

        # 2) elev_needed_no_elev
        min_floors = int(getattr(Thresholds, "MIN_FLOORS_FOR_ELEVATOR", 4))
        need_elev = (floors_tot >= min_floors).astype("float64")
        df["elev_needed_no_elev"] = need_elev * (1.0 - elev_f)

        # 3) balcony_vs_floor
        df["balcony_vs_floor"] = balcony.fillna(0) * np.maximum(floor_f, 0)

        # 4) garden_vs_urban
        if urban_type is not None:
            df["garden_vs_urban"] = garden.fillna(0) * (_lower_strip_safe(urban_type) == "urban").astype("float64")
        else:
            df["garden_vs_urban"] = np.nan

        # 5) parking_vs_central
        if PARKING_SPOT in df.columns and DISTANCE_TO_CENTER_KM in df.columns:
            dist_safe = np.maximum(dist_center, eps)
            df["parking_vs_central"] = np.divide(parking, dist_safe, where=dist_safe.astype("float64") > 0)
        else:
            df["parking_vs_central"] = np.nan

        # 6) humidity_x_ground  (scale humidity to [0,1] over [30..70])
        hum_scaled = _scale_01(humidity, 30.0, 70.0)
        df["humidity_x_ground"] = hum_scaled * is_ground.fillna(0)

        # 7) noise_x_lower  (lower floors suffer more)
        noise_scaled = _scale_01(noise, 30.0, 90.0)
        df["noise_x_lower"] = noise_scaled * (1.0 - df["floor_ratio"])

        # 8) energy_rank
        if energy is not None:
            e_norm = _lower_strip_safe(energy)
            rank_map = {k.lower(): v for k, v in Mappings.ENERGY_CLASS_RANK.items()}
            df["energy_rank"] = pd.to_numeric(e_norm.map(rank_map), errors="coerce")
        else:
            df["energy_rank"] = np.nan

        # 9) energy_x_size
        df["energy_x_size"] = df["energy_rank"] * df["log_size_m2"]

        # 10) pt_x_periphery
        if zone is not None:
            z_norm = _lower_strip_safe(zone)
            periph = (z_norm == Zone.PERIPHERY.value).astype("float64")
            df["pt_x_periphery"] = pt_near.fillna(0) * periph
        else:
            df["pt_x_periphery"] = np.nan

        # ---- dtype compaction (float32 where appropriate) -------------------
        to_f32 = [
            "log_size_m2",
            "sqm_per_room",
            "baths_per_100sqm",
            "rooms_per_100sqm",
            "city_zone_prior",
            "region_index_prior",
            "elev_x_floor",
            "no_elev_high_floor",
            "garage_vs_central",
            "attic_vs_floors",
            "floor_ratio",
            "elev_needed_no_elev",
            "balcony_vs_floor",
            "garden_vs_urban",
            "parking_vs_central",
            "humidity_x_ground",
            "noise_x_lower",
            "energy_rank",
            "energy_x_size",
            "pt_x_periphery",
        ]
        for col in to_f32:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

        return df