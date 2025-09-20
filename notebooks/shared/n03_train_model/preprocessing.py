from __future__ import annotations

"""
Preprocessing and feature preparation for training.
- No "magic" imports (no common_imports)
- FeatureEngineer: pure functions, no leakage by default
- Categorical domains: enforced with ordered categories where appropriate
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional, Tuple, Dict, Any, Mapping

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import logging

from notebooks.shared.common.constants import (
    LEAKY_FEATURES,
    PRICE_PER_SQM_CAPPED_VIOLATED,
    Cols,
    ENERGY_CLASS, URBAN_TYPE, REGION, ZONE, ORIENTATION, VIEW, CONDITION, HEATING,
    LAST_VERIFIED_TS, PRICE_PER_SQM, VALUATION_K, SIZE_M2, PRICE_PER_SQM_VS_REGION_AVG,
    HAS_GARDEN, HAS_BALCONY, GARAGE,
    Mappings, EnergyClass, Zone
)
from notebooks.shared.common.schema import get_all_fields
from notebooks.shared.common.utils import get_utc_now

# Additional leaky features that must never be used for model training
ML_LEAKY_FEATURES: set[str] = {
    PRICE_PER_SQM,
    PRICE_PER_SQM_VS_REGION_AVG,
    "_viz_price_per_sqm",
    "valuation_k_log",
    "price_per_sqm_capped",
    PRICE_PER_SQM_CAPPED_VIOLATED,
    "strongly_incoherent",
    "valuation_k_decile",
    "valuation_rank",
    "is_top_valuation",
}

def drop_leaky_and_target(df: pd.DataFrame, target: str, extra_leaky: Iterable[str] | None = None) -> pd.DataFrame:
    """
    Return a copy of `df` keeping only columns that are neither the target
    nor in the leaky-feature lists.
    """
    deny = set(LEAKY_FEATURES) | {target}
    if extra_leaky:
        deny |= set(extra_leaky)
    keep = [c for c in df.columns if c not in deny]
    return df.loc[:, keep].copy()

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class FeatureEngineerConfig:
    allow_target_derived: bool = False
    compute_semantic: bool = True
    add_month_sin_cos: bool = True

class FeatureEngineer:
    """
    Generates derived features and pre-training diagnostics.
    - Does not introduce leakage (by default).
    - Robust numeric calculations (handles NA/divisions).
    """

    def __init__(
        self,
        reference_time: Optional[datetime] = None,
        config: Optional[FeatureEngineerConfig] = None,
    ):
        self.reference_time = reference_time or get_utc_now()
        self.config = config or FeatureEngineerConfig()

    def prepare_for_training(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Pipeline:
          1) Basic temporal/structural features
          2) (Optional) semantic scores (luxury/env/risk)
          3) Final diagnostics (completeness)
        Returns: (df_enriched, report)
        """
        out = df.copy()
        report: Dict[str, Any] = {"features_created": [], "statistics": {}, "diagnostics": {}}

        out = self._add_basic_features(out, report)

        if self.config.compute_semantic:
            out = self._add_semantic_scores(out, report)

        report["diagnostics"] = self._diagnose_features(out)

        # Keep original behavior: cast float columns on the original df (no-op for `out`)
        for col in df.select_dtypes(include="float").columns:
            df[col] = df[col].astype("float32")
        
        return out, report

    def _add_basic_features(self, df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        # Timestamp → listing_month, missing_timestamp
        ts = pd.to_datetime(df.get(LAST_VERIFIED_TS, pd.NaT), utc=True, errors="coerce")
        df[Cols.LISTING_MONTH] = ts.dt.month.fillna(0).astype("Int16")
        df[Cols.MISSING_TIMESTAMP] = ts.isna().astype("Int8")

        # Optional: seasonal signals
        if self.config.add_month_sin_cos:
            # 1..12 → [0, 2π)
            month = df[Cols.LISTING_MONTH].fillna(0).astype("float32")
            # avoid 0 → compute sin/cos on (m-1)
            ang = (month - 1.0) * (2.0 * np.pi / 12.0)
            df[Cols.LISTING_MONTH_SIN] = np.sin(ang).astype("float32")
            df[Cols.LISTING_MONTH_COS] = np.cos(ang).astype("float32")
            report["features_created"] += [Cols.LISTING_MONTH_SIN, Cols.LISTING_MONTH_COS]

        # Building age
        if Cols.YEAR_BUILT in df.columns:
            # Note: YEAR_BUILT may be float or contain NA
            year = pd.to_numeric(df[Cols.YEAR_BUILT], errors="coerce")
            df[Cols.BUILDING_AGE_YEARS] = (self.reference_time.year - year).astype("Int32")
        else:
            df[Cols.BUILDING_AGE_YEARS] = pd.Series(pd.NA, index=df.index, dtype="Int32")

        # Ratios
        size = pd.to_numeric(df.get(SIZE_M2, np.nan), errors="coerce").replace(0, np.nan)
        rooms = pd.to_numeric(df.get(Cols.ROOMS, np.nan), errors="coerce")
        df[Cols.ROOMS_PER_SQM] = (rooms / size).astype("float32").fillna(0.0)

        # Amenity count (binary)
        for col in (HAS_GARDEN, HAS_BALCONY, GARAGE, Cols.HAS_ELEVATOR):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("Int8")
            else:
                df[col] = pd.Series(0, index=df.index, dtype="Int8")
        df[Cols.BASIC_AMENITY_COUNT] = (
            df[[HAS_GARDEN, HAS_BALCONY, GARAGE, Cols.HAS_ELEVATOR]].sum(axis=1).astype("Int16")
        )
        report["features_created"] += [Cols.ROOMS_PER_SQM, Cols.BASIC_AMENITY_COUNT]

        # Price per sqm (only if explicitly allowed → potential leakage)
        if PRICE_PER_SQM not in df.columns and self.config.allow_target_derived:
            if VALUATION_K in df.columns and SIZE_M2 in df.columns:
                v = pd.to_numeric(df[VALUATION_K], errors="coerce") * 1000.0
                s = pd.to_numeric(df[SIZE_M2], errors="coerce").replace(0, np.nan)
                df[PRICE_PER_SQM] = (v / s).astype("float32")
                report["features_created"].append(PRICE_PER_SQM)
                report["statistics"][PRICE_PER_SQM] = {
                    "mean": float(np.nanmean(df[PRICE_PER_SQM].to_numpy())),
                    "std": float(np.nanstd(df[PRICE_PER_SQM].to_numpy())),
                }

        # Cast floats to float32
        for col in df.select_dtypes(include="float").columns:
            df[col] = df[col].astype("float32")

        return df

    def _add_semantic_scores(self, df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        # Luxury score: scaled amenity count (0..1)
        amen_max = max(int(df[Cols.BASIC_AMENITY_COUNT].max(skipna=True) or 0), 1)
        df[Cols.LUXURY_SCORE] = (df[Cols.BASIC_AMENITY_COUNT] / amen_max).astype("float32")

        # Environmental score: rank of EnergyClass normalized to 0..1
        if ENERGY_CLASS in df.columns:
            rank_map = Mappings.ENERGY_CLASS_RANK  # A=7 .. G=1
            ranks = df[ENERGY_CLASS].map(rank_map).astype("float32")
            # normalize to [0,1] (G→0, A→1)
            df[Cols.ENV_SCORE] = ((ranks - 1.0) / (7.0 - 1.0)).fillna(0.5).astype("float32")
        else:
            df[Cols.ENV_SCORE] = pd.Series(0.5, index=df.index, dtype="float32")

        # Risk score: building age normalized to 0..1 (NA→0.0)
        age = pd.to_numeric(df.get(Cols.BUILDING_AGE_YEARS, np.nan), errors="coerce")
        max_age = float(np.nanmax(age.to_numpy())) if age.notna().any() else 0.0
        if max_age <= 0:
            df[Cols.RISK_SCORE] = pd.Series(0.0, index=df.index, dtype="float32")
        else:
            df[Cols.RISK_SCORE] = (age / max_age).fillna(0.0).astype("float32")

        report["features_created"] += [Cols.LUXURY_SCORE, Cols.ENV_SCORE, Cols.RISK_SCORE]

        # Cast floats to float32
        for col in df.select_dtypes(include="float").columns:
            df[col] = df[col].astype("float32")

        return df

    def _diagnose_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        # Today we target "property"; multi-asset support could be added later
        required = set(get_all_fields("property"))
        present = set(df.columns)

        # Cast floats to float32 (preserve original behavior)
        for col in df.select_dtypes(include="float").columns:
            df[col] = df[col].astype("float32")

        return {
            "missing_required": sorted(list(required - present)),
            "unknown_columns": sorted(list(present - required)),
            "n_columns": int(len(df.columns)),
        }

def enforce_categorical_domains(
    df: pd.DataFrame,
    location_weights: Mapping[str, float],
) -> pd.DataFrame:
    """
    Enforce consistent categorical domains:
    - Location: categories from config (order not relevant)
    - EnergyClass: A..G with descending efficiency order
    - Zone: ordered center > semi_center > periphery
    - Other semantic columns → generic 'category'
    Does not clamp values—only sets dtype to Categorical to avoid category explosion.
    """
    out = df.copy()

    # location
    if Cols.LOCATION in out.columns:
        locs = list(location_weights.keys())
        out[Cols.LOCATION] = pd.Categorical(out[Cols.LOCATION], categories=locs, ordered=False)

    # energy class (ordered by rank)
    if ENERGY_CLASS in out.columns:
        energy_order = [e.value for e in EnergyClass]  # ["A","B","C","D","E","F","G"]
        out[ENERGY_CLASS] = pd.Categorical(out[ENERGY_CLASS], categories=energy_order, ordered=True)

    # zone (ordered center > semi_center > periphery)
    if ZONE in out.columns:
        zone_order = [Zone.CENTER.value, Zone.SEMI_CENTER.value, Zone.PERIPHERY.value]
        out[ZONE] = pd.Categorical(out[ZONE], categories=zone_order, ordered=True)
    
    # Preserve original behavior: cast on the original df (no-op for `out`)
    for col in df.select_dtypes(include="float").columns:
        df[col] = df[col].astype("float32")

    # other semantic categoricals
    for col in (URBAN_TYPE, REGION, ORIENTATION, VIEW, CONDITION, HEATING):
        if col in out.columns:
            out[col] = out[col].astype("category")

    return out
