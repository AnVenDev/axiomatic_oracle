"""
Unified schema & validation utilities for RWA Property.

Scope
- Canonical field groups (core/structural/amenity/quality/score/derived).
- Suggested dtypes and column aliases for backward compatibility.
- DataFrame-level validation (presence, unknowns, dtype coercion, domain and cross-field checks).
- Convenience helpers to normalize booleans, column order and enforce categoricals.

Design
- Pure functions: no I/O, return dicts/DataFrames only.
- Best-effort conversions; strict mode available for enterprise workflows.
- Backward-compatible public API; extensions are opt-in via parameters.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Iterable, Tuple
import logging
import math

import pandas as pd                     # type: ignore
from pydantic import BaseModel, Field   # type: ignore

from shared.common.constants import (
    EnergyClass,
    ASSET_ID, ASSET_TYPE, LOCATION, REGION, URBAN_TYPE, ZONE,
    SIZE_M2, ROOMS, BATHROOMS, YEAR_BUILT, AGE_YEARS, FLOOR, BUILDING_FLOORS,
    IS_TOP_FLOOR, IS_GROUND_FLOOR,
    HAS_ELEVATOR, HAS_GARDEN, HAS_BALCONY, GARAGE, OWNER_OCCUPIED,
    PUBLIC_TRANSPORT_NEARBY, DISTANCE_TO_CENTER_KM, PARKING_SPOT, CELLAR, ATTIC, CONCIERGE,
    ENERGY_CLASS, HUMIDITY_LEVEL, TEMPERATURE_AVG, NOISE_LEVEL, AIR_QUALITY_INDEX,
    CONDITION_SCORE, RISK_SCORE, LUXURY_SCORE, ENV_SCORE,
    ORIENTATION, VIEW, CONDITION, HEATING,
    VALUATION_K, PRICE_PER_SQM, LAST_VERIFIED_TS, LISTING_MONTH,
    PRICE_PER_SQM_CAPPED, PRICE_PER_SQM_CAPPED_VIOLATED, LISTING_MONTH_SIN, LISTING_MONTH_COS,
    DAYS_SINCE_VERIFICATION, HOURS_SINCE_VERIFICATION, IS_STALE_30D, IS_STALE_60D, IS_STALE_90D,
    ANOMALY_SCORE, ANOMALY_FLAG, ANOMALY_LABEL, ANOMALY_SCORE_RAW,
    SEVERITY_SCORE, CONFIDENCE_SCORE, VALUE_SEGMENT, LUXURY_CATEGORY,
    DERIVED_FEATURES,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CORE_FIELDS", "PROPERTY_STRUCTURAL_FIELDS", "PROPERTY_AMENITY_FIELDS",
    "PROPERTY_QUALITY_FIELDS", "PROPERTY_SCORE_FIELDS", "PROPERTY_ADDITIONAL_FIELDS",
    "PROPERTY_DERIVED_FIELDS", "CATEGORICAL", "BOOLEAN", "NUMERIC", "DATETIME",
    "SUGGESTED_DTYPES", "COLUMN_ALIASES", "AssetSchema", "SCHEMA", "BOOLEAN_FIELDS", "RANGE_BOUNDS",
    "normalize_booleans", "enforce_cross_field_rules",
    "get_required_fields", "get_all_fields", "apply_aliases", "list_missing",
    "list_unknown", "coerce_dtypes", "enforce_domains", "enforce_categoricals",
    "normalize_column_order", "validate_df", "validate_and_coerce",
]

# ============================================================================
# Logical groups
# ============================================================================

CORE_FIELDS: List[str] = [
    ASSET_ID, ASSET_TYPE, LOCATION, VALUATION_K, PRICE_PER_SQM, LAST_VERIFIED_TS, LISTING_MONTH,
]

PROPERTY_STRUCTURAL_FIELDS: List[str] = [
    REGION, URBAN_TYPE, ZONE, SIZE_M2, ROOMS, BATHROOMS, YEAR_BUILT, AGE_YEARS,
    FLOOR, BUILDING_FLOORS, IS_TOP_FLOOR, IS_GROUND_FLOOR,
]

PROPERTY_AMENITY_FIELDS: List[str] = [
    HAS_ELEVATOR, HAS_GARDEN, HAS_BALCONY, GARAGE, OWNER_OCCUPIED,
    PUBLIC_TRANSPORT_NEARBY, DISTANCE_TO_CENTER_KM, PARKING_SPOT, CELLAR, ATTIC, CONCIERGE,
]

PROPERTY_QUALITY_FIELDS: List[str] = [
    ENERGY_CLASS, HUMIDITY_LEVEL, TEMPERATURE_AVG, NOISE_LEVEL, AIR_QUALITY_INDEX,
]

PROPERTY_SCORE_FIELDS: List[str] = [
    CONDITION_SCORE, RISK_SCORE, LUXURY_SCORE, ENV_SCORE,
]

PROPERTY_ADDITIONAL_FIELDS: List[str] = [
    ORIENTATION, VIEW, CONDITION, HEATING,
]

PROPERTY_DERIVED_FIELDS: List[str] = [
    PRICE_PER_SQM_CAPPED, PRICE_PER_SQM_CAPPED_VIOLATED,
    LISTING_MONTH_SIN, LISTING_MONTH_COS,
    DAYS_SINCE_VERIFICATION, HOURS_SINCE_VERIFICATION,
    IS_STALE_30D, IS_STALE_60D, IS_STALE_90D,
    ANOMALY_SCORE, ANOMALY_SCORE_RAW, ANOMALY_FLAG, ANOMALY_LABEL,
    SEVERITY_SCORE, CONFIDENCE_SCORE,
    VALUE_SEGMENT, LUXURY_CATEGORY,
]

# ============================================================================
# Expected macro types
# ============================================================================

CATEGORICAL: List[str] = [
    LOCATION, REGION, URBAN_TYPE, ZONE, ENERGY_CLASS, ORIENTATION, VIEW, CONDITION, HEATING,
    OWNER_OCCUPIED, PUBLIC_TRANSPORT_NEARBY, GARAGE, PARKING_SPOT, CELLAR, ATTIC, CONCIERGE,
    LUXURY_CATEGORY, VALUE_SEGMENT, ANOMALY_LABEL,
]

BOOLEAN: List[str] = [
    IS_TOP_FLOOR, IS_GROUND_FLOOR, HAS_ELEVATOR, HAS_GARDEN, HAS_BALCONY,
    IS_STALE_30D, IS_STALE_60D, IS_STALE_90D, ANOMALY_FLAG,
]

NUMERIC: List[str] = [
    SIZE_M2, ROOMS, BATHROOMS, YEAR_BUILT, AGE_YEARS, FLOOR, BUILDING_FLOORS,
    DISTANCE_TO_CENTER_KM, HUMIDITY_LEVEL, TEMPERATURE_AVG, NOISE_LEVEL, AIR_QUALITY_INDEX,
    CONDITION_SCORE, RISK_SCORE, LUXURY_SCORE, ENV_SCORE,
    VALUATION_K, PRICE_PER_SQM,
    PRICE_PER_SQM_CAPPED, PRICE_PER_SQM_CAPPED_VIOLATED,
    LISTING_MONTH_SIN, LISTING_MONTH_COS,
    DAYS_SINCE_VERIFICATION, HOURS_SINCE_VERIFICATION,
    ANOMALY_SCORE, ANOMALY_SCORE_RAW, SEVERITY_SCORE, CONFIDENCE_SCORE,
]

DATETIME: List[str] = [
    LAST_VERIFIED_TS,
]

# ============================================================================
# Suggested dtypes (pandas dtype strings)
# ============================================================================

SUGGESTED_DTYPES: Mapping[str, str] = {
    ASSET_ID: "object",
    SIZE_M2: "Int32",
    ROOMS: "Int16",
    BATHROOMS: "Int16",
    YEAR_BUILT: "Int16",
    AGE_YEARS: "Int16",
    FLOOR: "Int16",
    BUILDING_FLOORS: "Int16",
    VALUATION_K: "float32",
    PRICE_PER_SQM: "float32",
    CONDITION_SCORE: "float32",
    RISK_SCORE: "float32",
    LUXURY_SCORE: "float32",
    ENV_SCORE: "float32",
    LAST_VERIFIED_TS: "datetime64[ns, UTC]",
}

# ============================================================================
# Booleans normalization & bounds/cross-field rules
# ============================================================================

# Canonical boolean fields (0/1)
BOOLEAN_FIELDS: List[str] = [
    IS_TOP_FLOOR, IS_GROUND_FLOOR, HAS_ELEVATOR, HAS_GARDEN, HAS_BALCONY,
    GARAGE, PARKING_SPOT, CELLAR, ATTIC, CONCIERGE, ANOMALY_FLAG, OWNER_OCCUPIED, PUBLIC_TRANSPORT_NEARBY,
]

# Deterministic bounds for numeric columns (inclusive)
RANGE_BOUNDS: Mapping[str, Tuple[float, float]] = {
    SIZE_M2: (10.0, 1000.0),
    ROOMS: (1.0, 12.0),
    BATHROOMS: (1.0, 6.0),
    FLOOR: (0.0, 100.0),
    BUILDING_FLOORS: (1.0, 100.0),
    DISTANCE_TO_CENTER_KM: (0.05, 50.0),
    HUMIDITY_LEVEL: (30.0, 70.0),
}

_TRUE = {"1","true","t","yes","y","si","s"}
_FALSE = {"0","false","f","no","n"}

def normalize_booleans(df: pd.DataFrame, fields: Iterable[str] = BOOLEAN_FIELDS) -> pd.DataFrame:
    """Normalize boolean-like columns to 0/1 integers (best-effort, no raise)."""
    out = df.copy()
    for col in fields:
        if col not in out.columns:
            continue
        v = out[col]
        try:
            if pd.api.types.is_bool_dtype(v) or pd.api.types.is_integer_dtype(v):
                out[col] = v.astype("Int8")
                continue
            s = v.astype("string").str.strip().str.lower()
            out[col] = s.map(lambda x: 1 if x in _TRUE else (0 if x in _FALSE else None)).astype("Int8")
        except Exception as e:
            logger.debug("Boolean normalization skipped for %s: %s", col, e)
    return out

def enforce_cross_field_rules(df: pd.DataFrame) -> Dict[str, int]:
    """
    Cross-field invariant checks. Returns a dict with violation counters.
    Non-mutating; caller decides how to handle violations.
    Rules:
      - 0 <= floor < building_floors
      - attic==1 -> is_top_floor==1 and building_floors >= 5
    """
    v: Dict[str, int] = {}
    if FLOOR in df.columns and BUILDING_FLOORS in df.columns:
        try:
            bad = ((pd.to_numeric(df[FLOOR], errors="coerce") < 0) |
                   (pd.to_numeric(df[FLOOR], errors="coerce") >= pd.to_numeric(df[BUILDING_FLOORS], errors="coerce")))
            n = int(bad.fillna(False).sum())
            if n: v["floor_vs_building_floors"] = n
        except Exception:
            pass
    if ATTIC in df.columns and IS_TOP_FLOOR in df.columns and BUILDING_FLOORS in df.columns:
        try:
            attic_1 = pd.to_numeric(df[ATTIC], errors="coerce").fillna(0).astype(int) == 1
            top_ok = (pd.to_numeric(df[IS_TOP_FLOOR], errors="coerce").fillna(0).astype(int) == 1)
            bf_ok = (pd.to_numeric(df[BUILDING_FLOORS], errors="coerce").fillna(0) >= 5)
            bad = attic_1 & ~(top_ok & bf_ok)
            n = int(bad.fillna(False).sum())
            if n: v["attic_requires_top_floor_and_5plus"] = n
        except Exception:
            pass
    return v
# ============================================================================
# Pydantic schema (formal contract)
# ============================================================================

class AssetSchema(BaseModel):
    required: List[str] = Field(default_factory=list)
    optional: List[str] = Field(default_factory=list)
    dtypes: Dict[str, str] = Field(default_factory=dict)

    def all_fields(self) -> List[str]:
        # Preserve order and remove duplicates
        return list(dict.fromkeys([*self.required, *self.optional]))


SCHEMA: Mapping[str, AssetSchema] = {
    "property": AssetSchema(
        required=CORE_FIELDS
                 + PROPERTY_STRUCTURAL_FIELDS
                 + PROPERTY_AMENITY_FIELDS
                 + PROPERTY_QUALITY_FIELDS
                 + PROPERTY_SCORE_FIELDS
                 + PROPERTY_ADDITIONAL_FIELDS,
        optional=list(DERIVED_FEATURES) + PROPERTY_DERIVED_FIELDS,
        dtypes=dict(SUGGESTED_DTYPES),
    )
}

# ============================================================================
# API utilities
# ============================================================================

def get_required_fields(asset_type: str = "property") -> List[str]:
    return SCHEMA[asset_type].required


def get_all_fields(asset_type: str = "property") -> List[str]:
    return SCHEMA[asset_type].all_fields()


def apply_aliases(df: pd.DataFrame, aliases: Mapping[str, str] = COLUMN_ALIASES) -> pd.DataFrame:
    """Rename legacy columns to canonical names if present (no-op if absent)."""
    to_rename = {old: new for old, new in aliases.items() if old in df.columns and new not in df.columns}
    if to_rename:
        df = df.rename(columns=to_rename)
        logger.info("Applied column aliases: %s", to_rename)
    return df


def list_missing(df: pd.DataFrame, required: Iterable[str]) -> List[str]:
    return [c for c in required if c not in df.columns]


def list_unknown(df: pd.DataFrame, known: Iterable[str]) -> List[str]:
    known_set = set(known)
    return [c for c in df.columns if c not in known_set]


def _nullable_int_for(target: str) -> str:
    """Ensure pandas Nullable Int dtype is used (e.g., int32 -> Int32)."""
    if target.startswith("Int"):
        return target
    if target.startswith("int"):
        width = "".join(ch for ch in target if ch.isdigit()) or "32"
        return f"Int{width}"
    return target


def coerce_dtypes(df: pd.DataFrame, dtypes: Mapping[str, str]) -> pd.DataFrame:
    """Best-effort: coerce suggested dtypes where possible without raising."""
    out = df.copy()
    for col, dt in dtypes.items():
        if col not in out.columns:
            continue
        try:
            if dt.lower().startswith("int"):
                # Use nullable Int to preserve NaNs
                target = _nullable_int_for(dt)
                out[col] = pd.to_numeric(out[col], errors="coerce").astype(target)
            elif dt.lower().startswith("float"):
                out[col] = pd.to_numeric(out[col], errors="coerce").astype(dt)
            elif dt.lower().startswith("datetime64"):
                out[col] = pd.to_datetime(out[col], errors="coerce", utc=True)
            else:
                out[col] = out[col].astype(dt)
        except Exception as e:
            logger.warning("Unable to cast dtype for %s -> %s: %s", col, dt, e)
    return out


def enforce_domains(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """
    Check canonical domains (e.g., EnergyClass) and return violation counts.
    Does not mutate df; fixing is the caller's responsibility.
    """
    violations: Dict[str, Dict[str, int]] = {}

    if ENERGY_CLASS in df.columns:
        valid = {e.value for e in EnergyClass}
        bad_mask = ~df[ENERGY_CLASS].isin(valid) & df[ENERGY_CLASS].notna()
        n_bad = int(bad_mask.sum())
        if n_bad:
            violations[ENERGY_CLASS] = {"invalid_values": n_bad}

    if ZONE in df.columns:
        valid_zones = {"center", "semi_center", "periphery"}
        n_bad = int((~df[ZONE].isin(valid_zones) & df[ZONE].notna()).sum())
        if n_bad:
            violations[ZONE] = {"invalid_values": n_bad}

    # Numeric bounds (best-effort)
    for col, (lo, hi) in RANGE_BOUNDS.items():
        if col not in df.columns:
            continue
        try:
            xs = pd.to_numeric(df[col], errors="coerce")
            bad = xs.notna() & ((xs < lo) | (xs > hi) | ~pd.to_numeric(xs, errors="coerce").map(math.isfinite))
            n = int(bad.sum())
            if n:
                violations[col] = {"out_of_bounds": n, "range": [lo, hi]}
        except Exception:
            # Non-fatal: keep best-effort spirit
            pass

    return violations


def enforce_categoricals(df: pd.DataFrame, categoricals: Iterable[str] = CATEGORICAL) -> pd.DataFrame:
    """Cast present categorical columns to pandas 'category' dtype (best-effort)."""
    out = df.copy()
    for col in categoricals:
        if col in out.columns:
            try:
                out[col] = out[col].astype("category")
            except Exception as e:
                logger.debug("Skipped category cast for %s: %s", col, e)
    return out


def normalize_column_order(df: pd.DataFrame, asset_type: str = "property") -> pd.DataFrame:
    """Place required fields first, then the remaining columns in stable order."""
    schema = SCHEMA[asset_type]
    req = schema.required
    extras = [c for c in df.columns if c not in req]
    ordered = req + extras
    return df.loc[:, [c for c in ordered if c in df.columns]]


def validate_df(
    df: pd.DataFrame,
    asset_type: str = "property",
    *,
    coerce: bool = True,
    normalize_bools: bool = True,
    check_domains: bool = True,
    check_cross_fields: bool = True,
    strict: bool = False,
) -> Dict[str, object]:
    """
    Validate a DataFrame against the schema:
      - required columns present
      - unknown columns detected
      - suggested dtype coercion (optional, best-effort)
      - boolean normalization to 0/1 (optional)
      - canonical domain checks (optional)
      - cross-field rules (optional)
    Returns a validation report; does not mutate the input df.
    In strict mode, 'ok' is False if any violation occurs.
    """
    schema = SCHEMA[asset_type]
    dfx = apply_aliases(df)

    required = schema.required
    all_known = schema.all_fields()

    missing = list_missing(dfx, required)
    unknown = list_unknown(dfx, all_known)

    if coerce and schema.dtypes:
        dfx = coerce_dtypes(dfx, schema.dtypes)
    if normalize_bools:
        dfx = normalize_booleans(dfx)
    if coerce:
        dfx = enforce_categoricals(dfx)

    domain_violations = enforce_domains(dfx) if check_domains else {}
    cross_violations = enforce_cross_field_rules(dfx) if check_cross_fields else {}

    ok = (len(missing) == 0) and (not domain_violations) and (not cross_violations)
    if strict:
        ok = ok and (len(unknown) == 0)

    report: Dict[str, object] = {
        "ok": ok,
        "missing": missing,
        "unknown": unknown,
        "domain_violations": domain_violations,
        "cross_field_violations": cross_violations,
        "suggested_dtypes_applied": [c for c in schema.dtypes.keys() if c in df.columns] if coerce else [],
    }
    return report


def validate_and_coerce(
    df: pd.DataFrame,
    asset_type: str = "property",
    *,
    check_domains: bool = True,
    check_cross_fields: bool = True,
    normalize_bools: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Convenience variant: apply aliases/coerce/bools/categoricals/order and return (df_norm, report).
    The report reflects validation on the *original* df (transparency).
    """
    schema = SCHEMA[asset_type]
    dfx = apply_aliases(df)
    dfx = coerce_dtypes(dfx, schema.dtypes)
    if normalize_bools:
        dfx = normalize_booleans(dfx)
    dfx = enforce_categoricals(dfx)
    dfx = normalize_column_order(dfx, asset_type=asset_type)

    report = validate_df(df, asset_type=asset_type, coerce=True, normalize_bools=normalize_bools,
                         check_domains=check_domains, check_cross_fields=check_cross_fields)
    return dfx, report