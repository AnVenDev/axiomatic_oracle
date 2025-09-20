"""
Unified sanity-checks module for the Property RWA project.

Scope
- Dataset validation (schema, ranges, logical checks, domain checks, consistency, temporal)
- Schema-only validation
- Single-record validation & normalization for 'property'
- Pricing benchmarks (medians by location/zone) and ordering sanity checks

Design
- Pure utilities: return dicts/DataFrames; no I/O.
- Defensive logging; never crash on optional checks.
- Backward-compatible public API.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Set, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from notebooks.shared.common.constants import LEAKY_FEATURES, EXPECTED_PRED_RANGE
from notebooks.shared.common.schema import get_required_fields
from notebooks.shared.common.constants import (
    # core fields
    ASSET_ID, ASSET_TYPE, VALUATION_K, SIZE_M2, PRICE_PER_SQM,
    BUILDING_FLOORS, FLOOR, IS_TOP_FLOOR, IS_GROUND_FLOOR,
    ENERGY_CLASS, ENERGY_CLASSES,
    CONDITION, HEATING, ORIENTATION, VIEW,
    LAST_VERIFIED_TS, PREDICTION_TS, LAG_HOURS,
    # score columns (if present)
    CONDITION_SCORE, RISK_SCORE, LUXURY_SCORE, ENV_SCORE,
    # binary columns (if present)
    HAS_GARDEN, HAS_BALCONY, GARAGE, HAS_ELEVATOR,
    OWNER_OCCUPIED, PUBLIC_TRANSPORT_NEARBY,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Dataset-level
    "DataValidator",
    "validate_dataset",
    "validate_schema",
    "validate_property",
    # Simple helpers
    "validate_columns",
    "validate_allowed_values",
    "validate_nulls",
    # Pricing benchmarks
    "price_benchmark",
    "critical_city_order_check",
    # Gates
    "leakage_gate",
    "scale_gate",
]

# ---------------------------------------------------------------------------
# Simple helpers
# ---------------------------------------------------------------------------

def validate_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Return True if all required columns are present in the DataFrame."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        logger.error("Missing required columns: %s", missing)
        return False
    logger.debug("All required columns are present.")
    return True


def validate_allowed_values(df: pd.DataFrame, allowed_values: Dict[str, List[Any]]) -> bool:
    """Return False if any column contains values outside the allowed set."""
    valid = True
    for col, values in allowed_values.items():
        if col in df.columns:
            invalid_values = set(df[col].dropna()) - set(values)
            if invalid_values:
                logger.warning("Disallowed values in column %s: %s", col, invalid_values)
                valid = False
    return valid


def validate_nulls(df: pd.DataFrame, allow_nulls: Dict[str, bool]) -> bool:
    """Return False if a non-nullable column contains nulls."""
    valid = True
    for col, allow in allow_nulls.items():
        if col in df.columns:
            null_count = int(df[col].isna().sum())
            if not allow and null_count > 0:
                logger.error("Column %s has %d nulls but nulls are not allowed", col, null_count)
                valid = False
    return valid


# ---------------------------------------------------------------------------
# Local domains (record-level) — kept intentionally narrow and explicit
# ---------------------------------------------------------------------------

VALID_ORIENTATIONS: List[str] = [
    "North", "South", "East", "West",
    "North-East", "North-West", "South-East", "South-West",
]
VALID_VIEWS: List[str] = ["street", "inner courtyard", "garden", "park", "sea", "mountain", "landmarks"]
VALID_STATES: List[str] = ["new", "renovated", "good", "needs_renovation"]
VALID_HEATING: List[str] = ["autonomous", "centralized", "heat pump", "none"]

_SCORE_COLS = [CONDITION_SCORE, RISK_SCORE, LUXURY_SCORE, ENV_SCORE]
_BINARY_COLS = [
    HAS_GARDEN, HAS_BALCONY, GARAGE, HAS_ELEVATOR,
    OWNER_OCCUPIED, PUBLIC_TRANSPORT_NEARBY,
]
_CRITICAL_COLS = [ASSET_ID, ASSET_TYPE, VALUATION_K]

# Column names used by pricing benchmarks
PRICE_COL: str = "price_per_sqm"
LOCATION_COL: str = "location"
ZONE_COL: str = "zone"


# ---------------------------------------------------------------------------
# Dataset validator
# ---------------------------------------------------------------------------

class DataValidator:
    """End-to-end DataFrame validation against schema and business rules."""

    def __init__(self, asset_type: str = "property") -> None:
        self.asset_type = asset_type
        self.required: Set[str] = set(get_required_fields(asset_type))

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "asset_type": self.asset_type,
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "schema": self._check_schema(df),
            "ranges": self._check_ranges(df),
            "logic": self._check_logical(df),
            "domains": self._check_domains(df),
            "consistency": self._check_consistency(df),
            "temporal": self._check_temporal(df),
        }
        report["overall_passed"] = self._compute_overall(report)

        logger.info(
            "[DATA_VALIDATION] Completed: overall_passed=%s rows=%d",
            report["overall_passed"], len(df)
        )
        return report

    # ---- individual checks -------------------------------------------------

    def _check_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        present = set(df.columns)
        missing = sorted(self.required - present)
        extra = sorted(present - self.required)
        if missing:
            logger.error("Schema check failed; missing required fields: %s", missing)
        else:
            logger.info("Schema check passed.")
        return {"missing": missing, "extra": extra}

    def _check_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        # Scores must be within [0, 1]
        for col in _SCORE_COLS:
            if col in df:
                mask = ~pd.to_numeric(df[col], errors="coerce").between(0, 1, inclusive="both")
                count = int(mask.sum())
                out[col] = {
                    "invalid_count": count,
                    "invalid_pct": (float(count) / len(df) * 100.0) if len(df) else 0.0,
                }
                if count:
                    logger.warning("Values out of [0,1] for %s: %d rows", col, count)
        # Reasonable size guardrails
        if SIZE_M2 in df:
            size = pd.to_numeric(df[SIZE_M2], errors="coerce")
            small = int((size < 10).sum())
            large = int((size > 1000).sum())
            out[SIZE_M2] = {
                "too_small": small, "too_large": large,
                "too_small_pct": (float(small) / len(df) * 100.0) if len(df) else 0.0,
                "too_large_pct": (float(large) / len(df) * 100.0) if len(df) else 0.0,
            }
            if small or large:
                logger.warning("SIZE_M2 out of bounds (small=%d, large=%d)", small, large)
        return out

    def _check_logical(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Logical checks:
        - unique asset IDs
        - positive size/valuation
        - floor <= building_floors - 1
        """
        out: Dict[str, Any] = {}

        # Unique ID
        if ASSET_ID in df:
            is_unique = pd.Series(df[ASSET_ID]).is_unique
            out["unique_id"] = bool(is_unique)
            if not is_unique:
                logger.error("[VALIDATION] Non-unique Asset IDs detected")

        # Positive main metrics
        size_series = pd.to_numeric(df.get(SIZE_M2, pd.Series(dtype="float64")), errors="coerce")
        val_series = pd.to_numeric(df.get(VALUATION_K, pd.Series(dtype="float64")), errors="coerce")

        size_check = (size_series.dropna() > 0)
        positive_size = bool(size_check.all()) if len(size_check) > 0 else True

        val_check = (val_series.dropna() > 0)
        positive_value = bool(val_check.all()) if len(val_check) > 0 else True

        out["positive_size"] = positive_size
        out["positive_value"] = positive_value

        if not positive_size:
            logger.warning("[VALIDATION] Found non-positive sizes")
        if not positive_value:
            logger.warning("[VALIDATION] Found non-positive valuations")

        # Floor vs building
        if {FLOOR, BUILDING_FLOORS}.issubset(df.columns):
            floor_series = pd.to_numeric(df[FLOOR], errors="coerce")
            building_series = pd.to_numeric(df[BUILDING_FLOORS], errors="coerce")
            ok = bool((floor_series <= (building_series - 1)).all())
            out["floor_logic"] = ok
            if not ok:
                logger.warning("[VALIDATION] Inconsistent floor vs building_floors")

        return out

    def _check_domains(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Best-effort domain checks for known categorical columns (if present)."""
        out: Dict[str, Any] = {}

        # Energy class
        if ENERGY_CLASS in df:
            invalid = df[~df[ENERGY_CLASS].isin(ENERGY_CLASSES)][ENERGY_CLASS]
            out["energy_class"] = {"invalid_count": int(invalid.shape[0])}
            if len(invalid) > 0:
                logger.warning("[VALIDATION] Energy class out of domain: %d rows", len(invalid))

        # Orientation
        if ORIENTATION in df:
            invalid = df[~df[ORIENTATION].isin(VALID_ORIENTATIONS)][ORIENTATION]
            out["orientation"] = {"invalid_count": int(invalid.shape[0])}

        # View
        if VIEW in df:
            invalid = df[~df[VIEW].isin(VALID_VIEWS)][VIEW]
            out["view"] = {"invalid_count": int(invalid.shape[0])}

        # Condition
        if CONDITION in df:
            invalid = df[~df[CONDITION].isin(VALID_STATES)][CONDITION]
            out["condition"] = {"invalid_count": int(invalid.shape[0])}

        # Heating
        if HEATING in df:
            invalid = df[~df[HEATING].isin(VALID_HEATING)][HEATING]
            out["heating"] = {"invalid_count": int(invalid.shape[0])}

        return out

    def _check_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Cross-field consistency checks for top/ground floor flags."""
        out: Dict[str, Any] = {}
        # Top-floor flag
        if {IS_TOP_FLOOR, FLOOR, BUILDING_FLOORS}.issubset(df.columns):
            mask = (pd.to_numeric(df[IS_TOP_FLOOR], errors="coerce") == 1) & (
                pd.to_numeric(df[FLOOR], errors="coerce") != pd.to_numeric(df[BUILDING_FLOORS], errors="coerce") - 1
            )
            out["top_floor_flag_ok"] = (int(mask.sum()) == 0)
        # Ground-floor flag
        if {IS_GROUND_FLOOR, FLOOR}.issubset(df.columns):
            mask = (pd.to_numeric(df[IS_GROUND_FLOOR], errors="coerce") == 1) & (
                pd.to_numeric(df[FLOOR], errors="coerce") != 0
            )
            out["ground_floor_flag_ok"] = (int(mask.sum()) == 0)
        return out

    def _check_temporal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Basic temporal hygiene: missing timestamps and negative lag."""
        out: Dict[str, Any] = {}
        for col in (LAST_VERIFIED_TS, PREDICTION_TS):
            if col in df:
                invalid = int(pd.isna(df[col]).sum())
                out[col] = {
                    "invalid_count": invalid,
                    "invalid_pct": (float(invalid) / len(df) * 100.0) if len(df) else 0.0,
                }
        if LAG_HOURS in df:
            neg = int((pd.to_numeric(df[LAG_HOURS], errors="coerce") < 0).sum())
            out[LAG_HOURS] = {
                "negative_count": neg,
                "negative_pct": (float(neg) / len(df) * 100.0) if len(df) else 0.0,
            }
        return out

    def _compute_overall(self, rpt: Dict[str, Any]) -> bool:
        """Minimal overall pass flag based on schema and core logic checks."""
        ok = not rpt["schema"]["missing"]
        ok &= rpt["logic"].get("unique_id", True)
        ok &= rpt["logic"].get("positive_size", True)
        ok &= rpt["logic"].get("positive_value", True)
        return bool(ok)


# ---------------------------------------------------------------------------
# Public wrappers
# ---------------------------------------------------------------------------

def validate_dataset(
    df: pd.DataFrame,
    asset_type: str = "property",
    raise_on_failure: bool = True,
) -> Dict[str, Any]:
    """Validate a dataset; raise if `raise_on_failure` and overall fails."""
    validator = DataValidator(asset_type)
    report = validator.validate(df)
    if raise_on_failure and not report["overall_passed"]:
        raise RuntimeError(f"Dataset validation failed: {report['schema']['missing']}")
    return report


def validate_schema(df: pd.DataFrame, asset_type: str) -> None:
    """Assert that all required schema fields for `asset_type` are present."""
    required = set(get_required_fields(asset_type))
    missing = required - set(df.columns)
    if missing:
        raise AssertionError(f"[SCHEMA ERROR] Missing fields for '{asset_type}': {sorted(missing)}")
    logger.info("[SCHEMA] ✅ All required fields for '%s' present (%d fields).", asset_type, len(required))


def validate_property(prop_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate & normalize a single 'property' record (in-place best-effort).
    - Normalizes legacy/Italian keys
    - Enforces structural coherence (floors)
    - Clips/repairs score fields to [0,1]
    - Sets reasonable minimums for key magnitudes
    - Computes valuation/price fallbacks where missing
    - Ensures domain membership for categorical fields
    - Completes minimal schema (non-fatal)
    """
    original = dict(prop_data)  # snapshot for diff logging
    errors: List[str] = []
    flags: List[str] = []

    # 0) Legacy key normalization
    _normalize_legacy_keys(prop_data)

    # 1) Floor coherence
    floor_val = int(prop_data.get(FLOOR, prop_data.get("floor", 0)) or 0)
    bld_floors = int(prop_data.get(BUILDING_FLOORS, prop_data.get("building_floors", 0)) or 0)
    if floor_val > bld_floors:
        errors.append("floor_gt_building_floors")
        prop_data[FLOOR] = bld_floors
        flags.append("floor_adjusted")
        floor_val = bld_floors

    prop_data[IS_TOP_FLOOR] = int(floor_val == max(bld_floors - 1, 0))
    prop_data[IS_GROUND_FLOOR] = int(floor_val == 0)

    # 2) Score clipping to [0,1]
    for field in ("condition_score", "risk_score", "luxury_score", "env_score"):
        val = prop_data.get(field)
        if val is None:
            errors.append(f"{field}_missing")
            flags.append(f"{field}_missing")
            continue
        try:
            fval = float(val)
        except Exception:
            errors.append(f"{field}_non_numeric")
            flags.append(f"{field}_coerced_0")
            fval = 0.0
        if not (0.0 <= fval <= 1.0):
            errors.append(f"{field}_out_of_range")
            clipped = float(np.clip(fval, 0.0, 1.0))
            prop_data[field] = clipped
            flags.append(f"{field}_clipped")

    # 3) Reasonable minimums
    size = float(prop_data.get(SIZE_M2, prop_data.get("size_m2", 0)) or 0.0)
    if size < 20.0:
        errors.append("size_m2_too_small")
        prop_data[SIZE_M2] = 20.0
        flags.append("size_clamped")
        size = 20.0

    rooms = int(prop_data.get("rooms", prop_data.get("ROOMS", 0)) or 0)
    if rooms < 1:
        errors.append("rooms_too_few")
        prop_data["rooms"] = 1
        flags.append("rooms_clamped")

    baths = int(prop_data.get("bathrooms", prop_data.get("BATHROOMS", 0)) or 0)
    if baths < 1:
        errors.append("bathrooms_too_few")
        prop_data["bathrooms"] = 1
        flags.append("bathrooms_clamped")

    # 4) Valuation/price fallbacks
    valuation_k = float(prop_data.get(VALUATION_K, prop_data.get("valuation_k", 0)) or 0.0)
    if valuation_k < 10.0:
        errors.append("valuation_k_too_low_or_missing")
        fallback_price = (size * 500.0) / 1000.0  # heuristic 500 €/m²
        prop_data[VALUATION_K] = round(fallback_price, 2)
        flags.append("valuation_override")
        valuation_k = prop_data[VALUATION_K]

    pps = float(prop_data.get(PRICE_PER_SQM, prop_data.get("price_per_sqm", 0)) or 0.0)
    if pps <= 0.0 and size > 0:
        errors.append("price_per_sqm_non_positive_or_missing")
        prop_data[PRICE_PER_SQM] = round((valuation_k * 1000.0) / size, 2)
        flags.append("price_per_sqm_recomputed")

    # 5) Domain memberships
    if ENERGY_CLASS in prop_data and prop_data[ENERGY_CLASS] not in ENERGY_CLASSES:
        errors.append(f"invalid_energy_class:{prop_data.get(ENERGY_CLASS)}")
        prop_data[ENERGY_CLASS] = "C"
        flags.append("energy_class_reset")

    # Orientation
    orient_key = ORIENTATION if ORIENTATION in prop_data or "orientation" in prop_data else "orientation"
    if orient_key in prop_data and prop_data[orient_key] not in VALID_ORIENTATIONS:
        errors.append(f"invalid_orientation:{prop_data.get(orient_key)}")
        prop_data[orient_key] = VALID_ORIENTATIONS[0]
        flags.append("orientation_reset")

    # View
    view_key = VIEW if VIEW in prop_data or "view" in prop_data else "view"
    if view_key in prop_data and prop_data[view_key] not in VALID_VIEWS:
        errors.append(f"invalid_view:{prop_data.get(view_key)}")
        prop_data[view_key] = "street"
        flags.append("view_reset")

    # Condition
    cond_key = CONDITION if CONDITION in prop_data or "condition" in prop_data else "condition"
    if cond_key in prop_data and prop_data[cond_key] not in VALID_STATES:
        errors.append(f"invalid_state:{prop_data.get(cond_key)}")
        prop_data[cond_key] = "good"
        flags.append("state_reset")

    # Heating
    heat_key = HEATING if HEATING in prop_data or "heating" in prop_data else "heating"
    if heat_key in prop_data and prop_data[heat_key] not in VALID_HEATING:
        errors.append(f"invalid_heating:{prop_data.get(heat_key)}")
        prop_data[heat_key] = "autonomous"
        flags.append("heating_reset")

    # 6) Minimal schema completion (non-fatal)
    required = set(get_required_fields("property"))
    missing_keys = [k for k in required if k not in prop_data]
    if missing_keys:
        errors.append(f"missing_keys:{missing_keys}")
        flags.append("schema_incomplete")
        for k in missing_keys:
            if k == ASSET_ID:
                prop_data.setdefault(ASSET_ID, "unknown")
            elif k == ASSET_TYPE:
                prop_data.setdefault(ASSET_TYPE, "property")
            elif k == LAST_VERIFIED_TS:
                prop_data.setdefault(
                    LAST_VERIFIED_TS,
                    datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                )

    # 7) Metadata
    prop_data.setdefault("validation_errors", []).extend(errors)
    prop_data.setdefault("validation_flags", []).extend(flags)

    if errors:
        diff = {k: (original.get(k), prop_data.get(k)) for k in set(prop_data) | set(original)
                if original.get(k) != prop_data.get(k)}
        logger.warning(
            "[VALIDATION] Asset %s normalized. Errors=%s Flags=%s Changes=%s",
            prop_data.get(ASSET_ID, "unknown"),
            errors, flags, diff,
        )

    return prop_data


def _normalize_legacy_keys(prop_data: Dict[str, Any]) -> None:
    """Best-effort in-place mapping from Italian/legacy keys to canonical ones."""
    if "vista" in prop_data and VIEW not in prop_data and "view" not in prop_data:
        prop_data["view"] = prop_data.pop("vista")
    if "orientamento" in prop_data and ORIENTATION not in prop_data and "orientation" not in prop_data:
        prop_data["orientation"] = prop_data.pop("orientamento")
    if "stato" in prop_data and CONDITION not in prop_data and "condition" not in prop_data:
        prop_data["condition"] = prop_data.pop("stato")
    if "riscaldamento" in prop_data and HEATING not in prop_data and "heating" not in prop_data:
        prop_data["heating"] = prop_data.pop("riscaldamento")


# ---------------------------------------------------------------------------
# Pricing benchmarks
# ---------------------------------------------------------------------------

def price_benchmark(df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame | None]:
    """
    Compute price medians by city and optionally by (city, zone).

    Returns:
        (city_medians: Series, zone_pivot: DataFrame | None)
    """
    if PRICE_COL not in df.columns:
        raise ValueError(f"Missing column '{PRICE_COL}'.")
    # dropna to avoid NaN-medians
    city_med = (
        df[[LOCATION_COL, PRICE_COL]]
        .dropna()
        .groupby(LOCATION_COL, observed=True)[PRICE_COL]
        .median()
        .sort_values(ascending=False)
    )
    zone_med = None
    if ZONE_COL in df.columns:
        zone_med = (
            df[[LOCATION_COL, ZONE_COL, PRICE_COL]]
            .dropna()
            .groupby([LOCATION_COL, ZONE_COL], observed=True)[PRICE_COL]
            .median()
            .unstack(level=ZONE_COL)
        )
    return city_med, zone_med


def critical_city_order_check(
    city_med: pd.Series,
    min_ratio: float = 1.1,
    min_abs_diff: float = 80.0,
    require_both: bool = True,
) -> List[Dict[str, Any]]:
    """
    Check for significant gaps between adjacent cities in the price ranking.
    Emits an alert per adjacent pair comparing (prev_city vs current_city).
    """
    alerts: List[Dict[str, Any]] = []
    sorted_cities = city_med.dropna().sort_values(ascending=False)
    prev_city, prev_value = None, None
    for city, value in sorted_cities.items():
        if prev_value is not None:
            ratio = (prev_value / value) if value > 0 else float("inf")
            abs_diff = float(prev_value - value)
            alert_condition = (
                (ratio >= min_ratio) and (abs_diff >= min_abs_diff)
                if require_both else
                (ratio >= min_ratio) or (abs_diff >= min_abs_diff)
            )
            alerts.append({
                "city": prev_city,
                "ratio": float(ratio),
                "abs_diff": float(abs_diff),
                "passes": not alert_condition,
                "detail": f"{prev_city} vs {city}",
            })
        prev_city, prev_value = city, float(value)
    return alerts


# ---------------------------------------------------------------------------
# Gates (used in serving/monitoring pipelines)
# ---------------------------------------------------------------------------

def leakage_gate(columns: list[str]) -> tuple[bool, list[str]]:
    """Reject feature sets that contain leaky columns."""
    bad = sorted(set(columns) & LEAKY_FEATURES)
    return (len(bad) == 0, bad)


def scale_gate(value: float, expected: tuple[float, float] = EXPECTED_PRED_RANGE) -> tuple[bool, str]:
    """Ensure price-per-sqm-like values fall within an expected range."""
    lo, hi = expected
    ok = (value >= lo) and (value <= hi)
    return ok, ("" if ok else f"prediction {value:.2f} out of expected [{lo},{hi}]")
