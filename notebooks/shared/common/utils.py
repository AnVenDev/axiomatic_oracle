from __future__ import annotations

"""
Utility functions (notebooks/shared/common/utils.py)

Scope
- JSON encoding for numpy/pandas types (stable, compact canonical dumps)
- Deterministic global seeding (numpy + random)
- Robust ISO8601 parsing (UTC-aware)
- Lightweight dtype optimization for DataFrames
- Basic dataset diagnostics (log-only)
- Location canonicalization & city mapping helpers

Notes
- Pure, side-effect free except for logging and RNG seeding.
- Safe to import in notebooks and light-weight pipelines.
"""

import hashlib
import json
import logging
import random
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from shared.common.constants import (
    DEFAULT_REGION_BY_CITY,
    DEFAULT_URBAN_TYPE_BY_CITY,
    LOCATION,
    ZONE,
)

logger = logging.getLogger(__name__)

__all__ = [
    # JSON & hashing
    "NumpyJSONEncoder",
    "canonical_json_dumps",
    "sha256_hex",
    # Time / RNG
    "set_global_seed",
    "get_utc_now",
    "parse_iso8601",
    # DataFrame helpers
    "optimize_dtypes",
    "log_basic_diagnostics",
    # Location helpers
    "canonical_location",
    "derive_city_mappings",
    "normalize_location_weights",
]


# -----------------------------------------------------------------------------
# JSON utilities
# -----------------------------------------------------------------------------

class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder with first-class support for numpy, pandas, and datetimes."""
    def default(self, obj: Any) -> Any:  # noqa: D401
        # numpy scalars
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        # numpy & pandas containers
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        # datetimes
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)


def canonical_json_dumps(obj: Any) -> str:
    """
    Produce a stable, compact JSON string:
    - sort_keys=True to enable reproducible hashing
    - separators=(',', ':') to minimize size (useful for ≤1KB notes)
    - ensure_ascii=False to avoid unnecessary escapes
    """
    return json.dumps(
        obj,
        cls=NumpyJSONEncoder,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def sha256_hex(data: Union[str, bytes]) -> str:
    """Compute SHA-256 hex digest for given data (str encoded as UTF-8)."""
    b = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(b).hexdigest()


# -----------------------------------------------------------------------------
# Time / RNG
# -----------------------------------------------------------------------------

def set_global_seed(seed: int) -> np.random.Generator:
    """
    Set deterministic global RNG seeds (random + numpy) and return a local Generator.

    Returns:
        np.random.Generator seeded with `seed`.
    """
    random.seed(seed)
    np.random.seed(seed)  # legacy global RNG (still used by some libs)
    rng = np.random.default_rng(seed)  # preferred modern generator
    logger.info("[UTILS] Global seed set", extra={"seed": seed})
    return rng


def get_utc_now() -> datetime:
    """Return current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


def parse_iso8601(s: Optional[str]) -> Optional[datetime]:
    """
    Parse ISO8601 strings to UTC datetime.
    Accepts 'Z' suffix or explicit offsets; returns None on failure.
    """
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception as e:
        logger.warning("[UTILS] ISO8601 parsing failed", extra={"input": s, "error": str(e)})
        return None


# -----------------------------------------------------------------------------
# DataFrame helpers
# -----------------------------------------------------------------------------

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a memory-optimized copy of `df`:
    - float64 → float32
    - (pandas) Int64/int64 → Int16 or Int32 (nullable)
    - leave categories/objects untouched

    Notes
    - Best-effort: does not raise on conversion errors.
    - Preserves NaN semantics by using pandas nullable integer dtypes.
    """
    before = df.memory_usage(deep=True).sum()
    df_opt = df.copy()

    # Floats
    for col in df_opt.select_dtypes(include=["float64"]).columns:
        try:
            df_opt[col] = df_opt[col].astype("float32")
        except Exception:
            logger.debug("Skip float downcast for %s", col)

    # Signed integers (numpy int64) and pandas nullable Int64
    int_like = list(df_opt.select_dtypes(include=["int64"]).columns)
    int_like += [c for c in df_opt.columns if str(df_opt[c].dtype) == "Int64"]  # pandas nullable
    for col in sorted(set(int_like)):
        try:
            s = pd.to_numeric(df_opt[col], errors="coerce")
            mn, mx = s.min(), s.max()
            # Choose smallest sufficient nullable int width
            if pd.notna(mn) and pd.notna(mx):
                if -32768 <= mn <= mx <= 32767:
                    df_opt[col] = s.astype("Int16")
                else:
                    df_opt[col] = s.astype("Int32")
        except Exception:
            logger.debug("Skip int downcast for %s", col)

    after = df_opt.memory_usage(deep=True).sum()
    logger.debug(
        "[UTILS] Dtype optimization: %.1f KB → %.1f KB (Δ=%.1f KB)",
        before / 1024,
        after / 1024,
        (before - after) / 1024,
    )
    return df_opt


def log_basic_diagnostics(df: pd.DataFrame, log: logging.Logger = logger) -> None:
    """
    Log a few coarse dataset diagnostics (non-intrusive; no exceptions).
    """
    try:
        if LOCATION in df.columns:
            log.info("[UTILS] Distribution by location:\n%s", df[LOCATION].value_counts().to_string())
        if "valuation_k" in df.columns:  # kept literal for portability across callers
            log.info("[UTILS] Valuation min: %.2fk€", float(pd.to_numeric(df["valuation_k"], errors="coerce").min()))
            log.info("[UTILS] Valuation max: %.2fk€", float(pd.to_numeric(df["valuation_k"], errors="coerce").max()))
            log.info("[UTILS] Valuation mean: %.2fk€", float(pd.to_numeric(df["valuation_k"], errors="coerce").mean()))
        if {"size_m2", "valuation_k"}.issubset(df.columns):
            corr = (
                pd.to_numeric(df["size_m2"], errors="coerce")
                .corr(pd.to_numeric(df["valuation_k"], errors="coerce"))
            )
            if pd.notna(corr):
                log.info("[UTILS] Corr(size_m2, valuation_k): %.3f", float(corr))
    except Exception as e:
        log.debug("Diagnostics skipped: %s", e)


# -----------------------------------------------------------------------------
# Location helpers
# -----------------------------------------------------------------------------

def canonical_location(record: Union[Dict[str, Any], pd.Series]) -> str:
    """
    Return a record's canonical location string (best-effort).
    Preference order: LOCATION → ZONE → "".
    """
    try:
        if LOCATION in record and record[LOCATION]:
            return str(record[LOCATION])
        if ZONE in record and record[ZONE]:
            return str(record[ZONE])
    except Exception:
        pass
    return ""


def derive_city_mappings(
    source: Union[Dict[str, Any], List[str]],
    urban_override: Optional[Dict[str, str]] = None,
    region_override: Optional[Dict[str, str]] = None,
) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """
    Merge urban/region mappings for the provided set of cities.

    Args:
        source: config dict (with 'location_weights') or a list of city names.
        urban_override / region_override: optional override maps.

    Returns:
        (city_list, urban_type_by_city, region_by_city)
    """
    if isinstance(source, dict):
        locs = list((source.get("location_weights") or {}).keys())
        urban_override = urban_override or source.get("urban_type_by_city")
        region_override = region_override or source.get("region_by_city")
    elif isinstance(source, list):
        locs = source
    else:
        raise ValueError("source must be a dict (config) or a list of city names")

    urban_map = dict(DEFAULT_URBAN_TYPE_BY_CITY)
    region_map = dict(DEFAULT_REGION_BY_CITY)

    if urban_override:
        urban_map.update(urban_override)
    if region_override:
        region_map.update(region_override)

    # Fill missing cities with sensible defaults
    for city in set(locs) - set(urban_map):
        logger.warning("[UTILS] Missing urban type; defaulting to 'urban'", extra={"city": city})
        urban_map[city] = "urban"
    for city in set(locs) - set(region_map):
        logger.warning("[UTILS] Missing region; defaulting to 'unknown'", extra={"city": city})
        region_map[city] = "unknown"

    return locs, urban_map, region_map


def normalize_location_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize and validate location weights to a probability simplex.

    Raises:
        ValueError if empty or non-positive total.
    """
    if not weights:
        raise ValueError("location_weights is empty or None")
    total = float(sum(weights.values()))
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("sum(location_weights) must be > 0")

    normalized = {k: float(v) / total for k, v in weights.items()}
    s = sum(normalized.values())
    if not np.isclose(s, 1.0):
        # Final pass to ensure exact normalization in case of float drift
        normalized = {k: v / s for k, v in normalized.items()}
        logger.warning("[UTILS] Location weights did not sum to 1.0 after normalization; re-normalized.")
    return normalized
