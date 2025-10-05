from __future__ import annotations
"""
Pricing utilities (single source of truth).

Public API
- normalize_priors(raw) -> dict
- calculate_price_per_sqm_base(city, zone, city_base_prices, default_fallback)
- apply_pricing_pipeline(row, priors, base_price) -> float         # deterministic, no seasonality/noise
- apply_hedonic_adjustments(row, pricing_input, seasonality, city_base_prices, rng) -> float
- explain_price(row, priors, seasonality, city_base_prices) -> dict # transparent breakdown (no noise)

Design
- Pure/deterministic except for the optional, controlled noise in `apply_hedonic_adjustments`
  (pass a seeded RNG to make it reproducible).
- Zero I/O. Inputs are plain mappings; callers own schema enforcement.
- Priors are normalized to a fixed internal structure by `normalize_priors`.
"""

from typing import Any, Dict, Mapping, Optional, List, Tuple, Union
from datetime import datetime
import logging

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

# Canonical column names / values
from shared.common.constants import (
    LOCATION,
    ZONE,
    YEAR_BUILT,
    ENERGY_CLASS,
    IS_TOP_FLOOR,
    IS_GROUND_FLOOR,
    HAS_BALCONY,
    HAS_GARDEN,
    GARAGE,
    VIEW,
    VIEW_SEA,
    VIEW_LANDMARKS,
    ORIENTATION,
    HEATING,
    CONDITION,          # canonical column; we still accept legacy 'state'
    Cols,
)

__all__ = [
    "normalize_priors",
    "apply_pricing_pipeline",
    "apply_hedonic_adjustments",
    "explain_price",
    "calculate_price_per_sqm_base",
]

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Tunables / heuristics
# -----------------------------------------------------------------------------

ORIENTATION_BONUS: float = 1.05             # south-ish exposure bonus
HEATING_AUTONOMOUS_BONUS: float = 1.03      # autonomous heating bonus
DEFAULT_BASE_PRICE_FALLBACK: float = 3000.0 # €/m² when city/zone missing
NOISE_RANGE: Tuple[float, float] = (0.95, 1.05)

_SUNNY_ORIENTATIONS = {"south", "south-east", "south-west", "southeast", "southwest"}
_HEATING_AUTONOMOUS = "autonomous"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _get_month(row: Mapping[str, Any]) -> Optional[int]:
    """Extract month (1..12) from row. Accepts Cols.LISTING_MONTH or a legacy 'month' field."""
    month_val = row.get(Cols.LISTING_MONTH, row.get("month"))
    try:
        if month_val is None:
            return None
        m = int(month_val)
        return m if 1 <= m <= 12 else None
    except Exception:
        return None


def _norm_str(x: Any) -> str:
    """Lower/strip safe string normalization."""
    try:
        s = str(x)
    except Exception:
        return ""
    return s.strip().lower()


def _to_bool(x: Any) -> bool:
    """Best-effort boolean coercion for flags stored as 0/1, 'true'/'false', etc."""
    if isinstance(x, bool):
        return x
    s = _norm_str(x)
    if s in {"1", "true", "yes", "y", "t"}:
        return True
    if s in {"0", "false", "no", "n", "f"}:
        return False
    # Fallback: numeric non-zero is True
    try:
        return float(x) != 0.0
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Base €/m²
# -----------------------------------------------------------------------------

def calculate_price_per_sqm_base(
    city: str,
    zone: str,
    city_base_prices: Mapping[str, Mapping[str, float]],
    default_fallback: float = DEFAULT_BASE_PRICE_FALLBACK,
) -> float:
    """
    Resolve base €/m² from (city, zone). Fallbacks:
      1) City mean over known zones, if city exists.
      2) `default_fallback` otherwise.
    """
    try:
        city_prices = dict(city_base_prices.get(city, {}) or {})
        if zone in city_prices:
            return float(city_prices[zone])
        if city_prices:
            return float(np.mean(list(city_prices.values())))
        return float(default_fallback)
    except Exception:
        # Defensive fallback to keep pipelines resilient
        return float(default_fallback)


# -----------------------------------------------------------------------------
# Priors normalization
# -----------------------------------------------------------------------------

def normalize_priors(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Normalize priors into a unified schema:

      view_multipliers: { "sea":1.xx, "landmarks":1.xx }
      floor_modifiers:  { "is_top_floor":+Δ, "is_ground_floor":+Δ }   # Δ are additive to multiplier-1
      build_age:        { "new":+Δ, "recent":+Δ, "old":+Δ }           # Δ additive
      energy_class_multipliers: { "A":1.xx, ..., "G":0.xx }           # multiplicative
      state_modifiers:  { "new":1.xx, "renovated":1.xx, "good":1.0, "needs_renovation":0.xx } # multiplicative
      extras:           { "has_balcony":+Δ, "has_garage":+Δ, "has_garden":+Δ }                # Δ additive

    Notes:
      - Values under 'build_age' and 'extras' are additive deltas applied as (1 + delta).
      - Multiplicative sections (‘energy_class_multipliers’, ‘state_modifiers’, ‘view_multipliers’)
        are applied directly as multipliers.
    """
    src = dict(raw or {})
    build_age_raw = dict(src.get("build_age", {}) or {})

    # Lowercase keys for view multipliers, to be robust to row normalization
    vm = { _norm_str(k): float(v) for k, v in dict(src.get("view_multipliers", {VIEW_SEA: 1.0, VIEW_LANDMARKS: 1.0})).items() }

    return {
        "view_multipliers": vm,
        "floor_modifiers": dict(src.get("floor_modifiers", {"is_top_floor": 0.0, "is_ground_floor": 0.0})),
        "build_age": {
            "new": float(build_age_raw.get("new", 0.0)),
            "recent": float(build_age_raw.get("recent", 0.0)),
            "old": float(build_age_raw.get("old", 0.0)),
        },
        "energy_class_multipliers": dict(src.get("energy_class_multipliers", {})),
        "state_modifiers": dict(src.get("state_modifiers", {})),
        "extras": dict(src.get("extras", {"has_balcony": 0.0, "has_garage": 0.0, "has_garden": 0.0})),
    }


# -----------------------------------------------------------------------------
# Core pricing pipeline (deterministic, no seasonality/noise)
# -----------------------------------------------------------------------------

def apply_pricing_pipeline(row: Mapping[str, Any], priors: Mapping[str, Any], base_price: float) -> float:
    """
    Apply normalized priors to the base €/m² in a deterministic manner.
    Excludes local heuristics (orientation/heating), seasonality and noise.

    Sections:
      - build_age: (new/recent/old) → multiplicative (1 + delta)
      - floor_modifiers:            → multiplicative (1 + delta)
      - energy_class_multipliers:   → multiplicative (value)
      - state_modifiers:            → multiplicative (value)
      - extras: (balcony/garage/garden) → multiplicative (1 + delta)
      - view_multipliers:           → multiplicative (value)
    """
    price = float(base_price)
    pri = normalize_priors(priors)

    # Build age from YEAR_BUILT
    year = row.get(YEAR_BUILT)
    try:
        year_int = int(year) if year is not None else None
    except Exception:
        year_int = None

    if year_int is not None:
        current_year = datetime.now().year
        age = max(current_year - year_int, 0)
        if age <= 1:
            price *= 1.0 + float(pri["build_age"].get("new", 0.0))
        elif age <= 15:
            price *= 1.0 + float(pri["build_age"].get("recent", 0.0))
        elif age >= 50:
            price *= 1.0 + float(pri["build_age"].get("old", 0.0))

    # Floor modifiers
    if _to_bool(row.get(IS_TOP_FLOOR, False)):
        price *= 1.0 + float(pri["floor_modifiers"].get("is_top_floor", 0.0))
    if _to_bool(row.get(IS_GROUND_FLOOR, False)):
        price *= 1.0 + float(pri["floor_modifiers"].get("is_ground_floor", 0.0))

    # Energy class multiplier (expects A..G; unknown -> 1.0)
    energy_class = str(row.get(ENERGY_CLASS, "") or "")
    price *= float(pri["energy_class_multipliers"].get(energy_class, 1.0))

    # Condition/state multiplier (support both canonical CONDITION and legacy 'state')
    state_val = str(row.get(CONDITION, row.get("state", "good")) or "good")
    price *= float(pri["state_modifiers"].get(state_val, 1.0))

    # Extras
    if _to_bool(row.get(HAS_BALCONY, False)):
        price *= 1.0 + float(pri["extras"].get("has_balcony", 0.0))
    if _to_bool(row.get(GARAGE, False)):
        price *= 1.0 + float(pri["extras"].get("has_garage", 0.0))
    if _to_bool(row.get(HAS_GARDEN, False)):
        price *= 1.0 + float(pri["extras"].get("has_garden", 0.0))

    # View multiplier (case-insensitive)
    view_val = _norm_str(row.get(VIEW, ""))
    price *= float(pri["view_multipliers"].get(view_val, 1.0))

    return float(price)


# -----------------------------------------------------------------------------
# Full hedonic pipeline (heuristics + seasonality + noise)
# -----------------------------------------------------------------------------

def apply_hedonic_adjustments(
    row: Mapping[str, Any],
    pricing_input: Mapping[str, Any],
    seasonality: Mapping[int, float],
    city_base_prices: Mapping[str, Mapping[str, float]],
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Full pricing pipeline:
        base €/m² → priors → orientation/heating → seasonality → noise

    Notes:
      - To make the result deterministic, pass a seeded `rng` (e.g., np.random.default_rng(42)).
      - All multipliers are applied multiplicatively; priors are normalized internally.
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1) Base €/m²
    city = str(row.get(LOCATION, "") or "")
    zone = str(row.get(ZONE, "") or "")
    base_price = calculate_price_per_sqm_base(city, zone, city_base_prices, DEFAULT_BASE_PRICE_FALLBACK)

    # 2) Priors (normalized) without local heuristics
    price = apply_pricing_pipeline(row, normalize_priors(pricing_input), base_price)

    # 3) Local heuristics: orientation & heating (case-insensitive)
    orientation_val = _norm_str(row.get(ORIENTATION, ""))
    if orientation_val in _SUNNY_ORIENTATIONS:
        price *= ORIENTATION_BONUS

    heating_val = _norm_str(row.get(HEATING, ""))
    if heating_val == _HEATING_AUTONOMOUS:
        price *= HEATING_AUTONOMOUS_BONUS

    # 4) Seasonality by month (if provided)
    month_int = _get_month(row)
    if month_int in seasonality:
        try:
            price *= float(seasonality[month_int])  # type: ignore[index]
        except Exception:
            pass

    # 5) Controlled noise (optional)
    low, high = NOISE_RANGE
    try:
        price *= float(rng.uniform(low, high))
    except Exception:
        pass

    return float(price)


# -----------------------------------------------------------------------------
# Explainability (no noise)
# -----------------------------------------------------------------------------

def explain_price(
    row: Union[Mapping[str, Any], "pd.Series"],
    *,
    priors: Mapping[str, Any],
    seasonality: Mapping[int, float],
    city_base_prices: Mapping[str, Mapping[str, float]],
) -> Dict[str, Any]:
    """
    Produce a transparent breakdown of price (NO noise):
      - base
      - multipliers: ordered list of (name, multiplier)
      - composed_multiplier
      - final_no_noise

    Useful for audits and quality checks. Takes the same inputs as the main pipeline.
    """
    # Convert Series to mapping if needed
    try:
        if isinstance(row, pd.Series):
            row = row.to_dict()
    except Exception:
        pass

    rm: Mapping[str, Any] = dict(row)  # shallow copy for safety
    pri = normalize_priors(priors)

    city = str(rm.get(LOCATION, "") or "")
    zone = str(rm.get(ZONE, "") or "")
    base = calculate_price_per_sqm_base(city, zone, city_base_prices, DEFAULT_BASE_PRICE_FALLBACK)

    multipliers: List[Tuple[str, float]] = []

    # Build age
    year = rm.get(YEAR_BUILT)
    try:
        year_int = int(year) if year is not None else None
    except Exception:
        year_int = None

    if year_int is not None:
        current_year = datetime.now().year
        age = max(current_year - year_int, 0)
        if age <= 1:
            m = 1.0 + float(pri["build_age"].get("new", 0.0))
            if m != 1.0:
                multipliers.append(("build_age_new", m))
        elif age <= 15:
            m = 1.0 + float(pri["build_age"].get("recent", 0.0))
            if m != 1.0:
                multipliers.append(("build_age_recent", m))
        elif age >= 50:
            m = 1.0 + float(pri["build_age"].get("old", 0.0))
            if m != 1.0:
                multipliers.append(("build_age_old", m))

    # Floor
    if _to_bool(rm.get(IS_TOP_FLOOR, False)):
        m = 1.0 + float(pri["floor_modifiers"].get("is_top_floor", 0.0))
        if m != 1.0:
            multipliers.append(("is_top_floor", m))
    if _to_bool(rm.get(IS_GROUND_FLOOR, False)):
        m = 1.0 + float(pri["floor_modifiers"].get("is_ground_floor", 0.0))
        if m != 1.0:
            multipliers.append(("is_ground_floor", m))

    # Energy class
    ec = str(rm.get(ENERGY_CLASS, "") or "")
    m = float(pri["energy_class_multipliers"].get(ec, 1.0))
    if m != 1.0:
        multipliers.append((f"energy_class_{ec or 'unknown'}", m))

    # Condition/state
    state_val = str(rm.get(CONDITION, rm.get("state", "good")) or "good")
    m = float(pri["state_modifiers"].get(state_val, 1.0))
    if m != 1.0:
        multipliers.append((f"state_{state_val}", m))

    # Extras
    if _to_bool(rm.get(HAS_BALCONY, False)):
        m = 1.0 + float(pri["extras"].get("has_balcony", 0.0))
        if m != 1.0:
            multipliers.append(("has_balcony", m))
    if _to_bool(rm.get(GARAGE, False)):
        m = 1.0 + float(pri["extras"].get("has_garage", 0.0))
        if m != 1.0:
            multipliers.append(("has_garage", m))
    if _to_bool(rm.get(HAS_GARDEN, False)):
        m = 1.0 + float(pri["extras"].get("has_garden", 0.0))
        if m != 1.0:
            multipliers.append(("has_garden", m))

    # View (case-insensitive)
    v = _norm_str(rm.get(VIEW, ""))
    m = float(pri["view_multipliers"].get(v, 1.0))
    if m != 1.0:
        multipliers.append((f"view_{v or 'none'}", m))

    # Orientation heuristic
    orientation_val = _norm_str(rm.get(ORIENTATION, ""))
    if orientation_val in _SUNNY_ORIENTATIONS:
        multipliers.append(("orientation_southish", ORIENTATION_BONUS))

    # Heating heuristic
    heating_val = _norm_str(rm.get(HEATING, ""))
    if heating_val == _HEATING_AUTONOMOUS:
        multipliers.append(("heating_autonomous", HEATING_AUTONOMOUS_BONUS))

    # Seasonality
    m_int = _get_month(rm)
    if m_int in seasonality:
        try:
            sea_m = float(seasonality[m_int])  # type: ignore[index]
            if not np.isclose(sea_m, 1.0):
                multipliers.append((f"seasonality_{m_int}", sea_m))
        except Exception:
            pass

    # Compose
    composed = 1.0
    for _, mul in multipliers:
        composed *= float(mul)

    final_no_noise = float(base) * float(composed)

    return {
        "base": float(base),
        "multipliers": [(str(k), float(v)) for (k, v) in multipliers],
        "composed_multiplier": float(composed),
        "final_no_noise": float(final_no_noise),
    }
