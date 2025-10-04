from __future__ import annotations

"""
Synthetic asset builder for 'property' with hedonic pricing.

Design principles
- Pure function (no I/O), deterministic when a seeded RNG is provided.
- Strong typing and defensive defaults.
- Aligned with constants/schema and pricing normalization.
- Stable fallbacks for missing priors and city/zone base prices.
"""

from typing import Any, Dict, Mapping, Optional, Sequence
from datetime import datetime, timezone
import logging

import numpy as np  # type: ignore

from shared.common.utils import get_utc_now
from shared.common.constants import (
    ASSET_ID,
    ASSET_TYPE,
    LOCATION,
    REGION,
    URBAN_TYPE,
    ZONE,
    SIZE_M2,
    ROOMS,
    BATHROOMS,
    YEAR_BUILT,
    AGE_YEARS,
    FLOOR,
    BUILDING_FLOORS,
    IS_TOP_FLOOR,
    IS_GROUND_FLOOR,
    HAS_ELEVATOR,
    HAS_GARDEN,
    HAS_BALCONY,
    GARAGE,
    OWNER_OCCUPIED,
    PUBLIC_TRANSPORT_NEARBY,
    DISTANCE_TO_CENTER_KM,
    ENERGY_CLASS,
    HUMIDITY_LEVEL,
    TEMPERATURE_AVG,
    NOISE_LEVEL,
    AIR_QUALITY_INDEX,
    ORIENTATION,
    VIEW,
    CONDITION,
    HEATING,
    PARKING_SPOT,
    CELLAR,
    ATTIC,
    CONCIERGE,
    VALUATION_K,
    PRICE_PER_SQM,
    LAST_VERIFIED_TS,
    LISTING_MONTH,
    LUXURY_SCORE,
    ENV_SCORE,
    CONDITION_SCORE,
    RISK_SCORE,
    Cols,
)

from shared.common.pricing import apply_hedonic_adjustments
from shared.common.generation import (
    assign_zone_from_distance,
    simulate_condition_score,
)

__all__ = ["generate_property"]

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- #
# Internal helpers (pure, local scope)
# ----------------------------------------------------------------------------- #

def _choose_state(year_built: int, rng: np.random.Generator) -> str:
    """Pick a property condition informed by build year."""
    if year_built > 2020:
        return "new"
    if year_built > 2010:
        return str(rng.choice(["new", "renovated"], p=[0.3, 0.7]))
    return str(rng.choice(["needs_renovation", "good", "renovated"], p=[0.30, 0.50, 0.20]))


def _choose_orientation(rng: np.random.Generator) -> str:
    return str(
        rng.choice(
            ["North", "South", "East", "West", "North-East", "North-West", "South-East", "South-West"],
            p=[0.05, 0.25, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10],
        )
    )


def _choose_view(rng: np.random.Generator) -> str:
    return str(
        rng.choice(
            ["street", "inner courtyard", "garden", "park", "sea", "mountain", "landmarks"],
            p=[0.40, 0.30, 0.15, 0.08, 0.03, 0.02, 0.02],
        )
    )


def _choose_heating(rng: np.random.Generator) -> str:
    return str(rng.choice(["autonomous", "centralized", "heat pump", "none"], p=[0.60, 0.30, 0.08, 0.02]))


# ----------------------------------------------------------------------------- #
# Public API
# ----------------------------------------------------------------------------- #

def generate_property(
    *,
    index: int,
    config: Mapping[str, Any],
    locations: Sequence[str],
    urban_map: Mapping[str, str],
    region_map: Mapping[str, str],
    pricing_input: Optional[Mapping[str, Any]] = None,
    pricing: Optional[Mapping[str, Any]] = None,
    seasonality: Optional[Mapping[int, float]] = None,
    city_base_prices: Optional[Mapping[str, Mapping[str, float]]] = None,
    rng: Optional[np.random.Generator] = None,
    reference_time: Optional[datetime] = None,
    location: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a synthetic 'property' asset using hedonic priors and simple environment heuristics.

    Args:
        index: progressive index used to generate ASSET_ID.
        config: general generation/pricing config (used for a few thresholds).
        locations: list of eligible city names (canonical casing expected upstream).
        urban_map: city -> urban type mapping.
        region_map: city -> region mapping.
        pricing_input/pricing: priors (legacy or normalized). `pricing_input` takes precedence.
        seasonality: month -> seasonal multiplier.
        city_base_prices: base €/m² per city/zone.
        rng: optional np.random.Generator for determinism.
        reference_time: optional reference time for AGE_YEARS; defaults to now UTC.
        location: optional explicit city override; if None a random one is sampled.

    Returns:
        Dict[str, Any]: canonical asset record aligned with constants/schema.
    """
    rng = rng or np.random.default_rng()
    seasonality = seasonality or {}
    city_base_prices = city_base_prices or {}
    reference_time = reference_time or get_utc_now()
    # Pricing config precedence: normalized input wins; fallback to legacy `pricing`
    effective_pricing: Mapping[str, Any] = pricing_input or pricing or {}

    # ---------------------- Structural attributes ---------------------- #
    # Keep size/rooms/baths consistent and realistic for European apartments
    size_m2 = int(rng.integers(40, 200))
    rooms = int(np.clip(int(rng.normal(loc=max(2, size_m2 // 35), scale=1.0)), 2, 7))
    bathrooms = int(np.clip(int(rng.choice([1, 2, 2, 3], p=[0.55, 0.35, 0.05, 0.05])), 1, 3))

    year_built = int(rng.integers(1950, 2024))
    floor = int(rng.integers(0, 6))
    # Ensure at least one floor above current when possible
    building_floors = int(max(floor + 1, int(rng.integers(max(floor + 1, 3), 11))))
    is_top_floor = (floor == (building_floors - 1))
    is_ground_floor = (floor == 0)
    has_elevator = int(building_floors >= int(config.get("min_floors_for_elevator", 4)))

    has_garden = int(rng.random() < 0.30)
    has_balcony = int(rng.random() < 0.60)
    has_garage = int(rng.random() < 0.50)
    owner_occupied = int(rng.random() < 0.65)
    public_transport_nearby = int(rng.random() < 0.70)

    # ---------------------- Environmental / quality --------------------- #
    energy_class = str(rng.choice(["A", "B", "C", "D", "E", "F", "G"]))
    humidity = float(np.round(rng.uniform(30, 70), 1))
    temperature = float(np.round(rng.uniform(12, 25), 1))
    noise_level = int(rng.integers(20, 80))
    air_quality_index = int(rng.integers(30, 150))

    # ---------------------- Italian specifics -------------------------- #
    orientation = _choose_orientation(rng)
    view = _choose_view(rng)
    condition = _choose_state(year_built, rng)
    heating = _choose_heating(rng)

    # ---------------------- Location & geography ----------------------- #
    if location is None:
        if not locations:
            raise ValueError("`locations` must be non-empty when `location` is not provided.")
        location = str(rng.choice(list(locations)))
    urban_type = str(urban_map.get(location, "unknown"))
    region = str(region_map.get(location, "unknown"))

    # Exponential distance gives a realistic right-tail towards periphery
    distance_to_center_km = float(np.round(rng.exponential(scale=3.5), 2))
    zone = assign_zone_from_distance(distance_to_center_km, config.get("zone_thresholds_km", {}))

    # ---------------------- Time & age --------------------------------- #
    now_utc = reference_time.astimezone(timezone.utc)
    last_verified_ts = now_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    month = int(now_utc.month)
    age_years = int(max(0, now_utc.year - year_built))

    # ---------------------- Synthetic scores --------------------------- #
    condition_score = simulate_condition_score(humidity, temperature, energy_class, rng=rng)
    risk_score = float(np.round(min(1.0, max(0.0, (1 - condition_score) + rng.normal(0, 0.02))), 3))
    luxury_score = float(np.round(
        0.2 * has_garden
        + 0.2 * has_balcony
        + 0.2 * has_garage
        + 0.2 * (1 if energy_class in ["A", "B"] else 0)
        + 0.2 * (1 if size_m2 > 120 else 0),
        2,
    ))
    env_score = float(np.round(
        0.4 * (1 if air_quality_index < 80 else 0)
        + 0.3 * (1 if 35 <= humidity <= 60 else 0)
        + 0.3 * (1 if noise_level < 50 else 0),
        2,
    ))

    # ---------------------- Hedonic pricing (interim row) -------------- #
    # Fallback to "good" state if priors do not include the sampled `condition`.
    state_for_pricing = condition if condition in (effective_pricing.get("state_modifiers") or {}) else "good"
    interim = {
        LOCATION: location,
        ZONE: zone,
        YEAR_BUILT: year_built,
        IS_TOP_FLOOR: is_top_floor,
        IS_GROUND_FLOOR: is_ground_floor,
        ENERGY_CLASS: energy_class,
        "state": state_for_pricing,
        HAS_BALCONY: bool(has_balcony),
        HAS_GARDEN: bool(has_garden),
        GARAGE: bool(has_garage),
        Cols.LISTING_MONTH: month,
        VIEW: view,
        ORIENTATION: orientation,
        HEATING: heating,
    }

    price_per_sqm_val = apply_hedonic_adjustments(
        interim,
        effective_pricing,
        seasonality=seasonality,
        city_base_prices=city_base_prices,
        rng=rng,
    )
    valuation_k_val = float(np.round((price_per_sqm_val * size_m2) / 1000.0, 2))
    price_per_sqm_val = float(np.round(price_per_sqm_val, 2))

    # ---------------------- Canonical asset dict ------------------------ #
    asset: Dict[str, Any] = {
        ASSET_ID: f"asset_{index:06}",
        ASSET_TYPE: str(config.get(ASSET_TYPE, "property")),
        LOCATION: location,
        REGION: region,
        URBAN_TYPE: urban_type,
        ZONE: zone,
        SIZE_M2: size_m2,
        ROOMS: rooms,
        BATHROOMS: bathrooms,
        YEAR_BUILT: year_built,
        AGE_YEARS: age_years,
        FLOOR: floor,
        BUILDING_FLOORS: building_floors,
        IS_TOP_FLOOR: int(bool(is_top_floor)),
        IS_GROUND_FLOOR: int(bool(is_ground_floor)),
        HAS_ELEVATOR: int(bool(has_elevator)),
        HAS_GARDEN: int(bool(has_garden)),
        HAS_BALCONY: int(bool(has_balcony)),
        GARAGE: int(bool(has_garage)),
        OWNER_OCCUPIED: int(bool(owner_occupied)),
        PUBLIC_TRANSPORT_NEARBY: int(bool(public_transport_nearby)),
        DISTANCE_TO_CENTER_KM: distance_to_center_km,
        ENERGY_CLASS: energy_class,
        PRICE_PER_SQM: price_per_sqm_val,
        VALUATION_K: valuation_k_val,
        HUMIDITY_LEVEL: humidity,
        TEMPERATURE_AVG: temperature,
        NOISE_LEVEL: noise_level,
        AIR_QUALITY_INDEX: air_quality_index,
        LUXURY_SCORE: luxury_score,
        ENV_SCORE: env_score,
        CONDITION_SCORE: condition_score,
        RISK_SCORE: risk_score,
        ORIENTATION: orientation,
        VIEW: view,
        CONDITION: condition,
        HEATING: heating,
        PARKING_SPOT: int(bool((has_garage == 0) and (rng.random() < 0.30))),
        CELLAR: int(bool(rng.random() < 0.40)),
        ATTIC: int(bool((floor == building_floors - 1) and (rng.random() < 0.30))),
        CONCIERGE: int(bool((urban_type == "urban") and (size_m2 > 100) and (rng.random() < 0.30))),
        LAST_VERIFIED_TS: last_verified_ts,
        LISTING_MONTH: month,
    }

    logger.debug(
        "Generated property #%d @%s/%s | sqm=%s rooms=%s price_sqm=%.2f val_k=%.2f",
        index, location, zone, size_m2, rooms, price_per_sqm_val, valuation_k_val
    )
    return asset
