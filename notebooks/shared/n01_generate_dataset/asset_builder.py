# shared/asset_builder.py
from __future__ import annotations

from notebooks.shared.common.utils import get_utc_now

"""
Asset builder per 'property' (sintetico) con logica hedonic.
- Tipizzazione completa
- Default robusti e input validation
- Allineamento a constants/schema
- Compatibile con pricing normalizzato (shared.common.pricing)
"""

from typing import Any, Dict, Mapping, Optional, Sequence
from datetime import datetime
import logging

import numpy as np  # type: ignore

from notebooks.shared.common.constants import (
    ASSET_ID, ASSET_TYPE, LOCATION, REGION, URBAN_TYPE, ZONE,
    SIZE_M2, ROOMS, BATHROOMS, YEAR_BUILT, AGE_YEARS, FLOOR, BUILDING_FLOORS,
    IS_TOP_FLOOR, IS_GROUND_FLOOR, HAS_ELEVATOR, HAS_GARDEN, HAS_BALCONY, GARAGE,
    OWNER_OCCUPIED, PUBLIC_TRANSPORT_NEARBY, DISTANCE_TO_CENTER_KM,
    ENERGY_CLASS, HUMIDITY_LEVEL, TEMPERATURE_AVG, NOISE_LEVEL, AIR_QUALITY_INDEX,
    ORIENTATION, VIEW, CONDITION, HEATING, PARKING_SPOT, CELLAR, ATTIC, CONCIERGE,
    VALUATION_K, PRICE_PER_SQM, LAST_VERIFIED_TS, LISTING_MONTH,
    LUXURY_SCORE, ENV_SCORE, CONDITION_SCORE, RISK_SCORE,
    Cols,
)

from notebooks.shared.common.pricing import apply_hedonic_adjustments
from notebooks.shared.common.generation import (
    assign_zone_from_distance,
    random_recent_timestamp,
    simulate_condition_score,
)

__all__ = ["generate_property"]

logger = logging.getLogger(__name__)


def _choose_state(year_built: int, rng: np.random.Generator) -> str:
    """Seleziona lo stato/condizione dell'immobile in base all'anno di costruzione."""
    if year_built > 2020:
        return "new"
    if year_built > 2010:
        return str(rng.choice(["new", "renovated"], p=[0.3, 0.7]))
    return str(rng.choice(["needs_renovation", "good", "renovated"], p=[0.3, 0.5, 0.2]))


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
    Costruisce un asset 'property' sintetico usando priors (legacy o nuovi) e la logica hedonic.
    """
    if rng is None:
        rng = np.random.default_rng()
    seasonality = seasonality or {}
    city_base_prices = city_base_prices or {}

    # Scegli config pricing effettiva
    effective_pricing: Mapping[str, Any] = pricing_input or pricing or {}

    # --- Structural (dimensioni, piano, ecc.)
    size_m2 = int(rng.integers(40, 200))
    rooms = int(rng.integers(2, 7))
    bathrooms = int(rng.integers(1, 4))
    year_built = int(rng.integers(1950, 2024))
    floor = int(rng.integers(0, 5))
    building_floors = int(rng.integers(floor + 1, 10))
    is_top_floor = (floor == (building_floors - 1))
    is_ground_floor = (floor == 0)
    has_elevator = int(building_floors >= int(config.get("min_floors_for_elevator", 4)))
    has_garden = int(rng.random() < 0.30)
    has_balcony = int(rng.random() < 0.60)
    has_garage = int(rng.random() < 0.50)
    owner_occupied = int(rng.random() < 0.65)
    public_transport_nearby = int(rng.random() < 0.70)

    # --- Environmental / quality
    energy_class = str(rng.choice(["A", "B", "C", "D", "E", "F", "G"]))
    humidity = float(np.round(rng.uniform(30, 70), 1))
    temperature = float(np.round(rng.uniform(12, 25), 1))
    noise_level = int(rng.integers(20, 80))
    air_quality_index = int(rng.integers(30, 150))

    # --- Italian-specific
    orientation = _choose_orientation(rng)
    view = _choose_view(rng)

    # --- Condition/state
    condition = _choose_state(year_built, rng)
    heating = _choose_heating(rng)

    # --- Location & geography
    if location is None:
        if not locations:
            raise ValueError("`locations` non può essere vuoto se non passi `location` esplicita.")
        location = str(rng.choice(list(locations)))
    urban_type = str(urban_map.get(location, "unknown"))
    region = str(region_map.get(location, "unknown"))
    distance_to_center_km = float(np.round(rng.exponential(scale=3.5), 2))
    zone = assign_zone_from_distance(distance_to_center_km, config.get("zone_thresholds_km", {}))

    # --- Timestamp / age
    last_verified_ts = get_utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")
    month = (
        int(datetime.fromisoformat(last_verified_ts.replace("Z", "+00:00")).month)
        if last_verified_ts else None
    )
    age_years = (reference_time.year - year_built) if reference_time else None

    # --- Scores sintetici
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

    # --- Interim per pricing (hedonic)
    # Se lo 'state' non è contemplato nei modifiers, fallback a "good" per stabilità
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
    valuation_k_val = float(np.round((price_per_sqm_val * size_m2) / 1000, 2))
    price_per_sqm_val = float(np.round(price_per_sqm_val, 2))

    # --- Build asset dict (chiavi canoniche)
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
        IS_TOP_FLOOR: int(is_top_floor),
        IS_GROUND_FLOOR: int(is_ground_floor),
        HAS_ELEVATOR: has_elevator,
        HAS_GARDEN: has_garden,
        HAS_BALCONY: has_balcony,
        GARAGE: has_garage,
        OWNER_OCCUPIED: owner_occupied,
        PUBLIC_TRANSPORT_NEARBY: public_transport_nearby,
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
        PARKING_SPOT: int((has_garage == 0) and (rng.random() < 0.3)),
        CELLAR: int(rng.random() < 0.4),
        ATTIC: int((floor == building_floors - 1) and (rng.random() < 0.3)),
        CONCIERGE: int((urban_type == "urban") and (size_m2 > 100) and (rng.random() < 0.3)),
        LAST_VERIFIED_TS: last_verified_ts,
        LISTING_MONTH: month,
    }

    # Normalizza tipi per coerenza (aiuta schema/serving)
    for k in (HAS_ELEVATOR, HAS_GARDEN, HAS_BALCONY, GARAGE, OWNER_OCCUPIED,
              PUBLIC_TRANSPORT_NEARBY, PARKING_SPOT, CELLAR, ATTIC, CONCIERGE,
              IS_TOP_FLOOR, IS_GROUND_FLOOR):
        asset[k] = int(bool(asset[k]))

    logger.debug(
        "Generated property #%d @%s/%s | sqm=%s rooms=%s price_sqm=%.2f val_k=%.2f",
        index, location, zone, size_m2, rooms, price_per_sqm_val, valuation_k_val
    )
    return asset