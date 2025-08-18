from __future__ import annotations
"""
Pricing utilities (single source of truth).

- normalize_priors: uniforma i priors "legacy" a uno schema unificato
- apply_pricing_pipeline: applica i modificatori al prezzo base (senza rumore, senza euristiche extra)
- apply_hedonic_adjustments: pipeline completa (base → priors → euristiche → seasonality → noise)
- explain_price: breakdown trasparente dei moltiplicatori (senza rumore), per reporting/quality

Zero I/O. Tutte le funzioni sono pure/deterministiche ad eccezione del rumore controllato
in `apply_hedonic_adjustments` (disattivabile usando `rng` deterministico).
"""

from typing import Any, Dict, Mapping, Optional, List, Tuple, Union
from datetime import datetime
import logging
import numpy as np      # type: ignore
import pandas as pd     # type: ignore

# Colonne/domini
from notebooks.shared.common.constants import (
    LOCATION, ZONE, YEAR_BUILT, ENERGY_CLASS,
    IS_TOP_FLOOR, IS_GROUND_FLOOR,
    HAS_BALCONY, HAS_GARDEN, GARAGE,
    VIEW, VIEW_SEA, VIEW_LANDMARKS,
    ORIENTATION, HEATING,
    Cols,
)

logger = logging.getLogger(__name__)

__all__ = [
    "normalize_priors",
    "apply_pricing_pipeline",
    "apply_hedonic_adjustments",
    "explain_price",
    "calculate_price_per_sqm_base",
]

ORIENTATION_BONUS: float = 1.05          # South / SE / SW
HEATING_AUTONOMOUS_BONUS: float = 1.03
DEFAULT_BASE_PRICE_FALLBACK: float = 3000.0
NOISE_RANGE: Tuple[float, float] = (0.95, 1.05)

def _get_month(row: Mapping[str, Any]) -> Optional[int]:
    month_val = row.get(Cols.LISTING_MONTH, row.get("month"))
    try:
        if month_val is None:
            return None
        m = int(month_val)
        return m if 1 <= m <= 12 else None
    except Exception:
        return None

def calculate_price_per_sqm_base(
    city: str,
    zone: str,
    city_base_prices: Mapping[str, Mapping[str, float]],
    default_fallback: float = DEFAULT_BASE_PRICE_FALLBACK,
) -> float:
    """
    Base price per (city, zone). Se assente:
      - media delle zone per la città, se disponibile
      - altrimenti default_fallback
    """
    city_prices = city_base_prices.get(city, {})
    if zone in city_prices:
        return float(city_prices[zone])
    if city_prices:
        return float(np.mean(list(city_prices.values())))
    return float(default_fallback)

def normalize_priors(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Normalizza priors 'legacy' in schema unificato, compatibile con:
      - asset_factory.normalize_pricing_input
      - quality.decompose_price_per_sqm

    Chiavi finali:
      view_multipliers:{sea, landmarks}
      floor_modifiers:{is_top_floor, is_ground_floor}
      build_age:{new, recent, old}
      energy_class_multipliers:{A..G}
      state_modifiers:{new, renovated, good, needs_renovation, ...}
      extras:{has_balcony, has_garage, has_garden}
    """
    raw = dict(raw or {})
    build_age_raw = raw.get("build_age", {}) or {}

    return {
        "view_multipliers": raw.get("view_multipliers", {VIEW_SEA: 1.0, VIEW_LANDMARKS: 1.0}),
        "floor_modifiers": raw.get("floor_modifiers", {"is_top_floor": 0.0, "is_ground_floor": 0.0}),
        "build_age": {
            "new": float(build_age_raw.get("new", 0.0)),
            "recent": float(build_age_raw.get("recent", 0.0)),
            "old": float(build_age_raw.get("old", 0.0)),
        },
        "energy_class_multipliers": dict(raw.get("energy_class_multipliers", {})),
        "state_modifiers": dict(raw.get("state_modifiers", {})),
        "extras": raw.get("extras", {"has_balcony": 0.0, "has_garage": 0.0, "has_garden": 0.0}),
    }

def apply_pricing_pipeline(row: Mapping[str, Any], priors: Mapping[str, Any], base_price: float) -> float:
    """
    Applica i modificatori al prezzo base in modo sequenziale usando priors NORMALIZZATI.
    (Niente bonus orientamento/heating, niente seasonality, niente rumore.)
    """
    price = float(base_price)

    # Build age / year built
    year = row.get(YEAR_BUILT)
    try:
        year_int = int(year) if year is not None else None
    except Exception:
        year_int = None

    if year_int is not None:
        current_year = datetime.now().year
        age = max(current_year - year_int, 0)
        if age <= 1:
            price *= 1 + float(priors.get("build_age", {}).get("new", 0.0))
        elif age <= 15:
            price *= 1 + float(priors.get("build_age", {}).get("recent", 0.0))
        elif age >= 50:
            price *= 1 + float(priors.get("build_age", {}).get("old", 0.0))

    # Floor
    if bool(row.get(IS_TOP_FLOOR, False)):
        price *= 1 + float(priors.get("floor_modifiers", {}).get("is_top_floor", 0.0))
    if bool(row.get(IS_GROUND_FLOOR, False)):
        price *= 1 + float(priors.get("floor_modifiers", {}).get("is_ground_floor", 0.0))

    # Energy class
    energy_class = str(row.get(ENERGY_CLASS, "") or "")
    price *= float(priors.get("energy_class_multipliers", {}).get(energy_class, 1.0))

    # State
    state = str(row.get("state", "good") or "good")
    price *= float(priors.get("state_modifiers", {}).get(state, 1.0))

    # Extras
    if bool(row.get(HAS_BALCONY, False)):
        price *= 1 + float(priors.get("extras", {}).get("has_balcony", 0.0))
    if bool(row.get(GARAGE, False)):
        price *= 1 + float(priors.get("extras", {}).get("has_garage", 0.0))
    if bool(row.get(HAS_GARDEN, False)):
        price *= 1 + float(priors.get("extras", {}).get("has_garden", 0.0))

    # View multipliers
    view_val = str(row.get(VIEW, "") or "")
    price *= float(priors.get("view_multipliers", {}).get(view_val, 1.0))

    return float(price)

def apply_hedonic_adjustments(
    row: Mapping[str, Any],
    pricing_input: Mapping[str, Any],
    seasonality: Mapping[int, float],
    city_base_prices: Mapping[str, Mapping[str, float]],
    rng: Optional[np.random.Generator] = None
) -> float:
    """
    Applica l'intera pipeline di prezzo:
    base_price → priors (normalizzati) → orientation/heating → seasonality → noise

    Returns:
        price_per_sqm (float)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Base price per sqm
    base_price = calculate_price_per_sqm_base(
        str(row.get(LOCATION, "") or ""),
        str(row.get(ZONE, "") or ""),
        city_base_prices,
        default_fallback=DEFAULT_BASE_PRICE_FALLBACK,
    )

    # Priors normalizzati (accetta input legacy)
    priors = normalize_priors(pricing_input)

    # Pipeline principale (senza euristiche locali)
    price = apply_pricing_pipeline(row, priors, base_price)

    # Euristiche locali (orientamento, heating)
    orientation_val = str(row.get(ORIENTATION, "") or "")
    if orientation_val in {"South", "South-East", "South-West"}:
        price *= ORIENTATION_BONUS

    heating_val = str(row.get(HEATING, "") or "")
    if heating_val == "autonomous":
        price *= HEATING_AUTONOMOUS_BONUS

    # Seasonality
    month_int = _get_month(row)
    if month_int in seasonality:
        price *= float(seasonality[month_int])  # type: ignore[arg-type]

    # Controlled noise
    low, high = NOISE_RANGE
    price *= float(rng.uniform(low, high))

    return float(price)

def explain_price(
    row: Union[Mapping[str, Any], "pd.Series"],
    *,
    priors: Mapping[str, Any],
    seasonality: Mapping[int, float],
    city_base_prices: Mapping[str, Mapping[str, float]],
) -> Dict[str, Any]:
    """
    Ritorna un breakdown trasparente del prezzo (SENZA rumore):
      - base
      - multipliers (lista ordinata di (nome, moltiplicatore))
      - composed_multiplier (prodotto dei moltiplicatori)
      - final_no_noise (base * composed_multiplier)

    Non applica rumore; utile per report/quality.
    """
    try:
        if isinstance(row, pd.Series):
            row = row.to_dict()
    except Exception:
        # se pandas non è disponibile va bene lo stesso
        pass

    row_map: Mapping[str, Any] = dict(row)  # shallow copy
    pri = normalize_priors(priors)

    city = str(row_map.get(LOCATION, "") or "")
    zone = str(row_map.get(ZONE, "") or "")
    base = calculate_price_per_sqm_base(city, zone, city_base_prices, DEFAULT_BASE_PRICE_FALLBACK)

    multipliers: List[Tuple[str, float]] = []

    # 1) build age
    year = row_map.get(YEAR_BUILT)
    year_int = None
    try:
        if year is not None:
            year_int = int(year)
    except Exception:
        year_int = None
    if year_int is not None:
        current_year = datetime.now().year
        age = max(current_year - year_int, 0)
        if age <= 1:
            m = 1 + float(pri.get("build_age", {}).get("new", 0.0))
            if m != 1.0: multipliers.append(("build_age_new", m))
        elif age <= 15:
            m = 1 + float(pri.get("build_age", {}).get("recent", 0.0))
            if m != 1.0: multipliers.append(("build_age_recent", m))
        elif age >= 50:
            m = 1 + float(pri.get("build_age", {}).get("old", 0.0))
            if m != 1.0: multipliers.append(("build_age_old", m))

    # 2) floor
    if bool(row_map.get(IS_TOP_FLOOR, False)):
        m = 1 + float(pri.get("floor_modifiers", {}).get("is_top_floor", 0.0))
        if m != 1.0: multipliers.append(("is_top_floor", m))
    if bool(row_map.get(IS_GROUND_FLOOR, False)):
        m = 1 + float(pri.get("floor_modifiers", {}).get("is_ground_floor", 0.0))
        if m != 1.0: multipliers.append(("is_ground_floor", m))

    # 3) energy class
    ec = str(row_map.get(ENERGY_CLASS, "") or "")
    m = float(pri.get("energy_class_multipliers", {}).get(ec, 1.0))
    if m != 1.0: multipliers.append((f"energy_class_{ec or 'unknown'}", m))

    # 4) state
    state = str(row_map.get("state", "good") or "good")
    m = float(pri.get("state_modifiers", {}).get(state, 1.0))
    if m != 1.0: multipliers.append((f"state_{state}", m))

    # 5) extras
    if bool(row_map.get(HAS_BALCONY, False)):
        m = 1 + float(pri.get("extras", {}).get("has_balcony", 0.0))
        if m != 1.0: multipliers.append(("has_balcony", m))
    if bool(row_map.get(GARAGE, False)):
        m = 1 + float(pri.get("extras", {}).get("has_garage", 0.0))
        if m != 1.0: multipliers.append(("has_garage", m))
    if bool(row_map.get(HAS_GARDEN, False)):
        m = 1 + float(pri.get("extras", {}).get("has_garden", 0.0))
        if m != 1.0: multipliers.append(("has_garden", m))

    # 6) view
    v = str(row_map.get(VIEW, "") or "")
    m = float(pri.get("view_multipliers", {}).get(v, 1.0))
    if m != 1.0: multipliers.append((f"view_{v or 'none'}", m))

    # 7) orientation heuristic
    orientation_val = str(row_map.get(ORIENTATION, "") or "")
    if orientation_val in {"South", "South-East", "South-West"}:
        multipliers.append(("orientation_southish", ORIENTATION_BONUS))

    # 8) heating heuristic
    heating_val = str(row_map.get(HEATING, "") or "")
    if heating_val == "autonomous":
        multipliers.append(("heating_autonomous", HEATING_AUTONOMOUS_BONUS))

    # 9) seasonality
    m_int = _get_month(row_map)
    if m_int in seasonality:
        sea_m = float(seasonality[m_int])  # type: ignore[index]
        if sea_m != 1.0:
            multipliers.append((f"seasonality_{m_int}", sea_m))

    # Compose multiplier & final
    composed = 1.0
    for _, mul in multipliers:
        composed *= float(mul)
    final_no_noise = base * composed

    return {
        "base": float(base),
        "multipliers": [(str(k), float(v)) for (k, v) in multipliers],
        "composed_multiplier": float(composed),
        "final_no_noise": float(final_no_noise),
    }