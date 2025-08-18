# shared/generation.py
from __future__ import annotations

"""
Generation utilities per asset sintetici:
- Distribuzione stratificata delle location
- Assegnazione zona da distanza
- Simulazione di condition_score
- Timestamp recenti randomizzati

Nota: questo modulo NON contiene logica di pricing.
"""

from typing import List, Mapping, Optional
from datetime import datetime, timedelta, timezone
import logging

import numpy as np      # type: ignore

from notebooks.shared.common.constants import (
    DEFAULT_ZONE_THRESHOLDS,
    ZONE_CENTER, ZONE_SEMI_CENTER, ZONE_PERIPHERY,
)

logger = logging.getLogger(__name__)

__all__ = [
    "create_stratified_location_distribution",
    "assign_zone_from_distance",
    "simulate_condition_score",
    "random_recent_timestamp",
]

# ---------------------------------------------------------------------------
# Sampling / geography
# ---------------------------------------------------------------------------

def create_stratified_location_distribution(
    n_samples: int,
    location_weights: Mapping[str, float],
    rng: Optional[np.random.Generator] = None
) -> List[str]:
    """
    Crea una lista di location di lunghezza n_samples seguendo pesi target.
    """
    if n_samples <= 0:
        return []
    if not location_weights:
        raise ValueError("location_weights non può essere vuoto.")

    if rng is None:
        rng = np.random.default_rng()

    cities = list(location_weights.keys())
    probs = np.array([float(location_weights[c]) for c in cities], dtype=float)
    total_w = probs.sum()
    probs = (np.ones_like(probs) / len(probs)) if total_w <= 0 else (probs / total_w)

    counts = {city: int(round(n_samples * w)) for city, w in zip(cities, probs)}
    total = sum(counts.values())

    while total < n_samples:
        extra_city = str(rng.choice(cities, p=probs))
        counts[extra_city] += 1
        total += 1
    while total > n_samples:
        for city in cities:
            if counts[city] > 0 and total > n_samples:
                counts[city] -= 1
                total -= 1
            if total == n_samples:
                break

    locations: List[str] = []
    for city, cnt in counts.items():
        locations.extend([city] * cnt)
    rng.shuffle(locations)
    return locations[:n_samples]


def assign_zone_from_distance(
    distance_km: float,
    thresholds: Optional[Mapping[str, float]] = None
) -> str:
    """
    Assegna zona in base alla distanza dal centro, con soglie (km).
    thresholds default: DEFAULT_ZONE_THRESHOLDS
    """
    thr = dict(DEFAULT_ZONE_THRESHOLDS)
    if thresholds:
        thr.update(thresholds)

    center_thr = float(thr.get(ZONE_CENTER, DEFAULT_ZONE_THRESHOLDS[ZONE_CENTER]))
    semi_thr = float(thr.get(ZONE_SEMI_CENTER, DEFAULT_ZONE_THRESHOLDS[ZONE_SEMI_CENTER]))

    if distance_km < center_thr:
        return ZONE_CENTER
    if distance_km < semi_thr:
        return ZONE_SEMI_CENTER
    return ZONE_PERIPHERY

# ---------------------------------------------------------------------------
# Scoring / timestamps
# ---------------------------------------------------------------------------

def simulate_condition_score(
    humidity: float,
    temperature: float,
    energy_class: str,
    rng: Optional[np.random.Generator] = None
) -> float:
    """
    Simula un condition_score in [0,1] usando fattori ambientali ed efficienza energetica.
    """
    if rng is None:
        rng = np.random.default_rng()

    base = 0.85
    if humidity > 65:
        base -= 0.15
    elif humidity > 55:
        base -= 0.05

    if temperature < 14 or temperature > 24:
        base -= 0.07

    energy_adjust_map = {
        "A": +0.03, "B": +0.02, "C": 0.00,
        "D": -0.02, "E": -0.04, "F": -0.06, "G": -0.10,
    }
    base += float(energy_adjust_map.get(str(energy_class), 0.0))

    noise = float(rng.normal(0, 0.02))
    score = max(0.0, min(1.0, base + noise))
    return float(round(score, 3))


def random_recent_timestamp(
    reference_time: Optional[datetime],
    days_back: int = 60,
    rng: Optional[np.random.Generator] = None
) -> str:
    """
    Genera un timestamp ISO8601 (UTC, suffisso 'Z') casuale negli ultimi `days_back` giorni
    rispetto a `reference_time` (se None → now UTC).
    """
    if rng is None:
        rng = np.random.default_rng()
    if reference_time is None:
        reference_time = datetime.now(timezone.utc)

    delta_days = int(rng.integers(0, max(0, days_back) + 1))
    delta_hours = int(rng.integers(0, 24))
    delta_minutes = int(rng.integers(0, 60))

    dt = reference_time - timedelta(days=delta_days, hours=delta_hours, minutes=delta_minutes)
    return dt.replace(microsecond=0, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")