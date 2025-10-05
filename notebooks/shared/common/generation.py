from __future__ import annotations
"""
Deterministic helpers for synthetic data generation (Property domain).

Scope
- Stratified sampling of locations by target weights.
- Zone assignment from distance-to-center thresholds (single source of truth).
- Lightweight simulation of a condition_score in [0, 1].
- Random recent ISO-8601 timestamps (UTC, suffix 'Z').

Non-goals
- No pricing logic here (kept in `pricing.py`).
- No heavyweight dependencies; keep the import surface minimal.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Mapping, Optional

import logging
import numpy as np  # type: ignore

from shared.common.constants import Thresholds, Zone

__all__ = [
    "create_stratified_location_distribution",
    "assign_zone_from_distance",
    "simulate_condition_score",
    "random_recent_timestamp",
]

logger = logging.getLogger(__name__)

# =============================================================================
# Stratified sampling / geography
# =============================================================================


def create_stratified_location_distribution(
    n_samples: int,
    location_weights: Mapping[str, float],
    rng: Optional[np.random.Generator] = None,
) -> List[str]:
    """
    Build a list of `n_samples` locations distributed according to `location_weights`.

    Args:
        n_samples: Number of samples to generate (<=0 returns []).
        location_weights: Mapping {city -> weight}. Weights can be unnormalized.
        rng: Optional NumPy Generator; if None, uses default_rng().

    Behavior:
        - Normalizes weights to probabilities; if all weights <= 0, uses uniform.
        - Rounds expected counts, then adjusts with a minimal pass to reach n_samples.
        - Shuffles the final list for decorrelation.

    Raises:
        ValueError: if `location_weights` is empty.
    """
    if n_samples <= 0:
        return []
    if not location_weights:
        raise ValueError("location_weights must not be empty.")

    rng = rng or np.random.default_rng()

    cities = list(location_weights.keys())
    probs = np.asarray([float(location_weights[c]) for c in cities], dtype="float64")
    total_w = float(probs.sum())
    if total_w <= 0.0:
        probs[:] = 1.0 / len(probs)
    else:
        probs /= total_w

    # Integer allocation by rounded expectation
    counts = {city: int(round(n_samples * w)) for city, w in zip(cities, probs)}
    total = sum(counts.values())

    # Adjust to exact n_samples with minimal drift
    while total < n_samples:
        extra_city = str(rng.choice(cities, p=probs))
        counts[extra_city] += 1
        total += 1
    while total > n_samples:
        for city in cities:
            if counts[city] > 0:
                counts[city] -= 1
                total -= 1
                if total == n_samples:
                    break

    # Materialize & shuffle
    locations: List[str] = []
    for city, cnt in counts.items():
        if cnt > 0:
            locations.extend([city] * cnt)

    rng.shuffle(locations)
    return locations[:n_samples]


def assign_zone_from_distance(
    distance_km: float,
    thresholds: Optional[Mapping[str, float]] = None,
) -> str:
    """
    Assign a zone from distance-to-center (km) using canonical thresholds.

    Args:
        distance_km: Non-negative distance from the city center in kilometers.
        thresholds: Optional overrides. If provided, keys must match Zone values:
                    {"center": float, "semi_center": float}. Periphery is the fallback.

    Returns:
        Zone name (string): one of {"center", "semi_center", "periphery"}.
    """
    if distance_km < 0:
        # Defensive: keep generation robust.
        distance_km = 0.0

    thr = dict(Thresholds.DEFAULT_ZONE_THRESHOLDS_KM)
    if thresholds:
        thr.update(thresholds)

    center_thr = float(thr[Zone.CENTER.value])
    semi_thr = float(thr[Zone.SEMI_CENTER.value])

    if distance_km < center_thr:
        return Zone.CENTER.value
    if distance_km < semi_thr:
        return Zone.SEMI_CENTER.value
    return Zone.PERIPHERY.value


# =============================================================================
# Scoring / timestamps
# =============================================================================


def simulate_condition_score(
    humidity: float,
    temperature: float,
    energy_class: str,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Simulate a simple condition score in [0, 1] using environmental factors.

    Heuristics:
        - Start at base 0.85 and penalize for high humidity and off-comfort temperatures.
        - Adjust by energy class (A best → positive bump; G worst → negative bump).
        - Add small Gaussian noise (σ ≈ 0.02), clamp to [0, 1], round to 3 decimals.

    Args:
        humidity: Relative humidity percentage (expected 0..100; tolerant to out-of-range).
        temperature: Average temperature in °C.
        energy_class: One of {"A","B","C","D","E","F","G"}; unknown → 0 adjustment.
        rng: Optional NumPy Generator; if None, uses default_rng().

    Returns:
        float in [0, 1], rounded to 3 decimals.
    """
    rng = rng or np.random.default_rng()

    base = 0.85
    # Humidity penalties (soft thresholds)
    if humidity > 65:
        base -= 0.15
    elif humidity > 55:
        base -= 0.05

    # Temperature comfort band (roughly 18–24°C)
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
    rng: Optional[np.random.Generator] = None,
) -> str:
    """
    Generate an ISO-8601 UTC timestamp randomly within the last `days_back` days.

    Args:
        reference_time: Anchor time; if None, uses now (UTC).
        days_back: Non-negative range (days) to sample uniformly (random hours/minutes).
        rng: Optional NumPy Generator; if None, uses default_rng().

    Returns:
        ISO8601 string in UTC (no microseconds), e.g., "2025-01-31T13:05:00Z".
    """
    rng = rng or np.random.default_rng()
    if reference_time is None:
        reference_time = datetime.now(timezone.utc)
    if days_back < 0:
        days_back = 0

    delta_days = int(rng.integers(0, days_back + 1))
    delta_hours = int(rng.integers(0, 24))
    delta_minutes = int(rng.integers(0, 60))

    dt = reference_time - timedelta(days=delta_days, hours=delta_hours, minutes=delta_minutes)
    return dt.replace(microsecond=0, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")