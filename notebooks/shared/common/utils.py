from __future__ import annotations

"""
Utility functions:
- JSON encoding per tipi numpy/pandas
- Gestione seed globale deterministica
- Parsing ISO8601 robusto
- Ottimizzazione memoria DataFrame
- Diagnostica di base per dataset
- Funzioni di mapping e location canonica
"""

import json
import logging
import random
from datetime import datetime, date, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np      # type: ignore
import pandas as pd     # type: ignore

from notebooks.shared.common.constants import (
    LOCATION,
    ZONE,
    DEFAULT_URBAN_TYPE_BY_CITY,
    DEFAULT_REGION_BY_CITY
)

logger = logging.getLogger(__name__)

__all__ = [
    "NumpyJSONEncoder",
    "set_global_seed",
    "get_utc_now",
    "parse_iso8601",
    "optimize_dtypes",
    "log_basic_diagnostics",
    "canonical_location",
    "derive_city_mappings",
    "normalize_location_weights"
]

class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder che supporta tipi numpy, pandas e datetime."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)

def set_global_seed(seed: int) -> np.random.Generator:
    """
    Imposta il seed globale per garantire riproducibilità
    su `numpy` e `random`.

    Args:
        seed: Valore intero per inizializzare il RNG.
    Returns:
        Istanza np.random.Generator per uso locale.
    """
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    logger.info("[UTILS] Seed globale impostato", extra={"seed": seed})
    return rng

def get_utc_now() -> datetime:
    """Ritorna il datetime corrente in UTC (timezone-aware)."""
    return datetime.now(timezone.utc)

def parse_iso8601(s: str) -> Optional[datetime]:
    """
    Converte una stringa ISO8601 in `datetime` UTC.

    Args:
        s: Stringa in formato ISO8601.

    Returns:
        datetime con timezone UTC, oppure None se parsing fallisce.
    """
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception as e:
        logger.warning("[UTILS] Parsing ISO8601 fallito", extra={"input": s, "error": str(e)})
        return None

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restituisce una copia ottimizzata in termini di memoria del DataFrame,
    preservando valori NaN e riducendo la precisione dei tipi numerici quando possibile.

    - float64 → float32
    - int64   → Int16 o Int32 (nullable)
    """
    df_opt = df.copy()

    for col in df_opt.select_dtypes(include=["float64"]).columns:
        df_opt[col] = df_opt[col].astype("float32")

    for col in df_opt.select_dtypes(include=["int64"]).columns:
        mn, mx = df_opt[col].min(), df_opt[col].max()
        if pd.notna(mn) and pd.notna(mx):
            if -32768 <= mn <= mx <= 32767:
                df_opt[col] = df_opt[col].astype("Int16")  # nullable
            else:
                df_opt[col] = df_opt[col].astype("Int32")  # nullable

    logger.debug(
        "[UTILS] Ottimizzazione dtypes completata: "
        f"da {df.memory_usage(deep=True).sum()/1024:.1f} KB "
        f"a {df_opt.memory_usage(deep=True).sum()/1024:.1f} KB"
    )
    return df_opt

def log_basic_diagnostics(df: pd.DataFrame, log: logging.Logger = logger) -> None:
    """
    Logga statistiche di base sul dataset.
    """
    if LOCATION in df.columns:
        log.info("[UTILS] Distribuzione per location:\n%s", df[LOCATION].value_counts().to_string())
    if "valuation_k" in df.columns:
        log.info("[UTILS] Prezzo min: %.2fk€", df["valuation_k"].min())
        log.info("[UTILS] Prezzo max: %.2fk€", df["valuation_k"].max())
        log.info("[UTILS] Prezzo medio: %.2fk€", df["valuation_k"].mean())
    if {"size_m2", "valuation_k"}.issubset(df.columns):
        corr = df[["size_m2", "valuation_k"]].corr().iloc[0, 1]
        log.info("[UTILS] Corr size_m2 vs valuation_k: %.3f", corr)

def canonical_location(record: Union[Dict[str, Any], pd.Series]) -> str:
    """
    Ritorna la location canonica di un record.
    """
    if LOCATION in record and record[LOCATION]:
        return record[LOCATION]
    if ZONE in record and record[ZONE]:
        return record[ZONE]
    return ""

def derive_city_mappings(
    source: Union[Dict[str, Any], List[str]],
    urban_override: Optional[Dict[str, str]] = None,
    region_override: Optional[Dict[str, str]] = None
) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """
    Unisce mappature urban/region per le città fornite.
    """
    if isinstance(source, dict):
        locs = list(source.get("location_weights", {}).keys())
        urban_override = urban_override or source.get("urban_type_by_city")
        region_override = region_override or source.get("region_by_city")
    elif isinstance(source, list):
        locs = source
    else:
        raise ValueError("source deve essere dict o lista di città")

    urban_map = DEFAULT_URBAN_TYPE_BY_CITY.copy()
    region_map = DEFAULT_REGION_BY_CITY.copy()

    if urban_override:
        urban_map.update(urban_override)
    if region_override:
        region_map.update(region_override)

    for city in set(locs) - set(urban_map):
        logger.warning("[UTILS] Urban type mancante, fallback 'urban'", extra={"city": city})
        urban_map[city] = "urban"

    for city in set(locs) - set(region_map):
        logger.warning("[UTILS] Region mancante, fallback 'unknown'", extra={"city": city})
        region_map[city] = "unknown"

    return locs, urban_map, region_map

def normalize_location_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Normalizza e valida i pesi delle location.
    """
    if not weights:
        raise ValueError("location_weights vuoto o None")

    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("La somma di location_weights deve essere > 0")

    normalized = {k: v / total_weight for k, v in weights.items()}

    if not np.isclose(sum(normalized.values()), 1.0):
        logger.warning("[UTILS] Location weights non sommano esattamente a 1 dopo normalizzazione.")

    return normalized