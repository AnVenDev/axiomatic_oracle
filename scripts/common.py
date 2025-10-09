from datetime import datetime
import logging
import os
from typing import Any, Dict, Final, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

YEAR_BUILT: str = "year_built"

ASSET_ID = "asset_id"

DEFAULT_REGION_BY_CITY: Final[Dict[str, str]] = {
    "Milan": "north",
    "Rome": "center",
    "Turin": "north",
    "Naples": "south",
    "Bologna": "north",
    "Florence": "center",
    "Genoa": "north",
    "Palermo": "south",
    "Venice": "north",
    "Verona": "north",
    "Bari": "south",
    "Padua": "north",
    "Catania": "south",
    "Trieste": "north",
    "Cagliari": "south",
}

DEFAULT_URBAN_TYPE_BY_CITY: Final[Dict[str, str]] = {
    "Milan": "urban",
    "Rome": "urban",
    "Turin": "urban",
    "Naples": "urban",
    "Bologna": "urban",
    "Florence": "urban",
    "Genoa": "urban",
    "Palermo": "urban",
    "Venice": "urban",
    "Verona": "semiurban",
    "Bari": "urban",
    "Padua": "semiurban",
    "Catania": "urban",
    "Trieste": "semiurban",
    "Cagliari": "urban",
}

LOCATION = "location"

ASSET_CONFIG: Dict[str, Dict[str, Any]] = {
    "property": {
        # Raw features by semantic type (used by pipelines/validators).
        # NOTE: `city` is the primary location key produced by the generator;
        #       free-form `location` is excluded to avoid duplication.
        "categorical": [
            "city",
            "region", "zone",
            "energy_class", "condition", "heating", "view",
            "public_transport_nearby",
        ],
        "numeric": [
            "size_m2", "rooms", "bathrooms", "floor", "building_floors",
            "has_elevator", "has_garden", "has_balcony", "garage",
            "year_built", "listing_month",
        ],
        # Avoid duplicated semantics when both `city` and free-form `location` exist.
        "exclude": ["location"],

        # Domain normalization / synonyms (start from constants; allow overrides).
        "region_by_city": {**DEFAULT_REGION_BY_CITY},
        "urban_type_by_city": {**DEFAULT_URBAN_TYPE_BY_CITY},

        # Canonical city names (lowercase → Title Case).
        "city_synonyms": {
            "milano": "Milan", "firenze": "Florence", "roma": "Rome",
            "torino": "Turin", "napoli": "Naples", "genova": "Genoa",
            "venezia": "Venice", "cagliari": "Cagliari", "verona": "Verona",
            "trieste": "Trieste", "padova": "Padua", "bari": "Bari",
            "catania": "Catania", "palermo": "Palermo",
        },
    }
}

_LEVEL_MAP: Mapping[str, int] = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}

VIEW_LANDMARKS = "landmarks"
VIEW_SEA = "sea"

class Cols:
    # Core
    ASSET_ID = "asset_id"
    ASSET_TYPE = "asset_type"
    LOCATION = "location"
    VALUATION_K = "valuation_k"              # model-native unit: kEUR
    PRICE_PER_SQM = "price_per_sqm"
    LAST_VERIFIED_TS = "last_verified_ts"

    # Structural
    REGION = "region"
    URBAN_TYPE = "urban_type"
    ZONE = "zone"
    SIZE_M2 = "size_m2"
    ROOMS = "rooms"
    BATHROOMS = "bathrooms"
    YEAR_BUILT = "year_built"
    AGE_YEARS = "age_years"
    FLOOR = "floor"
    BUILDING_FLOORS = "building_floors"
    IS_TOP_FLOOR = "is_top_floor"
    IS_GROUND_FLOOR = "is_ground_floor"

    # Amenities
    HAS_ELEVATOR = "has_elevator"
    HAS_GARDEN = "has_garden"
    HAS_BALCONY = "has_balcony"
    GARAGE = "garage"
    OWNER_OCCUPIED = "owner_occupied"
    PUBLIC_TRANSPORT_NEARBY = "public_transport_nearby"
    DISTANCE_TO_CENTER_KM = "distance_to_center_km"
    PARKING_SPOT = "parking_spot"
    CELLAR = "cellar"
    ATTIC = "attic"
    CONCIERGE = "concierge"

    # Quality
    ENERGY_CLASS = "energy_class"
    HUMIDITY_LEVEL = "humidity_level"
    TEMPERATURE_AVG = "temperature_avg"
    NOISE_LEVEL = "noise_level"
    AIR_QUALITY_INDEX = "air_quality_index"

    # Scores
    CONDITION_SCORE = "condition_score"
    RISK_SCORE = "risk_score"
    LUXURY_SCORE = "luxury_score"
    ENV_SCORE = "env_score"

    # Domain-specific
    ORIENTATION = "orientation"
    VIEW = "view"
    CONDITION = "condition"
    HEATING = "heating"

    # Derived (generic)
    LISTING_QUARTER = "listing_quarter"
    LISTING_MONTH = "listing_month"
    DECADE_BUILT = "decade_built"
    BUILDING_AGE_YEARS = "building_age_years"
    ROOMS_PER_SQM = "rooms_per_sqm"
    BATHROOMS_PER_ROOM = "bathrooms_per_room"
    AVG_ROOM_SIZE = "avg_room_size"
    LOCATION_PREMIUM = "location_premium"
    BASIC_AMENITY_COUNT = "basic_amenity_count"
    PRICE_PER_SQM_VS_REGION_AVG = "price_per_sqm_vs_region_avg"
    PREDICTION_TS = "prediction_ts"
    LAG_HOURS = "lag_hours"
    MISSING_TIMESTAMP = "missing_timestamp"

    # Extended derived
    PRICE_PER_SQM_CAPPED = "price_per_sqm_capped"
    PRICE_PER_SQM_CAPPED_VIOLATED = "price_per_sqm_capped_violated"
    LISTING_MONTH_SIN = "listing_month_sin"
    LISTING_MONTH_COS = "listing_month_cos"
    DAYS_SINCE_VERIFICATION = "days_since_verification"
    HOURS_SINCE_VERIFICATION = "hours_since_verification"
    IS_STALE_30D = "is_stale_30d"
    IS_STALE_60D = "is_stale_60d"
    IS_STALE_90D = "is_stale_90d"
    ANOMALY_SCORE = "anomaly_score"
    ANOMALY_SCORE_RAW = "anomaly_score_raw"
    ANOMALY_FLAG = "anomaly_flag"
    ANOMALY_LABEL = "anomaly_label"
    SEVERITY_SCORE = "severity_score"
    CONFIDENCE_SCORE = "confidence_score"
    VALUE_SEGMENT = "value_segment"
    LUXURY_CATEGORY = "luxury_category"
    AGE_CATEGORY = "age_category"
    TIMESTAMP = "timestamp"

    # Interaction features (train=serve)
    GARAGE_VS_CENTRAL = "garage_vs_central"
    ATTIC_VS_FLOORS = "attic_vs_floors"

def _level_from_any(level: Union[int, str]) -> int:
    if isinstance(level, int):
        return level
    return _LEVEL_MAP.get(level.upper(), logging.INFO)

def configure_logger(
    level: Union[int, str] = logging.INFO,
    name: Optional[str] = None,
    json_format: Optional[bool] = None,
) -> logging.Logger:
    """
    Create or return a configured logger.

    Args:
        level: numeric or string level (e.g., "INFO").
        name: logger name; None → root logger.
        json_format: force JSON formatting; if None, honor env LOG_FORMAT=json|text.
    """
    lvl = _level_from_any(level)
    lg = logging.getLogger(name) if name else logging.getLogger()
    lg.setLevel(lvl)

    # Avoid duplicate stream handlers
    have_stream = any(isinstance(h, logging.StreamHandler) for h in lg.handlers)
    if not have_stream:
        handler = logging.StreamHandler()
        fmt = os.getenv("LOG_FORMAT", "").lower() if json_format is None else ("json" if json_format else "text")
        if fmt == "json":
            formatter = logging.Formatter(
                '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s",'
                '"msg":"%(message)s","module":"%(module)s","line":%(lineno)d}'
            )
        else:
            formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(formatter)
        lg.addHandler(handler)

    # If not root, prevent propagation to avoid duplicate logs
    if name:
        lg.propagate = False

    return lg

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

PRICE_COL: str = "price_per_sqm"
LOCATION_COL: str = "location"
ZONE_COL: str = "zone"

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# COSTANTI (estratto minimale da constants.py)
# =============================================================================

# Core columns
ASSET_ID: str         = "asset_id"
ASSET_TYPE: str       = "asset_type"
LOCATION: str         = "location"
VALUATION_K: str      = "valuation_k"        # unità modello: kEUR
PRICE_PER_SQM: str    = "price_per_sqm"
LAST_VERIFIED_TS: str = "last_verified_ts"

# Structural
REGION: str           = "region"
URBAN_TYPE: str       = "urban_type"
ZONE: str             = "zone"
SIZE_M2: str          = "size_m2"
ROOMS: str            = "rooms"
BATHROOMS: str        = "bathrooms"
YEAR_BUILT: str       = "year_built"
AGE_YEARS: str        = "age_years"
FLOOR: str            = "floor"
BUILDING_FLOORS: str  = "building_floors"
IS_TOP_FLOOR: str     = "is_top_floor"
IS_GROUND_FLOOR: str  = "is_ground_floor"

# Amenities
HAS_ELEVATOR: str              = "has_elevator"
HAS_GARDEN: str                = "has_garden"
HAS_BALCONY: str               = "has_balcony"
GARAGE: str                    = "garage"
OWNER_OCCUPIED: str            = "owner_occupied"
PUBLIC_TRANSPORT_NEARBY: str   = "public_transport_nearby"

# Quality
ENERGY_CLASS: str      = "energy_class"
HUMIDITY_LEVEL: str    = "humidity_level"
TEMPERATURE_AVG: str   = "temperature_avg"
NOISE_LEVEL: str       = "noise_level"
AIR_QUALITY_INDEX: str = "air_quality_index"

# Scores
CONDITION_SCORE: str = "condition_score"
RISK_SCORE: str      = "risk_score"
LUXURY_SCORE: str    = "luxury_score"
ENV_SCORE: str       = "env_score"

# Domain fields
ORIENTATION: str = "orientation"
VIEW: str        = "view"
CONDITION: str   = "condition"
HEATING: str     = "heating"

# Domains / allowed values
ENERGY_CLASSES: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G")
VALID_ORIENTATIONS: List[str] = [
    "North", "South", "East", "West",
    "North-East", "North-West", "South-East", "South-West",
]
VALID_VIEWS: List[str]   = ["street", "inner courtyard", "garden", "park", "sea", "mountain", "landmarks"]
VALID_STATES: List[str]  = ["new", "renovated", "good", "needs_renovation"]
VALID_HEATING: List[str] = ["autonomous", "centralized", "heat pump", "none"]

# (usate da price_benchmark per chiarezza semantica)
PRICE_COL: str    = PRICE_PER_SQM
LOCATION_COL: str = LOCATION
ZONE_COL: str     = ZONE

# =============================================================================
# Fallback schema minimale per record singoli
# =============================================================================
def get_required_fields(asset_type: str) -> List[str]:
    if asset_type == "property":
        # set minimo sufficiente per la validazione single-record
        return [
            ASSET_ID, ASSET_TYPE, LOCATION, SIZE_M2, ROOMS, BATHROOMS,
            FLOOR, BUILDING_FLOORS, ENERGY_CLASS, LAST_VERIFIED_TS,
        ]
    return [ASSET_ID, ASSET_TYPE, LAST_VERIFIED_TS]


# =============================================================================
# Helpers (legacy → canonico)
# =============================================================================
def _normalize_legacy_keys(prop_data: Dict[str, Any]) -> None:
    """Map in-place da chiavi italiane/legacy a canoniche (best-effort)."""
    if "vista" in prop_data and VIEW not in prop_data and "view" not in prop_data:
        prop_data["view"] = prop_data.pop("vista")
    if "orientamento" in prop_data and ORIENTATION not in prop_data and "orientation" not in prop_data:
        prop_data["orientation"] = prop_data.pop("orientamento")
    if "stato" in prop_data and CONDITION not in prop_data and "condition" not in prop_data:
        prop_data["condition"] = prop_data.pop("stato")
    if "riscaldamento" in prop_data and HEATING not in prop_data and "heating" not in prop_data:
        prop_data["heating"] = prop_data.pop("riscaldamento")


# =============================================================================
# PUBLIC: price_benchmark
# =============================================================================
def price_benchmark(df: pd.DataFrame) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
    """
    Mediane €/m² per città e, se presente, pivot per (città, zona).

    Returns:
        (city_medians: Series, zone_pivot: DataFrame | None)
    """
    if PRICE_COL not in df.columns:
        raise ValueError(f"Missing column '{PRICE_COL}'.")
    city_med = (
        df[[LOCATION_COL, PRICE_COL]]
        .dropna()
        .groupby(LOCATION_COL, observed=True)[PRICE_COL]
        .median()
        .sort_values(ascending=False)
    )
    zone_med: Optional[pd.DataFrame] = None
    if ZONE_COL in df.columns:
        zone_med = (
            df[[LOCATION_COL, ZONE_COL, PRICE_COL]]
            .dropna()
            .groupby([LOCATION_COL, ZONE_COL], observed=True)[PRICE_COL]
            .median()
            .unstack(level=ZONE_COL)
        )
    return city_med, zone_med


# =============================================================================
# PUBLIC: validate_property
# =============================================================================
def validate_property(prop_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valida & normalizza un singolo record 'property' (best-effort, in-place).

    - Normalizza chiavi legacy
    - Coerenza piani + flag top/ground
    - Clip scores in [0,1]
    - Minimi ragionevoli: size>=20, rooms>=1, bathrooms>=1
    - Fallback valuation/price se mancanti (euristica 500 €/m²)
    - Domini categorici (energy_class/orientation/view/condition/heating)
    - Completamento schema minimo (asset_id/type/last_verified_ts)
    """
    original = dict(prop_data)
    errors: List[str] = []
    flags:  List[str] = []

    # 0) Legacy mapping
    _normalize_legacy_keys(prop_data)

    # 1) Coerenza piani
    try:
        floor_val = int(prop_data.get(FLOOR, prop_data.get("floor", 0)) or 0)
    except Exception:
        floor_val = 0
    try:
        bld_floors = int(prop_data.get(BUILDING_FLOORS, prop_data.get("building_floors", 0)) or 0)
    except Exception:
        bld_floors = 0

    if floor_val > bld_floors:
        errors.append("floor_gt_building_floors")
        prop_data[FLOOR] = bld_floors
        flags.append("floor_adjusted")
        floor_val = bld_floors

    prop_data[IS_TOP_FLOOR]    = int(floor_val == max(bld_floors - 1, 0))
    prop_data[IS_GROUND_FLOOR] = int(floor_val == 0)

    # 2) Score clipping [0,1]
    for field in (CONDITION_SCORE, RISK_SCORE, LUXURY_SCORE, ENV_SCORE):
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

    # 3) Minimi ragionevoli
    size = float(prop_data.get(SIZE_M2, prop_data.get("size_m2", 0)) or 0.0)
    if size < 20.0:
        errors.append("size_m2_too_small")
        prop_data[SIZE_M2] = 20.0
        flags.append("size_clamped")
        size = 20.0

    rooms = int(prop_data.get(ROOMS, prop_data.get("ROOMS", 0)) or 0)
    if rooms < 1:
        errors.append("rooms_too_few")
        prop_data[ROOMS] = 1
        flags.append("rooms_clamped")

    baths = int(prop_data.get(BATHROOMS, prop_data.get("BATHROOMS", 0)) or 0)
    if baths < 1:
        errors.append("bathrooms_too_few")
        prop_data[BATHROOMS] = 1
        flags.append("bathrooms_clamped")

    # 4) Fallback valuation/price
    valuation_k = float(prop_data.get(VALUATION_K, prop_data.get("valuation_k", 0)) or 0.0)
    if valuation_k < 10.0:
        errors.append("valuation_k_too_low_or_missing")
        fallback_price = (size * 500.0) / 1000.0  # euristica 500 €/m²
        prop_data[VALUATION_K] = round(fallback_price, 2)
        flags.append("valuation_override")
        valuation_k = prop_data[VALUATION_K]

    pps = float(prop_data.get(PRICE_PER_SQM, prop_data.get("price_per_sqm", 0)) or 0.0)
    if pps <= 0.0 and size > 0:
        errors.append("price_per_sqm_non_positive_or_missing")
        prop_data[PRICE_PER_SQM] = round((valuation_k * 1000.0) / size, 2)
        flags.append("price_per_sqm_recomputed")

    # 5) Domini categorici
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

    # 6) Completamento schema minimo (non-fatale)
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

    # 7) Metadata log-friendly
    prop_data.setdefault("validation_errors", []).extend(errors)
    prop_data.setdefault("validation_flags", []).extend(flags)

    if errors:
        diff = {
            k: (original.get(k), prop_data.get(k))
            for k in set(prop_data) | set(original)
            if original.get(k) != prop_data.get(k)
        }
        logger.warning(
            "[VALIDATION] Asset %s normalized. Errors=%s Flags=%s Changes=%s",
            prop_data.get(ASSET_ID, "unknown"),
            errors,
            flags,
            diff,
        )

    return prop_data

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Chiavi canoniche minime (evitiamo import dal progetto)
LOCATION_KEY = "location"
ZONE_KEY = "zone"


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder con supporto first-class per numpy/pandas/datetime."""
    def default(self, obj: Any) -> Any:
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
        if isinstance(obj, (datetime,)):
            return obj.isoformat()
        return super().default(obj)


def _normalize_str(s: str) -> str:
    """
    Normalizzazione super-leggera e non distruttiva:
    - trim
    - collapse spazi multipli
    - rimuove separatori terminali comuni
    - non forza lingue o mappature città (evita sorprese)
    """
    s = (s or "").strip()
    if not s:
        return ""
    # se c'è una stringa tipo "Milan, Italy" o "Milan | center", prendi la parte di città
    for sep in ("|", ","):
        if sep in s:
            head = s.split(sep, 1)[0].strip()
            if head:
                s = head
                break
    # collassa whitespace
    s = " ".join(s.split())
    return s


def canonical_location(record_or_str: Union[Dict[str, Any], pd.Series, str]) -> str:
    """
    Restituisce una location canonica (best-effort).
    - Se input è dict/Series: preferisce 'location', altrimenti 'zone'.
    - Se input è str: normalizza leggermente e restituisce la stringa.
    Non solleva eccezioni: in caso di problemi ritorna "".
    """
    try:
        if isinstance(record_or_str, (dict, pd.Series)):
            rec = record_or_str
            if LOCATION_KEY in rec and rec[LOCATION_KEY]:
                return _normalize_str(str(rec[LOCATION_KEY]))
            if ZONE_KEY in rec and rec[ZONE_KEY]:
                return _normalize_str(str(rec[ZONE_KEY]))
            return ""
        # stringa
        return _normalize_str(str(record_or_str))
    except Exception as e:
        logger.debug("canonical_location: fallback empty due to %s", e)
        return ""


def get_utc_now() -> datetime:
    """Ritorna l'istante corrente in UTC (timezone-aware)."""
    return datetime.now(timezone.utc)

    
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

ZONE = "zone"
DEFAULT_BASE_PRICE_FALLBACK: float = 3000.0 # €/m² when city/zone missing
ENERGY_CLASS = "energy_class"
CONDITION = "condition"
IS_TOP_FLOOR = "is_top_floor"
IS_GROUND_FLOOR = "is_ground_floor"
HAS_GARDEN = "has_garden"
HAS_BALCONY = "has_balcony"
GARAGE = "garage"
ORIENTATION_BONUS: float = 1.05             # south-ish exposure bonus
HEATING_AUTONOMOUS_BONUS: float = 1.03      # autonomous heating bonus
_SUNNY_ORIENTATIONS = {"south", "south-east", "south-west", "southeast", "southwest"}
_HEATING_AUTONOMOUS = "autonomous"
ORIENTATION = "orientation"
HEATING = "heating"
VIEW = "view"

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