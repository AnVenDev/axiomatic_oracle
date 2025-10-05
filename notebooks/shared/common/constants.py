"""
Canonical constants for the RWA Property stack.

Purpose
- Typed enums for discrete domains (zones, energy classes, etc.).
- Canonical column names (single source of truth) and grouped sets.
- Versioning, network limits, seeds, and safe defaults.
- Centralized coefficients for conditional pricing (elevator×floor, garage×centrality, attic×floors, humidity).
- Backward-compatibility aliases for historical constant names.

Design principles
- Dependency-free; safe to import from any layer (generation, training, serving).
- Low-level primitives only (str/int/float/tuples); no heavy objects.
- Names are stable; prefer adding over renaming to preserve compatibility.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Final, Tuple

# =============================================================================
# City mappings & default domains
# =============================================================================

# English → Italian aliases (used by normalization where applicable).
CITY_ALIASES: Final[Dict[str, str]] = {
    "Naples": "Napoli",
    "Turin": "Torino",
    "Genoa": "Genova",
    "Venice": "Venezia",
    "Padua": "Padova",
}

# Default urban type by city: "urban" | "semiurban" | "rural".
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

# Macro-region per city (north|center|south).
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

# =============================================================================
# Enums (typed domains)
# =============================================================================

class AssetType(str, Enum):
    PROPERTY = "property"


class Zone(str, Enum):
    CENTER = "center"
    SEMI_CENTER = "semi_center"
    PERIPHERY = "periphery"


class UrbanType(str, Enum):
    URBAN = "urban"
    SEMIURBAN = "semiurban"
    RURAL = "rural"


class EnergyClass(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"


class ViewType(str, Enum):
    SEA = "sea"
    LANDMARKS = "landmarks"


# =============================================================================
# Versioning & chain limits
# =============================================================================

class Versions:
    DATASET: Final[str] = "v2"
    FEATURE_SET: Final[str] = "v1"
    MODEL_FAMILY: Final[str] = "axiomatic_rwa_property"


SEED: Final[int] = 42  # centralized RNG seed for reproducibility

SCHEMA_VERSION: Final[str] = "v2"
NOTE_MAX_BYTES: Final[int] = 1024            # Algorand note hard cap (bytes)
NOTE_PREFIX: Final[bytes] = b"AXM\x01"       # 4B prefix for indexer/explorer tagging
NETWORK: Final[str] = "testnet"              # default; actual runtime via env/secrets

# Features that must NOT be used as inputs (leakage guard).
LEAKY_FEATURES: Final[set[str]] = {
    "valuation_k", "price_per_sqm", "price_per_sqm_vs_region_avg",
    "target", "y", "label",
}

# Expected prediction range (sanity checks; unit depends on the downstream metric).
EXPECTED_PRED_RANGE: Final[tuple[float, float]] = (20.0, 20_000.0)  # €/m² (demo-safe)


# =============================================================================
# Canonical column names (single source of truth)
# =============================================================================

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


# =============================================================================
# Groups & sets
# =============================================================================

class Groups:
    DERIVED_FEATURES: Final[Tuple[str, ...]] = (
        Cols.LISTING_QUARTER,
        Cols.LISTING_MONTH,
        Cols.DECADE_BUILT,
        Cols.BUILDING_AGE_YEARS,
        Cols.ROOMS_PER_SQM,
        Cols.BATHROOMS_PER_ROOM,
        Cols.AVG_ROOM_SIZE,
        Cols.LOCATION_PREMIUM,
        Cols.BASIC_AMENITY_COUNT,
        Cols.PRICE_PER_SQM_VS_REGION_AVG,
        Cols.PREDICTION_TS,
        Cols.LAG_HOURS,
        Cols.MISSING_TIMESTAMP,
        # new interaction features are optional/derived:
        Cols.GARAGE_VS_CENTRAL,
        Cols.ATTIC_VS_FLOORS,
    )

    ENERGY_CLASSES: Final[Tuple[str, ...]] = tuple(ec.value for ec in EnergyClass)


# =============================================================================
# Thresholds & defaults
# =============================================================================

class Thresholds:
    MIN_SIZE_M2: Final[int] = 10
    MAX_SIZE_M2: Final[int] = 1000
    MAX_LAG_HOURS: Final[int] = 8760  # 1 year

    DEFAULT_ZONE_THRESHOLDS_KM: Final[Dict[str, float]] = {
        Zone.CENTER.value: 1.5,
        Zone.SEMI_CENTER.value: 5.0,
    }

    # Conditional-pricing coefficients (tunable; used by pricing/generation)
    ELEV_BONUS_PER_FLOOR: Final[float] = 0.010      # +1.0% per floor (>1) with elevator
    NO_ELEV_PENALTY_PER_FLOOR: Final[float] = 0.015 # -1.5% per floor (>2) without elevator
    GARAGE_CENTRALITY_COEF: Final[float] = 0.08     # bonus = coef / max(distance_km, CENTRALITY_EPS_KM)
    ATTIC_PER_FLOOR_COEF: Final[float] = 0.03       # +3% per building floor (penthouse premium)
    HUMIDITY_PENALTY_MAX: Final[float] = 0.10       # up to -10% at 70 on scale 30..70
    CENTRALITY_EPS_KM: Final[float] = 0.25          # to cap runaway multipliers near zero distance

    # Generation defaults
    MIN_FLOORS_FOR_ELEVATOR: Final[int] = 4
    ATTIC_TOPFLOOR_PRIOR: Final[float] = 0.35


class Defaults:
    """Reserved for future default knobs (scoring weights, etc.)."""
    pass


# =============================================================================
# Mappings
# =============================================================================

class Mappings:
    # Energy class rank (A best)
    ENERGY_CLASS_RANK: Final[Dict[str, int]] = {
        EnergyClass.A.value: 7,
        EnergyClass.B.value: 6,
        EnergyClass.C.value: 5,
        EnergyClass.D.value: 4,
        EnergyClass.E.value: 3,
        EnergyClass.F.value: 2,
        EnergyClass.G.value: 1,
    }

    # Zone ordering (center > semi_center > periphery)
    ZONE_ORDER: Final[Tuple[str, str, str]] = (
        Zone.CENTER.value,
        Zone.SEMI_CENTER.value,
        Zone.PERIPHERY.value,
    )


# =============================================================================
# Backward-compatibility aliases
# =============================================================================

# Core
ASSET_ID = Cols.ASSET_ID
ASSET_TYPE = Cols.ASSET_TYPE
LOCATION = Cols.LOCATION
VALUATION_K = Cols.VALUATION_K
PRICE_PER_SQM = Cols.PRICE_PER_SQM
LAST_VERIFIED_TS = Cols.LAST_VERIFIED_TS

# Structural
REGION = Cols.REGION
URBAN_TYPE = Cols.URBAN_TYPE
ZONE = Cols.ZONE
SIZE_M2 = Cols.SIZE_M2
ROOMS = Cols.ROOMS
BATHROOMS = Cols.BATHROOMS
YEAR_BUILT = Cols.YEAR_BUILT
AGE_YEARS = Cols.AGE_YEARS
FLOOR = Cols.FLOOR
BUILDING_FLOORS = Cols.BUILDING_FLOORS
IS_TOP_FLOOR = Cols.IS_TOP_FLOOR
IS_GROUND_FLOOR = Cols.IS_GROUND_FLOOR
TOP_FLOOR = IS_TOP_FLOOR
GROUND_FLOOR = IS_GROUND_FLOOR

# Amenities
HAS_ELEVATOR = Cols.HAS_ELEVATOR
HAS_GARDEN = Cols.HAS_GARDEN
HAS_BALCONY = Cols.HAS_BALCONY
GARAGE = Cols.GARAGE
OWNER_OCCUPIED = Cols.OWNER_OCCUPIED
PUBLIC_TRANSPORT_NEARBY = Cols.PUBLIC_TRANSPORT_NEARBY
DISTANCE_TO_CENTER_KM = Cols.DISTANCE_TO_CENTER_KM
PARKING_SPOT = Cols.PARKING_SPOT
CELLAR = Cols.CELLAR
ATTIC = Cols.ATTIC
CONCIERGE = Cols.CONCIERGE

# Quality
ENERGY_CLASS = Cols.ENERGY_CLASS
HUMIDITY_LEVEL = Cols.HUMIDITY_LEVEL
TEMPERATURE_AVG = Cols.TEMPERATURE_AVG
NOISE_LEVEL = Cols.NOISE_LEVEL
AIR_QUALITY_INDEX = Cols.AIR_QUALITY_INDEX

# Scores
CONDITION_SCORE = Cols.CONDITION_SCORE
RISK_SCORE = Cols.RISK_SCORE
LUXURY_SCORE = Cols.LUXURY_SCORE
ENV_SCORE = Cols.ENV_SCORE

# Domain
ORIENTATION = Cols.ORIENTATION
VIEW = Cols.VIEW
CONDITION = Cols.CONDITION
HEATING = Cols.HEATING

# View values
VIEW_SEA = ViewType.SEA.value
VIEW_LANDMARKS = ViewType.LANDMARKS.value

# Urban types
URBAN = UrbanType.URBAN.value
SEMIURBAN = UrbanType.SEMIURBAN.value
RURAL = UrbanType.RURAL.value

# Zones
ZONE_CENTER = Zone.CENTER.value
ZONE_SEMI_CENTER = Zone.SEMI_CENTER.value
ZONE_PERIPHERY = Zone.PERIPHERY.value

# Zone thresholds
DEFAULT_ZONE_THRESHOLDS = Thresholds.DEFAULT_ZONE_THRESHOLDS_KM

# Derived fields
LISTING_QUARTER = Cols.LISTING_QUARTER
LISTING_MONTH = Cols.LISTING_MONTH
DECADE_BUILT = Cols.DECADE_BUILT
BUILDING_AGE_YEARS = Cols.BUILDING_AGE_YEARS
ROOMS_PER_SQM = Cols.ROOMS_PER_SQM
BATHROOMS_PER_ROOM = Cols.BATHROOMS_PER_ROOM
AVG_ROOM_SIZE = Cols.AVG_ROOM_SIZE
LOCATION_PREMIUM = Cols.LOCATION_PREMIUM
BASIC_AMENITY_COUNT = Cols.BASIC_AMENITY_COUNT
PRICE_PER_SQM_VS_REGION_AVG = Cols.PRICE_PER_SQM_VS_REGION_AVG
PREDICTION_TS = Cols.PREDICTION_TS
LAG_HOURS = Cols.LAG_HOURS
MISSING_TIMESTAMP = Cols.MISSING_TIMESTAMP

# Extended derived
PRICE_PER_SQM_CAPPED = Cols.PRICE_PER_SQM_CAPPED
PRICE_PER_SQM_CAPPED_VIOLATED = Cols.PRICE_PER_SQM_CAPPED_VIOLATED
LISTING_MONTH_SIN = Cols.LISTING_MONTH_SIN
LISTING_MONTH_COS = Cols.LISTING_MONTH_COS
DAYS_SINCE_VERIFICATION = Cols.DAYS_SINCE_VERIFICATION
HOURS_SINCE_VERIFICATION = Cols.HOURS_SINCE_VERIFICATION
IS_STALE_30D = Cols.IS_STALE_30D
IS_STALE_60D = Cols.IS_STALE_60D
IS_STALE_90D = Cols.IS_STALE_90D
ANOMALY_SCORE = Cols.ANOMALY_SCORE
ANOMALY_SCORE_RAW = Cols.ANOMALY_SCORE_RAW
ANOMALY_FLAG = Cols.ANOMALY_FLAG
ANOMALY_LABEL = Cols.ANOMALY_LABEL
SEVERITY_SCORE = Cols.SEVERITY_SCORE
CONFIDENCE_SCORE = Cols.CONFIDENCE_SCORE
VALUE_SEGMENT = Cols.VALUE_SEGMENT
LUXURY_CATEGORY = Cols.LUXURY_CATEGORY
AGE_CATEGORY = Cols.AGE_CATEGORY
TIMESTAMP = Cols.TIMESTAMP

# Interaction features (aliases)
GARAGE_VS_CENTRAL = Cols.GARAGE_VS_CENTRAL
ATTIC_VS_FLOORS = Cols.ATTIC_VS_FLOORS

# Grouping (retro-compat)
DERIVED_FEATURES: Final[Tuple[str, ...]] = Groups.DERIVED_FEATURES

# Validation thresholds (retro-compat)
MIN_SIZE_M2 = Thresholds.MIN_SIZE_M2
MAX_SIZE_M2 = Thresholds.MAX_SIZE_M2
MAX_LAG_HOURS = Thresholds.MAX_LAG_HOURS

# Energy class values & group (retro-compat)
ENERGY_CLASS_A = EnergyClass.A.value
ENERGY_CLASS_B = EnergyClass.B.value
ENERGY_CLASS_C = EnergyClass.C.value
ENERGY_CLASS_D = EnergyClass.D.value
ENERGY_CLASS_E = EnergyClass.E.value
ENERGY_CLASS_F = EnergyClass.F.value
ENERGY_CLASS_G = EnergyClass.G.value
ENERGY_CLASSES = Groups.ENERGY_CLASSES


# =============================================================================
# Notebook-oriented defaults (kept minimal; legacy removed)
# =============================================================================

class AnalysisDefaults:
    """Lightweight hints for EDA—not used by core libraries."""
    VARIANCE_THRESHOLD: float = 0.01
    VIF_THRESHOLD: float = 10.0
    CORRELATION_THRESHOLD: float = 0.95
    OUTLIER_STD_THRESHOLD: float = 3.0
    DEFAULT_FIGSIZE: Tuple[int, int] = (12, 6)
    DEFAULT_PALETTE: str = "husl"
    DEFAULT_STYLE: str = "whitegrid"
    NUMERIC_EXCLUDE: Tuple[str, ...] = (
        Cols.ASSET_ID, Cols.ASSET_TYPE, Cols.LOCATION, Cols.REGION, Cols.URBAN_TYPE,
        Cols.ZONE, Cols.ORIENTATION, Cols.VIEW, Cols.CONDITION, Cols.HEATING,
        Cols.LAST_VERIFIED_TS, Cols.PREDICTION_TS,
    )
    CATEGORICAL_FOCUS: Tuple[str, ...] = (
        Cols.LOCATION, Cols.ZONE, Cols.URBAN_TYPE, Cols.REGION,
        Cols.ENERGY_CLASS, Cols.CONDITION, Cols.HEATING, Cols.VIEW,
    )


class MLDefaults:
    """Deprecated: kept for compatibility. No default RF/XGB params here."""
    pass

# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # mappings & defaults
    "CITY_ALIASES", "DEFAULT_URBAN_TYPE_BY_CITY", "DEFAULT_REGION_BY_CITY",
    # enums
    "AssetType", "Zone", "UrbanType", "EnergyClass", "ViewType",
    # versions/seed/constants
    "Versions", "SEED", "SCHEMA_VERSION", "NOTE_MAX_BYTES", "NOTE_PREFIX", "NETWORK",
    "LEAKY_FEATURES", "EXPECTED_PRED_RANGE",
    # namespaces / groups / thresholds / mappings
    "Cols", "Groups", "Thresholds", "Defaults", "Mappings",
    # retro-compat column names & groups
    "ASSET_ID", "ASSET_TYPE", "LOCATION", "VALUATION_K", "PRICE_PER_SQM",
    "LAST_VERIFIED_TS", "REGION", "URBAN_TYPE", "ZONE", "SIZE_M2", "ROOMS",
    "BATHROOMS", "YEAR_BUILT", "AGE_YEARS", "FLOOR", "BUILDING_FLOORS",
    "IS_TOP_FLOOR", "IS_GROUND_FLOOR", "TOP_FLOOR", "GROUND_FLOOR",
    "HAS_ELEVATOR", "HAS_GARDEN", "HAS_BALCONY", "GARAGE", "OWNER_OCCUPIED",
    "PUBLIC_TRANSPORT_NEARBY", "DISTANCE_TO_CENTER_KM", "PARKING_SPOT", "CELLAR",
    "ATTIC", "CONCIERGE", "ENERGY_CLASS", "HUMIDITY_LEVEL", "TEMPERATURE_AVG",
    "NOISE_LEVEL", "AIR_QUALITY_INDEX", "CONDITION_SCORE", "RISK_SCORE",
    "LUXURY_SCORE", "ENV_SCORE", "ORIENTATION", "VIEW", "CONDITION", "HEATING",
    "VIEW_SEA", "VIEW_LANDMARKS", "URBAN", "SEMIURBAN", "RURAL",
    "ZONE_CENTER", "ZONE_SEMI_CENTER", "ZONE_PERIPHERY", "DEFAULT_ZONE_THRESHOLDS",
    "LISTING_QUARTER", "LISTING_MONTH", "DECADE_BUILT", "BUILDING_AGE_YEARS",
    "ROOMS_PER_SQM", "BATHROOMS_PER_ROOM", "AVG_ROOM_SIZE", "LOCATION_PREMIUM",
    "BASIC_AMENITY_COUNT", "PRICE_PER_SQM_VS_REGION_AVG", "PREDICTION_TS",
    "LAG_HOURS", "MISSING_TIMESTAMP", "PRICE_PER_SQM_CAPPED",
    "PRICE_PER_SQM_CAPPED_VIOLATED", "LISTING_MONTH_SIN", "LISTING_MONTH_COS",
    "DAYS_SINCE_VERIFICATION", "HOURS_SINCE_VERIFICATION", "IS_STALE_30D",
    "IS_STALE_60D", "IS_STALE_90D", "ANOMALY_SCORE", "ANOMALY_SCORE_RAW",
    "ANOMALY_FLAG", "ANOMALY_LABEL", "SEVERITY_SCORE", "CONFIDENCE_SCORE",
    "VALUE_SEGMENT", "LUXURY_CATEGORY", "AGE_CATEGORY", "TIMESTAMP",
    "DERIVED_FEATURES",
    "ENERGY_CLASS_A", "ENERGY_CLASS_B", "ENERGY_CLASS_C", "ENERGY_CLASS_D",
    "ENERGY_CLASS_E", "ENERGY_CLASS_F", "ENERGY_CLASS_G", "ENERGY_CLASSES",
    # interaction feature aliases
    "GARAGE_VS_CENTRAL", "ATTIC_VS_FLOORS",
    # notebook-oriented defaults
    "AnalysisDefaults", "MLDefaults",
]