"""Schema unificato e validazione per il progetto RWA Property."""
from __future__ import annotations

from typing import Dict, List, Mapping, Iterable, Tuple
import logging
import pandas as pd                     # type: ignore
from pydantic import BaseModel, Field   # type: ignore

from notebooks.shared.common.constants import (
    # Enums/namespaces utili
    EnergyClass, Groups, Mappings, Cols,
    # Colonne (retrocompat)
    ASSET_ID, ASSET_TYPE, LOCATION, REGION, URBAN_TYPE, ZONE,
    SIZE_M2, ROOMS, BATHROOMS, YEAR_BUILT, AGE_YEARS, FLOOR, BUILDING_FLOORS,
    IS_TOP_FLOOR, IS_GROUND_FLOOR,
    HAS_ELEVATOR, HAS_GARDEN, HAS_BALCONY, GARAGE, OWNER_OCCUPIED,
    PUBLIC_TRANSPORT_NEARBY, DISTANCE_TO_CENTER_KM, PARKING_SPOT, CELLAR, ATTIC, CONCIERGE,
    ENERGY_CLASS, HUMIDITY_LEVEL, TEMPERATURE_AVG, NOISE_LEVEL, AIR_QUALITY_INDEX,
    CONDITION_SCORE, RISK_SCORE, LUXURY_SCORE, ENV_SCORE,
    ORIENTATION, VIEW, CONDITION, HEATING,
    VALUATION_K, PRICE_PER_SQM, LAST_VERIFIED_TS, LISTING_MONTH,
    PRICE_PER_SQM_CAPPED, PRICE_PER_SQM_CAPPED_VIOLATED, LISTING_MONTH_SIN, LISTING_MONTH_COS,
    DAYS_SINCE_VERIFICATION, HOURS_SINCE_VERIFICATION, IS_STALE_30D, IS_STALE_60D, IS_STALE_90D,
    ANOMALY_SCORE, ANOMALY_FLAG, ANOMALY_LABEL, ANOMALY_SCORE_RAW,
    SEVERITY_SCORE, CONFIDENCE_SCORE, VALUE_SEGMENT, LUXURY_CATEGORY,
    DERIVED_FEATURES,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CORE_FIELDS", "PROPERTY_STRUCTURAL_FIELDS", "PROPERTY_AMENITY_FIELDS",
    "PROPERTY_QUALITY_FIELDS", "PROPERTY_SCORE_FIELDS", "PROPERTY_ADDITIONAL_FIELDS",
    "PROPERTY_DERIVED_FIELDS", "CATEGORICAL", "BOOLEAN", "NUMERIC", "DATETIME",
    "SUGGESTED_DTYPES", "COLUMN_ALIASES", "AssetSchema", "SCHEMA",
    "get_required_fields", "get_all_fields", "apply_aliases", "list_missing",
    "list_unknown", "coerce_dtypes", "enforce_domains", "enforce_categoricals",
    "normalize_column_order", "validate_df", "validate_and_coerce",
]

# ---------------------------------------------------------------------------
# Gruppi logici (contratto)
# ---------------------------------------------------------------------------

CORE_FIELDS: List[str] = [
    ASSET_ID, ASSET_TYPE, LOCATION, VALUATION_K, PRICE_PER_SQM, LAST_VERIFIED_TS, LISTING_MONTH,
]

PROPERTY_STRUCTURAL_FIELDS: List[str] = [
    REGION, URBAN_TYPE, ZONE, SIZE_M2, ROOMS, BATHROOMS, YEAR_BUILT, AGE_YEARS,
    FLOOR, BUILDING_FLOORS, IS_TOP_FLOOR, IS_GROUND_FLOOR,
]

PROPERTY_AMENITY_FIELDS: List[str] = [
    HAS_ELEVATOR, HAS_GARDEN, HAS_BALCONY, GARAGE, OWNER_OCCUPIED,
    PUBLIC_TRANSPORT_NEARBY, DISTANCE_TO_CENTER_KM, PARKING_SPOT, CELLAR, ATTIC, CONCIERGE,
]

PROPERTY_QUALITY_FIELDS: List[str] = [
    ENERGY_CLASS, HUMIDITY_LEVEL, TEMPERATURE_AVG, NOISE_LEVEL, AIR_QUALITY_INDEX,
]

PROPERTY_SCORE_FIELDS: List[str] = [
    CONDITION_SCORE, RISK_SCORE, LUXURY_SCORE, ENV_SCORE,
]

PROPERTY_ADDITIONAL_FIELDS: List[str] = [
    ORIENTATION, VIEW, CONDITION, HEATING,
]

PROPERTY_DERIVED_FIELDS: List[str] = [
    PRICE_PER_SQM_CAPPED, PRICE_PER_SQM_CAPPED_VIOLATED,
    LISTING_MONTH_SIN, LISTING_MONTH_COS,
    DAYS_SINCE_VERIFICATION, HOURS_SINCE_VERIFICATION,
    IS_STALE_30D, IS_STALE_60D, IS_STALE_90D,
    ANOMALY_SCORE, ANOMALY_SCORE_RAW, ANOMALY_FLAG, ANOMALY_LABEL,
    SEVERITY_SCORE, CONFIDENCE_SCORE,
    VALUE_SEGMENT, LUXURY_CATEGORY,
]

# ---------------------------------------------------------------------------
# Tipi attesi per macro gruppi (aiutano coerce e validazione)
# ---------------------------------------------------------------------------

CATEGORICAL: List[str] = [
    LOCATION, REGION, URBAN_TYPE, ZONE, ENERGY_CLASS, ORIENTATION, VIEW, CONDITION, HEATING,
    OWNER_OCCUPIED, PUBLIC_TRANSPORT_NEARBY, GARAGE, PARKING_SPOT, CELLAR, ATTIC, CONCIERGE,
    LUXURY_CATEGORY, VALUE_SEGMENT, ANOMALY_LABEL,
]

BOOLEAN: List[str] = [
    IS_TOP_FLOOR, IS_GROUND_FLOOR, HAS_ELEVATOR, HAS_GARDEN, HAS_BALCONY,
    IS_STALE_30D, IS_STALE_60D, IS_STALE_90D, ANOMALY_FLAG,
]

NUMERIC: List[str] = [
    SIZE_M2, ROOMS, BATHROOMS, YEAR_BUILT, AGE_YEARS, FLOOR, BUILDING_FLOORS,
    DISTANCE_TO_CENTER_KM, HUMIDITY_LEVEL, TEMPERATURE_AVG, NOISE_LEVEL, AIR_QUALITY_INDEX,
    CONDITION_SCORE, RISK_SCORE, LUXURY_SCORE, ENV_SCORE,
    VALUATION_K, PRICE_PER_SQM,
    PRICE_PER_SQM_CAPPED, PRICE_PER_SQM_CAPPED_VIOLATED,
    LISTING_MONTH_SIN, LISTING_MONTH_COS,
    DAYS_SINCE_VERIFICATION, HOURS_SINCE_VERIFICATION,
    ANOMALY_SCORE, ANOMALY_SCORE_RAW, SEVERITY_SCORE, CONFIDENCE_SCORE,
]

DATETIME: List[str] = [
    LAST_VERIFIED_TS,
]

# ---------------------------------------------------------------------------
# Dtypes suggeriti (best-effort): pandas dtype strings
# ---------------------------------------------------------------------------

SUGGESTED_DTYPES: Mapping[str, str] = {
    ASSET_ID: "object",
    SIZE_M2: "Int32",           # nullable int
    ROOMS: "Int16",
    BATHROOMS: "Int16",
    YEAR_BUILT: "Int16",
    AGE_YEARS: "Int16",
    FLOOR: "Int16",
    BUILDING_FLOORS: "Int16",
    VALUATION_K: "float32",
    PRICE_PER_SQM: "float32",
    CONDITION_SCORE: "float32",
    RISK_SCORE: "float32",
    LUXURY_SCORE: "float32",
    ENV_SCORE: "float32",
    LAST_VERIFIED_TS: "datetime64[ns, UTC]",
}

# ---------------------------------------------------------------------------
# Alias/migrazioni colonne (per retrocompatibilità tra versioni dataset)
# ---------------------------------------------------------------------------

COLUMN_ALIASES: Mapping[str, str] = {
    # "energy_rating": ENERGY_CLASS,
    # "price_sq_m": PRICE_PER_SQM,
}

# ---------------------------------------------------------------------------
# Schema Pydantic (contratto formale)
# ---------------------------------------------------------------------------

class AssetSchema(BaseModel):
    required: List[str] = Field(default_factory=list)
    optional: List[str] = Field(default_factory=list)
    dtypes: Dict[str, str] = Field(default_factory=dict)

    def all_fields(self) -> List[str]:
        # preserva ordine e unicità
        return list(dict.fromkeys([*self.required, *self.optional]))


SCHEMA: Mapping[str, AssetSchema] = {
    "property": AssetSchema(
        required=CORE_FIELDS
                 + PROPERTY_STRUCTURAL_FIELDS
                 + PROPERTY_AMENITY_FIELDS
                 + PROPERTY_QUALITY_FIELDS
                 + PROPERTY_SCORE_FIELDS
                 + PROPERTY_ADDITIONAL_FIELDS,
        optional=list(DERIVED_FEATURES) + PROPERTY_DERIVED_FIELDS,
        dtypes=dict(SUGGESTED_DTYPES),
    )
}

# ---------------------------------------------------------------------------
# API utility
# ---------------------------------------------------------------------------

def get_required_fields(asset_type: str = "property") -> List[str]:
    return SCHEMA[asset_type].required


def get_all_fields(asset_type: str = "property") -> List[str]:
    return SCHEMA[asset_type].all_fields()


def apply_aliases(df: pd.DataFrame, aliases: Mapping[str, str] = COLUMN_ALIASES) -> pd.DataFrame:
    """Rinomina colonne legacy ai nomi canonici se presenti (no-op se assenti)."""
    to_rename = {old: new for old, new in aliases.items() if old in df.columns and new not in df.columns}
    if to_rename:
        df = df.rename(columns=to_rename)
        logger.info("Rinominati alias colonne: %s", to_rename)
    return df


def list_missing(df: pd.DataFrame, required: Iterable[str]) -> List[str]:
    return [c for c in required if c not in df.columns]


def list_unknown(df: pd.DataFrame, known: Iterable[str]) -> List[str]:
    known_set = set(known)
    return [c for c in df.columns if c not in known_set]


def _nullable_int_for(target: str) -> str:
    # Se target inizia con "Int" assumiamo già nullable; se "int" → mappiamo a Int32
    if target.startswith("Int"):
        return target
    if target.startswith("int"):
        width = "".join(ch for ch in target if ch.isdigit()) or "32"
        return f"Int{width}"
    return target


def coerce_dtypes(df: pd.DataFrame, dtypes: Mapping[str, str]) -> pd.DataFrame:
    """Best-effort: forza i dtypes suggeriti quando possibile (senza fallire)."""
    out = df.copy()
    for col, dt in dtypes.items():
        if col not in out.columns:
            continue
        try:
            if dt.lower().startswith("int"):
                # usa nullable Int
                target = _nullable_int_for(dt)
                out[col] = pd.to_numeric(out[col], errors="coerce").astype(target)
            elif dt.lower().startswith("float"):
                out[col] = pd.to_numeric(out[col], errors="coerce").astype(dt)
            elif dt.lower().startswith("datetime64"):
                out[col] = pd.to_datetime(out[col], errors="coerce", utc=True)
            else:
                out[col] = out[col].astype(dt)
        except Exception as e:
            logger.warning("Impossibile convertire dtype per %s→%s: %s", col, dt, e)
    return out


def enforce_domains(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """
    Controlla domini canonici (ad es. EnergyClass) e ritorna conteggi di violazioni.
    Non modifica il df: solo report (il fix è responsabilità del chiamante).
    """
    violations: Dict[str, Dict[str, int]] = {}

    if ENERGY_CLASS in df.columns:
        valid = {e.value for e in EnergyClass}
        bad_mask = ~df[ENERGY_CLASS].isin(valid) & df[ENERGY_CLASS].notna()
        n_bad = int(bad_mask.sum())
        if n_bad:
            violations[ENERGY_CLASS] = {"invalid_values": n_bad}

    if ZONE in df.columns:
        valid_zones = {"center", "semi_center", "periphery"}
        n_bad = int((~df[ZONE].isin(valid_zones) & df[ZONE].notna()).sum())
        if n_bad:
            violations[ZONE] = {"invalid_values": n_bad}

    return violations


def enforce_categoricals(df: pd.DataFrame, categoricals: Iterable[str] = CATEGORICAL) -> pd.DataFrame:
    """Casta le colonne categoriali presenti a dtype 'category' (best-effort)."""
    out = df.copy()
    for col in categoricals:
        if col in out.columns:
            try:
                out[col] = out[col].astype("category")
            except Exception as e:
                logger.debug("Skip category cast per %s: %s", col, e)
    return out


def normalize_column_order(df: pd.DataFrame, asset_type: str = "property") -> pd.DataFrame:
    """Riordina le colonne mettendo i required per primi, poi le altre in ordine stabile."""
    schema = SCHEMA[asset_type]
    req = schema.required
    extras = [c for c in df.columns if c not in req]
    ordered = req + extras
    return df.loc[:, [c for c in ordered if c in df.columns]]


def validate_df(
    df: pd.DataFrame,
    asset_type: str = "property",
    *,
    coerce: bool = True,
    check_domains: bool = True,
) -> Dict[str, object]:
    """
    Valida un DataFrame rispetto allo schema:
      - colonne richieste presenti
      - colonne sconosciute
      - dtypes coerenti (best-effort coerce, opzionale)
      - domini canonici (opzionale)
    Ritorna un report di validazione (non muta df).
    """
    schema = SCHEMA[asset_type]
    dfx = apply_aliases(df)

    required = schema.required
    all_known = schema.all_fields()

    missing = list_missing(dfx, required)
    unknown = list_unknown(dfx, all_known)

    if coerce and schema.dtypes:
        dfx = coerce_dtypes(dfx, schema.dtypes)
        dfx = enforce_categoricals(dfx)

    domain_violations = enforce_domains(dfx) if check_domains else {}

    ok = (len(missing) == 0) and (not domain_violations)

    report: Dict[str, object] = {
        "ok": ok,
        "missing": missing,
        "unknown": unknown,
        "domain_violations": domain_violations,
        "suggested_dtypes_applied": list(schema.dtypes.keys()) if coerce else [],
    }
    return report


def validate_and_coerce(
    df: pd.DataFrame,
    asset_type: str = "property",
    *,
    check_domains: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Variante comoda: applica coerce/ordering/categoricals e ritorna (df_norm, report).
    """
    schema = SCHEMA[asset_type]
    dfx = apply_aliases(df)
    dfx = coerce_dtypes(dfx, schema.dtypes)
    dfx = enforce_categoricals(dfx)
    dfx = normalize_column_order(dfx, asset_type=asset_type)

    report = validate_df(df, asset_type=asset_type, coerce=True, check_domains=check_domains)
    return dfx, report