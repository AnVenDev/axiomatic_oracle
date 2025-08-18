"""
FastAPI service exposing AI Oracle inference endpoints
(multi-RWA ready: initial asset_type 'property').

Run locally:
    uvicorn scripts.inference_api:app --reload --port 8000
"""

from __future__ import annotations

import os
import json
import time
import uuid
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Tuple

import difflib
from unicodedata import normalize as _u_norm

from notebooks.shared.common.config import ASSET_CONFIG
from notebooks.shared.common.constants import DEFAULT_REGION_BY_CITY, DEFAULT_URBAN_TYPE_BY_CITY
import numpy as np                                      # type: ignore
import pandas as pd                                     # type: ignore
from fastapi import Body, FastAPI, HTTPException, Query, Path as FPath  # type: ignore
from fastapi.middleware.cors import CORSMiddleware                      # type: ignore
from jsonschema import ValidationError, validate as jsonschema_validate # type: ignore
from pydantic import BaseModel, Field, ConfigDict                        # type: ignore

# ---- shared imports (logger, utils, sanity, pricing) ----
from notebooks.shared.common.config import configure_logger
from notebooks.shared.common.utils import NumpyJSONEncoder, get_utc_now, canonical_location
from notebooks.shared.common.sanity_checks import validate_property
from notebooks.shared.common.constants import ASSET_ID, LOCATION
from notebooks.shared.common.pricing import explain_price
from notebooks.shared.common.sanity_checks import price_benchmark

# ---- model registry (caricamento modelli fitted + meta/manifest) ----
from scripts.model_registry import (
    ModelNotFoundError,
    RegistryLookupError,
    cache_stats,
    discover_models_for_asset,
    get_model_metadata,
    get_model_paths,
    get_pipeline,
    health_check_model,
    list_tasks,
    refresh_cache,
)

from scripts.blockchain_publisher import publish_ai_prediction

# =============================================================================
# App & Config
# =============================================================================
API_VERSION = "0.6.1"

app = FastAPI(
    title="AI Oracle Inference API",
    version=API_VERSION,
    description="Inference service for multi-RWA asset valuation (initial: property).",
)

# CORS: restringi in prod con variabile d'ambiente
_allowed = os.getenv("ALLOWED_ORIGINS", "*")
allow_origins = [o.strip() for o in _allowed.split(",")] if _allowed != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=(allow_origins != ["*"]),
    allow_methods=["*"],
    allow_headers=["*"],
)

# logger
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_JSON = os.getenv("LOG_JSON", "true").lower() in {"1", "true", "yes", "y"}
logger = configure_logger(level=LOG_LEVEL, name="api", json_format=LOG_JSON)

# Flags runtime
STRICT_RAW_FEATURES = os.getenv("STRICT_RAW_FEATURES", "1").lower() in {"1", "true", "yes", "y"}
ALLOW_BASELINE_FALLBACK = os.getenv("ALLOW_BASELINE_FALLBACK", "0").lower() in {"1", "true", "yes", "y"}
INFERENCE_DEBUG = os.getenv("INFERENCE_DEBUG", "0").lower() in {"1", "true", "yes", "y"}
REDACT_API_LOGS = os.getenv("REDACT_API_LOGS", "1").lower() in {"1", "true", "yes", "y"}

# Paths base → notebooks/outputs
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", "notebooks/outputs"))
LOGS_DIR = Path(os.getenv("AI_ORACLE_LOG_DIR", OUTPUTS_DIR / "logs"))
LOGS_DIR.mkdir(parents=True, exist_ok=True)
API_LOG_PATH = LOGS_DIR / "api_inference_log.jsonl"

# Schemas (configurabili)
SCHEMAS_DIR = Path(os.getenv("SCHEMAS_DIR", "schemas"))
_schema_path_v2 = SCHEMAS_DIR / "output_schema_v2.json"
_schema_path_v1 = SCHEMAS_DIR / "output_schema_v1.json"
if _schema_path_v2.exists():
    OUTPUT_SCHEMA = json.loads(_schema_path_v2.read_text(encoding="utf-8"))
    SCHEMA_VERSION = "v2"
elif _schema_path_v1.exists():
    OUTPUT_SCHEMA = json.loads(_schema_path_v1.read_text(encoding="utf-8"))
    SCHEMA_VERSION = OUTPUT_SCHEMA.get("$id", "v1")
else:
    OUTPUT_SCHEMA = {}
    SCHEMA_VERSION = "v0-fallback"
    logger.warning("No output schema found (v2/v1). Schema validation disabled.")

# =============================================================================
# Request Models (flessibili: extra='allow')
# =============================================================================
class PropertyPredictRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # accetta campi extra; li filtriamo noi
    location: Optional[str] = None
    size_m2: Optional[float] = Field(None, gt=0)
    rooms: Optional[int] = Field(None, ge=0)
    bathrooms: Optional[int] = Field(None, ge=0)
    year_built: Optional[int] = Field(None, ge=1800, le=datetime.utcnow().year)
    floor: Optional[int] = Field(None, ge=0)
    building_floors: Optional[int] = Field(None, ge=1)
    has_elevator: Optional[int] = Field(None, ge=0, le=1)
    has_garden: Optional[int] = Field(None, ge=0, le=1)
    has_balcony: Optional[int] = Field(None, ge=0, le=1)
    has_garage: Optional[int] = Field(None, ge=0, le=1)
    energy_class: Optional[str] = None
    humidity_level: Optional[float] = Field(None, ge=0, le=100)
    temperature_avg: Optional[float] = None
    noise_level: Optional[int] = Field(None, ge=0, le=150)
    air_quality_index: Optional[int] = Field(None, ge=0, le=500)
    age_years: Optional[int] = Field(None, ge=0)

REQUEST_MODELS: Dict[str, Any] = {"property": PropertyPredictRequest}

# =============================================================================
# Helpers — logging, mappature, validazione record
# =============================================================================
LOCATION_MAP_PATH = os.getenv("LOCATION_MAP_JSON", "").strip()

_DEFAULT_LOCATION_MAP = {
    "milan":    {"region": "Lombardia", "zone": "milano"},
    "milano":   {"region": "Lombardia", "zone": "milano"},
    "rome":     {"region": "Lazio",     "zone": "roma"},
    "roma":     {"region": "Lazio",     "zone": "roma"},
    "florence": {"region": "Toscana",   "zone": "firenze"},
    "firenze":  {"region": "Toscana",   "zone": "firenze"},
}

def _load_location_map() -> Dict[str, Dict[str, str]]:
    try:
        if LOCATION_MAP_PATH:
            p = Path(LOCATION_MAP_PATH)
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                return {str(k).lower(): {**v} for k, v in data.items() if isinstance(v, dict)}
    except Exception as e:
        logger.warning("LOCATION_MAP_JSON load failed: %s", e)
    return _DEFAULT_LOCATION_MAP

_KNOWN_CATS: Dict[str, set] = {"region": set(), "zone": set(), "city": set()}

# Italian regions -> macro (north/center/south)
_REGION_TO_MACRO = {
    # NORTH
    "lombardia":"north","piemonte":"north","liguria":"north","veneto":"north",
    "friuliveneziagiulia":"north","trentinoaltoadige":"north","emiliaromagna":"north",
    "valledaosta":"north","valledaoste":"north",
    # CENTER
    "lazio":"center","toscana":"center","umbria":"center","marche":"center","abruzzo":"center","molise":"center",
    # SOUTH & ISLANDS
    "campania":"south","puglia":"south","calabria":"south","basilicata":"south",
    "sicilia":"south","sardegna":"south",
}

# Zone synonyms -> canonical categories used in training
_ZONE_SYNONYMS = {
    "centro":"center", "centrostorico":"center", "historiccenter":"center",
    "semicentro":"semi_center", "semicenter":"semi_center", "semi-centro":"semi_center",
    "periferia":"periphery", "periferico":"periphery", "suburb":"periphery",
}

def _slug(s: str) -> str:
    s = _u_norm("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return "".join(ch for ch in s.lower() if ch.isalnum())

def _unwrap_for_introspection(obj: Any) -> Any:
    """
    Raggiunge ColumnTransformer/OneHotEncoder e wrapper comuni:
    GridSearchCV/RandomizedSearchCV/TransformedTargetRegressor/Pipeline.
    """
    tried = set()
    def _walk(x):
        if id(x) in tried or x is None:
            return
        tried.add(id(x))
        yield x
        for attr in ("best_estimator_", "estimator", "regressor_", "regressor",
                     "pipeline", "model", "final_estimator"):
            if hasattr(x, attr):
                try:
                    yield from _walk(getattr(x, attr))
                except Exception:
                    pass
        try:
            from sklearn.pipeline import Pipeline  # type: ignore
            if isinstance(x, Pipeline):
                for _, step in x.steps:
                    yield from _walk(step)
        except Exception:
            pass
    for node in _walk(obj):
        yield node

def _refresh_known_categories_from_pipeline(pipeline) -> None:
    try:
        from sklearn.compose import ColumnTransformer  # type: ignore
        for node in _unwrap_for_introspection(pipeline):
            if isinstance(node, ColumnTransformer):
                for _tname, trans, cols_in in node.transformers:
                    cats = getattr(trans, "categories_", None)
                    if cats is None:
                        continue
                    cols = list(cols_in) if isinstance(cols_in, (list, tuple)) else [cols_in]
                    for col, cat_vals in zip(cols, cats):
                        col = str(col)
                        if col in _KNOWN_CATS:
                            _KNOWN_CATS[col].update(map(str, cat_vals))
        if any(_KNOWN_CATS[k] for k in _KNOWN_CATS):
            logger.info(
                "known cats loaded: city=%d region=%d zone=%d",
                len(_KNOWN_CATS["city"]), len(_KNOWN_CATS["region"]), len(_KNOWN_CATS["zone"])
            )
    except Exception as e:
        logger.debug("Could not introspect categories: %s", e)

def _to_macro_region(value: Optional[str], city: Optional[str] = None) -> Optional[str]:
    if not value and not city:
        return value
    if value:
        v = _slug(str(value))
        if v in {"north","center","south"}:
            return value.strip().lower()
        if v in _REGION_TO_MACRO:
            return _REGION_TO_MACRO[v]
    # fallback: deduci dalla città
    try:
        prop_cfg = ASSET_CONFIG["property"]
        by_city = (prop_cfg.get("region_by_city") or DEFAULT_REGION_BY_CITY) or {}
        if city:
            key = str(city).strip().title()
            macro = by_city.get(key)
            if macro:
                return str(macro).lower()
    except Exception:
        pass
    return value

def _closest_known(name: str, value: str) -> Optional[str]:
    known = _KNOWN_CATS.get(name) or set()
    if not known:
        return value
    v_slug = _slug(value)
    cat_slugs = {cat: _slug(cat) for cat in known}
    for cat, s in cat_slugs.items():
        if s == v_slug:
            return cat
    match = difflib.get_close_matches(v_slug, list(cat_slugs.values()), n=1, cutoff=0.8)
    if match:
        target_slug = match[0]
        for cat, s in cat_slugs.items():
            if s == target_slug:
                return cat
    return value

def _normalize_to_known(name: str, value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    v = str(value).strip()
    if name == "region":
        v = _to_macro_region(v, None) or v
    if name == "zone":
        z = _slug(v)
        v = _ZONE_SYNONYMS.get(z, v)
    return _closest_known(name, v)

def log_jsonl(record: dict, path: Path = API_LOG_PATH) -> None:
    payload = {**record, "_logged_at": get_utc_now()}
    line = json.dumps(payload, cls=NumpyJSONEncoder, ensure_ascii=False)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    with os.fdopen(fd, "a", encoding="utf-8") as f:
        f.write(line + "\n")

_SENSITIVE_KEYS = {
    "address", "note", "notes", "email", "phone", "lat", "lon", "lng", "latitude", "longitude",
    "coordinates", "gps", "contact"
}
def _redact(obj: Any) -> Any:
    """Redige chiavi potenzialmente sensibili nei log JSONL."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k.lower() in _SENSITIVE_KEYS:
                out[k] = "***"
            else:
                out[k] = _redact(v)
        return out
    if isinstance(obj, list):
        return [_redact(x) for x in obj]
    return obj

def _predict_one(pipeline_obj, df: pd.DataFrame) -> float:
    """
    Predizione scalare robusta.
    """
    # 1) API scikit (consistente con tutto il pipeline, incl. inverse_transform target)
    if hasattr(pipeline_obj, "predict"):
        y = pipeline_obj.predict(df)
        if isinstance(y, (list, tuple, np.ndarray)):
            return float(np.ravel(y)[0])
        return float(y)

    # 2) Callable
    if callable(pipeline_obj):
        return float(pipeline_obj(df))

    # 3) Baseline (solo se esplicitamente abilitata)
    if ALLOW_BASELINE_FALLBACK:
        s = df.get("size_m2")
        if s is not None and len(s) > 0 and pd.notna(s.iloc[0]):
            return float(max(1.0, 0.8 * float(s.iloc[0])))

    raise RuntimeError("Model object has no predict/callable interface")

# --- expected features resolver (supporta monkeypatch dei test) ---
_EXPECTED_FEATURES: Optional[List[str]] = None  # i test patchano questa variabile

def collect_expected_features(meta: Optional[dict], pipeline: Optional[object], payload_keys: Optional[List[str]] = None) -> List[str]:
    """
    Ordine di priorità:
      1) variabile module-level _EXPECTED_FEATURES (monkeypatch nei test)
      2) meta["expected_features"] | meta["features"] | meta["input_features"]
      3) meta["preprocessing"]["feature_names_out"] | ["feature_names"]
      4) pipeline.ColumnTransformer(raw input) se presente
      5) fallback soft: payload_keys (se fornite)
    """
    # 1) test override
    if isinstance(_EXPECTED_FEATURES, list) and _EXPECTED_FEATURES:
        logger.info("expected_features resolved: source=%s count=%d sample=%s",
                    "test_override(_EXPECTED_FEATURES)", len(_EXPECTED_FEATURES), _EXPECTED_FEATURES[:10])
        return list(_EXPECTED_FEATURES)

    def _post_enc(src: str) -> bool:
        return "post-encoding" in src

    cands: List[Tuple[str, List[str]]] = []
    # 2) meta keys flat
    if isinstance(meta, dict):
        for k in ("expected_features", "features", "input_features"):
            v = meta.get(k)
            if isinstance(v, list) and v:
                cands.append((f"meta.{k}", v))
        # 3) meta nested
        prep = meta.get("preprocessing")
        if isinstance(prep, dict):
            v = prep.get("feature_names_out") or prep.get("feature_names")
            if isinstance(v, list) and v:
                cands.append(("meta.preprocessing.feature_names_out/feature_names (post-encoding!)", v))

    # 4) pipeline raw input dal ColumnTransformer
    if pipeline is not None:
        try:
            from sklearn.pipeline import Pipeline  # type: ignore
            from sklearn.compose import ColumnTransformer  # type: ignore
            if isinstance(pipeline, Pipeline):
                for _, step in pipeline.steps:
                    if isinstance(step, ColumnTransformer):
                        cols: List[str] = []
                        for _tname, _trans, cols_in in step.transformers:
                            if cols_in in (None, "drop"):
                                continue
                            if isinstance(cols_in, (list, tuple, np.ndarray)):
                                cols.extend([str(c) for c in cols_in])
                        if cols:
                            _seen = set()
                            raw_cols = [c for c in cols if not (c in _seen or _seen.add(c))]
                            logger.info("expected_features resolved: source=%s count=%d sample=%s",
                                        "pipeline.ColumnTransformer(raw input)", len(raw_cols), raw_cols[:10])
                            return list(raw_cols)
        except Exception:
            pass
        try:
            fn = getattr(pipeline, "get_feature_names_out", None)
            if callable(fn):
                v = list(fn())
                if v:
                    cands.append(("pipeline.get_feature_names_out (post-encoding!)", v))
        except Exception:
            pass

    for src, v in cands:
        if not v:
            continue
        if STRICT_RAW_FEATURES and _post_enc(src):
            logger.warning("Skipping %s due to STRICT_RAW_FEATURES=1", src)
            continue
        logger.info("expected_features resolved: source=%s count=%d sample=%s", src, len(v), v[:10])
        return list(v)

    # 5) ultima spiaggia: chiavi del payload
    if isinstance(payload_keys, list) and payload_keys:
        logger.info("expected_features resolved: source=%s count=%d sample=%s",
                    "payload_keys (fallback)", len(payload_keys), payload_keys[:10])
        return list(payload_keys)

    return []

_SAFE_DERIVED = {
    "age_years","luxury_score","env_score",
    "location","city",
    "is_top_floor","listing_month", ASSET_ID
}

_KEY_ALIASES = {
    "sqm":"size_m2","size":"size_m2","m2":"size_m2",
    "year":"year_built","built_year":"year_built",
    "balcony":"has_balcony","garden":"has_garden","garage":"has_garage",
    "air_quality":"air_quality_index","noise":"noise_level",
    "valuation":"valuation_k","price_k":"valuation_k",
    "n_rooms":"rooms","room_count":"rooms",
    "n_bathrooms":"bathrooms","bathroom_count":"bathrooms",
    "elevator":"has_elevator",
    "city_name":"city"
}

def _canonicalize_keys(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {_KEY_ALIASES.get(k, k): v for k, v in rec.items()}

def _autofill_safe(rec: Dict[str, Any]) -> Dict[str, Any]:
    r = dict(rec)

    # ---- 0) Canonicalizza chiavi (alias) ----
    r = _canonicalize_keys(r)

    # ---- 1) age_years ----
    if "age_years" not in r and r.get("year_built") not in (None, ""):
        try:
            r["age_years"] = max(0, datetime.utcnow().year - int(r["year_built"]))
        except Exception:
            pass

    # ---- 2) luxury/env score ----
    if "luxury_score" not in r:
        g = 1.0 if bool(r.get("has_garden", 0)) else 0.0
        b = 1.0 if bool(r.get("has_balcony", 0)) else 0.0
        ga = 1.0 if bool(r.get("has_garage", 0)) else 0.0
        r["luxury_score"] = (g + b + ga) / 3.0

    if "env_score" not in r:
        try:
            aq = float(r.get("air_quality_index", 0.0))
            nz = float(r.get("noise_level", 0.0))
            r["env_score"] = float(np.clip((aq / 100.0) * (1.0 - nz / 100.0), 0.0, 1.0))
        except Exception:
            r["env_score"] = None

    # ---- 3) canonical location ----
    if LOCATION in r and r[LOCATION]:
        try:
            r[LOCATION] = canonical_location(r)
        except Exception:
            pass

    # ---- 4) listing_month ----
    if "listing_month" not in r or r.get("listing_month") in (None, "", 0):
        try:
            r["listing_month"] = int(datetime.utcnow().month)
        except Exception:
            r["listing_month"] = None

    # ---- 5) is_top_floor ----
    try:
        if "is_top_floor" not in r and r.get("floor") is not None and r.get("building_floors") is not None:
            r["is_top_floor"] = int(r.get("floor") == r.get("building_floors"))
    except Exception:
        pass

    # ---- 6) CITY from location + synonyms (sempre presente lato modello) ----
    try:
        prop_cfg = ASSET_CONFIG["property"]
        city_syn = {str(k).strip().lower(): str(v).strip().title()
                    for k, v in (prop_cfg.get("city_synonyms") or {}).items()}
        # se manca city, usa location "as city"
        if not r.get("city"):
            loc = str(r.get(LOCATION, "")).strip()
            if loc:
                key = loc.lower()
                r["city"] = city_syn.get(key, loc.title())
        else:
            key = str(r["city"]).strip().lower()
            r["city"] = city_syn.get(key, str(r["city"]).strip().title())
    except Exception:
        # fallback: title case
        if r.get("city"):
            r["city"] = str(r["city"]).strip().title()

    # ---- 7) REGION macro (north/center/south) ----
    try:
        prop_cfg = ASSET_CONFIG["property"]
        by_city = (prop_cfg.get("region_by_city") or DEFAULT_REGION_BY_CITY) or {}
        if r.get("region"):
            # converte eventuali nomi italiani in macro-region
            r["region"] = _to_macro_region(str(r["region"]), r.get("city"))
        else:
            if r.get("city"):
                r["region"] = str(by_city.get(str(r["city"]).strip().title(), "")).lower() or None
        # snap a categorie note (se il modello le espone)
        if r.get("region"):
            r["region"] = _normalize_to_known("region", r["region"])
    except Exception:
        pass

    # ---- 8) URBAN TYPE by city (se manca) ----
    try:
        prop_cfg = ASSET_CONFIG["property"]
        urb_by_city = (prop_cfg.get("urban_type_by_city") or DEFAULT_URBAN_TYPE_BY_CITY) or {}
        if not r.get("urban_type") and r.get("city"):
            r["urban_type"] = urb_by_city.get(str(r["city"]).strip().title())
    except Exception:
        pass

    # ---- 9) ZONE: normalizza sinonimi; se manca usa distanza dal centro ----
    try:
        z = r.get("zone")
        if z:
            r["zone"] = _normalize_to_known("zone", z)
        else:
            # prova con distanza → thresholds da config
            prop_cfg = ASSET_CONFIG["property"]
            th = prop_cfg.get("zone_thresholds_km") or {"center": 1.5, "semi_center": 5.0}
            d = r.get("distance_to_center_km")
            if d is not None:
                d = float(d)
                if d <= float(th.get("center", 1.5)):
                    r["zone"] = "center"
                elif d <= float(th.get("semi_center", 5.0)):
                    r["zone"] = "semi_center"
                else:
                    r["zone"] = "periphery"
        if r.get("zone"):
            r["zone"] = _normalize_to_known("zone", r["zone"])
    except Exception:
        pass

    # ---- 10) Snap anche CITY alle categorie note, se disponibili ----
    try:
        if r.get("city"):
            r["city"] = _closest_known("city", str(r["city"]))
    except Exception:
        pass

    return r

def validate_input_record(record: Dict[str, Any], all_expected: List[str], *, strict: bool = False, drop_extras: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Canonicalizza → deriva campi sicuri → filtra/tollera extras → valida con shared.sanity_checks.validate_property."""
    base = _autofill_safe(_canonicalize_keys(record))
    allowed = set(all_expected) | _SAFE_DERIVED
    extras = [k for k in list(base.keys()) if k not in allowed]
    if drop_extras:
        for k in extras:
            base.pop(k, None)
    elif strict and extras:
        raise ValueError(f"Unexpected extra features: {extras}")
    report = validate_property(base)
    if strict and not report.get("ok", True):
        raise ValueError(f"Property validation failed: {report.get('errors') or report}")
    return base, report

# ---- Predict with confidence (forest variance o manifest RMSE) ----
_Z = {0.80: 1.282, 0.90: 1.645, 0.95: 1.960, 0.98: 2.326, 0.99: 2.576}
def _z_for_conf(conf: float) -> float: return _Z.get(round(conf, 2), 1.960)

def _split_preprocessor_and_model(pipeline_obj: Any) -> Tuple[Any, Any]:
    """
    Se è una Pipeline sklearn: ritorna (preprocessor, final_model_step).
    Altrimenti: (None, pipeline_obj).
    """
    try:
        from sklearn.pipeline import Pipeline  # type: ignore
        if isinstance(pipeline_obj, Pipeline):
            pre = pipeline_obj[:-1]
            final_model = pipeline_obj.steps[-1][1]
            return pre, final_model
    except Exception:
        pass
    return None, pipeline_obj

def _unwrap_final_estimator(model: Any) -> Tuple[Any, Optional[Any]]:
    """
    Ritorna (estimatore_finale, ttr) dove:
      - estimatore_finale può essere la RandomForestRegressor (con .estimators_)
      - ttr è il TransformedTargetRegressor se presente (altrimenti None)
    """
    ttr = None
    candidate = model
    for node in _unwrap_for_introspection(model):
        # cattura TTR se presente
        if node.__class__.__name__ == "TransformedTargetRegressor":
            ttr = node
        if hasattr(node, "estimators_"):
            candidate = node
    return candidate, ttr

def _inverse_target_if_needed(y: np.ndarray, ttr: Any) -> np.ndarray:
    """
    Applica l'inverse-transform del target se il modello è un TransformedTargetRegressor.
    Supporta sia transformer_.inverse_transform sia inverse_func.
    """
    if ttr is None:
        return y
    try:
        invf = getattr(ttr, "inverse_func", None)
        if callable(invf):
            return np.asarray(invf(np.asarray(y)))
    except Exception:
        pass
    try:
        tr = getattr(ttr, "transformer_", None) or getattr(ttr, "transformer", None)
        if tr is not None and hasattr(tr, "inverse_transform"):
            y2 = np.asarray(y).reshape(-1, 1)
            return np.asarray(tr.inverse_transform(y2)).ravel()
    except Exception:
        pass
    # se non riusciamo ad invertire, restituiamo y così com'è
    return y

def predict_with_confidence(
    pipeline_obj,
    record: Dict[str, Any],
    all_expected: List[str],
    manifest_path: Path,
    confidence: float = 0.95
) -> Dict[str, Any]:
    df = pd.DataFrame([{k: record.get(k, np.nan) for k in all_expected}], columns=all_expected)
    # Diagnostica DF input al modello
    try:
        nonnull_total = int(df.notnull().sum().sum())
        logger.info("DF shape=%s nonnull=%s", df.shape, nonnull_total)
        logger.info("First 10 cols: %s", all_expected[:10])
        if nonnull_total == 0:
            logger.warning("All values are NaN after aligning with expected features. Probabile mismatch raw vs post-encoding.")
    except Exception:
        pass

    # Predizione coerente (pipeline.predict → scala corretta)
    y_hat = float(_predict_one(pipeline_obj, df))

    # Prova a stimare una CI dai forest (se possibile), altrimenti fallback globale
    try:
        pre, model_step = _split_preprocessor_and_model(pipeline_obj)
        X = df
        if pre is not None and hasattr(pre, "transform"):
            try:
                X = pre.transform(df)
                shape = getattr(X, "shape", None)
                nnz = getattr(X, "nnz", None)
                if nnz is not None:
                    logger.info("Preprocessor out: shape=%s nnz=%s", shape, int(nnz))
                else:
                    nonnull = int(np.isfinite(X).sum()) if hasattr(np, "isfinite") else None
                    logger.info("Preprocessor out: shape=%s nonnull=%s", shape, nonnull)
            except Exception:
                X = df

        model_final, ttr = _unwrap_final_estimator(model_step)

        if hasattr(model_final, "estimators_") and isinstance(model_final.estimators_, (list, tuple)) and len(model_final.estimators_) >= 3:
            # per-tree in spazio del regressore interno
            per_tree_raw = np.array([float(np.ravel(est.predict(X))[0]) for est in model_final.estimators_], dtype=float)
            # inverse target se TTR
            per_tree = _inverse_target_if_needed(per_tree_raw, ttr)
            m = float(per_tree.mean())
            s = float(per_tree.std(ddof=1)) if len(per_tree) > 1 else 0.0
            z = _z_for_conf(confidence)
            ci = z * s
            return {
                "prediction": round(m, 2),                # stessa scala di y_hat
                "point_pred": round(y_hat, 2),
                "uncertainty": round(s, 2),
                "confidence": float(confidence),
                "confidence_interval": (round(m - ci, 2), round(m + ci, 2)),
                "ci_margin": round(ci, 2),
                "method": "forest_variance",
                "n_estimators": len(model_final.estimators_),
            }

        # Se non è una foresta o non disponibili estimators_ → fallback
        raise RuntimeError("No per-tree variance available")
    except Exception as e:
        logger.debug("Forest variance unavailable (%s); using global sigma fallback.", e)
        # Fallback: usa RMSE/MAE dal manifest, oppure 10% del valore
        sigma = None
        try:
            if manifest_path and manifest_path.exists():
                mf = json.loads(manifest_path.read_text(encoding="utf-8"))
                metrics = (mf.get("metrics") or {}).get("validation") or (mf.get("metrics") or {}).get("valid") or {}
                rmse = metrics.get("rmse") or metrics.get("RMSE")
                mae = metrics.get("mae") or metrics.get("MAE")
                if isinstance(rmse, (int, float)) and rmse > 0:
                    sigma = float(rmse)
                elif isinstance(mae, (int, float)) and mae > 0:
                    sigma = float(mae) * 1.253314
        except Exception:
            pass
        if sigma is None:
            sigma = max(1.0, abs(y_hat) * 0.10)
        z = _z_for_conf(confidence)
        ci = z * float(sigma)
        return {
            "prediction": round(y_hat, 2),               # coerenza con point_pred
            "point_pred": round(y_hat, 2),
            "uncertainty": round(float(sigma), 2),
            "confidence": float(confidence),
            "confidence_interval": (round(y_hat - ci, 2), round(y_hat + ci, 2)),
            "ci_margin": round(ci, 2),
            "method": "global_sigma",
            "n_estimators": None,
        }

# =============================================================================
# Endpoints
# =============================================================================
@app.get("/health")
def health() -> dict:
    try:
        health_info = health_check_model("property", "value_regressor")
        cache_info = cache_stats()
        return {
            "status": "ok" if health_info.get("status") == "healthy" else "degraded",
            "model_health": health_info,
            "cache_stats": cache_info,
            "asset_types": list(REQUEST_MODELS.keys()),
            "schema_version": SCHEMA_VERSION,
            "api_version": API_VERSION,
            "time": get_utc_now(),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/predict/{asset_type}")
def predict(
    asset_type: str = FPath(...),
    publish: bool = Query(False),
    payload: dict = Body(...),
) -> dict:
    asset_type = asset_type.lower()
    if asset_type not in REQUEST_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported asset_type: {asset_type}")

    # 1) Carica artefatti fitted + meta
    try:
        pipeline = get_pipeline(asset_type, "value_regressor")
        # paths può non contenere 'manifest' in ambienti mock → gestisci None
        try:
            paths = get_model_paths(asset_type, "value_regressor")
        except Exception:
            paths = {}
        meta = get_model_metadata(asset_type, "value_regressor") or {}

        # 2) Expected features robuste (supporto ai test)
        payload_keys = list(payload.keys()) if isinstance(payload, dict) else None
        all_expected: List[str] = collect_expected_features(meta, pipeline, payload_keys=payload_keys)

        # fallback extra: se ancora vuoto, prova a usare il manifest tramite registry.expected_features (se disponibile)
        if not all_expected:
            try:
                from scripts.model_registry import expected_features as reg_expected_features
                cat, num = reg_expected_features(meta, paths.get("manifest"), asset_type=asset_type)
                all_expected = list(cat) + list(num)
            except Exception:
                pass

        if not all_expected:
            raise RuntimeError("Empty expected features")

    except (RegistryLookupError, ModelNotFoundError) as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Artifacts error: {e}")

    # 3) Canonicalizza/valida il payload
    ModelCls = REQUEST_MODELS[asset_type]
    try:
        # diagnostica allineamento PRIMA della validazione/canonicalizzazione
        try:
            payload_keys_dbg = list(payload.keys()) if isinstance(payload, dict) else []
            allowed_dbg = set(all_expected) | _SAFE_DERIVED
            extras_dropped_dbg = [k for k in payload_keys_dbg if k not in allowed_dbg]
            missing_keys_dbg = [k for k in all_expected if k not in payload_keys_dbg]
            logger.info(
                "input alignment: payload_keys=%d expected=%d extras_dropped=%d missing_expected=%d",
                len(payload_keys_dbg), len(all_expected), len(extras_dropped_dbg), len(missing_keys_dbg)
            )
            if extras_dropped_dbg[:5]:
                logger.debug("extras_dropped sample=%s", extras_dropped_dbg[:5])
            if missing_keys_dbg[:5]:
                logger.debug("missing_expected sample=%s", missing_keys_dbg[:5])
        except Exception:
            pass

        _refresh_known_categories_from_pipeline(pipeline)
        logger.debug("known region (sample 8): %s", list(sorted(_KNOWN_CATS.get("region", [])))[:8])
        logger.debug("known zone   (sample 8): %s", list(sorted(_KNOWN_CATS.get("zone", [])))[:8])

        # pydantic per tipo/shape, poi validazione propria
        req_obj = ModelCls(**payload)
        rec_in = req_obj.model_dump(exclude_none=False)  # tieni anche None per autofill
        rec, vreport = validate_input_record(rec_in, all_expected, strict=False, drop_extras=False)
        logger.debug("record keys after validate: %s", sorted(list(rec.keys()))[:30])
        logger.info("geo check: region=%r in_known=%s | zone=%r in_known=%s",
                    rec.get("region"), rec.get("region") in _KNOWN_CATS.get("region", set()),
                    rec.get("zone"),   rec.get("zone")   in _KNOWN_CATS.get("zone", set()))
        if not rec.get(ASSET_ID):
            rec[ASSET_ID] = f"{asset_type}_{uuid.uuid4().hex[:10]}"
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid payload: {e}")

    # 4) Predizione + CI + latency
    t0 = time.perf_counter()
    try:
        manifest_path = Path(paths.get("manifest")) if paths and paths.get("manifest") else Path()
        conf = predict_with_confidence(
            pipeline_obj=pipeline,
            record=rec,
            all_expected=all_expected,
            manifest_path=manifest_path,
            confidence=0.95,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    # 5) Build response (schema v2) — publish sempre presente
    ci_low, ci_high = conf["confidence_interval"]
    response: Dict[str, Any] = {
        "schema_version": "v2",
        "asset_id": rec[ASSET_ID],
        "asset_type": asset_type,
        "timestamp": get_utc_now(),
        "metrics": {
            "valuation_k": float(conf["prediction"]),
            "point_pred_k": float(conf.get("point_pred", conf["prediction"])),
            "uncertainty_k": float(conf["uncertainty"]),
            "confidence": float(conf.get("confidence", 0.95)),
            "confidence_low_k": float(ci_low),
            "confidence_high_k": float(ci_high),
            "ci_margin_k": float(conf["ci_margin"]),
            "latency_ms": float(latency_ms),
            "ci_method": conf.get("method"),
            "n_estimators": conf.get("n_estimators"),
        },
        "flags": {
            "anomaly": (not vreport.get("ok", True)),
            "drift_detected": False,
            "needs_review": (not vreport.get("ok", True)),
        },
        "model_meta": {
            "value_model_version": meta.get("model_version"),
            "value_model_name": meta.get("model_class") or type(getattr(pipeline, "steps", [[None, pipeline]])[-1][1]).__name__,
            "n_features_total": len(all_expected),
        },
        "model_health": {
            "status": "ok",
            "model_path": str(paths.get("pipeline")) if paths and paths.get("pipeline") else "",
            "metadata_valid": True,
            "metrics": meta.get("metrics", {}),
        },
        "validation": {
            "ok": bool(vreport.get("ok", True)),
            "warnings": vreport.get("warnings"),
            "errors": vreport.get("errors"),
        },
        "drift": {"message": None},
        "offchain_refs": {"detail_report_hash": None, "sensor_batch_hash": None},
        "cache_hit": False,
        "schema_validation_error": "",
        "blockchain_txid": "",
        "asa_id": "",
        "publish": {"status": "skipped"},
    }

    # Breakdown & benchmark opzionali
    try:
        br = explain_price(rec)
        response.setdefault("explanations", {})["pricing_breakdown"] = br
    except Exception:
        pass
    try:
        pb = price_benchmark(location=rec.get("location"), valuation_k=float(conf["prediction"]))
        if pb:
            response.setdefault("sanity", {})["price_benchmark"] = pb
            if pb.get("out_of_band", False):
                response["flags"]["price_out_of_band"] = True
                response["flags"]["needs_review"] = True
    except Exception:
        pass

    # 6) Publish (blockchain) opzionale
    if publish:
        try:
            result = publish_ai_prediction(response)
            response["blockchain_txid"] = result.get("blockchain_txid")
            response["asa_id"] = result.get("asa_id")
            response["publish"] = {
                "status": "success",
                "txid": result.get("blockchain_txid"),
                "asa_id": result.get("asa_id"),
            }
        except Exception as e:
            response["publish"] = {"status": "error", "error": str(e)}

    # 7) Schema validation (best-effort)
    if OUTPUT_SCHEMA:
        try:
            jsonschema_validate(instance=response, schema=OUTPUT_SCHEMA)
        except ValidationError as ve:
            response["schema_validation_error"] = str(ve).split("\n", 1)[0][:240]
        except Exception as e:
            response["schema_validation_error"] = f"Schema check failed: {e}"[:240]

    # 8) API log (best-effort)
    try:
        log_jsonl({
            "event": "prediction",
            "asset_type": asset_type,
            "request": _redact(payload) if REDACT_API_LOGS else payload,
            "response": response
        })
    except Exception:
        pass

    return response

# ---- models & cache mgmt ----
@app.get("/models/{asset_type}")
def list_models(asset_type: str) -> dict:
    return {
        "asset_type": asset_type,
        "tasks": list_tasks(asset_type),
        "discovered_models": [p.name for p in discover_models_for_asset(asset_type)],
    }

@app.post("/models/{asset_type}/{task}/refresh")
def refresh_model_cache(asset_type: str, task: str) -> dict:
    refresh_cache(asset_type, task)
    return {"status": "cache_refreshed", "asset_type": asset_type, "task": task}

@app.get("/models/{asset_type}/{task}/health")
def model_health(asset_type: str, task: str) -> dict:
    return health_check_model(asset_type, task)

# ---- logs browsing ----
@app.get("/logs/api")
def get_api_logs() -> List[dict]:
    try:
        if not API_LOG_PATH.exists():
            return []
        with API_LOG_PATH.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Log read error: {e}")

@app.get("/logs/published")
def get_published_assets():
    try:
        p = LOGS_DIR / "published_assets.json"
        if not p.exists():
            return []
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Read error: {e}")

@app.get("/logs/detail_reports")
def get_detail_reports():
    try:
        detail_dir = LOGS_DIR / "detail_reports"
        if not detail_dir.exists():
            return []
        reports = []
        for fp in sorted(detail_dir.glob("*.json")):
            try:
                reports.append(json.loads(fp.read_text(encoding="utf-8")))
            except Exception:
                continue
        return reports
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Read error: {e}")

# ---- local dev runner ----
if __name__ == "__main__":
    import uvicorn  # type: ignore
    uvicorn.run("scripts.inference_api:app", host="127.0.0.1", port=8000, reload=True)