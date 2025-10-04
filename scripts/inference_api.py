# (file) scripts/inference_api.py
# Module: inference_api.py — FastAPI service for AI Oracle inference & PoVal™.

from __future__ import annotations

# =========================
# Standard library imports
# =========================
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import difflib
from unicodedata import normalize as _u_norm

# ===========
# Third-party
# ===========
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from fastapi import (  # type: ignore
    Body,
    Depends,
    FastAPI,
    HTTPException,
    Path as FPath,
    Query,
    Request,
    status,
)
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from fastapi.responses import JSONResponse  # type: ignore
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer  # type: ignore
try:
    from fastapi.responses import FileResponse  # type: ignore
except Exception:  # fallback per versioni vecchie
    from starlette.responses import FileResponse  # type: ignore
from jsonschema import ValidationError, validate as jsonschema_validate  # type: ignore
from pydantic import BaseModel, ConfigDict, Field  # type: ignore

# ========
# Local os
# ========
import sys
from pathlib import Path as _Path

from notebooks.shared.common.config import ASSET_CONFIG, configure_logger
from notebooks.shared.common.constants import (
    ASSET_ID,
    DEFAULT_REGION_BY_CITY,
    DEFAULT_URBAN_TYPE_BY_CITY,
    LOCATION,
)
from notebooks.shared.common.pricing import explain_price
from notebooks.shared.common.sanity_checks import price_benchmark, validate_property
from notebooks.shared.common.utils import NumpyJSONEncoder, canonical_location, get_utc_now

from scripts.algorand_utils import get_tx_note_info  # used by /verify
from scripts.attestation_registry import AttestationRegistry
from scripts.blockchain_publisher import publish_ai_prediction
from scripts.canon import build_p1_from_response, canonical_note_bytes_p1
from scripts.logger_utils import save_audit_bundle
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
from scripts.secrets_manager import get_account, get_network

# =============================================================================
# Import bootstrap: ensure notebooks/ on path for shared imports
# =============================================================================
ROOT = _Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
NB_DIR = ROOT / "notebooks"
if NB_DIR.exists() and str(NB_DIR) not in sys.path:
    sys.path.insert(0, str(NB_DIR))

# =============================================================================
# App & Config
# =============================================================================
API_VERSION = "0.8.0"

app = FastAPI(
    title="AI Oracle Inference API",
    version=API_VERSION,
    description="Inference service for multi-RWA asset valuation (initial: property).",
)

# CORS (restrict in prod via env)
_allowed = os.getenv("ALLOWED_ORIGINS", "*")
allow_origins = [o.strip() for o in _allowed.split(",")] if _allowed != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=(allow_origins != ["*"]),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logger
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_JSON = os.getenv("LOG_JSON", "true").lower() in {"1", "true", "yes", "y"}
logger = configure_logger(level=LOG_LEVEL, name="api", json_format=LOG_JSON)
logger.info("AI Oracle API starting (api_version=%s, log_json=%s, level=%s)", API_VERSION, LOG_JSON, LOG_LEVEL)

# Runtime flags
STRICT_RAW_FEATURES = os.getenv("STRICT_RAW_FEATURES", "1").lower() in {"1", "true", "yes", "y"}
ALLOW_BASELINE_FALLBACK = os.getenv("ALLOW_BASELINE_FALLBACK", "0").lower() in {"1", "true", "yes", "y"}
INFERENCE_DEBUG = os.getenv("INFERENCE_DEBUG", "0").lower() in {"1", "true", "yes", "y"}
REDACT_API_LOGS = os.getenv("REDACT_API_LOGS", "1").lower() in {"1", "true", "yes", "y"}
logger.info(
    "flags: STRICT_RAW_FEATURES=%s ALLOW_BASELINE_FALLBACK=%s INFERENCE_DEBUG=%s REDACT_API_LOGS=%s",
    STRICT_RAW_FEATURES, ALLOW_BASELINE_FALLBACK, INFERENCE_DEBUG, REDACT_API_LOGS
)

# Paths base → notebooks/outputs
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", "notebooks/outputs"))
LOGS_DIR = Path(os.getenv("AI_ORACLE_LOG_DIR", OUTPUTS_DIR / "logs"))
LOGS_DIR.mkdir(parents=True, exist_ok=True)
API_LOG_PATH = LOGS_DIR / "api_inference_log.jsonl"
registry = AttestationRegistry()

# Schemas (configurabili) — best-effort load (non far fallire l'import)
SCHEMAS_DIR = Path(os.getenv("SCHEMAS_DIR", "schemas"))
try:
    OUTPUT_SCHEMA = json.loads((SCHEMAS_DIR / "output_schema_v2.json").read_text(encoding="utf-8"))
except Exception:
    OUTPUT_SCHEMA = {}
SCHEMA_VERSION = "v2"
try:
    P1_SCHEMA = json.loads((SCHEMAS_DIR / "poval_v1_compact.schema.json").read_text(encoding="utf-8"))
except Exception:
    P1_SCHEMA = {}

# PoVal & timing guards
NOTE_MAX_BYTES = int(os.getenv("NOTE_MAX_BYTES", "1024"))
P1_TS_SKEW_PAST = int(os.getenv("P1_TS_SKEW_PAST", "600"))   # 10 min
P1_TS_SKEW_FUTURE = int(os.getenv("P1_TS_SKEW_FUTURE", "120"))  # 2 min

# =============================================================================
# Security MUST: Auth + Rate-limit + Body-limit
# =============================================================================
_auth = HTTPBearer(auto_error=False)
API_KEY = os.getenv("API_KEY")  # if unset => open mode

def require_auth(cred: HTTPAuthorizationCredentials = Depends(_auth)):
    """SECURITY: Simple static bearer token check (disable if API_KEY unset)."""
    if not API_KEY:
        return
    if cred is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
    token = cred.credentials or ""
    if token != API_KEY:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token")

# --- Rate limit (token bucket per IP) ---
RATE_LIMIT_RPS = float(os.getenv("RATE_LIMIT_RPS", "5"))  # demo default: 5 req/sec per IP
_rate_bucket: Dict[str, Tuple[float, float]] = {}  # ip -> (tokens, last_ts)

def require_ratelimit(request: Request):
    """PERF/SECURITY: Naive in-process rate limit per client IP."""
    ip = request.client.host if request.client else "local"
    capacity = max(1.0, RATE_LIMIT_RPS)
    refill = RATE_LIMIT_RPS
    now = time.time()
    tokens, last = _rate_bucket.get(ip, (capacity, now))
    tokens = min(capacity, tokens + (now - last) * refill)
    if tokens < 1.0:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit")
    _rate_bucket[ip] = (tokens - 1.0, now)

# --- Body size limit (413) ---
MAX_BODY_BYTES = int(os.getenv("MAX_BODY_BYTES", str(256 * 1024)))

@app.middleware("http")
async def _limit_body_mw(request: Request, call_next):
    """SECURITY: Reject bodies larger than MAX_BODY_BYTES with 413."""
    cl = request.headers.get("content-length")
    try:
        if cl and int(cl) > MAX_BODY_BYTES:
            return JSONResponse(
                {"detail": "Payload too large"},
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )
    except Exception:
        pass  # header invalido → lascia passare
    return await call_next(request)

# =============================================================================
# Request Models (flexible: extra='allow')
# =============================================================================
class PropertyPredictRequest(BaseModel):
    """Model for 'property' asset prediction; extra fields are accepted."""
    model_config = ConfigDict(extra="allow")  # accept extra fields; we filter later
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
LOCATION_MAP_PATH = os.getenv("LOCATION_MAP_JSON", "").strip()  # reserved for future use
_KNOWN_CATS: Dict[str, set] = {"region": set(), "zone": set(), "city": set()}

# Italian regions -> macro (north/center/south)
_REGION_TO_MACRO = {
    "lombardia": "north",
    "piemonte": "north",
    "liguria": "north",
    "veneto": "north",
    "friuliveneziagiulia": "north",
    "trentinoaltoadige": "north",
    "emiliaromagna": "north",
    "valledaosta": "north",
    "valledaoste": "north",
    "lazio": "center",
    "toscana": "center",
    "umbria": "center",
    "marche": "center",
    "abruzzo": "center",
    "molise": "center",
    "campania": "south",
    "puglia": "south",
    "calabria": "south",
    "basilicata": "south",
    "sicilia": "south",
    "sardegna": "south",
}

_ZONE_SYNONYMS = {
    "centro": "center",
    "centrostorico": "center",
    "historiccenter": "center",
    "semicentro": "semi_center",
    "semicenter": "semi_center",
    "semi-centro": "semi_center",
    "periferia": "periphery",
    "periferico": "periphery",
    "suburb": "periphery",
}

def _slug(s: str) -> str:
    """Normalize to ASCII lower + strip non-alnum."""
    s = _u_norm("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return "".join(ch for ch in s.lower() if ch.isalnum())

def utc_now_iso_z() -> str:
    """Return UTC now as ISO string with 'Z' suffix."""
    try:
        ts = get_utc_now()
        if not isinstance(ts, str):
            ts = ts.isoformat()
        return ts.replace("+00:00", "Z")
    except Exception:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _unwrap_for_introspection(obj: Any, max_depth: int = 10) -> Any:
    """Traverse nested estimators/pipelines with depth limit."""
    tried = set()
    def _walk(x, depth=0):
        if depth > max_depth:
            return
        if id(x) in tried or x is None:
            return
        tried.add(id(x))
        yield x
        for attr in ("best_estimator_","estimator","regressor_","regressor","pipeline","model","final_estimator"):
            if hasattr(x, attr):
                try:
                    child = getattr(x, attr)
                    if child is not None:
                        yield from _walk(child, depth + 1)
                except Exception:
                    pass
        try:
            from sklearn.pipeline import Pipeline
            if isinstance(x, Pipeline):
                for _, step in x.steps:
                    yield from _walk(step, depth + 1)
        except Exception:
            pass
    for node in _walk(obj):
        yield node

def _unwrap_estimator(obj):
    """Estrae l'estimatore 'vero' attraversando wrapper comuni (Pipeline, *SearchCV, TTR)."""
    tried = set()
    while obj is not None and id(obj) not in tried:
        tried.add(id(obj))
        try:
            from sklearn.pipeline import Pipeline  # type: ignore
            if isinstance(obj, Pipeline):
                obj = obj.steps[-1][1]
                continue
        except Exception:
            pass
        if getattr(obj, "__class__", type(None)).__name__ == "TransformedTargetRegressor":
            reg = getattr(obj, "regressor", None)
            obj = reg or obj
            continue
        for attr in ("best_estimator_", "estimator"):
            new = getattr(obj, attr, None)
            if new is not None and new is not obj:
                obj = new
                break
        else:
            break
    return obj

def _raw_columns_from_pipeline(pipeline) -> List[str]:
    """Ritorna i NOMI RAW dal/i ColumnTransformer presenti (anche annidati)."""
    cols: List[str] = []
    seen = set()
    stack = [pipeline]
    try:
        from sklearn.compose import ColumnTransformer  # type: ignore
        from sklearn.pipeline import Pipeline  # type: ignore
    except Exception:
        return cols
    while stack:
        x = stack.pop()
        if id(x) in seen:
            continue
        seen.add(id(x))
        if isinstance(x, Pipeline):
            for _, step in x.steps:
                stack.append(step)
            continue
        if isinstance(x, ColumnTransformer):
            for _tname, _trans, cols_in in x.transformers:
                if cols_in in (None, "drop"):
                    continue
                if isinstance(cols_in, (list, tuple, np.ndarray)):
                    cols.extend([str(c) for c in cols_in])
                else:
                    cols.append(str(cols_in))
        for attr in ("best_estimator_", "estimator", "regressor", "pipeline", "model"):
            child = getattr(x, attr, None)
            if child is not None:
                stack.append(child)
    out, seen = [], set()
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

def _refresh_known_categories_from_pipeline(pipeline) -> None:
    """Collect known categories visiting SOLO gli step della Pipeline (no recursion)."""
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        ct_list = []
        if isinstance(pipeline, Pipeline):
            for _, step in pipeline.steps:
                if isinstance(step, ColumnTransformer):
                    ct_list.append(step)
        elif isinstance(pipeline, ColumnTransformer):
            ct_list.append(pipeline)
        for ct in ct_list:
            for _tname, trans, cols_in in ct.transformers:
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
                len(_KNOWN_CATS["city"]), len(_KNOWN_CATS["region"]), len(_KNOWN_CATS["zone"]),
            )
    except RecursionError:
        logger.warning("Recursion limit hit during pipeline introspection, skipping")
        return
    except Exception as e:
        logger.debug("Could not introspect categories (flat): %s", e)

def _to_macro_region(value: Optional[str], city: Optional[str] = None) -> Optional[str]:
    """Map Italian region names/cities to macro area (north/center/south) when possible."""
    if not value and not city:
        return value
    if value:
        v = _slug(str(value))
        if v in {"north", "center", "south"}:
            try:
                return value.strip().lower()
            except Exception:
                return str(value).lower()
        if v in _REGION_TO_MACRO:
            return _REGION_TO_MACRO[v]
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
    """Return exact/closest category match (slug-based) for the given field name."""
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
    """Normalize region/zone values to known categories using synonyms and slugs."""
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
    """Append a JSONL record to API log (best-effort)."""
    payload = {**record, "_logged_at": get_utc_now()}
    line = json.dumps(payload, cls=NumpyJSONEncoder, ensure_ascii=False)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    with os.fdopen(fd, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# --- Missing & dtype cleaners (evita pd.NA in sklearn) -----------------------
def _to_numpy_nan(x):
    try:
        import pandas as _pd
        if x is _pd.NA:
            return np.nan
    except Exception:
        pass
    try:
        if isinstance(x, float) and (x != x):  # NaN
            return np.nan
    except Exception:
        pass
    return x

def _is_nullable_int_or_bool_dtype(dtype) -> bool:
    s = str(dtype)
    return s.startswith("Int") or s == "boolean"

def _clean_missing_df(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    df = df.applymap(_to_numpy_nan)
    for c in df.columns:
        dt = df[c].dtype
        if _is_nullable_int_or_bool_dtype(dt):
            df[c] = df[c].astype("float64")
    return df

# SECURITY: Sensitive keys to redact from logs.
_SENSITIVE_KEYS = {
    "address","note","notes","email","phone","lat","lon","lng","latitude","longitude","coordinates","gps","contact",
}

def _redact(obj: Any) -> Any:
    """SECURITY: Recursively redact sensitive fields (no PII in logs)."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k] = "***" if k.lower() in _SENSITIVE_KEYS else _redact(v)
        return out
    if isinstance(obj, list):
        return [_redact(x) for x in obj]
    return obj

def _predict_one(pipeline_obj, df: pd.DataFrame) -> float:
    """Run prediction on a single-row DataFrame, with baseline fallback if enabled."""
    df = _clean_missing_df(df)
    if hasattr(pipeline_obj, "predict"):
        logger.info("predict_one: pipeline_obj=%s", type(pipeline_obj).__name__)
        try:
            y = pipeline_obj.predict(df)
            logger.info("predict_one: raw predict() -> %s", np.ravel(y)[0] if isinstance(y, (list, tuple, np.ndarray)) else y)
            if isinstance(y, (list, tuple, np.ndarray)):
                return float(np.ravel(y)[0])
            return float(y)
        except RecursionError:
            logger.warning("predict_one: RecursionError on pipeline.predict; trying last step estimator")
            try:
                from sklearn.pipeline import Pipeline  # type: ignore
                if isinstance(pipeline_obj, Pipeline):
                    base = pipeline_obj.steps[-1][1]
                    y = base.predict(df)
                    logger.info("predict_one: last-step estimator predict -> %s", np.ravel(y)[0] if isinstance(y, (list, tuple, np.ndarray)) else y)
                    if isinstance(y, (list, tuple, np.ndarray)):
                        return float(np.ravel(y)[0])
                    return float(y)
            except Exception as e:
                logger.warning("predict_one: last-step estimator failed: %s", e)
            if ALLOW_BASELINE_FALLBACK:
                s = df.get("size_m2")
                if s is not None and len(s) > 0 and pd.notna(s.iloc[0]):
                    fb = float(max(1.0, 0.8 * float(s.iloc[0])))
                    logger.info("predict_one: baseline fallback -> %s", fb)
                    return fb
            raise
    if callable(pipeline_obj):
        logger.info("predict_one: pipeline_obj is callable; calling...")
        return float(pipeline_obj(df))
    if ALLOW_BASELINE_FALLBACK:
        s = df.get("size_m2")
        if s is not None and len(s) > 0 and pd.notna(s.iloc[0]):
            fb = float(max(1.0, 0.8 * float(s.iloc[0])))
            logger.info("predict_one: callable fallback -> %s", fb)
            return fb
    raise RuntimeError("Model object has no predict/callable interface")

_EXPECTED_FEATURES: Optional[List[str]] = None

def collect_expected_features(
    meta: Optional[dict],
    pipeline: Optional[object],
    payload_keys: Optional[List[str]] = None,
) -> List[str]:
    """
    Ordine di priorità per i nomi RAW attesi:
    1) meta.feature_order
    2) meta.expected_features / features / input_features
    3) ColumnTransformer RAW columns (anche annidato)
    4) payload_keys (fallback estremo)
    """
    # 0) override runtime per test
    if isinstance(_EXPECTED_FEATURES, list) and _EXPECTED_FEATURES:
        logger.info("expected_features: test_override (_EXPECTED_FEATURES), n=%d", len(_EXPECTED_FEATURES))
        out = list(_EXPECTED_FEATURES)
    else:
        # 1) meta.feature_order
        fo = (meta or {}).get("feature_order")
        if isinstance(fo, list) and fo:
            ordered, s = [], set()
            for x in fo:
                x = str(x)
                if x not in s:
                    s.add(x)
                    ordered.append(x)
            out = ordered
            logger.info("expected_features: meta.feature_order, n=%d, head=%s", len(out), out[:10])
        else:
            # 2) altre chiavi meta
            out = None
            if isinstance(meta, dict):
                for k in ("expected_features", "features", "input_features"):
                    v = meta.get(k)
                    if isinstance(v, list) and v:
                        out = list(map(str, v))
                        logger.info("expected_features: meta.%s, n=%d, head=%s", k, len(out), out[:10])
                        break
            # 3) CT RAW
            if out is None and pipeline is not None:
                root = _unwrap_estimator(pipeline)
                raw_cols = _raw_columns_from_pipeline(root)
                if raw_cols:
                    out = raw_cols
                    logger.info("expected_features: ColumnTransformer RAW, n=%d, head=%s", len(out), out[:10])
            # 4) payload_keys
            if out is None and isinstance(payload_keys, list) and payload_keys:
                out = list(map(str, payload_keys))
                logger.info("expected_features: payload_keys fallback, n=%d, head=%s", len(out), out[:10])
            if out is None:
                raise RuntimeError("Empty expected features (no meta, no ColumnTransformer RAW, no payload_keys)")

    # --- HARDENING: assicurati che 'city' sia presente se uno step a monte la richiede.
    # Aggiungerla è innocuo per i ColumnTransformer (colonne extra sono ignorate se non selezionate).
    if "city" not in out:
        out = list(out) + ["city"]

    return out

_SAFE_DERIVED = {
    "age_years",
    "luxury_score",
    "env_score",
    "location",
    "city",
    "is_top_floor",
    "listing_month",
    ASSET_ID,
}

_KEY_ALIASES = {
    "sqm": "size_m2",
    "size": "size_m2",
    "m2": "size_m2",
    "year": "year_built",
    "built_year": "year_built",
    "balcony": "has_balcony",
    "garden": "has_garden",
    "garage": "has_garage",
    "air_quality": "air_quality_index",
    "noise": "noise_level",
    "valuation": "valuation_k",
    "price_k": "valuation_k",
    "n_rooms": "rooms",
    "room_count": "rooms",
    "n_bathrooms": "bathrooms",
    "bathroom_count": "bathrooms",
    "elevator": "has_elevator",
    "city_name": "city",
}

def _canonicalize_keys(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Apply field aliases to a record."""
    return {_KEY_ALIASES.get(k, k): v for k, v in rec.items()}

def _autofill_safe(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Compute derived fields + normalize location/city/region/zone (SAFE for None)."""
    r = dict(rec)
    r = _canonicalize_keys(r)

    # Derived fields
    if "age_years" not in r and r.get("year_built") not in (None, ""):
        try:
            r["age_years"] = max(0, datetime.utcnow().year - int(r["year_built"]))
        except Exception:
            pass
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

    # Location normalization
    if LOCATION in r and r[LOCATION]:
        try:
            if isinstance(r[LOCATION], str) and r[LOCATION].strip():
                r[LOCATION] = canonical_location(r)
            else:
                r[LOCATION] = None
        except Exception:
            pass

    # Listing month
    if "listing_month" not in r or r.get("listing_month") in (None, "", 0):
        try:
            r["listing_month"] = int(datetime.utcnow().month)
        except Exception:
            r["listing_month"] = None

    # is_top_floor
    try:
        if "is_top_floor" not in r and r.get("floor") is not None and r.get("building_floors") is not None:
            r["is_top_floor"] = int(r.get("floor") == r.get("building_floors"))
    except Exception:
        pass

    # City normalization (SAFE: no str(None))
    try:
        prop_cfg = ASSET_CONFIG["property"]
        city_syn = {
            str(k).strip().lower(): str(v).strip().title()
            for k, v in (prop_cfg.get("city_synonyms") or {}).items()
        }
        def _title_or_none(x):
            if isinstance(x, str) and x.strip():
                return x.strip().title()
            return None
        if not r.get("city"):
            loc = r.get(LOCATION)
            if isinstance(loc, str) and loc.strip():
                key = loc.strip().lower()
                r["city"] = city_syn.get(key, _title_or_none(loc))
            else:
                r["city"] = None
        else:
            cval = r.get("city")
            if isinstance(cval, str) and cval.strip():
                key = cval.strip().lower()
                r["city"] = city_syn.get(key, _title_or_none(cval))
            elif cval is None:
                r["city"] = None
    except Exception:
        if isinstance(r.get("city"), str):
            r["city"] = r["city"].strip().title() or None

    # Region normalization (macro)
    try:
        prop_cfg = ASSET_CONFIG["property"]
        by_city = (prop_cfg.get("region_by_city") or DEFAULT_REGION_BY_CITY) or {}
        if r.get("region"):
            if isinstance(r["region"], str) and r["region"].strip():
                r["region"] = _to_macro_region(r["region"], r.get("city"))
            else:
                r["region"] = None
        else:
            city_val = r.get("city")
            if isinstance(city_val, str) and city_val.strip():
                r["region"] = (by_city.get(city_val.strip().title()) or None)
            else:
                r["region"] = None
        if r.get("region"):
            r["region"] = _normalize_to_known("region", r["region"])
    except Exception:
        pass

    # Zone normalization
    try:
        z = r.get("zone")
        if isinstance(z, str) and z.strip():
            r["zone"] = _normalize_to_known("zone", z.strip())
        else:
            prop_cfg = ASSET_CONFIG["property"]
            th = prop_cfg.get("zone_thresholds_km") or {"center": 1.5, "semi_center": 5.0}
            d = r.get("distance_to_center_km")
            if d is not None:
                try:
                    d = float(d)
                    if d <= float(th.get("center", 1.5)):
                        r["zone"] = "center"
                    elif d <= float(th.get("semi_center", 5.0)):
                        r["zone"] = "semi_center"
                    else:
                        r["zone"] = "periphery"
                except Exception:
                    r["zone"] = None
            else:
                r["zone"] = None
        if r.get("zone"):
            r["zone"] = _normalize_to_known("zone", r["zone"])
    except Exception:
        pass

    try:
        if r.get("city"):
            r["city"] = _closest_known("city", str(r["city"]))
    except Exception:
        pass

    # --- DEFAULTS “safe” per step custom PRE-imputer (no change nella logica del modello) ---
    r.setdefault("heating", "standard")
    r.setdefault("urban_type", "residential")
    r.setdefault("view", "standard")
    if "public_transport_nearby" not in r or r.get("public_transport_nearby") in (None, ""):
        r["public_transport_nearby"] = 1
    if "condition" not in r or r.get("condition") in (None, ""):
        r["condition"] = "good"
    # alias legacy: 'garage' → 'has_garage'
    if "has_garage" not in r and "garage" in r:
        try:
            r["has_garage"] = int(bool(r.get("garage")))
        except Exception:
            r["has_garage"] = 0
    r.setdefault("has_garage", 0)

    return r

def validate_input_record(
    record: Dict[str, Any],
    all_expected: List[str],
    *,
    strict: bool = False,
    drop_extras: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Validate & sanitize a property record against expected features."""
    base = _autofill_safe(_canonicalize_keys(record))
    # Se il pipeline richiede 'city' ma non è tra gli expected, tienila comunque
    allowed = set(all_expected) | _SAFE_DERIVED | {"city"}
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

def _ts_to_seconds(ts) -> int:
    """Convert timestamp to seconds (handles both int and ISO string)."""
    if isinstance(ts, (int, float)):
        return int(ts)
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            return int(dt.timestamp())
        except Exception:
            pass
    return int(time.time())

_Z = {0.80: 1.282, 0.90: 1.645, 0.95: 1.960, 0.98: 2.326, 0.99: 2.576}
def _z_for_conf(conf: float) -> float:
    """Return z-score for given confidence level (rounded key)."""
    return _Z.get(round(conf, 2), 1.960)

def _split_preprocessor_and_model(pipeline_obj: Any) -> Tuple[Any, Any]:
    """Return (preprocessor, model) if pipeline; otherwise (None, obj)."""
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
    ttr = None
    m = model
    seen = set()
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(m, Pipeline):
            m = m.steps[-1][1]
    except Exception:
        pass
    if m.__class__.__name__ == "TransformedTargetRegressor":
        ttr = m
        base = getattr(m, "regressor", None)
        if base is not None:
            m = base
    for attr in ("best_estimator_", "estimator"):
        if id(m) in seen:
            break
        seen.add(id(m))
        new_m = getattr(m, attr, None)
        if new_m is not None and new_m is not m:
            m = new_m
        else:
            break
    return m, ttr

def _inverse_target_if_needed(y: np.ndarray, ttr: Any) -> np.ndarray:
    """Apply inverse transformation on target if TransformedTargetRegressor used."""
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
    return y

def predict_with_confidence(
    pipeline_obj,
    record: Dict[str, Any],
    all_expected: List[str],
    manifest_path: Path,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Return prediction + uncertainty via forest variance or global sigma fallback."""
    df_raw = pd.DataFrame([{k: record.get(k, np.nan) for k in all_expected}], columns=all_expected)
    df = _clean_missing_df(df_raw)

    try:
        nonnull_total = int(df.notnull().sum().sum())
        logger.info("DF shape=%s nonnull=%s", df.shape, nonnull_total)
        logger.info("DF head (first 8 cols)=%s", {c: df.iloc[0][c] for c in all_expected[:8]})
        if nonnull_total == 0:
            logger.warning("All values are NaN after aligning with expected features. Probable raw vs post-encoding mismatch.")
    except Exception:
        pass

    try:
        y_hat = float(_predict_one(pipeline_obj, df))
        logger.info("predict_with_confidence: y_hat(point)=%s", y_hat)
    except RecursionError:
        sigma = max(1.0, abs(float(df.get("size_m2", pd.Series([0])).iloc[0] or 0.0)) * 0.10)
        z = _z_for_conf(confidence)
        ci = z * float(sigma)
        out = {
            "prediction": 0.0,
            "point_pred": 0.0,
            "uncertainty": round(float(sigma), 2),
            "confidence": float(confidence),
            "confidence_interval": (round(0.0 - ci, 2), round(0.0 + ci, 2)),
            "ci_margin": round(ci, 2),
            "method": "global_sigma",
            "n_estimators": None,
        }
        logger.info("predict_with_confidence: RecursionError fallback -> %s", out)
        return out

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
            except Exception as e:
                logger.warning("Preprocessor.transform failed; using raw df. err=%s", e)
                X = df
        model_final, ttr = _unwrap_final_estimator(model_step)

        # Forest-like variance
        if hasattr(model_final, "estimators_") and isinstance(model_final.estimators_, (list, tuple)) and len(model_final.estimators_) >= 3:
            per_tree_raw = np.array([float(np.ravel(est.predict(X))[0]) for est in model_final.estimators_], dtype=float)
            per_tree = _inverse_target_if_needed(per_tree_raw, ttr)
            m = float(per_tree.mean())
            s = float(per_tree.std(ddof=1)) if len(per_tree) > 1 else 0.0
            z = _z_for_conf(confidence)
            ci = z * s
            out = {
                "prediction": round(m, 2),
                "point_pred": round(y_hat, 2),
                "uncertainty": round(s, 2),
                "confidence": float(confidence),
                "confidence_interval": (round(m - ci, 2), round(m + ci, 2)),
                "ci_margin": round(ci, 2),
                "method": "forest_variance",
                "n_estimators": len(model_final.estimators_),
            }
            logger.info("predict_with_confidence: forest_variance -> %s", out)
            return out

        raise RuntimeError("No per-tree variance available")
    except Exception as e:
        logger.debug("Forest variance unavailable (%s); using global sigma fallback.", e)
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
        out = {
            "prediction": round(y_hat, 2),
            "point_pred": round(y_hat, 2),
            "uncertainty": round(float(sigma), 2),
            "confidence": float(confidence),
            "confidence_interval": (round(y_hat - ci, 2), round(y_hat + ci, 2)),
            "ci_margin": round(ci, 2),
            "method": "global_sigma",
            "n_estimators": None,
        }
        logger.info("predict_with_confidence: global_sigma -> %s", out)
        return out

# =============================================================================
# Endpoints
# =============================================================================
@app.get("/health")
def health() -> dict:
    """Health check for model registry and cache."""
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

@app.post("/predict/{asset_type}", dependencies=[Depends(require_auth), Depends(require_ratelimit)])
def predict(
    asset_type: str = FPath(...),
    publish: bool = Query(False),
    attestation_only: bool = Query(False, description="If true, return PoVal p1 only (publish optional)"),
    payload: dict = Body(...),
) -> dict:
    """Predict valuation, build PoVal p1, optionally publish on-chain, return response v2."""
    asset_type = asset_type.lower()
    if asset_type not in REQUEST_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported asset_type: {asset_type}")

    logger.info(
        "predict: START asset_type=%s publish=%s attestation_only=%s payload_keys=%s",
        asset_type, publish, attestation_only, list(payload.keys()) if isinstance(payload, dict) else None
    )

    # 1) Load fitted artifacts + meta
    try:
        pipeline = get_pipeline(asset_type, "value_regressor")
        try:
            paths = get_model_paths(asset_type, "value_regressor")
        except Exception:
            paths = {}
        meta = get_model_metadata(asset_type, "value_regressor") or {}
        logger.info(
            "predict: artifacts loaded pipeline=%s model_meta.version=%s model_path=%s",
            type(pipeline).__name__, meta.get("model_version"), (paths or {}).get("pipeline")
        )

        # --- PATCH: ripulisce l'OUTPUT di PriorsGuard.transform (se presente) ---
        try:
            from sklearn.pipeline import Pipeline as _SkPipeline
            if isinstance(pipeline, _SkPipeline):
                new_steps = []
                for name, step in pipeline.steps:
                    clsname = getattr(step, "__class__", type(None)).__name__
                    if clsname == "PriorsGuard" and hasattr(step, "transform"):
                        orig_transform = step.transform
                        def _wrapped_transform(X, _orig=orig_transform):
                            Y = _orig(X)
                            try:
                                if isinstance(Y, pd.DataFrame):
                                    Y = _clean_missing_df(Y)
                            except Exception:
                                pass
                            return Y
                        step.transform = _wrapped_transform  # monkey patch
                    new_steps.append((name, step))
                pipeline.steps = new_steps
                logger.info("predict: PriorsGuard cleaner patch applied.")
            else:
                logger.debug("predict: pipeline is not sklearn.Pipeline; patch not needed.")
        except Exception as _e:
            logger.debug("predict: PriorsGuard patch skipped: %s", _e)

        # 2) Resolve expected raw features
        payload_keys = list(payload.keys()) if isinstance(payload, dict) else None
        all_expected: List[str] = collect_expected_features(meta, pipeline, payload_keys=payload_keys)
        if not all_expected:
            try:
                from scripts.model_registry import expected_features as reg_expected_features
                cat, num = reg_expected_features(meta, paths.get("manifest"), asset_type=asset_type)
                all_expected = list(cat) + list(num)
            except Exception:
                pass
        if not all_expected:
            raise RuntimeError("Empty expected features")
        logger.info("predict: expected_features n=%d head=%s", len(all_expected), all_expected[:10])
    except (RegistryLookupError, ModelNotFoundError) as e:
        logger.error("predict: registry/model error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error("predict: artifacts error: %s", e)
        raise HTTPException(status_code=500, detail=f"Artifacts error: {e}")

    # 3) Canonicalize/validate payload
    ModelCls = REQUEST_MODELS[asset_type]
    try:
        _refresh_known_categories_from_pipeline(pipeline)
        req_obj = ModelCls(**payload)
        rec_in = req_obj.model_dump(exclude_none=False)
        rec, vreport = validate_input_record(rec_in, all_expected, strict=False, drop_extras=False)
        if not rec.get(ASSET_ID):
            rec[ASSET_ID] = f"{asset_type}_{uuid.uuid4().hex[:10]}"
        logger.info(
            "predict: record normalized keys=%d sample=%s validation_ok=%s warnings=%s",
            len(rec), {k: rec[k] for k in list(rec)[:8]}, vreport.get("ok", True), bool(vreport.get("warnings"))
        )
    except Exception as e:
        logger.error("predict: invalid payload: %s", e)
        raise HTTPException(status_code=422, detail=f"Invalid payload: {e}")

    # 4) Predict + CI + latency
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
        logger.error("predict: inference error: %s", e)
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info("predict: conf=%s latency_ms=%s", conf, latency_ms)

    # 5) Build response (schema v2)
    ci_low, ci_high = conf["confidence_interval"]
    response: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "asset_id": rec[ASSET_ID],
        "asset_type": asset_type,
        "timestamp": utc_now_iso_z(),
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
            "model_hash": meta.get("model_hash"),
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
    logger.info("predict: RESPONSE.metrics=%s", response["metrics"])

    # Optional: breakdown & sanity benchmark
    try:
        br = explain_price(rec)
        response.setdefault("explanations", {})["pricing_breakdown"] = br
    except Exception as e:
        logger.debug("explain_price failed: %s", e)
    try:
        pb = price_benchmark(location=rec.get("location"), valuation_k=float(conf["prediction"]))
        if pb:
            response.setdefault("sanity", {})["price_benchmark"] = pb
            if pb.get("out_of_band", False):
                response["flags"]["price_out_of_band"] = True
                response["flags"]["needs_review"] = True
    except Exception as e:
        logger.debug("price_benchmark failed: %s", e)

    # 6) Build PoVal p1 (always)
    canonical_input_subset = {k: rec.get(k, None) for k in all_expected}
    response["canonical_input"] = canonical_input_subset  # used by builder as clean fallback
    p1, dbg = build_p1_from_response(response, allowed_input_keys=all_expected)
    p1_bytes, p1_sha, p1_size = canonical_note_bytes_p1(p1)
    logger.info("predict: p1 built size=%s sha256=%s ih=%s", p1_size, p1_sha, dbg.get("ih"))

    # Guardrail: note size
    if int(p1_size) > NOTE_MAX_BYTES:
        logger.error("predict: p1_too_large: %sB > NOTE_MAX_BYTES=%s", p1_size, NOTE_MAX_BYTES)
        raise HTTPException(status_code=413, detail=f"p1_too_large: {p1_size}B > NOTE_MAX_BYTES={NOTE_MAX_BYTES}")

    # Anti-replay (asset_id + p1 sha256)
    asset_id = response["asset_id"]
    if registry.seen(p1_sha, asset_id):
        logger.error("predict: replay detected asset_id=%s p1_sha256=%s", asset_id, p1_sha)
        raise HTTPException(status_code=409, detail=f"replay: asset_id={asset_id} p1_sha256={p1_sha}")

    # Audit bundle (always created) — ZIP served by /audit/{rid}
    rid = uuid.uuid4().hex[:12]
    bundle_dir = Path(os.getenv("OUTPUTS_DIR", "notebooks/outputs")) / "audit" / rid
    bundle_dir.mkdir(parents=True, exist_ok=True)
    save_audit_bundle(
        bundle_dir,
        p1_bytes=p1_bytes,
        p1_sha256=p1_sha,
        canonical_input=canonical_input_subset,
    )
    response["audit_bundle"] = {"rid": rid}
    logger.info("predict: audit_bundle rid=%s dir=%s", rid, str(bundle_dir))

    # Attach attestation preview to response
    response["attestation"] = {
        "p1": p1,
        "p1_sha256": p1_sha,
        "p1_size_bytes": int(p1_size),
        "canonical_input": canonical_input_subset,
        "input_hash": dbg.get("ih"),
    }

    # 7) Publish (blockchain)
    net = f"algorand-{get_network()}"
    if publish:
        try:
            logger.info("publish: submitting p1 to network=%s ...", net)
            result = publish_ai_prediction(response, p1_bytes=p1_bytes, p1_sha256=p1_sha)
            response["publish"] = {
                "status": "ok",
                "txid": result.get("blockchain_txid"),
                "asa_id": result.get("asa_id"),
                "network": net,
                "note_size": result.get("note_size"),
                "note_sha256": result.get("note_sha256"),
                "is_compacted": result.get("is_compacted"),
                "confirmed_round": result.get("confirmed_round"),
                "explorer_url": get_tx_note_info(result.get("blockchain_txid")).get("explorer_url", None),
            }
            response["blockchain_txid"] = response["publish"]["txid"]
            response["asa_id"] = response["publish"]["asa_id"]
            logger.info("publish: OK txid=%s round=%s", response["blockchain_txid"], response["publish"].get("confirmed_round"))
            try:
                issuer = ""
                try:
                    acc = get_account(require_signing=False)
                    issuer = getattr(acc, "address", "") or ""
                except Exception:
                    pass
                registry.record(
                    p1_sha,
                    asset_id,
                    response["blockchain_txid"],
                    get_network(),
                    issuer,
                    int(p1.get("ts", 0)),
                )
                (bundle_dir / "tx.txt").write_text(str(response["blockchain_txid"] or ""), encoding="utf-8")
            except Exception as e:
                logger.debug("publish: audit/record failed: %s", e)

        except TypeError:
            logger.info("publish: falling back to legacy publisher signature")
            try:
                result = publish_ai_prediction(response)
                response["publish"] = {
                    "status": "ok",
                    "txid": result.get("blockchain_txid"),
                    "asa_id": result.get("asa_id"),
                    "network": net,
                    "note_size": result.get("note_size"),
                    "note_sha256": result.get("note_sha256"),
                    "is_compacted": result.get("is_compacted"),
                    "confirmed_round": result.get("confirmed_round"),
                    "explorer_url": get_tx_note_info(result.get("blockchain_txid")).get("explorer_url", None),
                }
                response["blockchain_txid"] = response["publish"]["txid"]
                response["asa_id"] = response["publish"]["asa_id"]
                logger.info("publish(legacy): OK txid=%s", response["blockchain_txid"])
            except Exception as e2:
                logger.error("publish(legacy): error: %s", e2)
                response["publish"] = {"status": "error", "error": str(e2)}
        except Exception as e:
            logger.error("publish: error: %s", e)
            response["publish"] = {"status": "error", "error": str(e)}

    # 8) Schema validation v2 (best-effort)
    if OUTPUT_SCHEMA:
        try:
            jsonschema_validate(instance=response, schema=OUTPUT_SCHEMA)
        except ValidationError as ve:
            response["schema_validation_error"] = str(ve).split("\n", 1)[0][:240]
            logger.warning("schema v2 validation error: %s", response["schema_validation_error"])
        except Exception as e:
            response["schema_validation_error"] = f"Schema check failed: {e}"[:240]
            logger.warning("schema v2 check failed: %s", response["schema_validation_error"])

    # 9) API log (best-effort)
    try:
        rec_log = {
            "event": "prediction",
            "asset_type": asset_type,
            "request": _redact(payload) if REDACT_API_LOGS else payload,
            "response_meta": {
                "schema_version": response.get("schema_version"),
                "asset_id": response.get("asset_id"),
                "timestamp": response.get("timestamp"),
                "metrics": {
                    "valuation_k": response.get("metrics", {}).get("valuation_k"),
                    "uncertainty_k": response.get("metrics", {}).get("uncertainty_k"),
                    "latency_ms": response.get("metrics", {}).get("latency_ms"),
                },
                "model_meta": {
                    "value_model_name": response.get("model_meta", {}).get("value_model_name"),
                    "value_model_version": response.get("model_meta", {}).get("value_model_version"),
                },
                "attestation": {"p1_sha256": p1_sha, "p1_size_bytes": p1_size},
                "publish_status": response.get("publish", {}).get("status"),
            },
            "_logged_at": utc_now_iso_z(),
        }
        if os.getenv("LOG_FULL_RESPONSES", "0").lower() in {"1", "true", "yes", "y"}:
            rec_log["response"] = {
                "timestamp": response.get("timestamp"),
                "asset_id": response.get("asset_id"),
                "metrics": response.get("metrics"),
                "model_meta": response.get("model_meta"),
            }
        if not publish:
            log_jsonl(rec_log)
    except Exception as e:
        print(f"[warn] log_jsonl failed: {e}")

    # 10) attestation_only
    if attestation_only:
        out = {
            "asset_id": response["asset_id"],
            "attestation": response["attestation"]["p1"],
            "attestation_sha256": response["attestation"]["p1_sha256"],
            "attestation_size": response["attestation"]["p1_size_bytes"],
            "published": response.get("publish", {}).get("status") == "ok",
            "txid": response.get("blockchain_txid") or None,
            "network": net if publish else None,
            "audit_bundle_path": response.get("audit_bundle_path"),  # lasciato invariato per compat
        }
        logger.info("predict: attestation_only OUT=%s", {k: out[k] for k in out if k != "attestation"})
        return out

    logger.info("predict: END asset_id=%s valuation_k=%s", response["asset_id"], response["metrics"]["valuation_k"])
    return response

RID_RE = re.compile(r"^[a-zA-Z0-9_-]{6,64}$")

@app.get("/audit/{rid}", dependencies=[Depends(require_auth), Depends(require_ratelimit)])
def download_audit_bundle(rid: str):
    """
    Return a ZIP of the audit bundle created in /predict (p1.json, p1.sha256, canonical_input.json, etc.).
    SECURITY: Uses validated 'rid' and does not expose real paths.
    """
    if not RID_RE.match(rid):
        raise HTTPException(status_code=400, detail="invalid rid")
    base = Path(os.getenv("OUTPUTS_DIR", "notebooks/outputs")) / "audit" / rid
    if not base.exists():
        raise HTTPException(status_code=404, detail="not found")
    tmpdir = tempfile.mkdtemp(prefix="poval-")
    zip_path = shutil.make_archive(Path(tmpdir) / f"axiomatic_audit_{rid}", "zip", base)
    filename = f"axiomatic_audit_{rid}.zip"
    return FileResponse(zip_path, media_type="application/zip", filename=filename)

# ---- VERIFY endpoint (One-Click Verify) ----
@app.post("/verify", dependencies=[Depends(require_auth), Depends(require_ratelimit)])
def verify(body: dict = Body(...)) -> dict:
    """
    Verify PoVal note on-chain.

    Modes:
    - p1 (compact): validate schema + value-in-interval + ACJ-1 hash consistency.
    - legacy: minimal checks for backward compatibility.
    """
    txid = str(body.get("txid") or "").strip()
    if not txid:
        raise HTTPException(422, "Missing 'txid'")

    note = get_tx_note_info(txid)  # Indexer vs Algod logic inside
    note_json = note.get("note_json")
    onchain_sha = note.get("note_sha256")
    explorer_url = note.get("explorer_url")

    expected_sha = body.get("attestation_sha256") or body.get("expected_sha256")
    p1_from_body = None
    att = body.get("attestation") if isinstance(body.get("attestation"), dict) else None
    if att and isinstance(att.get("p1"), dict):
        p1_from_body = att["p1"]
    elif isinstance(body.get("prediction"), dict):
        att2 = body["prediction"].get("attestation")
        if isinstance(att2, dict) and isinstance(att2.get("p1"), dict):
            p1_from_body = att2["p1"]

    if expected_sha is None and isinstance(p1_from_body, dict):
        try:
            _, expected_sha, _ = canonical_note_bytes_p1(p1_from_body)
        except Exception:
            expected_sha = None

    ok = False
    reason = None
    mode = None

    try:
        if isinstance(note_json, dict) and note_json.get("s") == "p1":
            mode = "p1"
            try:
                if P1_SCHEMA:
                    jsonschema_validate(instance=note_json, schema=P1_SCHEMA)
            except ValidationError as ve:
                return {
                    "txid": txid,
                    "verified": False,
                    "mode": mode,
                    "reason": f"schema_invalid: {str(ve).splitlines()[0]}",
                    "note_size": note.get("note_size"),
                    "note_sha256": onchain_sha,
                    "confirmed_round": note.get("confirmed_round"),
                    "explorer_url": explorer_url,
                    "note_preview": note_json if INFERENCE_DEBUG else None,
                }

            time_out_of_window = False
            try:
                now = int(time.time())
                ts_sec = int(datetime.fromisoformat(str(note_json["ts"]).replace('Z', '+00:00')).timestamp())
                if ts_sec < (now - P1_TS_SKEW_PAST) or ts_sec > (now + P1_TS_SKEW_FUTURE):
                    time_out_of_window = True
            except Exception:
                time_out_of_window = False

            v = float(note_json["v"])
            lo, hi = float(note_json["u"][0]), float(note_json["u"][1])
            if not (lo <= v <= hi):
                ok, reason = False, f"value_out_of_range:{v}∉[{lo},{hi}]"
            else:
                ok = True

            if ok and expected_sha and onchain_sha and expected_sha != onchain_sha:
                ok = False
                reason = "sha_mismatch"

            if ok and onchain_sha:
                try:
                    from scripts.canon import canonicalize_jcs
                    bytes_rebuilt = canonicalize_jcs(note_json)
                    rebuilt_sha = hashlib.sha256(bytes_rebuilt).hexdigest()
                    if rebuilt_sha != onchain_sha:
                        ok = False
                        reason = "onchain_hash_mismatch"
                except Exception:
                    pass

            if ok and time_out_of_window and INFERENCE_DEBUG:
                note.setdefault("debug", {})  # type: ignore[assignment]
                note["debug"]["timestamp_out_of_window"] = True  # type: ignore[index]

        elif isinstance(note_json, dict) and ("ref" in note_json or "schema_version" in note_json):
            mode = "legacy"
            ok = True
        else:
            reason = "unsupported_or_empty_note"

    except Exception as e:
        ok = False
        reason = f"verify_error:{e}"

    return {
        "txid": txid,
        "verified": bool(ok),
        "mode": mode,
        "note_sha256": onchain_sha,
        "expected_sha256": expected_sha,
        "note_size": note.get("note_size"),
        "confirmed_round": note.get("confirmed_round"),
        "explorer_url": explorer_url,
        "reason": None if ok else (reason or "mismatch"),
        "note_preview": note_json if INFERENCE_DEBUG else None,
    }

# ---- models & cache mgmt ----
@app.get("/models/{asset_type}", dependencies=[Depends(require_auth), Depends(require_ratelimit)])
def list_models(asset_type: str) -> dict:
    """List tasks and discovered models for a given asset type."""
    return {
        "asset_type": asset_type,
        "tasks": list_tasks(asset_type),
        "discovered_models": [p.name for p in discover_models_for_asset(asset_type)],
    }

@app.post("/models/{asset_type}/{task}/refresh", dependencies=[Depends(require_auth), Depends(require_ratelimit)])
def refresh_model_cache(asset_type: str, task: str) -> dict:
    """Refresh in-memory model cache for the (asset_type, task) pair."""
    refresh_cache(asset_type, task)
    return {"status": "cache_refreshed", "asset_type": asset_type, "task": task}

@app.get("/models/{asset_type}/{task}/health", dependencies=[Depends(require_auth), Depends(require_ratelimit)])
def model_health(asset_type: str, task: str) -> dict:
    """Return model health info."""
    return health_check_model(asset_type, task)

# ---- logs browsing ----
@app.get("/logs/api", dependencies=[Depends(require_auth), Depends(require_ratelimit)])
def get_api_logs() -> List[dict]:
    """Return last API JSONL entries (best-effort)."""
    try:
        if not API_LOG_PATH.exists():
            return []
        with API_LOG_PATH.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Log read error: {e}")

@app.get("/logs/published", dependencies=[Depends(require_auth), Depends(require_ratelimit)])
def get_published_assets():
    """Return published assets list (from JSON file, if present)."""
    try:
        p = LOGS_DIR / "published_assets.json"
        if not p.exists():
            return []
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Read error: {e}")

@app.get("/logs/detail_reports", dependencies=[Depends(require_auth), Depends(require_ratelimit)])
def get_detail_reports():
    """Return detail reports aggregated from logs/detail_reports/*.json."""
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