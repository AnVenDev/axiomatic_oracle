"""
Module: inference_api.py — FastAPI service for AI Oracle inference & PoVal™.

Responsibilities:
- Predict valuations (multi-RWA-ready; current asset_type: 'property').
- Build PoVal p1 attestation (ACJ-1 canonical JSON) and size-guard.
- Optional on-chain publish (Algorand) and One-Click Verify.
- Lightweight security: bearer auth, rate limit, body size guard.
- Logging without PII (redacted fields), audit bundle export (zip).

NOTE:
- ACJ-1 canonicalization uses sorted keys + compact separators + UTF-8 bytes.
- PoVal p1 MUST respect NOTE_MAX_BYTES to fit Algorand note field.
- Verify should read historical tx via Indexer (Algod is only for pending).

Run locally:
    uvicorn scripts.inference_api:app --reload --port 8000
"""

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
# NOTE: Support import from both notebooks/shared and shared on sys.path.
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

# Runtime flags
STRICT_RAW_FEATURES = os.getenv("STRICT_RAW_FEATURES", "1").lower() in {"1", "true", "yes", "y"}
ALLOW_BASELINE_FALLBACK = os.getenv("ALLOW_BASELINE_FALLBACK", "0").lower() in {"1", "true", "yes", "y"}
INFERENCE_DEBUG = os.getenv("INFERENCE_DEBUG", "0").lower() in {"1", "true", "yes", "y"}
REDACT_API_LOGS = os.getenv("REDACT_API_LOGS", "1").lower() in {"1", "true", "yes", "y"}

# Paths base → notebooks/outputs
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", "notebooks/outputs"))
LOGS_DIR = Path(os.getenv("AI_ORACLE_LOG_DIR", OUTPUTS_DIR / "logs"))
LOGS_DIR.mkdir(parents=True, exist_ok=True)
API_LOG_PATH = LOGS_DIR / "api_inference_log.jsonl"
registry = AttestationRegistry()

# Schemas (configurabili)
SCHEMAS_DIR = Path(os.getenv("SCHEMAS_DIR", "schemas"))
OUTPUT_SCHEMA = json.loads((SCHEMAS_DIR / "output_schema_v2.json").read_text(encoding="utf-8"))
SCHEMA_VERSION = "v2"
P1_SCHEMA = json.loads((SCHEMAS_DIR / "poval_v1_compact.schema.json").read_text(encoding="utf-8"))

# PoVal & timing guards
NOTE_MAX_BYTES = int(os.getenv("NOTE_MAX_BYTES", "1024"))
P1_TS_SKEW_PAST = int(os.getenv("P1_TS_SKEW_PAST", "600"))  # 10 min
P1_TS_SKEW_FUTURE = int(os.getenv("P1_TS_SKEW_FUTURE", "120"))  # 2 min

# =============================================================================
# Security MUST: Auth + Rate-limit + Body-limit
# =============================================================================
# --- Auth (Bearer) ---
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
        # NOTE: extremely defensive fallback
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _unwrap_for_introspection(obj: Any) -> Any:
    """Traverse nested estimators/pipelines to expose inner nodes for inspection."""
    tried = set()

    def _walk(x):
        if id(x) in tried or x is None:
            return
        tried.add(id(x))
        yield x
        for attr in (
            "best_estimator_",
            "estimator",
            "regressor_",
            "regressor",
            "pipeline",
            "model",
            "final_estimator",
        ):
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
    """NOTE: Introspect ColumnTransformer to collect known categories for normalization."""
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
                len(_KNOWN_CATS["city"]),
                len(_KNOWN_CATS["region"]),
                len(_KNOWN_CATS["zone"]),
            )
    except Exception as e:
        logger.debug("Could not introspect categories: %s", e)

def _to_macro_region(value: Optional[str], city: Optional[str] = None) -> Optional[str]:
    """Map Italian region names/cities to macro area (north/center/south) when possible."""
    if not value and not city:
        return value
    if value:
        v = _slug(str(value))
        if v in {"north", "center", "south"}:
            return value.strip().lower()
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

# SECURITY: Sensitive keys to redact from logs.
_SENSITIVE_KEYS = {
    "address",
    "note",
    "notes",
    "email",
    "phone",
    "lat",
    "lon",
    "lng",
    "latitude",
    "longitude",
    "coordinates",
    "gps",
    "contact",
}

def _redact(obj: Any) -> Any:
    """SECURITY: Recursively redact sensitive fields (no PII in logs)."""
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
    """Run prediction on a single-row DataFrame, with baseline fallback if enabled."""
    if hasattr(pipeline_obj, "predict"):
        y = pipeline_obj.predict(df)
        if isinstance(y, (list, tuple, np.ndarray)):
            return float(np.ravel(y)[0])
        return float(y)
    if callable(pipeline_obj):
        return float(pipeline_obj(df))
    if ALLOW_BASELINE_FALLBACK:
        s = df.get("size_m2")
        if s is not None and len(s) > 0 and pd.notna(s.iloc[0]):
            return float(max(1.0, 0.8 * float(s.iloc[0])))
    raise RuntimeError("Model object has no predict/callable interface")

_EXPECTED_FEATURES: Optional[List[str]] = None

def collect_expected_features(
    meta: Optional[dict],
    pipeline: Optional[object],
    payload_keys: Optional[List[str]] = None,
) -> List[str]:
    """
    Resolve expected raw input feature list using (priority):
    1) meta.feature_order, 2) meta.expected_features/features/input_features,
    3) pipeline raw ColumnTransformer columns, 4) pipeline.get_feature_names_out (post-encoding!),
    5) payload_keys fallback.
    """
    if isinstance(_EXPECTED_FEATURES, list) and _EXPECTED_FEATURES:
        logger.info(
            "expected_features resolved: source=%s count=%d sample=%s",
            "test_override(_EXPECTED_FEATURES)",
            len(_EXPECTED_FEATURES),
            _EXPECTED_FEATURES[:10],
        )
        return list(_EXPECTED_FEATURES)

    # Priority 0: explicit order → raw → input hashing consistency
    try:
        fo = (meta or {}).get("feature_order")
        if isinstance(fo, list) and fo:
            seen = set()
            ordered = [str(x) for x in fo if not (x in seen or seen.add(x))]
            logger.info(
                "expected_features resolved: source=%s count=%d sample=%s",
                "meta.feature_order",
                len(ordered),
                ordered[:10],
            )
            return ordered
    except Exception:
        pass

    def _post_enc(src: str) -> bool:
        return "post-encoding" in src

    cands: List[Tuple[str, List[str]]] = []
    if isinstance(meta, dict):
        for k in ("expected_features", "features", "input_features"):
            v = meta.get(k)
            if isinstance(v, list) and v:
                cands.append((f"meta.{k}", v))
        prep = meta.get("preprocessing")
        if isinstance(prep, dict):
            v = prep.get("feature_names_out") or prep.get("feature_names")
            if isinstance(v, list) and v:
                cands.append(("meta.preprocessing.feature_names_out/feature_names (post-encoding!)", v))

    if pipeline is not None:
        try:
            from sklearn.compose import ColumnTransformer  # type: ignore
            from sklearn.pipeline import Pipeline  # type: ignore

            if isinstance(pipeline, Pipeline):
                cols: List[str] = []
                for _, step in pipeline.steps:
                    if isinstance(step, ColumnTransformer):
                        for _tname, _trans, cols_in in step.transformers:
                            if cols_in in (None, "drop"):
                                continue
                            if isinstance(cols_in, (list, tuple, np.ndarray)):
                                cols.extend([str(c) for c in cols_in])
                if cols:
                    _seen = set()
                    raw_cols = [c for c in cols if not (c in _seen or _seen.add(c))]
                    logger.info(
                        "expected_features resolved: source=%s count=%d sample=%s",
                        "pipeline.ColumnTransformer(raw input)",
                        len(raw_cols),
                        raw_cols[:10],
                    )
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

    if isinstance(payload_keys, list) and payload_keys:
        logger.info(
            "expected_features resolved: source=%s count=%d sample=%s",
            "payload_keys (fallback)",
            len(payload_keys),
            payload_keys[:10],
        )
        return list(payload_keys)

    return []

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
    """NOTE: Compute cheap derived fields and normalize location/city/region/zone."""
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
            r[LOCATION] = canonical_location(r)
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
        if (
            "is_top_floor" not in r
            and r.get("floor") is not None
            and r.get("building_floors") is not None
        ):
            r["is_top_floor"] = int(r.get("floor") == r.get("building_floors"))
    except Exception:
        pass

    # City normalization (via synonyms)
    try:
        prop_cfg = ASSET_CONFIG["property"]
        city_syn = {
            str(k).strip().lower(): str(v).strip().title()
            for k, v in (prop_cfg.get("city_synonyms") or {}).items()
        }
        if not r.get("city"):
            loc = str(r.get(LOCATION, "")).strip()
            if loc:
                key = loc.lower()
                r["city"] = city_syn.get(key, loc.title())
        else:
            key = str(r["city"]).strip().lower()
            r["city"] = city_syn.get(key, str(r["city"]).strip().title())
    except Exception:
        if r.get("city"):
            r["city"] = str(r["city"]).strip().title()

    # Region macro and normalization
    try:
        prop_cfg = ASSET_CONFIG["property"]
        by_city = (prop_cfg.get("region_by_city") or DEFAULT_REGION_BY_CITY) or {}
        if r.get("region"):
            r["region"] = _to_macro_region(str(r["region"]), r.get("city"))
        else:
            if r.get("city"):
                r["region"] = str(by_city.get(str(r["city"]).strip().title(), "")).lower() or None
        if r.get("region"):
            r["region"] = _normalize_to_known("region", r["region"])
    except Exception:
        pass

    # Urban type by city
    try:
        prop_cfg = ASSET_CONFIG["property"]
        urb_by_city = (prop_cfg.get("urban_type_by_city") or DEFAULT_URBAN_TYPE_BY_CITY) or {}
        if not r.get("urban_type") and r.get("city"):
            r["urban_type"] = urb_by_city.get(str(r["city"]).strip().title())
    except Exception:
        pass

    # Zone from thresholds or normalization
    try:
        z = r.get("zone")
        if z:
            r["zone"] = _normalize_to_known("zone", z)
        else:
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

    try:
        if r.get("city"):
            r["city"] = _closest_known("city", str(r["city"]))
    except Exception:
        pass
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
    """Try to extract (final_base_estimator, TransformedTargetRegressor_if_any)."""
    ttr = None
    candidate = model
    for node in _unwrap_for_introspection(model):
        if node.__class__.__name__ == "TransformedTargetRegressor":
            ttr = node
        if hasattr(node, "estimators_"):
            candidate = node
    return candidate, ttr

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
    df = pd.DataFrame([{k: record.get(k, np.nan) for k in all_expected}], columns=all_expected)
    try:
        nonnull_total = int(df.notnull().sum().sum())
        logger.info("DF shape=%s nonnull=%s", df.shape, nonnull_total)
        logger.info("First 10 cols: %s", all_expected[:10])
        if nonnull_total == 0:
            logger.warning(
                "All values are NaN after aligning with expected features. Probable raw vs post-encoding mismatch."
            )
    except Exception:
        pass

    y_hat = float(_predict_one(pipeline_obj, df))

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

        # NOTE: Estimate CI from per-tree variance (if forest-like model available).
        if hasattr(model_final, "estimators_") and isinstance(model_final.estimators_, (list, tuple)) and len(model_final.estimators_) >= 3:
            per_tree_raw = np.array(
                [float(np.ravel(est.predict(X))[0]) for est in model_final.estimators_], dtype=float
            )
            per_tree = _inverse_target_if_needed(per_tree_raw, ttr)
            m = float(per_tree.mean())
            s = float(per_tree.std(ddof=1)) if len(per_tree) > 1 else 0.0
            z = _z_for_conf(confidence)
            ci = z * s
            return {
                "prediction": round(m, 2),
                "point_pred": round(y_hat, 2),
                "uncertainty": round(s, 2),
                "confidence": float(confidence),
                "confidence_interval": (round(m - ci, 2), round(m + ci, 2)),
                "ci_margin": round(ci, 2),
                "method": "forest_variance",
                "n_estimators": len(model_final.estimators_),
            }

        raise RuntimeError("No per-tree variance available")
    except Exception as e:
        logger.debug("Forest variance unavailable (%s); using global sigma fallback.", e)
        # PERF: Cheap fallback — use validation RMSE/MAE or 10% of |y_hat| as sigma.
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
            "prediction": round(y_hat, 2),
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

    # 1) Load fitted artifacts + meta
    try:
        pipeline = get_pipeline(asset_type, "value_regressor")
        try:
            paths = get_model_paths(asset_type, "value_regressor")
        except Exception:
            paths = {}
        meta = get_model_metadata(asset_type, "value_regressor") or {}

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
    except (RegistryLookupError, ModelNotFoundError) as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
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
    except Exception as e:
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
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

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
            "value_model_name": meta.get("model_class")
            or type(getattr(pipeline, "steps", [[None, pipeline]])[-1][1]).__name__,
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

    # Optional: breakdown & sanity benchmark
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

    # 6) Build PoVal p1 (always)
    canonical_input_subset = {k: rec.get(k, None) for k in all_expected}
    response["canonical_input"] = canonical_input_subset  # used by builder as clean fallback
    p1, dbg = build_p1_from_response(response, allowed_input_keys=all_expected)
    p1_bytes, p1_sha, p1_size = canonical_note_bytes_p1(p1)

    # Guardrail: note size
    if int(p1_size) > NOTE_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"p1_too_large: {p1_size}B > NOTE_MAX_BYTES={NOTE_MAX_BYTES}",
        )

    # Anti-replay (asset_id + p1 sha256)
    asset_id = response["asset_id"]
    if registry.seen(p1_sha, asset_id):
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
            result = publish_ai_prediction(response, p1_bytes=p1_bytes, p1_sha256=p1_sha)
        except TypeError:
            # NOTE: Backward-compat signature fallback (update publisher signature to accept p1_*).
            result = publish_ai_prediction(response)

            # WARN: This assignment block currently lives ONLY in the TypeError fallback.
            # If publish_ai_prediction succeeds with new signature, publish fields below will not be set.
            # Consider moving the block outside the except to apply in both cases.
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

            # Backward compatibility fields
            response["blockchain_txid"] = response["publish"]["txid"]
            response["asa_id"] = response["publish"]["asa_id"]

            # Audit trace & anti-replay record
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
                    result.get("blockchain_txid"),
                    get_network(),
                    issuer,
                    int(p1.get("ts", 0)),
                )
                (bundle_dir / "tx.txt").write_text(str(result.get("blockchain_txid") or ""), encoding="utf-8")
            except Exception:
                pass
        except Exception as e:
            response["publish"] = {"status": "error", "error": str(e)}

    # 8) Schema validation v2 (best-effort)
    if OUTPUT_SCHEMA:
        try:
            jsonschema_validate(instance=response, schema=OUTPUT_SCHEMA)
        except ValidationError as ve:
            response["schema_validation_error"] = str(ve).split("\n", 1)[0][:240]
        except Exception as e:
            response["schema_validation_error"] = f"Schema check failed: {e}"[:240]

    # 9) API log (best-effort)
    try:
        rec_log = {
            "event": "prediction",
            "asset_type": asset_type,
            # SECURITY: redact potential PII from inbound payloads.
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
            "_logged_at": utc_now_iso_z(),  # helpful for UI sorting
        }

        # Optional: include a light subset of response
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
        # WARN: Do not break inference on logging issues.
        print(f"[warn] log_jsonl failed: {e}")

    # 10) attestation_only: return only PoVal p1 + minimal metadata
    if attestation_only:
        return {
            "asset_id": response["asset_id"],
            "attestation": response["attestation"]["p1"],
            "attestation_sha256": response["attestation"]["p1_sha256"],
            "attestation_size": response["attestation"]["p1_size_bytes"],
            "published": response.get("publish", {}).get("status") == "ok",
            "txid": response.get("blockchain_txid") or None,
            "network": net if publish else None,
            "audit_bundle_path": response.get("audit_bundle_path"),
        }

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

    NOTE:
    - get_tx_note_info(txid) should prefer Indexer for historical tx and fallback to Algod for pending ones.
    - Indexer ingestion may lag by a few seconds (add retry/backoff in algorand_utils).
    """
    txid = str(body.get("txid") or "").strip()
    if not txid:
        raise HTTPException(422, "Missing 'txid'")

    # 0) On-chain lookup
    note = get_tx_note_info(txid)  # NOTE: Indexer vs Algod behavior defined in algorand_utils
    note_json = note.get("note_json")
    onchain_sha = note.get("note_sha256")
    explorer_url = note.get("explorer_url")

    # 1) Optional expected values from client (sha or p1)
    expected_sha = body.get("attestation_sha256") or body.get("expected_sha256")
    p1_from_body = None
    att = body.get("attestation") if isinstance(body.get("attestation"), dict) else None
    if att and isinstance(att.get("p1"), dict):
        p1_from_body = att["p1"]
    elif isinstance(body.get("prediction"), dict):
        att2 = body["prediction"].get("attestation")
        if isinstance(att2, dict) and isinstance(att2.get("p1"), dict):
            p1_from_body = att2["p1"]

    # Derive expected sha from provided p1 if needed
    if expected_sha is None and isinstance(p1_from_body, dict):
        try:
            # NOTE: derive expected hash using ACJ-1 canonical bytes from client-provided p1.
            _, expected_sha, _ = canonical_note_bytes_p1(p1_from_body)
        except Exception:
            expected_sha = None  # continue anyway

    ok = False
    reason = None
    mode = None

    try:
        # --- PoVal p1 (compact) ------------------------------------------------
        if isinstance(note_json, dict) and note_json.get("s") == "p1":
            mode = "p1"

            # Schema p1
            try:
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

            # NOTE: Timestamp window is an information signal, NOT a hard fail.
            time_out_of_window = False
            try:
                now = int(time.time())
                ts_sec = _ts_to_seconds(note_json["ts"])
                if ts_sec < (now - P1_TS_SKEW_PAST) or ts_sec > (now + P1_TS_SKEW_FUTURE):
                    time_out_of_window = True
            except Exception:
                time_out_of_window = False

            # Value inside declared confidence interval
            v = float(note_json["v"])
            lo, hi = float(note_json["u"][0]), float(note_json["u"][1])
            if not (lo <= v <= hi):
                ok, reason = False, f"value_out_of_range:{v}∉[{lo},{hi}]"
            else:
                ok = True

            # Compare expected vs on-chain sha (if provided)
            if ok and expected_sha and onchain_sha and expected_sha != onchain_sha:
                ok = False
                reason = "sha_mismatch"

            # Optional hardening: recompute on-chain note ACJ-1 bytes → sha, compare
            if ok and onchain_sha:
                try:
                    from scripts.canon import canonicalize_jcs
                    bytes_rebuilt = canonicalize_jcs(note_json)
                    rebuilt_sha = hashlib.sha256(bytes_rebuilt).hexdigest()
                    if rebuilt_sha != onchain_sha:
                        # WARN: Non-canonical or altered note — fail verification.
                        ok = False
                        reason = "onchain_hash_mismatch"
                except Exception:
                    # Non-blocking: mismatch already covered above.
                    pass

            # If ok but timestamp outside window → annotate in debug only
            if ok and time_out_of_window and INFERENCE_DEBUG:
                note.setdefault("debug", {})  # type: ignore[assignment]
                note["debug"]["timestamp_out_of_window"] = True  # type: ignore[index]

        # --- Legacy ------------------------------------------------------------
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
        # SECURITY: Do not expose full on-chain payload in production.
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
