"""
FastAPI service exposing AI Oracle inference endpoints 
(multi-RWA ready: initial asset_type 'property').

Run locally:
    uvicorn scripts.inference_api:app --reload --port 8000
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi import Path as FPath
from jsonschema import ValidationError
from jsonschema import validate as jsonschema_validate
from pydantic import BaseModel, Field, model_validator

from scripts.blockchain_publisher import publish_ai_prediction
from scripts.model_registry import (
    _PIPELINE_TTL_CACHE,
    ModelNotFoundError,
    RegistryLookupError,
    cache_stats,
    discover_models_for_asset,
    get_model_metadata,
    get_pipeline,
    health_check_model,
    list_tasks,
    refresh_cache,
    validate_model_compatibility,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
LOG_PATH = Path("data/api_inference_log.jsonl")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

SCHEMA_PATH = Path("schemas/output_schema_v1.json")
EXAMPLE_PATH = Path("schemas/output_example.json")

try:
    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        OUTPUT_SCHEMA = json.load(f)
    SCHEMA_VERSION = OUTPUT_SCHEMA.get("$id", "v1")
except Exception as e:
    OUTPUT_SCHEMA = {}
    SCHEMA_VERSION = "v0-fallback"
    print(f"[WARN] Could not load schema: {e}")

API_VERSION = "0.3.0"

ALLOWED_ENERGY_CLASSES = {"A", "B", "C", "D", "E", "F", "G"}


# -----------------------------------------------------------------------------
# Pydantic Request Models
# -----------------------------------------------------------------------------
class PropertyPredictRequest(BaseModel):
    location: str
    size_m2: float = Field(..., gt=0)
    rooms: int = Field(..., ge=0)
    bathrooms: int = Field(..., ge=0)
    year_built: int = Field(..., ge=1800, le=datetime.utcnow().year)
    floor: int = Field(..., ge=0)
    building_floors: int = Field(..., ge=1)
    has_elevator: int = Field(..., ge=0, le=1)
    has_garden: int = Field(..., ge=0, le=1)
    has_balcony: int = Field(..., ge=0, le=1)
    garage: int = Field(..., ge=0, le=1)
    energy_class: str
    humidity_level: float = Field(..., ge=0, le=100)
    temperature_avg: float
    noise_level: int = Field(..., ge=0, le=150)
    air_quality_index: int = Field(..., ge=0, le=500)
    age_years: Optional[int] = Field(None, ge=0)

    @model_validator(mode="after")
    def validate_and_normalize(self):
        if self.floor >= self.building_floors:
            raise ValueError("floor must be < building_floors")
        ec = self.energy_class.upper()
        if ec not in ALLOWED_ENERGY_CLASSES:
            raise ValueError(
                f"energy_class must be one of {sorted(ALLOWED_ENERGY_CLASSES)}"
            )
        object.__setattr__(self, "energy_class", ec)
        if self.age_years is None:
            object.__setattr__(
                self, "age_years", datetime.utcnow().year - self.year_built
            )
        return self


REQUEST_MODELS: Dict[str, Any] = {"property": PropertyPredictRequest}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def log_jsonl(record: dict):
    record["_logged_at"] = datetime.utcnow().isoformat() + "Z"
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def hash_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def build_response(
    asset_type: str,
    valuation_k: float,
    model_meta: dict,
    publish: bool,
    asset_id: Optional[str] = None,
):
    if not asset_id:
        asset_id = f"{asset_type}_{uuid.uuid4().hex[:10]}"

    dataset_hash = model_meta.get("dataset_hash_sha256") or model_meta.get(
        "dataset_hash"
    )
    model_hash = None
    model_path_hint = model_meta.get("model_path")
    if model_path_hint:
        mp = Path(model_path_hint)
        if mp.exists():
            model_hash = hash_file(mp)

    # Base response
    response = {
        "schema_version": SCHEMA_VERSION,
        "asset_id": asset_id,
        "asset_type": asset_type,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "metrics": {"valuation_base_k": round(float(valuation_k), 3)},
        "flags": {"anomaly": False, "needs_review": False},
        "model_meta": {
            "value_model_version": model_meta.get("model_version"),
            "value_model_name": model_meta.get("model_class"),
        },
        "offchain_refs": {"detail_report_hash": None, "sensor_batch_hash": None},
    }

    if dataset_hash:
        response["model_meta"]["dataset_hash"] = dataset_hash
    if model_hash:
        response["model_meta"]["model_hash"] = model_hash[:32]

    # Blockchain Integration
    if publish:
        try:
            result = publish_ai_prediction(response)  # <-- chiama publisher
            response["blockchain_txid"] = result["blockchain_txid"]
            response["asa_id"] = result["asa_id"]
            response["publish"] = {
                "status": "success",
                "txid": result.get("blockchain_txid"),
                "asa_id": result.get("asa_id"),
            }
        except Exception as e:
            response["publish"] = {"status": "error", "error": str(e)}

    return response


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(
    title="AI Oracle Inference API",
    version=API_VERSION,
    description="Inference service for multi-RWA asset valuation (initial: property).",
)


@app.get("/health")
def health():
    try:
        health_info = health_check_model("property", "value_regressor")
        cache_info = cache_stats()
        return {
            "status": "ok" if health_info["status"] == "healthy" else "degraded",
            "model_health": health_info,
            "cache_stats": cache_info,
            "asset_types": list(REQUEST_MODELS.keys()),
            "schema_version": SCHEMA_VERSION,
            "api_version": API_VERSION,
            "time": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/predict/{asset_type}")
def predict(
    asset_type: str = FPath(...),
    publish: bool = Query(False),
    payload: dict = Body(...),
):
    asset_type = asset_type.lower()
    if asset_type not in REQUEST_MODELS:
        raise HTTPException(
            status_code=400, detail=f"Unsupported asset_type: {asset_type}"
        )

    ModelCls = REQUEST_MODELS[asset_type]
    try:
        req_obj = ModelCls(**payload)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid payload: {e}")

    try:
        pipeline = get_pipeline(asset_type, "value_regressor", fallback_latest=True)
        raw_meta = get_model_metadata(
            asset_type, "value_regressor", fallback_latest=True
        )
        meta = raw_meta if raw_meta else {}
        health = health_check_model(asset_type, "value_regressor")
        if health["status"] != "healthy":
            raise HTTPException(status_code=503, detail=health["error"])
    except (RegistryLookupError, ModelNotFoundError) as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Validate compatibility
    expected_features = list(req_obj.model_dump().keys())
    if not validate_model_compatibility(pipeline, expected_features):
        raise HTTPException(status_code=422, detail="Model-input schema mismatch")

    start = time.time()
    df_input = pd.DataFrame([req_obj.model_dump()])
    try:
        X = pipeline[:-1].transform(df_input)  # trasforma senza predict finale
        X = pd.DataFrame(
            X, columns=pipeline[-1].feature_name_
        )  # fornisci nomi previsti
        pred = pipeline[-1].predict(X)[0]  # LGBMRegressor

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    latency_ms = (time.time() - start) * 1000

    response = build_response(asset_type, pred, meta, publish)
    response["metrics"]["latency_ms"] = round(latency_ms, 2)
    response["model_health"] = health
    response["cache_hit"] = meta.get("model_path") in _PIPELINE_TTL_CACHE

    try:
        jsonschema_validate(instance=response, schema=OUTPUT_SCHEMA)
    except ValidationError as ve:
        response["schema_validation_error"] = str(ve).split("\n", 1)[0][:240]
    except Exception as e:
        response["schema_validation_error"] = f"Schema check failed: {e}"[:240]

    log_jsonl(
        {
            "event": "prediction",
            "asset_type": asset_type,
            "request": req_obj.model_dump(),
            "response": response,
        }
    )
    return response


@app.get("/models/{asset_type}")
def list_models(asset_type: str):
    return {
        "asset_type": asset_type,
        "tasks": list_tasks(asset_type),
        "discovered_models": [p.name for p in discover_models_for_asset(asset_type)],
    }


@app.post("/models/{asset_type}/{task}/refresh")
def refresh_model_cache(asset_type: str, task: str):
    refresh_cache(asset_type, task)
    return {"status": "cache_refreshed", "asset_type": asset_type, "task": task}


@app.get("/models/{asset_type}/{task}/health")
def model_health(asset_type: str, task: str):
    return health_check_model(asset_type, task)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("scripts.inference_api:app", host="127.0.0.1", port=8000, reload=True)
