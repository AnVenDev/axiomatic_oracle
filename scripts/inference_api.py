"""
inference_api.py
FastAPI service exposing AI Oracle inference endpoints (multi-RWA ready: initial asset_type 'property').

Run locally:
    uvicorn scripts.inference_api:app --reload --port 8000

Endpoints:
    GET  /health
    POST /predict/{asset_type}?publish=false

Roadmap:
    - Add anomaly detection
    - Real Algorand publish integration
    - Add additional asset types and request models
    - Auth / rate limiting / version negotiation
"""

from __future__ import annotations

import pandas as pd
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Path as FPath, Query, Body
from pydantic import BaseModel, Field, model_validator

from scripts.model_registry import (
    get_pipeline,
    get_model_metadata,
    RegistryLookupError,
    ModelNotFoundError
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
LOG_PATH = Path("data/api_inference_log.jsonl")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

ALLOWED_ENERGY_CLASSES = {"A", "B", "C", "D", "E", "F", "G"}

app = FastAPI(
    title="AI Oracle Inference API",
    version="0.1.0",
    description="Inference service for multi-RWA asset valuation (initial: property)."
)

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
    age_years: int = Field(..., ge=0)

    @model_validator(mode="after")
    def validate_building_and_energy(self):
        # floor consistency
        if self.floor >= self.building_floors:
            raise ValueError("floor must be < building_floors")
        # energy class check
        if self.energy_class not in ALLOWED_ENERGY_CLASSES:
            raise ValueError(f"energy_class must be one of {sorted(ALLOWED_ENERGY_CLASSES)}")
        return self


REQUEST_MODELS = {
    "property": PropertyPredictRequest
}

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def log_jsonl(record: dict):
    record["_logged_at"] = datetime.utcnow().isoformat() + "Z"
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def build_response(
    asset_type: str,
    valuation_k: float,
    model_meta: dict,
    publish: bool,
    asset_id: Optional[str] = None
):
    if not asset_id:
        asset_id = f"{asset_type}_{uuid.uuid4().hex[:10]}"

    response = {
        "asset_id": asset_id,
        "asset_type": asset_type,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "metrics": {
            "valuation_base_k": round(float(valuation_k), 3)
        },
        "flags": {
            "anomaly": False,
            "needs_review": False
        },
        "model_meta": {
            "value_model_version": model_meta.get("model_version"),
            "value_model_name": model_meta.get("model_class")
        },
        "offchain_refs": {
            "detail_report_hash": None,
            "sensor_batch_hash": None
        }
    }

    if publish:
        # Simulated publish stub
        response["publish"] = {
            "status": "simulated",
            "txid": f"SIM-{uuid.uuid4().hex[:16]}"
        }

    return response


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    try:
        meta = get_model_metadata("property", "value_regressor")
        return {
            "status": "ok",
            "asset_types": list(REQUEST_MODELS.keys()),
            "property_model_version": meta.get("model_version") if meta else None,
            "time": datetime.utcnow().isoformat() + "Z"
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.post("/predict/{asset_type}")
def predict(
    asset_type: str = FPath(..., description="Asset type (e.g., property)"),
    publish: bool = Query(False, description="Simulate on-chain publish"),
    payload: dict = Body(...)
):
    asset_type = asset_type.lower()

    if asset_type not in REQUEST_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported asset_type: {asset_type}")

    ModelCls = REQUEST_MODELS[asset_type]

    # Validate input -> Pydantic model
    try:
        req_obj = ModelCls(**payload)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid payload: {e}")

    # Load pipeline + metadata
    try:
        pipeline = get_pipeline(asset_type, "value_regressor")
        meta = get_model_metadata(asset_type, "value_regressor") or {}
    except (RegistryLookupError, ModelNotFoundError) as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Prediction
    start = time.time()
    df_input = pd.DataFrame([req_obj.model_dump()])
    pred = pipeline.predict(df_input)[0]
    latency_ms = (time.time() - start) * 1000

    response = build_response(
        asset_type=asset_type,
        valuation_k=pred,
        model_meta=meta,
        publish=publish
    )
    response["latency_ms"] = round(latency_ms, 2)

    # Log
    log_jsonl({
        "event": "prediction",
        "asset_type": asset_type,
        "request": req_obj.model_dump(),
        "response": response
    })

    return response


# -----------------------------------------------------------------------------
# Main (diagnostic run)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # For direct execution (optional)
    import uvicorn
    uvicorn.run("scripts.inference_api:app", host="127.0.0.1", port=8000, reload=True)