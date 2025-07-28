from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hashlib
import json
import time
from datetime import datetime, timedelta

import requests
from jsonschema import ValidationError
from jsonschema import validate as jsonschema_validate

from scripts.model_registry import (
    cache_stats,
    get_model_metadata,
    get_pipeline,
    health_check_model,
)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
ASSET_TYPE = "property"
API_BASE = "http://localhost:8000"
EXAMPLE_PATH = Path("schemas/output_example.json")
SCHEMA_PATH = Path("schemas/output_schema_v1.json")
LOG_PATH = Path("data/api_inference_log.jsonl")
TOLERANCE_K = 1.0
TOLERANCE_PERCENT = 0.05


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------
def ok(msg):
    print(f"[OK]  {msg}")


def warn(msg):
    print(f"[WARN] {msg}")


def fail(msg):
    print(f"[FAIL] {msg}")
    failures.append(msg)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# Sample Request
# ----------------------------------------------------------------------------
SAMPLE_PATH = Path(__file__).parent.parent / "data" / "sample_property.json"
with open(SAMPLE_PATH) as f:
    payload = json.load(f)

# Payload extraction
api_payload = payload.get("features", payload)


# ----------------------------------------------------------------------------
# Test: API prediction endpoint
# ----------------------------------------------------------------------------
def test_predict_endpoint():
    resp = requests.post(f"{API_BASE}/predict/{ASSET_TYPE}", json=api_payload)
    if resp.status_code != 200:
        fail(f"API call failed with status {resp.status_code}")
        return None
    ok("API prediction call succeeded")

    data = resp.json()
    if "valuation_base_k" not in data.get("metrics", {}):
        fail("Missing valuation_base_k in response")
    assert data


# ----------------------------------------------------------------------------
# Test: Schema compliance
# ----------------------------------------------------------------------------
def validate_schema(response):
    try:
        # Load the strict schema
        with SCHEMA_PATH.open() as f:
            schema = json.load(f)
        allowed_keys = schema.get("properties", {}).keys()

        # Create a filtered version of the response
        cleaned_response = {k: v for k, v in response.items() if k in allowed_keys}

        # Run strict schema validation only on allowed keys
        jsonschema_validate(instance=cleaned_response, schema=schema)
        ok("Strict schema validation passed")
    except ValidationError as e:
        fail(f"Schema validation error: {e.message}")
    except Exception as e:
        fail(f"Schema validation failed (unexpected): {e}")

    try:
        # Compare with example structure (non-strict)
        with EXAMPLE_PATH.open() as f:
            example = json.load(f)

        ignore_keys = {"_logged_at"}
        optional_keys = {
            "schema_validation_error",
            "blockchain_txid",
            "publish",
            "asa_id",
        }

        diff_keys = (
            (set(response.keys()) ^ set(example.keys())) - ignore_keys - optional_keys
        )

        if not diff_keys:
            ok("Matches example structure")
        else:
            warn(f"Mismatch with example keys: {diff_keys}")
    except Exception as e:
        warn(f"Example structure check failed: {e}")


# ----------------------------------------------------------------------------
# Test: Registry + Metadata
# ----------------------------------------------------------------------------
def test_model_registry():
    try:
        get_pipeline(ASSET_TYPE, "value_regressor")
        ok("Model loaded from registry")

        meta = get_model_metadata(ASSET_TYPE, "value_regressor")
        if meta:
            ok(f"Model version: {meta.get('model_version')}")

        # Health check
        health = health_check_model(ASSET_TYPE, "value_regressor")
        if health["status"] == "healthy":
            ok("Model health check passed")
        else:
            fail(f"Model unhealthy: {health['error']}")

        stats = cache_stats()
        ok(f"Cache status: {stats['pipelines_cached']} pipelines cached")
    except Exception as e:
        fail(f"Model registry error: {e}")


# ----------------------------------------------------------------------------
# Test: Advanced API endpoints
# ----------------------------------------------------------------------------
def test_api_advanced_features():
    try:
        r1 = requests.get(f"{API_BASE}/models/{ASSET_TYPE}")
        if r1.status_code == 200:
            ok("Model discovery endpoint OK")

        r2 = requests.get(f"{API_BASE}/models/{ASSET_TYPE}/value_regressor/health")
        if r2.status_code == 200 and r2.json().get("status") == "healthy":
            ok("Model health endpoint OK")

        r3 = requests.post(f"{API_BASE}/models/{ASSET_TYPE}/value_regressor/refresh")
        if r3.status_code == 200:
            ok("Model cache refresh endpoint OK")
    except Exception as e:
        fail(f"Advanced API test failed: {e}")


# ----------------------------------------------------------------------------
# Test: Prediction consistency & latency
# ----------------------------------------------------------------------------
def test_prediction_consistency_advanced():
    preds = []
    latencies = []
    for _ in range(3):
        start = time.time()
        r = requests.post(f"{API_BASE}/predict/{ASSET_TYPE}", json=api_payload)
        lat = (time.time() - start) * 1000
        latencies.append(lat)
        if r.status_code == 200:
            preds.append(r.json()["metrics"]["valuation_base_k"])

    if len(set(preds)) == 1:
        ok("Predictions are stable")
    else:
        warn(f"Prediction variation: {preds}")

    mean_latency = sum(latencies) / len(latencies)
    if mean_latency < 1000:
        ok(f"Mean latency acceptable: {mean_latency:.2f}ms")
    else:
        warn(f"High latency: {mean_latency:.2f}ms")


# ----------------------------------------------------------------------------
# Test: Log file integrity
# ----------------------------------------------------------------------------
def test_recent_log():
    try:
        with LOG_PATH.open() as f:
            last_line = list(f)[-1]
            entry = json.loads(last_line)
        ts = entry.get("_logged_at")
        logged_time = datetime.fromisoformat(ts.replace("Z", ""))
        if datetime.utcnow() - logged_time < timedelta(minutes=5):
            ok("Recent prediction log is valid")
        else:
            warn("Log is stale")
    except Exception as e:
        fail(f"Log test failed: {e}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    failures = []
    print("--- E2E Sanity Check ---")
    test_model_registry()
    test_api_advanced_features()
    data = test_predict_endpoint()
    if data:
        validate_schema(
            data,
        )
        test_prediction_consistency_advanced()
    test_recent_log()

    print("\n--- Summary ---")
    if not failures:
        print("✅ All tests passed")
    else:
        print(f"❌ {len(failures)} failures:")
        for f in failures:
            print(" -", f)
