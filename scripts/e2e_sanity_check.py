"""
e2e_sanity_check.py
End-to-end sanity check for AI Oracle pipeline.
"""

from __future__ import annotations
import json
import time
import math
import hashlib
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd          # <--- assicurati ci sia
import requests
from jsonschema import validate as jsonschema_validate, ValidationError

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DATASET_PATH = Path("data/property_dataset_v1.csv")
MODEL_META_PATH = Path("models/property/value_regressor_v1_meta.json")
MODEL_PATH = Path("models/property/value_regressor_v1.joblib")
SCHEMA_PATH = Path("schemas/output_example.json")
LOG_PATH = Path("data/api_inference_log.jsonl")
API_BASE = "http://127.0.0.1:8000"
ASSET_TYPE = "property"
TOLERANCE_K = 25.0

REQUIRED_DATASET_COLUMNS = {
    "asset_id","asset_type","location","size_m2","rooms","bathrooms","year_built",
    "age_years","floor","building_floors","has_elevator","has_garden","has_balcony","garage",
    "energy_class","humidity_level","temperature_avg","noise_level","air_quality_index",
    "valuation_k","condition_score","risk_score","last_verified_ts"
}

# ---------------------------------------------------------------------
def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def section(title: str):
    print(f"\n=== {title} ===")

def fail(msg: str, failures: list):
    print(f"[FAIL] {msg}")
    failures.append(msg)

def ok(msg: str):
    print(f"[OK] {msg}")

def warn(msg: str):
    print(f"[WARN] {msg}")

# ---------------------------------------------------------------------
def main():
    failures = []
    df = None
    local_pred = None
    api_prediction_data = None
    schema = None
    sample = None

    # DATASET
    section("DATASET")
    if not DATASET_PATH.exists():
        fail(f"Dataset not found: {DATASET_PATH}", failures)
    else:
        try:
            df = pd.read_csv(DATASET_PATH)
            missing = REQUIRED_DATASET_COLUMNS - set(df.columns)
            if missing:
                fail(f"Missing required columns: {sorted(missing)}", failures)
            else:
                ok(f"Dataset loaded rows={len(df)} cols={len(df.columns)}")
        except Exception as e:
            fail(f"Error reading dataset: {e}", failures)

    # MODEL
    section("MODEL")
    if not MODEL_PATH.exists():
        fail(f"Model file missing: {MODEL_PATH}", failures)
    else:
        ok(f"Model file found: {MODEL_PATH.name} (hash={file_sha256(MODEL_PATH)[:16]})")
    if not MODEL_META_PATH.exists():
        fail(f"Metadata file missing: {MODEL_META_PATH}", failures)
    else:
        try:
            meta = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
            ok("Metadata loaded.")
            r2 = meta.get("metrics", {}).get("r2")
            if r2 is None:
                warn("R² not in metadata.")
            else:
                ok(f"R²={r2}")
        except Exception as e:
            fail(f"Cannot parse metadata: {e}", failures)

    # LOCAL PREDICTION
    section("LOCAL PREDICTION")
    if df is None:
        fail("Dataset not loaded -> skipping local prediction.", failures)
    else:
        try:
            import joblib
            pipe = joblib.load(MODEL_PATH)
            sample = df.sample(1, random_state=42).iloc[0].to_dict()
            for drop_key in ["valuation_k","condition_score","risk_score","last_verified_ts",
                             "asset_id","asset_type"]:
                sample.pop(drop_key, None)
            if "energy_class" not in sample:
                fail("energy_class missing in sample row", failures)
            else:
                local_pred = float(pipe.predict(pd.DataFrame([sample]))[0])
                ok(f"Local prediction valuation_k={local_pred:.3f}")
        except Exception as e:
            fail(f"Local prediction failed: {e}", failures)

    # SCHEMA
    section("SCHEMA")
    if not SCHEMA_PATH.exists():
        fail(f"Schema file missing: {SCHEMA_PATH}", failures)
    else:
        try:
            schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
            if "schema_version" in schema["properties"]:
                ok("Schema loaded with schema_version property.")
            else:
                warn("schema_version property not found in schema.")
        except Exception as e:
            fail(f"Error reading schema: {e}", failures)

    # API HEALTH
    section("API HEALTH")
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        if resp.status_code == 200:
            hjson = resp.json()
            if hjson.get("status") == "ok":
                ok("Health endpoint OK.")
            else:
                warn(f"Health status != ok ({hjson.get('status')})")
        else:
            fail(f"/health HTTP {resp.status_code}", failures)
    except Exception as e:
        fail(f"Health request failed: {e}", failures)

    # API PREDICTION
    section("API PREDICTION")
    if sample is None:
        fail("No sample available for API prediction.", failures)
    else:
        api_payload = dict(sample)
        if "age_years" not in api_payload and "year_built" in api_payload:
            api_payload["age_years"] = datetime.utcnow().year - int(api_payload["year_built"])
        try:
            r = requests.post(f"{API_BASE}/predict/{ASSET_TYPE}?publish=true",
                              json=api_payload, timeout=8)
            if r.status_code != 200:
                fail(f"API prediction failed HTTP {r.status_code} body={r.text}", failures)
            else:
                api_prediction_data = r.json()
                if "schema_validation_error" in api_prediction_data:
                    fail(f"API schema validation reported: {api_prediction_data['schema_validation_error']}", failures)
                else:
                    ok("API prediction returned (no schema_validation_error).")
                if "metrics" in api_prediction_data and "valuation_base_k" in api_prediction_data["metrics"]:
                    ok("valuation_base_k present in API metrics.")
                else:
                    fail("Missing valuation_base_k in API metrics", failures)
                if schema:
                    try:
                        jsonschema_validate(api_prediction_data, schema)
                        ok("API payload conforms to schema.")
                    except ValidationError as ve:
                        fail(f"API response fails schema: {ve.message}", failures)
        except Exception as e:
            fail(f"API request error: {e}", failures)

    # PREDICTION CONSISTENCY
    section("PREDICTION CONSISTENCY")
    if local_pred is not None and api_prediction_data:
        api_val = api_prediction_data["metrics"]["valuation_base_k"]
        diff = abs(api_val - local_pred)
        if diff <= TOLERANCE_K:
            ok(f"Local vs API diff acceptable (Δ={diff:.3f} k€)")
        else:
            fail(f"Local vs API diff TOO LARGE (Δ={diff:.3f} k€ > {TOLERANCE_K})", failures)
    else:
        warn("Skipping consistency (missing local or API prediction).")

    # LOGGING
    section("LOGGING")
    if not LOG_PATH.exists():
        fail(f"Log file missing: {LOG_PATH}", failures)
    else:
        try:
            lines = LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
            if not lines:
                fail("Log file empty.", failures)
            else:
                recent_cutoff = datetime.utcnow() - timedelta(minutes=5)
                found_recent = False
                for line in reversed(lines[-50:]):
                    try:
                        rec = json.loads(line)
                        ts = rec.get("_logged_at")
                        if ts:
                            dt = datetime.fromisoformat(ts.replace("Z", ""))
                            if dt >= recent_cutoff and rec.get("event") == "prediction":
                                found_recent = True
                                break
                    except Exception:
                        continue
                if found_recent:
                    ok("Recent prediction log entry found.")
                else:
                    fail("No recent prediction log entry (<5 min).", failures)
        except Exception as e:
            fail(f"Error reading log: {e}", failures)

    # SUMMARY
    section("SUMMARY")
    if failures:
        print("❌ E2E SANITY CHECK FAILED")
        for f in failures:
            print(" -", f)
        exit(1)
    else:
        print("✅ E2E SANITY CHECK PASSED")


if __name__ == "__main__":
    main()