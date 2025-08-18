# tests/test_e2e_sanity_check.py
from __future__ import annotations

import os
import sys
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest                                               # type: ignore
import requests                                             # type: ignore
from jsonschema import ValidationError                      # type: ignore
from jsonschema import validate as jsonschema_validate      # type: ignore

# Import helpers dal registry
from scripts.model_registry import (
    cache_stats,
    get_model_metadata,
    get_pipeline,
    health_check_model,
)

# ----------------------------------------------------------------------------
# Skip condition (evita rossi se l'API non è avviata)
# ----------------------------------------------------------------------------
pytestmark = pytest.mark.skipif(
    os.getenv("E2E_ENABLE", "0") != "1",
    reason="E2E disabled (set E2E_ENABLE=1 to run)",
)

# ----------------------------------------------------------------------------
# Config (allineata a notebooks/outputs)
# ----------------------------------------------------------------------------
ASSET_TYPE = os.getenv("E2E_ASSET_TYPE", "property")
API_BASE = os.getenv("E2E_API_BASE", "http://127.0.0.1:8000")

SCHEMAS_DIR = Path(os.getenv("SCHEMAS_DIR", "schemas"))
SCHEMA_V2 = SCHEMAS_DIR / "output_schema_v2.json"
SCHEMA_V1 = SCHEMAS_DIR / "output_schema_v1.json"
EXAMPLE_PATH = SCHEMAS_DIR / "output_example.json"

OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", "notebooks/outputs"))
LOG_DIR = Path(os.getenv("AI_ORACLE_LOG_DIR", OUTPUTS_DIR / "logs"))
LOG_PATH = LOG_DIR / "api_inference_log.jsonl"

# tolleranze informative (non hard-fail)
TOLERANCE_K = 1.0
TOLERANCE_PERCENT = 0.05

# Per l'esecuzione come script standalone (facoltativo)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Stato globale solo se lanci come __main__
failures: list[str] = []


# ----------------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------------
def ok(msg: str):
    print(f"[OK]  {msg}")


def warn(msg: str):
    print(f"[WARN] {msg}")


def fail(msg: str):
    print(f"[FAIL] {msg}")
    failures.append(msg)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _pick_valuation(metrics: dict | None):
    """Compat: v2 -> valuation_k ; v1 -> valuation_base_k."""
    if not isinstance(metrics, dict):
        return None
    return metrics.get("valuation_k") or metrics.get("valuation_base_k")


def _load_schema_for_response(resp: dict) -> dict | None:
    """
    Carica lo schema coerente con la response, se disponibile:
    - v2 -> output_schema_v2.json
    - v1 -> output_schema_v1.json
    Se lo schema mancante, ritorna None (skip validazione stretta).
    """
    ver = str(resp.get("schema_version") or "").lower()
    if ver == "v2" and SCHEMA_V2.exists():
        return json.loads(SCHEMA_V2.read_text(encoding="utf-8"))
    if ver == "v1" and SCHEMA_V1.exists():
        return json.loads(SCHEMA_V1.read_text(encoding="utf-8"))
    return None


def _api_up(timeout_s: float = 5.0) -> bool:
    """Best-effort: prova /health per capire se l'API è su."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=timeout_s)
        return r.status_code == 200
    except Exception:
        return False


def _parse_iso(ts: str) -> datetime:
    """Parsa ISO8601; supporta suffisso 'Z'."""
    if not ts:
        return datetime.now(timezone.utc)
    if ts.endswith("Z"):
        ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts)


# ----------------------------------------------------------------------------
# Sample Request
# ----------------------------------------------------------------------------
# Prima prova a leggere data/sample_property.json; se manca, usa un fallback coerente.
SAMPLE_PATH = ROOT / "data" / "sample_property.json"
try:
    if SAMPLE_PATH.exists():
        payload_file = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
        api_payload = payload_file.get("features", payload_file)
    else:
        raise FileNotFoundError
except Exception:
    # Fallback coerente con i pydantic models del backend
    api_payload = {
        "location": "Milan",
        "size_m2": 95,
        "rooms": 4,
        "bathrooms": 2,
        "year_built": 1999,
        "floor": 2,
        "building_floors": 6,
        "has_elevator": 1,
        "has_garden": 0,
        "has_balcony": 1,
        "garage": 1,
        "energy_class": "B",
        "humidity_level": 50.0,
        "temperature_avg": 20.5,
        "noise_level": 40,
        "air_quality_index": 70,
    }


# ----------------------------------------------------------------------------
# Tests (funzioni “e2e style”: possono essere chiamate da pytest o da __main__)
# ----------------------------------------------------------------------------
def test_predict_endpoint():
    assert _api_up(), "API not reachable at /health (start the server or set E2E_ENABLE=0 to skip)"
    resp = requests.post(f"{API_BASE}/predict/{ASSET_TYPE}", json=api_payload, timeout=8)
    if resp.status_code != 200:
        fail(f"API call failed with status {resp.status_code} | body={resp.text[:240]}")
        return None

    ok("API prediction call succeeded")
    data = resp.json()

    # deve esserci una valutazione (v1 o v2)
    val = _pick_valuation(data.get("metrics", {}))
    if val is None:
        fail("Missing valuation value in response (metrics.valuation_k or .valuation_base_k)")
    return data


def validate_schema(response: dict):
    """
    Valida contro lo schema “coerente” se presente.
    Inoltre fa un confronto morbido con l'example (se c'è).
    """
    try:
        schema = _load_schema_for_response(response)
        if schema:
            jsonschema_validate(instance=response, schema=schema)
            ok(f"Strict schema validation passed (schema_version={response.get('schema_version')})")
        else:
            warn("Schema file not found for this schema_version; skipping strict validation.")
    except ValidationError as e:
        fail(f"Schema validation error: {e.message}")
    except Exception as e:
        fail(f"Schema validation failed (unexpected): {e}")

    # Confronto non-strict con example (se esiste; tipicamente v1)
    try:
        if EXAMPLE_PATH.exists():
            example = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
            ignore_keys = {"_logged_at"}
            optional_keys = {"schema_validation_error", "blockchain_txid", "publish", "asa_id", "model_health"}
            diff_keys = (set(response.keys()) ^ set(example.keys())) - ignore_keys - optional_keys
            if not diff_keys:
                ok("Matches example structure (non-strict)")
            else:
                warn(f"Mismatch with example keys: {diff_keys}")
    except Exception as e:
        warn(f"Example structure check failed: {e}")


def test_model_registry():
    try:
        pipe = get_pipeline(ASSET_TYPE, "value_regressor")
        assert pipe is not None
        ok("Model loaded from registry")

        meta = get_model_metadata(ASSET_TYPE, "value_regressor")
        if meta:
            ok(f"Model version: {meta.get('model_version')}")

        health = health_check_model(ASSET_TYPE, "value_regressor")
        if health.get("status") == "healthy":
            ok("Model health check passed")
        else:
            fail(f"Model unhealthy: {health.get('error', 'unknown')}")
            return

        stats = cache_stats()
        ok(f"Cache status: {stats.get('pipelines_cached', 0)} pipelines cached")
    except Exception as e:
        fail(f"Model registry error: {e}")


def test_api_advanced_features():
    try:
        r1 = requests.get(f"{API_BASE}/models/{ASSET_TYPE}", timeout=5)
        if r1.status_code == 200:
            ok("Model discovery endpoint OK")
        else:
            warn(f"Model discovery endpoint status={r1.status_code}")

        r2 = requests.get(f"{API_BASE}/models/{ASSET_TYPE}/value_regressor/health", timeout=5)
        if r2.status_code == 200 and r2.json().get("status") in {"healthy", "unhealthy"}:
            ok("Model health endpoint OK")
        else:
            warn(f"Model health endpoint status={r2.status_code} body={r2.text[:200]}")

        r3 = requests.post(f"{API_BASE}/models/{ASSET_TYPE}/value_regressor/refresh", timeout=5)
        if r3.status_code == 200:
            ok("Model cache refresh endpoint OK")
        else:
            warn(f"Model cache refresh endpoint status={r3.status_code}")
    except Exception as e:
        fail(f"Advanced API test failed: {e}")


def test_prediction_consistency_advanced():
    preds = []
    latencies = []
    for _ in range(3):
        t0 = time.perf_counter()
        r = requests.post(f"{API_BASE}/predict/{ASSET_TYPE}", json=api_payload, timeout=8)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)
        if r.status_code == 200:
            val = _pick_valuation(r.json().get("metrics", {}))
            preds.append(val)

    if len(preds) >= 2 and len(set(preds)) == 1:
        ok(f"Predictions are stable: {preds[0]}")
    else:
        warn(f"Prediction variation: {preds}")

    mean_latency = sum(latencies) / max(len(latencies), 1)
    if mean_latency < 1000:
        ok(f"Mean latency acceptable: {mean_latency:.2f} ms")
    else:
        warn(f"High latency: {mean_latency:.2f} ms")


def test_recent_log():
    try:
        if not LOG_PATH.exists():
            warn(f"Log file not found: {LOG_PATH}")
            return
        with LOG_PATH.open("r", encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
        if not lines:
            warn("Log file is empty")
            return
        entry = json.loads(lines[-1])
        ts = entry.get("_logged_at")
        if not ts:
            warn("Last log entry missing _logged_at")
            return
        logged_time = _parse_iso(ts)
        age = datetime.now(timezone.utc) - logged_time
        if age < timedelta(minutes=5):
            ok("Recent prediction log is valid")
        else:
            warn(f"Log is stale (age ~ {age})")
    except Exception as e:
        fail(f"Log test failed: {e}")


# ----------------------------------------------------------------------------
# Standalone main (facoltativo): permette di lanciarlo anche con `python tests/...py`
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    failures = []
    print("--- E2E Sanity Check ---")
    if not _api_up():
        print("API not reachable. Start the server or set E2E_ENABLE=1 when running under pytest.")
        sys.exit(1)
    test_model_registry()
    test_api_advanced_features()
    data = test_predict_endpoint()
    if data:
        validate_schema(data)
        test_prediction_consistency_advanced()
    test_recent_log()

    print("\n--- Summary ---")
    if not failures:
        print("✅ All tests passed")
        sys.exit(0)
    else:
        print(f"❌ {len(failures)} failures:")
        for f in failures:
            print(" -", f)
        sys.exit(1)