# scripts/sample_property.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import os
from pathlib import Path

# Per generare PoVal p1 dai sample v2
try:
    from scripts.canon import build_p1_from_response, canonical_note_bytes_p1
except Exception:  # pragma: no cover
    build_p1 = None  # type: ignore
    build_p1_from_response = None  # type: ignore
    canonical_note_bytes_p1 = None  # type: ignore

def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

# -----------------------------------------------------------------------------
# Esempio di payload di RICHIESTA per /predict/property
# -----------------------------------------------------------------------------
sample_property_request: Dict[str, Any] = {
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
    "garage": 1,  # alias accettato → verrà canonicalizzato a has_garage
    "energy_class": "B",
    "humidity_level": 50.0,
    "temperature_avg": 20.5,
    "noise_level": 40,
    "air_quality_index": 70,
    # Il backend calcola age_years, luxury_score, env_score
}

# ---- helpers per risolvere il path del modello (solo scopo demo) ----
def _models_base() -> Path:
    for env in ("AI_ORACLE_MODELS_BASE", "MODELS_ROOT"):
        v = os.getenv(env)
        if v:
            p = Path(v)
            if p.exists():
                return p.resolve()
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "notebooks" / "outputs" / "modeling" / "property",
        root / "notebooks" / "outputs" / "modeling",
        root / "shared" / "outputs" / "models" / "property",
        root / "models" / "property",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return candidates[0]

def _resolve_model_path(version: str = "v2") -> str:
    base = _models_base()
    prop_dir = base if base.name == "property" else (base / "property")
    target = prop_dir / f"value_regressor_{version}.joblib"
    if target.exists():
        return str(target)
    artifacts = (base.parent if base.name == "property" else base) / "artifacts"
    for name in ("rf_champion_A.joblib", "rf_champion_B.joblib"):
        cand = artifacts / name
        if cand.exists():
            return str(cand)
    return str(target)

# -----------------------------------------------------------------------------
# ESEMPI DI RISPOSTA — SCHEMA v2
# -----------------------------------------------------------------------------
def make_sample_response_v2(
    asset_id: str = "property_001",
    *,
    valuation_k: float = 610.5,
    point_pred_k: Optional[float] = None,
    uncertainty_k: float = 18.4,
    conf: float = 0.95,
    ci_low_k: float = 575.0,
    ci_high_k: float = 646.0,
    ci_method: str = "forest_variance",
    n_estimators: Optional[int] = 200,
    latency_ms: float = 12.8,
) -> Dict[str, Any]:
    resp = {
        "schema_version": "v2",
        "asset_id": asset_id,
        "asset_type": "property",
        "timestamp": _now_iso(),
        "metrics": {
            "valuation_k": float(valuation_k),
            "point_pred_k": float(point_pred_k if point_pred_k is not None else valuation_k),
            "uncertainty_k": float(uncertainty_k),
            "confidence": float(conf),
            "confidence_low_k": float(ci_low_k),
            "confidence_high_k": float(ci_high_k),
            "ci_margin_k": round(float(ci_high_k) - float(valuation_k), 3),
            "latency_ms": float(latency_ms),
            "ci_method": ci_method,
            "n_estimators": n_estimators,
        },
        "flags": {"anomaly": False, "drift_detected": False, "needs_review": False},
        "model_meta": {
            "value_model_version": "v2",
            "value_model_name": "RandomForestRegressor",
            "n_features_total": 20,
            "model_hash": "fakehashpropertymodel1234567890abcdef",
        },
        "model_health": {
            "status": "ok",
            "model_path": _resolve_model_path("v2"),
            "metadata_valid": True,
            "metrics": {"rmse": 22.1, "mae": 15.4, "r2": 0.87},
        },
        "validation": {"ok": True, "warnings": None, "errors": None},
        "drift": {"message": None},
        "offchain_refs": {"detail_report_hash": None, "sensor_batch_hash": None},
        "cache_hit": False,
        "schema_validation_error": "",
        "blockchain_txid": "",
        "asa_id": "",
        "publish": {"status": "not_attempted"},
    }

    # Se disponibile, allega anche PoVal p1 (come fa l'API)
    if build_p1_from_response and canonical_note_bytes_p1:
        try:
            p1, _dbg = build_p1_from_response(resp, allowed_input_keys=[])
            resp.setdefault("attestation", {})["p1"] = p1
            b, h, n = canonical_note_bytes_p1(p1)
            resp["attestation"].update({"p1_sha256": h, "p1_size_bytes": int(n)})
        except Exception:
            pass
    return resp

# Un singolo sample v2 pronto all’uso + lista
sample_response_v2: Dict[str, Any] = make_sample_response_v2()
multiple_samples_v2: List[Dict[str, Any]] = [
    make_sample_response_v2(
        asset_id=f"property_{i:03}",
        valuation_k=550.0 + i * 12.5,
        ci_low_k=530.0 + i * 12.0,
        ci_high_k=570.0 + i * 13.0,
        latency_ms=10.0 + i,
    )
    for i in range(5)
]

# -----------------------------------------------------------------------------
# LEGACY — SCHEMA v1 (ancora supportato)
# -----------------------------------------------------------------------------
sample_response_v1: Dict[str, Any] = {
    "asset_id": "property_001",
    "asset_type": "property",
    "timestamp": _now_iso(),
    "schema_version": "v1",
    "metrics": {"valuation_base_k": 601.1},
    "flags": {"anomaly": False, "drift_detected": False, "needs_review": False},
    "model_meta": {
        "value_model_version": "v1",
        "value_model_name": "LGBMRegressor",
        "model_hash": "fakehashpropertymodel123",
    },
    "offchain_refs": {"detail_report_hash": None, "sensor_batch_hash": None},
    "model_health": {"status": "ok", "metadata_valid": True, "metrics": {}},
    "cache_hit": False,
    "publish": {"status": "not_attempted"},
}
multiple_samples_v1: List[Dict[str, Any]] = [
    {
        "asset_id": f"property_{i:03}",
        "asset_type": "property",
        "timestamp": _now_iso(),
        "schema_version": "v1",
        "metrics": {"valuation_base_k": round(555.0 + i * 10.0, 3)},
        "flags": {"anomaly": False, "drift_detected": False, "needs_review": False},
        "model_meta": {
            "value_model_version": "v1",
            "value_model_name": "LGBMRegressor",
            "model_hash": f"fakehash_{i}",
        },
        "offchain_refs": {"detail_report_hash": None, "sensor_batch_hash": None},
        "model_health": {"status": "ok", "metadata_valid": True, "metrics": {}},
        "cache_hit": False,
        "publish": {"status": "not_attempted"},
    }
    for i in range(5)
]

# -----------------------------------------------------------------------------
# Alias attesi dai test
# -----------------------------------------------------------------------------
sample_response = sample_response_v2
multiple_samples = multiple_samples_v2
