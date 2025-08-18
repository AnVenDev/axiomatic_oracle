# tests/conftest.py
from __future__ import annotations
import os, sys, json, shutil
from pathlib import Path
import pytest   # type: ignore

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

# PYTHONPATH: progetto + notebooks (per import "shared.common.*")
for p in (ROOT, NB):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# cartelle dei notebooks
modeling_root = NB / "outputs" / "modeling"
artifacts_dir = modeling_root / "artifacts"
property_dir  = modeling_root / "property"

# fallback features allineate al payload dell'API
DEFAULT_CAT = ["location", "energy_class"]
DEFAULT_NUM = [
    "size_m2", "rooms", "bathrooms", "year_built", "floor", "building_floors",
    "has_elevator", "has_garden", "has_balcony", "garage",
    "humidity_level", "temperature_avg", "noise_level", "air_quality_index",
    "age_years",
]

def _ensure_property_layout_from_artifacts() -> None:
    if not artifacts_dir.exists():
        return
    property_dir.mkdir(parents=True, exist_ok=True)

    # scegli un .joblib dagli artifacts
    cand = next((p for p in [
        artifacts_dir / "rf_champion_A.joblib",
        artifacts_dir / "rf_champion_B.joblib",
        *sorted(artifacts_dir.glob("*.joblib"))
    ] if p.exists()), None)
    if not cand:
        return

    # copia come value_regressor_v1.joblib
    dst = property_dir / "value_regressor_v1.joblib"
    if not dst.exists():
        shutil.copy2(cand, dst)

    # meta: prova dal manifest, altrimenti fallback non-vuoto
    meta = property_dir / "value_regressor_v1_meta.json"
    if not meta.exists():
        feats_cat, feats_num, metrics = [], [], {}
        manifest = modeling_root / "training_manifest.json"
        if manifest.exists():
            try:
                m = json.loads(manifest.read_text(encoding="utf-8"))
                # prova 1: manifest["model"]["features"] = {"categorical": [...], "numeric": [...]}
                mf_feats = (m.get("model") or {}).get("features") or {}
                feats_cat = mf_feats.get("categorical", []) or []
                feats_num = mf_feats.get("numeric", []) or []
                # prova 2: alcune pipeline salvano in m["feature_config"]
                if not (feats_cat or feats_num):
                    mf_feats = (m.get("feature_config") or {})
                    feats_cat = mf_feats.get("categorical", []) or []
                    feats_num = mf_feats.get("numeric", []) or []
                metrics = (m.get("metrics") or {}).get("validation", {}) or {}
            except Exception:
                pass

        # fallback sicuro se vuote
        if not (feats_cat or feats_num):
            feats_cat = DEFAULT_CAT[:]
            feats_num = DEFAULT_NUM[:]

        meta_obj = {
            "model_version": "v1",
            "model_class": "RandomForestRegressor",
            "features_categorical": feats_cat,
            "features_numeric": feats_num,
            "metrics": metrics or {"r2": 0.6, "mae_k": 70.0, "rmse_k": 87.0},
            "model_path": str(dst),
        }
        meta.write_text(json.dumps(meta_obj, indent=2), encoding="utf-8")

    # porta anche il manifest accanto (facoltativo)
    src_manifest = modeling_root / "training_manifest.json"
    dst_manifest = property_dir / "training_manifest.json"
    if src_manifest.exists() and not dst_manifest.exists():
        shutil.copy2(src_manifest, dst_manifest)

@pytest.fixture(scope="session", autouse=True)
def _prepare_models_and_env():
    # prepara la struttura property/ dai notebooks/artifacts
    _ensure_property_layout_from_artifacts()

    # fissa le ENV *prima* dell'import dell'API/registry
    if modeling_root.exists():
        os.environ.setdefault("AI_ORACLE_MODELS_BASE", str(modeling_root))
        os.environ.setdefault("MODELS_ROOT", str(modeling_root))

    # disattiva logging su file durante i test
    os.environ.setdefault("AI_ORACLE_DISABLE_API_LOG", "1")