"""
model_registry.py
Centralized registry & loader for AI Oracle model pipelines (Multi-RWA ready).

Refactor notes (aligned with notebooks + /shared):
- Models root prefers ../shared/outputs/models (override with env MODELS_ROOT).
- Loads ONLY fitted models, with auto-fallback to the newest fitted version.
- Accepts explicit registry entries but gracefully falls back to dynamic discovery.
- Exposes expected feature lists, preferring training_manifest.json over meta.json.
- Enriches metadata with sha256, size, timestamps, and validation metrics.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore

# =============================================================================
# Configuration
# =============================================================================

# Preferred root: env → ../shared/outputs/models → ./shared/outputs/models → ../models → ./models
def _resolve_models_root() -> Path:
    """Risolve la root dei modelli con supporto per struttura notebooks"""
    candidates: List[Path] = []
    
    # Check env first
    env_root = os.getenv("MODELS_ROOT") or os.getenv("AI_ORACLE_MODELS_BASE")
    if env_root and env_root.strip():
        candidates.append(Path(env_root))
    
    # Path specifici per la struttura notebooks
    candidates += [
        Path("notebooks/outputs/modeling"),           # 🔴 NUOVO: struttura notebooks
        Path("./notebooks/outputs/modeling"),         # 🔴 NUOVO: relativo
        Path("../notebooks/outputs/modeling"),        # 🔴 NUOVO: da tests/
        Path("outputs/modeling"),                     # Se eseguito da notebooks/
        Path("../shared/outputs/models"),
        Path("./shared/outputs/models"),
        Path("../models"),
        Path("./models"),
    ]
    
    for c in candidates:
        if c.exists():
            # Verifica contenuto
            if (c / "property").exists():
                print(f"✅ Using models root: {c}")
                return c.resolve()
            # Check anche in artifacts/ per i modelli
            if (c / "artifacts").exists():
                artifacts = c / "artifacts"
                if any(artifacts.glob("*.joblib")):
                    print(f"✅ Found models in artifacts: {c}")
                    return c.resolve()
    
    # Default
    default = Path("./models")
    print(f"⚠️ Using default (may be empty): {default}")
    return default.resolve()

def _resolve_path(asset_type: str, task: str, preferred_version: Optional[str] = None) -> Path:
    """
    Supporta sia struttura standard che notebooks/outputs/modeling/artifacts/
    """
    at = _normalize_key(asset_type)
    tk = _normalize_key(task)
    
    # Check struttura standard
    standard_path = MODELS_BASE / at / f"{tk}_v1.joblib"
    if standard_path.exists():
        return standard_path
    
    # Check in artifacts (struttura notebooks)
    artifacts_path = MODELS_BASE / "artifacts" / f"rf_champion_A.joblib"
    if artifacts_path.exists():
        return artifacts_path
    
    artifacts_path_b = MODELS_BASE / "artifacts" / f"rf_champion_B.joblib"
    if artifacts_path_b.exists():
        return artifacts_path_b
    
    # Discovery
    for pattern in [f"{tk}_*.joblib", "rf_*.joblib", "*.joblib"]:
        for candidate in MODELS_BASE.rglob(pattern):
            try:
                pl = joblib.load(candidate)
                if _is_fitted_pipeline(pl):
                    return candidate
            except:
                continue
    
    raise ModelNotFoundError(f"No fitted model found for '{asset_type}/{task}'")

MODELS_BASE: Path = _resolve_models_root()

MODEL_EXT = ".joblib"
META_SUFFIX = "_meta.json"
VERSION_RE = re.compile(r"_(v\d+(?:[a-z0-9\-_\.]*)?)\.joblib$", re.IGNORECASE)
TASK_DEFAULT = "value_regressor"

logger = logging.getLogger("model_registry")
if not logger.handlers:
    # non invasivo: lascia che l'app imposti i handlers; qui solo livello
    logger.setLevel(logging.INFO)

# Registry (optional): (asset_type, task) -> relative path (under MODELS_BASE)
MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "property": {
        "value_regressor": "property/value_regressor_v2.joblib",
    }
}

# In-memory caches
_PIPELINE_TTL_CACHE: Dict[str, Tuple[Any, float]] = {}
_METADATA_CACHE: Dict[Path, dict] = {}
CACHE_TTL_SECONDS = 3600


# =============================================================================
# Exceptions
# =============================================================================
class ModelNotFoundError(Exception):
    """Raised when a model path does not exist on disk."""


class RegistryLookupError(Exception):
    """Raised when (asset_type, task) pair is not defined in MODEL_REGISTRY."""


# =============================================================================
# Internal helpers
# =============================================================================
def _normalize_key(s: str) -> str:
    return s.strip().lower()


def _file_sha256(path: Path, chunk_size: int = 1 << 20) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _list_version_files(asset_type: str, task: str) -> List[Path]:
    """All model files for task under asset_type folder, sorted desc by version number."""
    at_dir = MODELS_BASE / _normalize_key(asset_type)
    if not at_dir.exists():
        return []
    # pattern: <task>_v*.joblib
    files = sorted(at_dir.glob(f"{task}_v*.joblib"))
    # sort by numeric version desc when possible
    def _ver_num(p: Path) -> int:
        m = VERSION_RE.search(p.name)
        if m:
            try:
                return int(m.group(1)[1:])
            except Exception:
                return -1
        return -1
    return sorted(files, key=_ver_num, reverse=True)


def _is_fitted_pipeline(pl: Any) -> bool:
    try:
        if hasattr(pl, "steps"):  # sklearn Pipeline
            check_is_fitted(pl.steps[-1][1])
        else:
            check_is_fitted(pl)
        return True
    except Exception:
        return False


def _suggest_similar_models(asset_type: str, task: str, max_suggestions: int = 3) -> List[str]:
    """Suggest similar tasks registered for the given asset_type."""
    at = _normalize_key(asset_type)
    if at not in MODEL_REGISTRY:
        return []
    all_tasks = list(MODEL_REGISTRY[at].keys())
    return get_close_matches(task, all_tasks, n=max_suggestions, cutoff=0.6)


def _metadata_path_for(model_path: Path) -> Path:
    """
    Given: property/value_regressor_v1.joblib
    Expect: property/value_regressor_v1_meta.json
    """
    stem = model_path.name.replace(MODEL_EXT, "")
    return model_path.parent / f"{stem}{META_SUFFIX}"


def _manifest_path_for(asset_type: str) -> Path:
    """training_manifest.json placed at the asset_type folder root."""
    return (MODELS_BASE / _normalize_key(asset_type)) / "training_manifest.json"


# =============================================================================
# Path resolution (registry → discovery) and fitted fallback
# =============================================================================
def _resolve_registered(asset_type: str, task: str) -> Optional[Path]:
    at = _normalize_key(asset_type)
    tk = _normalize_key(task)
    try:
        rel = MODEL_REGISTRY[at][tk]
        full = MODELS_BASE / rel
        return full if full.exists() else None
    except KeyError:
        return None


def _resolve_path(asset_type: str, task: str, preferred_version: Optional[str] = None) -> Path:
    """
    Resolution order:
    1) If preferred_version is provided: try models/<asset_type>/<task>_<preferred>.joblib
    2) Registered path from MODEL_REGISTRY
    3) Discovery: pick newest **fitted** version in models/<asset_type>/
    """
    at = _normalize_key(asset_type)
    tk = _normalize_key(task)

    # 1) explicit preferred version
    if preferred_version:
        p = MODELS_BASE / at / f"{tk}_{preferred_version}.joblib"
        if p.exists():
            return p
        logger.warning(f"Preferred version not found: {p.name}. Falling back…")

    # 2) registry entry
    reg = _resolve_registered(at, tk)
    if reg is not None:
        if reg.exists():
            return reg
        logger.warning(f"Registry path missing: {reg}. Falling back…")

    # 3) discovery: newest **fitted** model
    for cand in _list_version_files(at, tk):
        try:
            pl = joblib.load(cand)
            if _is_fitted_pipeline(pl):
                return cand
        except Exception:
            continue

    suggestions = _suggest_similar_models(at, tk)
    hint = f" Did you mean: {suggestions}?" if suggestions else ""
    raise ModelNotFoundError(f"No fitted model found for '{asset_type}/{task}' under {MODELS_BASE}.{hint}")


# =============================================================================
# Public API
# =============================================================================
def get_pipeline(
    asset_type: str,
    task: str = TASK_DEFAULT,
    *,
    preferred_version: Optional[str] = None,
) -> Any:
    """
    Return a loaded (and TTL-cached) model pipeline (only if fitted).
    """
    model_path = _resolve_path(asset_type, task, preferred_version=preferred_version)
    now = time.time()
    cache_key = str(model_path.resolve())
    if cache_key in _PIPELINE_TTL_CACHE:
        pipeline, ts = _PIPELINE_TTL_CACHE[cache_key]
        if now - float(ts) < CACHE_TTL_SECONDS:
            return pipeline

    pipeline = joblib.load(model_path)
    if not _is_fitted_pipeline(pipeline):
        # try fallback to another fitted candidate
        for cand in _list_version_files(asset_type, task):
            if cand == model_path:
                continue
            try:
                pl2 = joblib.load(cand)
                if _is_fitted_pipeline(pl2):
                    pipeline = pl2
                    model_path = cand
                    break
            except Exception:
                continue
        else:
            raise ModelNotFoundError(f"Model at {model_path} is not fitted and no fallback found")

    _PIPELINE_TTL_CACHE[cache_key] = (pipeline, now)
    logger.info(f"Model loaded: {model_path.name}")
    return pipeline


def get_model_paths(
    asset_type: str,
    task: str = TASK_DEFAULT,
    *,
    preferred_version: Optional[str] = None,
) -> Dict[str, Path]:
    """Return dict with 'pipeline', 'meta', 'manifest' paths for the resolved model."""
    pipeline_path = _resolve_path(asset_type, task, preferred_version=preferred_version)
    meta_path = _metadata_path_for(pipeline_path)
    manifest_path = _manifest_path_for(asset_type)
    return {"pipeline": pipeline_path, "meta": meta_path, "manifest": manifest_path}


# --- NEW: shared schema/config probing ---------------------------------------
from importlib import import_module

def _features_from_shared(asset_type: str) -> Optional[tuple[list[str], list[str]]]:
    """
    Tenta di leggere le feature da /shared:
    Priorità (first-hit wins):
      1) shared.common.schema.get_feature_spec(asset_type) -> {"categorical":[...], "numeric":[...]}
      2) shared.common.schema.FEATURE_SPEC[asset_type]      -> idem
      3) shared.common.schema.FEATURES_CATEGORICAL[asset], FEATURES_NUMERIC[asset]
      4) shared.common.config.FEATURE_SPEC (stesse varianti)
    """
    candidates = [
        ("shared.common.schema", "get_feature_spec", "func"),
        ("shared.common.schema", "FEATURE_SPEC", "map"),
        ("shared.common.schema", "FEATURES_CATEGORICAL", "split"),
        ("shared.common.config", "get_feature_spec", "func"),
        ("shared.common.config", "FEATURE_SPEC", "map"),
        ("shared.common.config", "FEATURES_CATEGORICAL", "split"),
    ]
    for mod_name, attr, mode in candidates:
        try:
            mod = import_module(mod_name)
            obj = getattr(mod, attr)
            if mode == "func" and callable(obj):
                spec = obj(asset_type)
                if isinstance(spec, dict):
                    cat = list(spec.get("categorical", []) or [])
                    num = list(spec.get("numeric", []) or [])
                    return cat, num
            elif mode == "map" and isinstance(obj, dict) and asset_type in obj:
                spec = obj[asset_type]
                cat = list(spec.get("categororical", spec.get("categorical", [])) or [])
                num = list(spec.get("numeric", []) or [])
                return cat, num
            elif mode == "split":
                cats = getattr(mod, "FEATURES_CATEGORICAL", None)
                nums = getattr(mod, "FEATURES_NUMERIC", None)
                if isinstance(cats, dict) and isinstance(nums, dict) and asset_type in cats and asset_type in nums:
                    return list(cats[asset_type] or []), list(nums[asset_type] or [])
        except Exception:
            continue
    return None

def _find_manifest(models_base: Path, asset_type: str) -> Path:
    """
    Cerca training_manifest.json in più posti comuni:
      1) <models_base>/<asset_type>/training_manifest.json
      2) <models_base>/training_manifest.json
      3) ../shared/outputs/<asset_type>/training_manifest.json
      4) ./shared/outputs/<asset_type>/training_manifest.json
    Ritorna il primo percorso esistente (o quello #1 anche se non esiste).
    """
    at = _normalize_key(asset_type)
    candidates = [
        models_base / at / "training_manifest.json",
        models_base / "training_manifest.json",
        Path("../shared/outputs") / at / "training_manifest.json",
        Path("./shared/outputs") / at / "training_manifest.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]  # default (potrebbe non esistere)

def get_model_paths(
    asset_type: str,
    task: str = TASK_DEFAULT,
    *,
    preferred_version: Optional[str] = None,
) -> Dict[str, Path]:
    pipeline_path = _resolve_path(asset_type, task, preferred_version=preferred_version)
    meta_path = _metadata_path_for(pipeline_path)
    manifest_path = _find_manifest(MODELS_BASE, asset_type)
    return {"pipeline": pipeline_path, "meta": meta_path, "manifest": manifest_path}

def expected_features(meta: dict, manifest_path: Path, asset_type: str | None = None) -> tuple[list[str], list[str]]:
    # 1) prova da /shared (se asset_type è noto)
    if asset_type:
        shared_spec = _features_from_shared(asset_type)
        if shared_spec:
            cat, num = shared_spec
            # dedup/no-overlap
            seen = set(); cat = [c for c in cat if not (c in seen or seen.add(c))]
            num = [c for c in num if c not in set(cat)]
            return cat, num

    # 2) training manifest
    cat = list(meta.get("features_categorical", []) or [])
    num = list(meta.get("features_numeric", []) or [])
    if manifest_path.exists():
        try:
            mf = json.loads(manifest_path.read_text(encoding="utf-8"))
            feats = mf.get("model", {}).get("feature_list") or mf.get("model", {}).get("features", {})
            if isinstance(feats, dict):
                cat = list(feats.get("categorical", cat) or cat)
                num = list(feats.get("numeric", num) or num)
        except Exception:
            pass

    seen = set(); cat = [c for c in cat if not (c in seen or seen.add(c))]
    num = [c for c in num if c not in set(cat)]
    return cat, num

def _load_nb_outputs(asset_type: str) -> dict:
    """
    Prova a caricare outputs utili dei notebooks per arricchire i metadata:
      - ../shared/outputs/<asset>/metrics/*.json (es. valid_metrics.json)
      - ../shared/outputs/<asset>/dataset_stats.json
    Merge non distruttivo in un dict {"metrics": {...}, "dataset_stats": {...}}
    """
    at = _normalize_key(asset_type)
    root_candidates = [Path("../shared/outputs") / at, Path("./shared/outputs") / at]
    out: dict = {}
    for root in root_candidates:
        try:
            metrics_dir = root / "metrics"
            if metrics_dir.exists():
                for fp in metrics_dir.glob("*.json"):
                    try:
                        part = json.loads(fp.read_text(encoding="utf-8"))
                        out.setdefault("metrics", {}).update(part)
                    except Exception:
                        pass
            ds = root / "dataset_stats.json"
            if ds.exists():
                out["dataset_stats"] = json.loads(ds.read_text(encoding="utf-8"))
        except Exception:
            continue
    return out

def get_model_metadata(
    asset_type: str,
    task: str = TASK_DEFAULT,
    *,
    preferred_version: Optional[str] = None,
) -> Optional[dict]:
    paths = get_model_paths(asset_type, task, preferred_version=preferred_version)
    meta_path = paths["meta"]; pipeline_path = paths["pipeline"]; manifest_path = paths["manifest"]

    if not meta_path.exists():
        return None

    if meta_path not in _METADATA_CACHE:
        md = json.loads(meta_path.read_text(encoding="utf-8"))
        md.setdefault("model_path", str(pipeline_path))
        sha = _file_sha256(pipeline_path)
        if sha:
            md.setdefault("model_hash", sha)
        try:
            md["model_size_mb"] = round(pipeline_path.stat().st_size / (1024 * 1024), 2)
            md["last_modified"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(pipeline_path.stat().st_mtime))
        except Exception:
            pass

        # Preferisci metrics dal manifest → poi notebook outputs → poi meta.json
        try:
            if manifest_path.exists():
                mf = json.loads(manifest_path.read_text(encoding="utf-8"))
                metrics = (mf.get("metrics") or {}).get("validation") or (mf.get("metrics") or {}).get("valid") or {}
                if metrics:
                    md.setdefault("metrics", metrics)
        except Exception:
            pass

        # Merge outputs notebooks (non distruttivo)
        try:
            nb_out = _load_nb_outputs(asset_type)
            if nb_out.get("metrics"):
                md.setdefault("metrics", {}).update(nb_out["metrics"])
            if nb_out.get("dataset_stats"):
                md.setdefault("dataset_stats", nb_out["dataset_stats"])
        except Exception:
            pass

        _METADATA_CACHE[meta_path] = md

    return _METADATA_CACHE[meta_path]


def validate_model_compatibility(pipeline: Any, expected_features_list: List[str]) -> bool:
    """
    Checks if the model's expected input features match the expected list (order-insensitive).
    """
    if hasattr(pipeline, "feature_names_in_"):
        pipeline_features = list(pipeline.feature_names_in_)
        return set(pipeline_features) == set(expected_features_list)
    # If the attribute is not available, assume compatible (sklearn versions vary)
    return True


def health_check_model(asset_type: str, task: str = TASK_DEFAULT, *, preferred_version: Optional[str] = None) -> dict:
    """
    Returns a health diagnostic for the given model:
    - Load success
    - Metadata presence
    - Size and last modification
    - Training metrics (if available)
    """
    try:
        paths = get_model_paths(asset_type, task, preferred_version=preferred_version)
        pipeline = get_pipeline(asset_type, task, preferred_version=preferred_version)
        meta = get_model_metadata(asset_type, task, preferred_version=preferred_version)

        return {
            "status": "healthy",
            "model_path": str(paths["pipeline"]),
            "size_mb": round(paths["pipeline"].stat().st_size / (1024 * 1024), 2),
            "last_modified": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(paths["pipeline"].stat().st_mtime)),
            "metadata_valid": bool(meta),
            "metrics": (meta or {}).get("metrics"),
            "fitted": _is_fitted_pipeline(pipeline),
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


# =============================================================================
# Discovery / helpers (backward-compatible)
# =============================================================================
def list_asset_types() -> List[str]:
    # present folders under MODELS_BASE OR keys in registry
    assets = set(MODEL_REGISTRY.keys())
    if MODELS_BASE.exists():
        for p in MODELS_BASE.iterdir():
            if p.is_dir():
                assets.add(p.name)
    return sorted(assets)


def list_tasks(asset_type: str) -> List[str]:
    at = _normalize_key(asset_type)
    tasks = set(MODEL_REGISTRY.get(at, {}).keys())
    # discover tasks by scanning filenames
    for p in discover_models_for_asset(at):
        parsed = parse_task_and_version(p.name)
        if parsed:
            t, _ = parsed
            tasks.add(t)
    if not tasks:
        raise RegistryLookupError(f"Unknown asset_type or no tasks found: {asset_type}")
    return sorted(tasks)


def model_exists(asset_type: str, task: str = TASK_DEFAULT) -> bool:
    try:
        _ = _resolve_path(asset_type, task)
        return True
    except (RegistryLookupError, ModelNotFoundError):
        return False


def refresh_cache(asset_type: Optional[str] = None, task: Optional[str] = None) -> None:
    """Clear cached entries (all or selected) so next call reloads from disk."""
    if asset_type and task:
        try:
            path = _resolve_path(asset_type, task)
            _PIPELINE_TTL_CACHE.pop(str(path.resolve()), None)
            _METADATA_CACHE.pop(_metadata_path_for(path), None)
        except (RegistryLookupError, ModelNotFoundError):
            pass
    else:
        _PIPELINE_TTL_CACHE.clear()
        _METADATA_CACHE.clear()


def cache_stats() -> dict:
    now = time.time()
    cache_info: Dict[str, dict] = {}
    for model_path, (_pipeline, timestamp) in _PIPELINE_TTL_CACHE.items():
        if not isinstance(timestamp, (int, float)):
            continue
        age = now - timestamp
        cache_info[str(model_path)] = {
            "age_seconds": round(age, 1),
            "expires_in": round(CACHE_TTL_SECONDS - age, 1),
            "expired": age >= CACHE_TTL_SECONDS,
        }
    active_models = sum(1 for v in cache_info.values() if not v["expired"])
    expired_models = len(cache_info) - active_models
    return {
        "pipelines_cached": active_models,
        "pipelines_expired": expired_models,
        "metadata_cached": len(_METADATA_CACHE),
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "models": cache_info,
    }


def discover_models_for_asset(asset_type: str) -> List[Path]:
    """Scan the asset_type directory for *.joblib models (not only those registered)."""
    at_dir = MODELS_BASE / _normalize_key(asset_type)
    if not at_dir.exists():
        return []
    return sorted(at_dir.glob(f"*{MODEL_EXT}"))


def parse_task_and_version(model_filename: str) -> Optional[Tuple[str, str]]:
    """From value_regressor_v1.joblib -> ("value_regressor", "v1")"""
    if not model_filename.endswith(MODEL_EXT):
        return None
    m = VERSION_RE.search(model_filename)
    if not m:
        return None
    version = m.group(1)
    task_part = model_filename[: model_filename.rfind("_" + version)]
    return task_part, version


def suggest_task_versions(asset_type: str, task: str) -> List[str]:
    """Return a list of versioned model filenames matching a task prefix."""
    matches = []
    for p in discover_models_for_asset(asset_type):
        parsed = parse_task_and_version(p.name)
        if parsed and parsed[0] == task:
            matches.append(p.name)
    return sorted(matches)


def latest_version_filename(asset_type: str, task: str) -> Optional[str]:
    """Pick the highest version number available for task under asset_type."""
    cands = suggest_task_versions(asset_type, task)
    if not cands:
        return None
    def _vn(name: str) -> int:
        m = VERSION_RE.search(name)
        if not m:
            return -1
        try:
            return int(m.group(1)[1:])
        except Exception:
            return -1
    return sorted(cands, key=_vn, reverse=True)[0]


# =============================================================================
# Diagnostics when run directly
# =============================================================================
if __name__ == "__main__":
    print("MODELS_BASE:", MODELS_BASE)
    print("Available asset types:", list_asset_types())
    for at in list_asset_types():
        print(f"\nAsset Type: {at}")
        for t in list_tasks(at):
            ok = model_exists(at, t)
            print(f"  Task: {t} -> exists: {ok}")
            if ok:
                paths = get_model_paths(at, t)
                meta = get_model_metadata(at, t)
                print(
                    f"    pipeline: {paths['pipeline'].name} | "
                    f"version: {(meta or {}).get('model_version')} | "
                    f"hash: {(meta or {}).get('model_hash', '')[:16]}"
                )
        discovered = discover_models_for_asset(at)
        if discovered:
            print("  Discovered files:")
            for p in discovered:
                parsed = parse_task_and_version(p.name)
                pv = f"{parsed[0]} ({parsed[1]})" if parsed else "unparsed"
                print(f"    - {p.name} -> {pv}")
    print("\nCache stats:", cache_stats())