"""
model_registry.py
Centralized registry & loader for AI Oracle model pipelines (Multi-RWA ready).

Responsibilities:
- Map (asset_type, task) → local filesystem path
- Lazy-load & TTL-cache pipelines using joblib
- Provide enriched metadata with hash and model_path
- Validate model compatibility with expected features
- Suggest similar tasks on failure (fuzzy matching)
- Perform model health diagnostics (size, timestamp, metrics)
- Prepare stub for remote model access (S3, IPFS, ASA)
- Enable optional logging and async support (future API-ready)

Usage:
    from scripts.model_registry import get_pipeline, get_model_metadata

    pipe = get_pipeline("property", "value_regressor")
    meta = get_model_metadata("property", "value_regressor")

Conventions:
- Each asset_type has a subfolder under /models
- Model filename: <task>_<version>.joblib  (e.g. value_regressor_v1.joblib)
- Metadata file:  <task>_<version>_meta.json
- Side artifacts (hash, timestamp) enriched automatically
"""

import json
import os
import re
import hashlib
import joblib
import logging
from pathlib import Path
from __future__ import annotations
from typing import Dict, Optional, List, Iterable, Tuple
from difflib import get_close_matches

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
ENV_BASE = os.getenv("AI_ORACLE_MODELS_BASE")
DEFAULT_BASE = Path(__file__).resolve().parent.parent / "models"
MODELS_BASE = Path(ENV_BASE).resolve() if ENV_BASE else DEFAULT_BASE

MODEL_EXT = ".joblib"
META_SUFFIX = "_meta.json"
VERSION_PATTERN = re.compile(r"_(v\d+(?:[a-z0-9\-_\.]*)?)\.joblib$", re.IGNORECASE)

logger = logging.getLogger("model_registry")
logger.setLevel(logging.INFO)

# Registry: (asset_type, task) -> relative path
MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "property": {
        "value_regressor": "property/value_regressor_v1.joblib",
        # Future:
        # "anomaly_model": "property/anomaly_iforest_v0.joblib"
    },
    # "art": { ... }
}

# In-memory caches
_PIPELINE_CACHE: Dict[Path, object] = {}
_METADATA_CACHE: Dict[Path, dict] = {}
_PIPELINE_TTL_CACHE = {}
CACHE_TTL_SECONDS = 3600

logger = logging.getLogger("model_registry")
logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------
class ModelNotFoundError(Exception):
    """Raised when a model path does not exist on disk."""


class RegistryLookupError(Exception):
    """Raised when (asset_type, task) pair is not defined in MODEL_REGISTRY."""

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------
def _normalize_key(s: str) -> str:
    return s.strip().lower()


def _suggest_similar_models(asset_type: str, task: str, max_suggestions: int = 3) -> list:
    """
    Suggest similar tasks (by name) registered for the given asset_type.
    Uses fuzzy string matching to help resolve typos or confusion.
    """
    at = _normalize_key(asset_type)
    if at not in MODEL_REGISTRY:
        return []

    all_tasks = list(MODEL_REGISTRY[at].keys())
    return get_close_matches(task, all_tasks, n=max_suggestions, cutoff=0.6)


def _resolve_path(asset_type: str, task: str, fallback_latest: bool = False) -> Path:
    at = _normalize_key(asset_type)
    tk = _normalize_key(task)

    try:
        rel_path = MODEL_REGISTRY[at][tk]
        full_path = MODELS_BASE / rel_path
        if full_path.exists():
            return full_path
    except KeyError:
        raise RegistryLookupError(
            f"Unknown asset_type/task: '{asset_type}' / '{task}'. "
            f"Defined asset_types: {list(MODEL_REGISTRY.keys())}"
        )

    # If file not found but fallback is allowed
    if fallback_latest:
        latest = latest_version_filename(at, tk)
        if latest:
            fallback_path = MODELS_BASE / at / latest
            if fallback_path.exists():
                print(f"[Fallback] Using latest available model: {fallback_path.name}")
                return fallback_path

    # Optional: Try remote download fallback
    # from scripts.remote_registry import RemoteModelRegistry
    # try:
    #     await RemoteModelRegistry().download_model(at, tk, version=latest or "v1")
    #     if fallback_path.exists():
    #         return fallback_path
    # except Exception as e:
    #     print(f"[Remote fallback failed] {e}")

    suggestions = _suggest_similar_models(asset_type, task)
    hint = f" Did you mean: {suggestions}?" if suggestions else ""
    raise ModelNotFoundError(f"Model file not found at: {full_path}.{hint}")


def _metadata_path_for(model_path: Path) -> Path:
    """
    Given: property/value_regressor_v1.joblib
    Expect: property/value_regressor_v1_meta.json
    """
    stem = model_path.name.replace(MODEL_EXT, "")
    meta_name = f"{stem}{META_SUFFIX}"
    return model_path.parent / meta_name


def _file_hash_sha256(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def get_pipeline(asset_type: str, task: str = "value_regressor", fallback_latest: bool = True):
    """
    Return a loaded (and cached) model pipeline with TTL support.

    :param asset_type: e.g. "property"
    :param task: e.g. "value_regressor"
    :param fallback_latest: load latest available model if registered one is missing
    :return: scikit-learn Pipeline object
    """
    model_path = _resolve_path(asset_type, task, fallback_latest=fallback_latest)
    now = time.time()

    # Check TTL cache
    if model_path in _PIPELINE_TTL_CACHE:
        pipeline, ts = _PIPELINE_TTL_CACHE[model_path]
        if now - ts < CACHE_TTL_SECONDS:
            return pipeline  # Cache still valid

    # Load and cache model
    pipeline = joblib.load(model_path)
    _PIPELINE_TTL_CACHE[model_path] = (pipeline, now)
    logger.info(f"Model loaded: {model_path.name}")
    return pipeline

def validate_model_compatibility(pipeline, expected_features: list) -> bool:
    """
    Checks if the model's expected input features match the expected list.

    Returns:
        True if compatible, False otherwise.
    """
    if hasattr(pipeline, "feature_names_in_"):
        pipeline_features = list(pipeline.feature_names_in_)
        return set(pipeline_features) == set(expected_features)
    return True  # assume compatible if attribute not available


def get_model_metadata(asset_type: str, task: str = "value_regressor", fallback_latest: bool = True) -> Optional[dict]:
    """
    Return metadata dict if side-car JSON exists, else None.
    Adds `model_path` and (if absent) `model_hash` convenience fields.
    Optionally falls back to the latest available version if registered one is missing.
    """
    model_path = _resolve_path(asset_type, task, fallback_latest=fallback_latest)
    meta_path = _metadata_path_for(model_path)

    if not meta_path.exists():
        return None

    if meta_path not in _METADATA_CACHE:
        with meta_path.open("r", encoding="utf-8") as f:
            _METADATA_CACHE[meta_path] = json.load(f)

        # Post-process enrichment
        _METADATA_CACHE[meta_path].setdefault("model_path", str(model_path))
        if "model_hash" not in _METADATA_CACHE[meta_path]:
            h = _file_hash_sha256(model_path)
            if h:
                _METADATA_CACHE[meta_path]["model_hash"] = h

    return _METADATA_CACHE[meta_path]


def get_model_size(model_path: Path) -> float:
    return round(model_path.stat().st_size / 1_048_576, 2)  # MB


def get_last_modified(model_path: Path) -> str:
    return time.ctime(model_path.stat().st_mtime)


def health_check_model(asset_type: str, task: str = "value_regressor") -> dict:
    """
    Returns a health diagnostic for the given model:
    - Load success
    - Metadata presence
    - Size and last modification
    - Training metrics (if available)
    """
    try:
        pipeline = get_pipeline(asset_type, task)
        model_path = _resolve_path(asset_type, task)
        meta = get_model_metadata(asset_type, task)

        return {
            "status": "healthy",
            "model_path": str(model_path),
            "size_mb": get_model_size(model_path),
            "last_modified": get_last_modified(model_path),
            "metadata_valid": bool(meta),
            "metrics": meta.get("metrics") if meta else None,
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def list_asset_types() -> List[str]:
    return list(MODEL_REGISTRY.keys())


def list_tasks(asset_type: str) -> List[str]:
    at = _normalize_key(asset_type)
    if at not in MODEL_REGISTRY:
        raise RegistryLookupError(f"Unknown asset_type: {asset_type}")
    return list(MODEL_REGISTRY[at].keys())


def model_exists(asset_type: str, task: str = "value_regressor") -> bool:
    try:
        _resolve_path(asset_type, task)
        return True
    except (RegistryLookupError, ModelNotFoundError):
        return False


def refresh_cache(asset_type: Optional[str] = None, task: Optional[str] = None) -> None:
    """
    Clear cached entries (all or selected) so next call reloads from disk.
    """
    if asset_type and task:
        try:
            path = _resolve_path(asset_type, task)
            _PIPELINE_CACHE.pop(path, None)
            _METADATA_CACHE.pop(_metadata_path_for(path), None)
        except (RegistryLookupError, ModelNotFoundError):
            pass
    else:
        _PIPELINE_CACHE.clear()
        _METADATA_CACHE.clear()


def cache_stats() -> dict:
    return {
        "pipelines_cached": len(_PIPELINE_CACHE),
        "metadata_cached": len(_METADATA_CACHE),
        "model_paths": [str(p) for p in _PIPELINE_CACHE.keys()]
    }


# -----------------------------------------------------------------------------
# Version / discovery helpers
# -----------------------------------------------------------------------------
def discover_models_for_asset(asset_type: str) -> List[Path]:
    """
    Scan the asset_type directory for *.joblib models (not only those registered).
    """
    at_dir = MODELS_BASE / _normalize_key(asset_type)
    if not at_dir.exists():
        return []
    return sorted(at_dir.glob(f"*{MODEL_EXT}"))


def parse_task_and_version(model_filename: str) -> Optional[Tuple[str, str]]:
    """
    From a filename like value_regressor_v1.joblib -> ("value_regressor", "v1")
    """
    if not model_filename.endswith(MODEL_EXT):
        return None
    m = VERSION_PATTERN.search(model_filename)
    if not m:
        return None
    version = m.group(1)
    task_part = model_filename[: model_filename.rfind("_" + version)]
    return task_part, version


def suggest_task_versions(asset_type: str, task: str) -> List[str]:
    """
    Return a list of versioned model filenames matching a task prefix.
    """
    matches = []
    for p in discover_models_for_asset(asset_type):
        parsed = parse_task_and_version(p.name)
        if parsed:
            t, v = parsed
            if t == task:
                matches.append(p.name)
    return matches


def latest_version_filename(asset_type: str, task: str) -> Optional[str]:
    """
    Heuristic: pick highest version number (lexicographic on extracted version).
    """
    candidates = suggest_task_versions(asset_type, task)
    if not candidates:
        return None
    # Simple lexicographic sort; adapt if version semantics become complex
    return sorted(candidates)[-1]

# -----------------------------------------------------------------------------
# Remote Registry Support
# -----------------------------------------------------------------------------

class RemoteModelRegistry:
    def __init__(self, backend="s3"):
        self.backend = backend

    async def download_model(self, asset_type: str, task: str, version: str):
        """
        Download model from remote backend.
        Implement S3/IPFS/ASA download logic here.
        """
        raise NotImplementedError("Remote download not yet implemented")

    async def check_remote_availability(self, asset_type: str, task: str, version: str) -> bool:
        """
        Check if model exists remotely (stub).
        """
        return False

# -----------------------------------------------------------------------------
# Diagnostics when run directly
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("MODELS_BASE:", MODELS_BASE)
    print("Available asset types (registry):", list_asset_types())
    for at in list_asset_types():
        print(f"\nAsset Type: {at}")
        for t in list_tasks(at):
            exists = model_exists(at, t)
            print(f"  Task: {t} -> exists: {exists}")
            if exists:
                meta = get_model_metadata(at, t)
                if meta:
                    print(f"    version: {meta.get('model_version')} | "
                          f"R²: {meta.get('metrics', {}).get('r2')} | "
                          f"hash: {meta.get('model_hash', '')[:16]}")
        # Discovery (even unregistered)
        discovered = discover_models_for_asset(at)
        if discovered:
            print("  Discovered files:")
            for p in discovered:
                parsed = parse_task_and_version(p.name)
                pv = f"{parsed[0]} ({parsed[1]})" if parsed else "unparsed"
                print(f"    - {p.name} -> {pv}")
    print("\nCache stats:", cache_stats())