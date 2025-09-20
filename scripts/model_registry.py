"""
model_registry.py
Centralized registry & loader for AI Oracle model pipelines (Multi-RWA ready).

Key points
- Models root prefers notebooks outputs (override with env MODELS_ROOT / AI_ORACLE_MODELS_BASE).
- Loads ONLY fitted models (fallback to newest fitted among discovered candidates).
- Registry optional; graceful discovery if missing.
- Exposes expected feature lists (pref: shared spec → training_manifest.json → meta.json).
- Enriches metadata with sha256, size, timestamps, and validation metrics.

SECURITY
- No secret material handled here; FS reads only. Do not log PII.

NOTE
- Public functions used by the API layer:
    get_pipeline, get_model_paths, get_model_metadata, health_check_model,
    list_tasks, discover_models_for_asset, refresh_cache, cache_stats,
    expected_features
"""

from __future__ import annotations

# =========================
# Standard library imports
# =========================
import hashlib
import json
import logging
import os
import re
import time
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ===========
# Third-party
# ===========
import joblib  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore

# =============================================================================
# Configuration
# =============================================================================

def _resolve_models_root() -> Path:
    """
    Resolve the base directory for models.
    Order:
      0) env MODELS_ROOT / AI_ORACLE_MODELS_BASE
      1) notebooks/outputs/modeling
      2) ./notebooks/outputs/modeling
      3) ../notebooks/outputs/modeling
      4) outputs/modeling
      5) ../shared/outputs/models
      6) ./shared/outputs/models
      7) ../models
      8) ./models
    First directory that exists wins.
    """
    candidates: List[Path] = []
    env_root = os.getenv("MODELS_ROOT") or os.getenv("AI_ORACLE_MODELS_BASE")
    if env_root and env_root.strip():
        candidates.append(Path(env_root))

    candidates += [
        Path("notebooks/outputs/modeling"),
        Path("./notebooks/outputs/modeling"),
        Path("../notebooks/outputs/modeling"),
        Path("outputs/modeling"),
        Path("../shared/outputs/models"),
        Path("./shared/outputs/models"),
        Path("../models"),
        Path("./models"),
    ]

    for c in candidates:
        if c.exists():
            return c.resolve()

    return Path("./models").resolve()


MODELS_BASE: Path = _resolve_models_root()

MODEL_EXT = ".joblib"
META_SUFFIX = "_meta.json"
VERSION_RE = re.compile(r"_(v\d+(?:[a-z0-9\-_\.]*)?)\.joblib$", re.IGNORECASE)
TASK_DEFAULT = "value_regressor"

logger = logging.getLogger("model_registry")
if not logger.handlers:
    logger.setLevel(logging.INFO)

# Optional registry: (asset_type, task) -> relative path under MODELS_BASE
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


def _is_fitted_pipeline(pl: Any) -> bool:
    """
    Best-effort 'is fitted' check. If pipeline has steps, check last estimator;
    else check the object itself.
    """
    try:
        if hasattr(pl, "steps"):
            check_is_fitted(pl.steps[-1][1])
        else:
            check_is_fitted(pl)
        return True
    except Exception:
        return False


def _metadata_path_for(model_path: Path) -> Path:
    """property/value_regressor_v1.joblib -> property/value_regressor_v1_meta.json"""
    stem = model_path.name.replace(MODEL_EXT, "")
    return model_path.parent / f"{stem}{META_SUFFIX}"


def _find_manifest(models_base: Path, asset_type: str) -> Path:
    """
    Look for training_manifest.json in common locations:
      1) <models_base>/<asset_type>/training_manifest.json
      2) <models_base>/training_manifest.json
      3) <models_base>/artifacts/training_manifest.json
      4) ../shared/outputs/<asset_type>/training_manifest.json
      5) ./shared/outputs/<asset_type>/training_manifest.json
    Returns the first existing path or the first candidate if none exist.
    """
    at = _normalize_key(asset_type)
    candidates = [
        models_base / at / "training_manifest.json",
        models_base / "training_manifest.json",
        models_base / "artifacts" / "training_manifest.json",
        Path("../shared/outputs") / at / "training_manifest.json",
        Path("./shared/outputs") / at / "training_manifest.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _resolve_registered(asset_type: str, task: str) -> Optional[Path]:
    at = _normalize_key(asset_type)
    tk = _normalize_key(task)
    try:
        rel = MODEL_REGISTRY[at][tk]
        full = MODELS_BASE / rel
        return full if full.exists() else None
    except KeyError:
        return None


def _list_version_files(asset_type: str, task: str) -> List[Path]:
    """
    Collect plausible model candidates:
      - <MODELS_BASE>/<asset_type>/<task>_v*.joblib
      - <MODELS_BASE>/<asset_type>/*.joblib
      - <MODELS_BASE>/artifacts/*.joblib  (notebooks style)
    Sorted by modification time (newest first).
    """
    at_dir = MODELS_BASE / _normalize_key(asset_type)
    cands: List[Path] = []
    if at_dir.exists():
        cands += list(at_dir.glob(f"{task}_v*{MODEL_EXT}"))
        cands += list(at_dir.glob(f"*{MODEL_EXT}"))
    art = MODELS_BASE / "artifacts"
    if art.exists():
        cands += list(art.glob(f"*{MODEL_EXT}"))

    # Unique + sort by mtime desc
    uniq: Dict[Path, Path] = {}
    for p in cands:
        try:
            uniq[p.resolve()] = p
        except Exception:
            uniq[p] = p
    cands = list(uniq.values())
    cands.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    return cands


def _resolve_path(asset_type: str, task: str, preferred_version: Optional[str] = None) -> Path:
    """
    Resolution order:
      1) Preferred version (if provided)
      2) Registry entry
      3) Discovery: newest **fitted** among plausible candidates
    """
    at = _normalize_key(asset_type)
    tk = _normalize_key(task)

    # 1) preferred explicit file
    if preferred_version:
        p = MODELS_BASE / at / f"{tk}_{preferred_version}.joblib"
        if p.exists():
            return p
        logger.warning("Preferred version not found: %s", p)

    # 2) registry
    reg = _resolve_registered(at, tk)
    if reg is not None:
        if reg.exists():
            return reg
        logger.warning("Registry path missing: %s", reg)

    # 3) discovery → pick first that is fitted
    for cand in _list_version_files(at, tk):
        try:
            pl = joblib.load(cand)
            if _is_fitted_pipeline(pl):
                return cand
        except Exception:
            continue

    raise ModelNotFoundError(f"No fitted model found for '{asset_type}/{task}' under {MODELS_BASE}.")


def _feature_order_path_for(model_path: Path) -> Path:
    """property/value_regressor_v1.joblib -> property/feature_order.json"""
    return model_path.parent / "feature_order.json"


# =============================================================================
# Public API
# =============================================================================
def get_pipeline(
    asset_type: str,
    task: str = TASK_DEFAULT,
    *,
    preferred_version: Optional[str] = None,
) -> Any:
    """Return a loaded (and TTL-cached) fitted model pipeline."""
    model_path = _resolve_path(asset_type, task, preferred_version=preferred_version)
    now = time.time()
    cache_key = str(model_path.resolve())
    if cache_key in _PIPELINE_TTL_CACHE:
        pipeline, ts = _PIPELINE_TTL_CACHE[cache_key]
        if now - float(ts) < CACHE_TTL_SECONDS:
            return pipeline

    pipeline = joblib.load(model_path)
    if not _is_fitted_pipeline(pipeline):
        # Try fallback among other candidates
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
    logger.info("Model loaded: %s", model_path.name)
    return pipeline


def get_model_paths(
    asset_type: str,
    task: str = TASK_DEFAULT,
    *,
    preferred_version: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Return dict with 'pipeline', 'meta', 'manifest' (and 'feature_order' if present)
    paths for the resolved model.
    """
    pipeline_path = _resolve_path(asset_type, task, preferred_version=preferred_version)
    meta_path = _metadata_path_for(pipeline_path)
    manifest_path = _find_manifest(MODELS_BASE, asset_type)
    forder_path = _feature_order_path_for(pipeline_path)

    out: Dict[str, Path] = {"pipeline": pipeline_path, "meta": meta_path, "manifest": manifest_path}
    if forder_path.exists():
        out["feature_order"] = forder_path
    return out


def _features_from_shared(asset_type: str) -> Optional[tuple[list[str], list[str]]]:
    """
    Try to read feature spec from /shared if available:
      1) shared.common.schema.get_feature_spec(asset_type) -> {"categorical":[...], "numeric":[...]}
      2) shared.common.schema.FEATURE_SPEC[asset_type]
      3) shared.common.schema.FEATURES_CATEGORICAL/NUMERIC
      4) shared.common.config.* (same shapes)

    Returns (categorical, numeric) or None.
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
                # tolerate typos like 'categororical'
                cat = list(spec.get("categororical", spec.get("categorical", [])) or [])
                num = list(spec.get("numeric", []) or [])
                return cat, num
            elif mode == "split":
                cats = getattr(mod, "FEATURES_CATEGORICAL", None)
                nums = getattr(mod, "FEATURES_NUMERIC", None)
                if (
                    isinstance(cats, dict)
                    and isinstance(nums, dict)
                    and asset_type in cats
                    and asset_type in nums
                ):
                    return list(cats[asset_type] or []), list(nums[asset_type] or [])
        except Exception:
            continue
    return None


def expected_features(
    meta: dict,
    manifest_path: Path,
    asset_type: Optional[str] = None,
) -> tuple[list[str], list[str]]:
    """
    Returns (categorical, numeric).
    Priority:
      1) shared spec (if available for asset_type)
      2) training_manifest.json (model.feature_list or model.features)
      3) meta.json (features_categorical / features_numeric)
    """
    if asset_type:
        shared_spec = _features_from_shared(asset_type)
        if shared_spec:
            cat, num = shared_spec
            seen = set()
            cat = [c for c in cat if not (c in seen or seen.add(c))]
            num = [c for c in num if c not in set(cat)]
            return cat, num

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

    seen = set()
    cat = [c for c in cat if not (c in seen or seen.add(c))]
    num = [c for c in num if c not in set(cat)]
    return cat, num


def _load_nb_outputs(asset_type: str) -> dict:
    """
    Load additional notebook outputs (metrics/dataset stats) to enrich metadata:
      ../shared/outputs/<asset>/metrics/*.json
      ../shared/outputs/<asset>/dataset_stats.json
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


def get_feature_order(
    asset_type: str,
    task: str = TASK_DEFAULT,
    *,
    preferred_version: Optional[str] = None,
) -> list[str]:
    """
    Resolve the *raw* feature order to be used for canonical input/hash.
    Priority:
      1) meta.json -> "feature_order" (array)
      2) training_manifest.json -> paths.feature_order_path (file JSON)
      3) well-known file: <MODELS_BASE>/<asset_type>/feature_order.json
    Returns [] if none found.
    """
    paths = get_model_paths(asset_type, task, preferred_version=preferred_version)
    meta_path = paths["meta"]
    manifest_path = paths["manifest"]

    # 1) inline meta.json
    try:
        if meta_path.exists():
            md = json.loads(meta_path.read_text(encoding="utf-8"))
            fo = md.get("feature_order")
            if isinstance(fo, list) and fo:
                seen = set()
                return [str(x) for x in fo if not (x in seen or seen.add(x))]
    except Exception:
        pass

    # 2) manifest -> external file path
    def _read_fo_file(p: Path) -> list[str]:
        try:
            if p.exists():
                raw = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    seen = set()
                    return [str(x) for x in raw if not (x in seen or seen.add(x))]
                if isinstance(raw, dict):
                    arr = raw.get("feature_order") or raw.get("features") or []
                    if isinstance(arr, list) and arr:
                        seen = set()
                        return [str(x) for x in arr if not (x in seen or seen.add(x))]
        except Exception:
            pass
        return []

    if manifest_path.exists():
        try:
            mf = json.loads(manifest_path.read_text(encoding="utf-8"))
            pstr = (mf.get("paths") or {}).get("feature_order_path")
            if pstr:
                cand = Path(pstr)
                # robust resolution: relative to manifest, models base, or asset dir
                for base in [manifest_path.parent, MODELS_BASE, MODELS_BASE / _normalize_key(asset_type)]:
                    p = (base / pstr).resolve() if not cand.is_absolute() else cand
                    if p.exists():
                        fo = _read_fo_file(p)
                        if fo:
                            return fo
                # last resort: absolute path
                if cand.is_absolute() and cand.exists():
                    fo = _read_fo_file(cand)
                    if fo:
                        return fo
        except Exception:
            pass

    # 3) default well-known path
    default_fo = (MODELS_BASE / _normalize_key(asset_type) / "feature_order.json").resolve()
    fo = _read_fo_file(default_fo)
    if fo:
        return fo

    return []


def get_model_metadata(
    asset_type: str,
    task: str = TASK_DEFAULT,
    *,
    preferred_version: Optional[str] = None,
) -> Optional[dict]:
    """
    Aggregate metadata for a resolved model:
    - model path, sha256, size, last_modified
    - metrics (prefer manifest → notebooks → meta.json)
    - feature order & counts
    Fields are normalized for API/UI expectations.
    """
    paths = get_model_paths(asset_type, task, preferred_version=preferred_version)
    meta_path = paths["meta"]
    pipeline_path = paths["pipeline"]
    manifest_path = paths["manifest"]

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
            md["last_modified"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(pipeline_path.stat().st_mtime)
            )
        except Exception:
            pass

        # Prefer metrics from manifest → then notebooks → then meta.json
        try:
            if manifest_path.exists():
                mf = json.loads(manifest_path.read_text(encoding="utf-8"))
                metrics = (mf.get("metrics") or {}).get("validation") or (mf.get("metrics") or {}).get("valid") or {}
                if metrics:
                    md.setdefault("metrics", metrics)
        except Exception:
            pass

        try:
            nb_out = _load_nb_outputs(asset_type)
            if nb_out.get("metrics"):
                md.setdefault("metrics", {}).update(nb_out["metrics"])
            if nb_out.get("dataset_stats"):
                md.setdefault("dataset_stats", nb_out["dataset_stats"])
        except Exception:
            pass

        # --- Feature order & n_features_total --------------------------------
        try:
            forder = get_feature_order(asset_type, task, preferred_version=preferred_version)
        except Exception:
            forder = []
        if forder:
            md.setdefault("feature_order", forder)
            md.setdefault("n_features", len(forder))

        # --- Normalize names for API/UI expectations -------------------------
        vmv = md.get("model_version") or md.get("value_model_version")
        vmn = md.get("model_class") or md.get("value_model_name")
        nft = md.get("n_features") or (
            len(md.get("feature_order", []))
            if isinstance(md.get("feature_order"), list)
            else None
        )
        if vmv:
            md.setdefault("value_model_version", vmv)
        if vmn:
            md.setdefault("value_model_name", vmn)
        if nft:
            md.setdefault("n_features_total", int(nft))

        _METADATA_CACHE[meta_path] = md

    return _METADATA_CACHE[meta_path]


def validate_model_compatibility(pipeline: Any, expected_features_list: List[str]) -> bool:
    """Order-insensitive check for feature set compatibility (best effort)."""
    if hasattr(pipeline, "feature_names_in_"):
        pipeline_features = list(pipeline.feature_names_in_)
        return set(pipeline_features) == set(expected_features_list)
    return True


def health_check_model(
    asset_type: str,
    task: str = TASK_DEFAULT,
    *,
    preferred_version: Optional[str] = None,
) -> dict:
    """Quick diagnostic for a given model."""
    try:
        paths = get_model_paths(asset_type, task, preferred_version=preferred_version)
        pipeline = get_pipeline(asset_type, task, preferred_version=preferred_version)
        meta = get_model_metadata(asset_type, task, preferred_version=preferred_version)

        return {
            "status": "healthy",
            "model_path": str(paths["pipeline"]),
            "size_mb": round(paths["pipeline"].stat().st_size / (1024 * 1024), 2),
            "last_modified": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(paths["pipeline"].stat().st_mtime)
            ),
            "metadata_valid": bool(meta),
            "metrics": (meta or {}).get("metrics"),
            "n_features_total": (meta or {}).get("n_features_total") or (meta or {}).get("n_features"),
            "fitted": _is_fitted_pipeline(pipeline),
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


# =============================================================================
# Discovery / helpers (backward-compatible)
# =============================================================================
def list_asset_types() -> List[str]:
    assets = set(MODEL_REGISTRY.keys())
    if MODELS_BASE.exists():
        for p in MODELS_BASE.iterdir():
            if p.is_dir():
                assets.add(p.name)
    return sorted(assets)


def discover_models_for_asset(asset_type: str) -> List[Path]:
    at_dir = MODELS_BASE / _normalize_key(asset_type)
    res: List[Path] = []
    if at_dir.exists():
        res += list(at_dir.glob(f"*{MODEL_EXT}"))
    art = MODELS_BASE / "artifacts"
    if art.exists():
        res += list(art.glob(f"*{MODEL_EXT}"))
    # unique
    seen: set[Path] = set()
    out: List[Path] = []
    for p in res:
        key = p.resolve()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return sorted(out)


def parse_task_and_version(model_filename: str) -> Optional[Tuple[str, str]]:
    """value_regressor_v1.joblib -> ('value_regressor', 'v1')"""
    if not model_filename.endswith(MODEL_EXT):
        return None
    m = VERSION_RE.search(model_filename)
    if not m:
        return None
    version = m.group(1)
    task_part = model_filename[: model_filename.rfind("_" + version)]
    return task_part, version


def list_tasks(asset_type: str) -> List[str]:
    at = _normalize_key(asset_type)
    tasks = set(MODEL_REGISTRY.get(at, {}).keys())
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
    """Clear cached entries (all or selected)."""
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


# =============================================================================
# Diagnostics
# =============================================================================
if __name__ == "__main__":
    print("MODELS_BASE:", MODELS_BASE)
    print("Available asset types:", list_asset_types())
    for at in list_asset_types():
        print(f"\nAsset Type: {at}")
        try:
            tasks = list_tasks(at)
        except Exception as e:
            print("  (no tasks)", e)
            continue
        for t in tasks:
            ok = model_exists(at, t)
            print(f"  Task: {t} -> exists: {ok}")
            if ok:
                paths = get_model_paths(at, t)
                meta = get_model_metadata(at, t)
                print(
                    f"    pipeline: {paths['pipeline'].name} | "
                    f"version: {(meta or {}).get('model_version')} | "
                    f"hash: {(meta or {}).get('model_hash', '')[:16]} | "
                    f"n_features_total: {(meta or {}).get('n_features_total')}"
                )
    print("\nCache stats:", cache_stats())
