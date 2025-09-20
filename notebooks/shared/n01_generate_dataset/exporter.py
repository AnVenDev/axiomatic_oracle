from __future__ import annotations
import logging

from notebooks.shared.common.utils import canonical_json_dumps, sha256_hex

"""
Export utilities:
- Atomic writes for CSV/Parquet
- Quality report + manifest with checksums and metadata
- Overwrite protection and robust path handling
"""

import io
import json
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import pandas as pd # type: ignore

from notebooks.shared.common.constants import SCHEMA_VERSION, Versions
from notebooks.shared.common.schema import get_required_fields
from notebooks.shared.common.quality import get_top_outliers


logger = logging.getLogger(__name__)

def _atomic_replace(tmp: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, dest)

def atomic_write_csv(path: Path, df: pd.DataFrame, *, index: bool = False) -> None:
    """
    Scrive CSV in modo atomico: path.tmp → rename.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=index)
    _atomic_replace(tmp, path)

def atomic_write_parquet(path: Path, df: pd.DataFrame, *, compression: Optional[str] = None) -> None:
    """
    Scrive Parquet in modo atomico (compressione opzionale).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    kwargs = {}
    if compression:
        kwargs["compression"] = compression
    df.to_parquet(tmp, index=False, **kwargs)
    _atomic_replace(tmp, path)

def write_json(path: Path, obj: Any) -> None:
    """
    Scrive JSON con indentazione; crea la dir se necessario.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)

def compute_file_hash(path: Path, algo: str = "sha256") -> str:
    """
    Calcola hash del file a blocchi (default sha256).
    """
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def compute_dataframe_digest(df: pd.DataFrame, *, algo: str = "sha256") -> str:
    """
    Calcola un digest stabile del contenuto del DataFrame (ordinando colonne).
    Utile per legare modello/dataset senza affidarsi solo all'hash del file.
    """
    h = hashlib.new(algo)
    cols = sorted(df.columns)
    # Scrivi in CSV in-memory per evitare dipendenza da parquet engine
    buf = io.StringIO()
    df.to_csv(buf, index=False, columns=cols)
    h.update(buf.getvalue().encode("utf-8"))
    return h.hexdigest()

def _validate_required_fields(df: pd.DataFrame, asset_type: str) -> Tuple[bool, Iterable[str]]:
    required = get_required_fields(asset_type)
    missing = [f for f in required if f not in df.columns]
    return (len(missing) == 0), missing

def _sanitize_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Rende il config JSON-serializzabile (Path→str, ecc.).
    """
    out: Dict[str, Any] = {}
    for k, v in dict(cfg).items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [str(x) if isinstance(x, Path) else x for x in v]
        elif isinstance(v, dict):
            out[k] = _sanitize_config(v)
        else:
            out[k] = v
    return out

def export_dataset(
    df: pd.DataFrame,
    config: Mapping[str, Any],
    report: Mapping[str, Any],
    logger,
    *,
    format: str = "csv",           # "csv" | "parquet"
    compression: Optional[str] = None,
    index: bool = False,
    no_overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Persist dataset, quality report (JSON + YAML se disponibile), outliers opzionali e manifest.

    Ritorna:
        manifest (dict) completo con checksum e metadati.
    """
    asset_type = str(config.get("asset_type", "property"))
    required_ok, missing = _validate_required_fields(df, asset_type)
    if not required_ok:
        logger.warning("Dataset missing required fields for %s: %s", asset_type, missing)

    # Riordino colonne: required first, poi extras (senza perdere colonne)
    required_fields = list(get_required_fields(asset_type))
    extras = [c for c in df.columns if c not in required_fields]
    ordered_cols = required_fields + extras
    df_to_save = df.loc[:, ordered_cols].copy()

    payload_bytes = df_to_save.to_csv(index=index).encode("utf-8") if format=="csv" else b""
    dataset_sha256 = sha256_hex(payload_bytes) if payload_bytes else None


    # Paths (accettiamo sia legacy flat che settings tipizzati)
    paths = config.get("paths") or {}
    output_path = Path(paths.get("output_path") or config.get("output_path") or "data/dataset.csv")
    snapshot_dir = Path(paths.get("snapshot_dir") or config.get("snapshot_dir") or "data/snapshots")
    log_dir = Path(paths.get("log_dir") or config.get("log_dir") or "logs")

    # Protezione overwrite
    if no_overwrite and output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")

    # Scrittura dataset (atomica)
    if format.lower() == "csv":
        if output_path.suffix.lower() != ".csv":
            output_path = output_path.with_suffix(".csv")
        atomic_write_csv(output_path, df_to_save, index=index)
    elif format.lower() == "parquet":
        if output_path.suffix.lower() != ".parquet":
            output_path = output_path.with_suffix(".parquet")
        atomic_write_parquet(output_path, df_to_save, compression=compression)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info("✅ Saved dataset to %s", output_path)

    # Quality report JSON
    quality_json = log_dir / "quality_report.json"
    write_json(quality_json, dict(report))
    logger.info("✅ Saved quality report JSON to %s", quality_json)

    # YAML (best-effort)
    quality_yaml = log_dir / "quality_report.yaml"
    try:
        import yaml  # type: ignore
        with open(quality_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(dict(report), f, sort_keys=False, allow_unicode=True)
        logger.info("✅ Saved quality report YAML to %s", quality_yaml)
        has_yaml = True
    except Exception:
        logger.warning("PyYAML not available or failed; skipping YAML report.")
        has_yaml = False

    # Outliers (best-effort, se modulo disponibile)
    try:
        n = int(config.get("top_outliers_n", 30))
        outliers_df = get_top_outliers(df_to_save, n=n)
        outliers_path = log_dir / "top_outliers.csv"
        atomic_write_csv(outliers_path, outliers_df, index=False)
        logger.info("✅ Saved top %d outliers to %s", n, outliers_path)
    except Exception as e:
        logger.debug("Skipping outliers export: %s", e)

    # Manifests & checksums
    dataset_hash = compute_file_hash(output_path)
    report_hash = compute_file_hash(quality_json)
    data_digest = compute_dataframe_digest(df_to_save)

    manifest: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "versions": {
            "dataset": Versions.DATASET,
            "feature_set": Versions.FEATURE_SET,
            "model_family": Versions.MODEL_FAMILY,
        },
        "asset_type": asset_type,
        "paths": {
            "dataset": str(output_path),
            "quality_report_json": str(quality_json),
        },
        "hashes": {
            "dataset_file_sha256": dataset_hash,
            "quality_report_sha256": report_hash,
            "data_digest_sha256": data_digest,
        },
        "shape": {
            "rows": int(len(df_to_save)),
            "columns": list(map(str, df_to_save.columns)),
        },
        "config_snapshot": _sanitize_config(config),
        "report_summary": {
            "keys": sorted(list(report.keys())),
        },
    }

    if has_yaml:
        manifest.setdefault("paths", {})["quality_report_yaml"] = str(quality_yaml)

    manifest.update({
        "schema_version": SCHEMA_VERSION,
        "checksums": {"dataset_sha256": dataset_sha256},
    })

    # Manifest hash (sul JSON del manifest)
    manifest_bytes = canonical_json_dumps(manifest).encode("utf-8")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest["manifest_hash"] = manifest_hash

    # Persist manifest in snapshot dir
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    manifest_name = f"manifest_{datetime.utcnow():%Y%m%dT%H%M%SZ}.json"
    manifest_path = snapshot_dir / manifest_name
    write_json(manifest_path, manifest)
    logger.info("✅ Saved manifest to %s (hash=%s)", manifest_path, manifest_hash)

    return manifest