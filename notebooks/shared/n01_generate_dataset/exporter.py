from __future__ import annotations
import logging

from shared.common.utils import canonical_json_dumps, sha256_hex

"""
Export utilities:
- Atomic writes for CSV/Parquet
- Quality report + manifest with checksums and metadata
- Overwrite protection and robust path handling
"""

import io
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import pandas as pd  # type: ignore

from shared.common.constants import SCHEMA_VERSION, Versions
from shared.common.schema import get_required_fields
from shared.common.quality import get_top_outliers

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Atomic writers
# --------------------------------------------------------------------------- #

def _atomic_replace(tmp: Path, dest: Path) -> None:
    """Atomically move the temporary file into place (same filesystem)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, dest)


def atomic_write_csv(path: Path, df: pd.DataFrame, *, index: bool = False) -> None:
    """Write CSV atomically: write to path.tmp then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=index)
    _atomic_replace(tmp, path)


def atomic_write_parquet(path: Path, df: pd.DataFrame, *, compression: Optional[str] = None) -> None:
    """Write Parquet atomically (optional compression)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    kwargs: Dict[str, Any] = {}
    if compression:
        kwargs["compression"] = compression
    try:
        df.to_parquet(tmp, index=False, **kwargs)
    except Exception as e:
        # Surface a clear hint if an engine is missing
        raise RuntimeError(
            f"Failed to write Parquet. Ensure a parquet engine is installed (pyarrow or fastparquet). Details: {e}"
        ) from e
    _atomic_replace(tmp, path)


def write_json(path: Path, obj: Any) -> None:
    """
    Atomically write JSON using canonical, deterministic serialization.
    - Ensures robust handling of numpy/pandas types.
    - Stable key ordering and minimal separators.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(canonical_json_dumps(obj), encoding="utf-8")
    _atomic_replace(tmp, path)


# --------------------------------------------------------------------------- #
# Hashing helpers
# --------------------------------------------------------------------------- #

def compute_file_hash(path: Path, algo: str = "sha256") -> str:
    """Compute a file hash in blocks (default sha256)."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dataframe_digest(df: pd.DataFrame, *, algo: str = "sha256") -> str:
    """
    Compute a stable content digest over the DataFrame rows/columns.
    - Sorts columns for stability.
    - Uses in-memory CSV to avoid parquet engine dependency.
    """
    h = hashlib.new(algo)
    cols = sorted(df.columns)
    buf = io.StringIO()
    df.to_csv(buf, index=False, columns=cols)
    h.update(buf.getvalue().encode("utf-8"))
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _validate_required_fields(df: pd.DataFrame, asset_type: str) -> Tuple[bool, Iterable[str]]:
    required = get_required_fields(asset_type)
    missing = [f for f in required if f not in df.columns]
    return (len(missing) == 0), missing


def _sanitize_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Make config JSON-serializable (e.g., Path → str), recursively."""
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


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

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
    Persist the dataset, quality report (JSON [+ YAML if available]),
    optional outliers CSV, and a manifest with checksums and metadata.

    Returns:
        A manifest (dict) including paths, checksums, and summary metadata.
    """
    asset_type = str(config.get("asset_type", "property"))
    required_ok, missing = _validate_required_fields(df, asset_type)
    if not required_ok:
        logger.warning("Dataset missing required fields for %s: %s", asset_type, missing)

    # Reorder columns: required first, then extras (without dropping any)
    required_fields = list(get_required_fields(asset_type))
    extras = [c for c in df.columns if c not in required_fields]
    ordered_cols = required_fields + extras
    df_to_save = df.loc[:, ordered_cols].copy()

    # In-memory payload hash (format-aware, always computed)
    fmt = format.lower()
    if fmt == "csv":
        payload_bytes = df_to_save.to_csv(index=index).encode("utf-8")
    elif fmt == "parquet":
        bio = io.BytesIO()
        kwargs: Dict[str, Any] = {}
        if compression:
            kwargs["compression"] = compression
        try:
            df_to_save.to_parquet(bio, index=False, **kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to serialize Parquet in-memory. Install a parquet engine (pyarrow/fastparquet). Details: {e}"
            ) from e
        payload_bytes = bio.getvalue()
    else:
        raise ValueError(f"Unsupported format: {format}")
    dataset_sha256 = sha256_hex(payload_bytes)

    # Paths: support both legacy flat and typed settings
    paths = config.get("paths") or {}
    output_path = Path(paths.get("output_path") or config.get("output_path") or "data/dataset.csv")
    snapshot_dir = Path(paths.get("snapshot_dir") or config.get("snapshot_dir") or "data/snapshots")
    log_dir = Path(paths.get("log_dir") or config.get("log_dir") or "logs")

    # Overwrite protection
    if no_overwrite and output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")

    # Write dataset (atomic)
    if fmt == "csv":
        if output_path.suffix.lower() != ".csv":
            output_path = output_path.with_suffix(".csv")
        atomic_write_csv(output_path, df_to_save, index=index)
    else:  # parquet
        if output_path.suffix.lower() != ".parquet":
            output_path = output_path.with_suffix(".parquet")
        atomic_write_parquet(output_path, df_to_save, compression=compression)

    logger.info("✅ Saved dataset to %s", output_path)

    # Quality report (JSON, always) — atomic & canonical
    quality_json = log_dir / "quality_report.json"
    write_json(quality_json, dict(report))
    logger.info("✅ Saved quality report JSON to %s", quality_json)

    # YAML (best-effort)
    quality_yaml = log_dir / "quality_report.yaml"
    has_yaml = False
    try:
        import yaml  # type: ignore
        quality_yaml.parent.mkdir(parents=True, exist_ok=True)
        with open(quality_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(dict(report), f, sort_keys=False, allow_unicode=True)
        logger.info("✅ Saved quality report YAML to %s", quality_yaml)
        has_yaml = True
    except Exception:
        logger.warning("PyYAML not available or failed; skipping YAML report.")
        has_yaml = False

    # Outliers (best-effort)
    outliers_path: Optional[Path] = None
    try:
        n = int(config.get("top_outliers_n", 30))
        outliers_df = get_top_outliers(df_to_save, n=n)
        outliers_path = log_dir / "top_outliers.csv"
        atomic_write_csv(outliers_path, outliers_df, index=False)
        logger.info("✅ Saved top %d outliers to %s", n, outliers_path)
    except Exception as e:
        logger.debug("Skipping outliers export: %s", e)

    # Checksums
    dataset_file_sha256 = compute_file_hash(output_path)
    quality_report_sha256 = compute_file_hash(quality_json)
    data_digest_sha256 = compute_dataframe_digest(df_to_save)

    # Manifest
    manifest: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "versions": {
            "dataset": Versions.DATASET,
            "feature_set": Versions.FEATURE_SET,
            "model_family": Versions.MODEL_FAMILY,
        },
        "schema_version": SCHEMA_VERSION,
        "asset_type": asset_type,
        "paths": {
            "dataset": str(output_path),
            "quality_report_json": str(quality_json),
        },
        "hashes": {
            # On-disk files
            "dataset_file_sha256": dataset_file_sha256,
            "quality_report_sha256": quality_report_sha256,
            # Logical payload/digest
            "dataset_payload_sha256": dataset_sha256,
            "data_digest_sha256": data_digest_sha256,
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
        manifest["paths"]["quality_report_yaml"] = str(quality_yaml)
    if outliers_path is not None:
        manifest["paths"]["top_outliers_csv"] = str(outliers_path)

    # Manifest hash over canonical JSON
    manifest_bytes = canonical_json_dumps(manifest).encode("utf-8")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest["manifest_hash"] = manifest_hash

    # Persist manifest in snapshot dir (atomic)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    manifest_name = f"manifest_{datetime.utcnow():%Y%m%dT%H%M%SZ}.json"
    manifest_path = snapshot_dir / manifest_name
    write_json(manifest_path, manifest)
    logger.info("✅ Saved manifest to %s (hash=%s)", manifest_path, manifest_hash)

    return manifest