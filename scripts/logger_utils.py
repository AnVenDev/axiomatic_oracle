# scripts/logger_utils.py
from __future__ import annotations
"""
Module: logger_utils.py — File-backed logging utilities (atomic & JSON-safe)

Responsibilities
- Persist publication artifacts in both JSONL (append-only) and JSON array formats.
- Save detailed prediction payloads for audit/debug (with SHA-256 for integrity).
- Write minimal audit bundles for on-chain attestations (p1, hashes, canonical input).
- Provide small helpers (atomic writes, file SHA-256).

Design notes
- JSONL is the source of truth for append-only logs (robust under concurrency).
- The JSON array file is kept for backward compatibility with existing tooling/tests.
- All writes are atomic (tmp + os.replace, or O_APPEND for JSONL).
- Timestamps use UTC ISO via shared utils.

SECURITY
- Do not log secrets/PII. Callers must pre-redact sensitive fields.
- Filenames are sanitized to avoid path traversal & invalid characters.

ENV
- OUTPUTS_DIR (default: notebooks/outputs)
- AI_ORACLE_LOG_DIR (default: <OUTPUTS_DIR>/logs)
- AI_ORACLE_DISABLE_API_LOG=1|true to disable all writes (no-op).
"""

# =========================
# Standard library imports
# =========================
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional

# ===========
# Shared deps
# ===========
# Provides: get_utc_now() -> ISO UTC string; NumpyJSONEncoder safe for numpy/pandas objects
from notebooks.shared.common.utils import get_utc_now, NumpyJSONEncoder  # type: ignore

__all__ = [
    "log_asset_publication",
    "append_jsonl",
    "save_publications_to_json",
    "compute_file_hash",
    "save_prediction_detail",
    "save_audit_bundle",
]

# =============================================================================
# Paths (honor AI_ORACLE_LOG_DIR; fallback to notebooks/outputs/logs)
# =============================================================================
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", "notebooks/outputs"))
LOG_DIR = Path(os.getenv("AI_ORACLE_LOG_DIR", OUTPUTS_DIR / "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE_JSON = LOG_DIR / "published_assets.json"      # array (compatibility)
LOG_FILE_JSONL = LOG_DIR / "published_assets.jsonl"    # append-only
DETAIL_DIR = LOG_DIR / "detail_reports"
DETAIL_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Low-level helpers
# =============================================================================
def _atomic_write_json(path: Path, payload: Any) -> None:
    """
    Atomically write a JSON document to `path`.
    Uses a temp file + os.replace (atomic on POSIX/NTFS).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as tmp:
        json.dump(payload, tmp, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)

def _atomic_append_jsonl(payload: Dict[str, Any], path: Path) -> None:
    """
    Append a single JSON line atomically.
    Uses O_APPEND to avoid interleaving under concurrent writers.
    """
    line = json.dumps(payload, cls=NumpyJSONEncoder, ensure_ascii=False)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    with os.fdopen(fd, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def _file_sha256_stream(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 of a file by streaming chunks (default 1 MiB)."""
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def _logging_enabled() -> bool:
    """
    Global kill-switch for file writes.
    AI_ORACLE_DISABLE_API_LOG=1|true → disable.
    """
    return os.getenv("AI_ORACLE_DISABLE_API_LOG", "0").lower() not in {"1", "true", "yes", "y"}

def _safe_name(name: str, *, maxlen: int = 80) -> str:
    """
    Convert any string into a safe filename: allow [a-zA-Z0-9._-] only, trim to `maxlen`.
    Empty results become 'item'.
    """
    base = re.sub(r"[^a-zA-Z0-9._-]", "-", str(name or "").strip())
    base = base.strip("-._") or "item"
    return base[:maxlen]

# =============================================================================
# Public API
# =============================================================================
def log_asset_publication(data: Dict[str, Any]) -> None:
    """
    Log a single publication event:
    - Always appends to JSONL (append-only).
    - Also mirrors to the JSON array file for compatibility with tests/tools.

    NOTE: Caller is responsible for redaction of sensitive fields.
    """
    if not _logging_enabled():
        return

    enriched = {**data, "logged_at": get_utc_now()}

    # JSONL append-only (concurrency-friendly)
    _atomic_append_jsonl(enriched, LOG_FILE_JSONL)

    # Maintain JSON array file (compat)
    items: List[Dict[str, Any]] = []
    if LOG_FILE_JSON.exists():
        try:
            existing = json.loads(LOG_FILE_JSON.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                items = existing
        except Exception:
            items = []
    items.append(enriched)
    _atomic_write_json(LOG_FILE_JSON, items)

def append_jsonl(record: Dict[str, Any], path: Optional[Path] = None) -> None:
    """
    Generic JSONL append with automatic timestamp.
    Useful for inference/monitoring logs.
    """
    if not _logging_enabled():
        return
    path = path or LOG_FILE_JSONL
    payload = {**record, "_logged_at": get_utc_now()}
    _atomic_append_jsonl(payload, path)

def save_publications_to_json(results: List[Dict[str, Any]], filename: Path = LOG_FILE_JSON) -> None:
    """
    Overwrite the JSON array file with a list of result dicts.
    Kept for compatibility with batch flows.
    """
    if not _logging_enabled():
        return
    enriched = [{**r, "logged_at": get_utc_now()} for r in results]
    _atomic_write_json(filename, enriched)

def compute_file_hash(path: Path) -> str:
    """Return SHA-256 hex digest of the file at `path`."""
    return _file_sha256_stream(path)

def save_prediction_detail(prediction: Dict[str, Any], *, filename: Optional[Path] = None) -> str:
    """
    Save the full prediction payload under logs/detail_reports/<asset_id>.json
    (or at a custom `filename`) and return the resulting file's SHA-256.

    Returns "" if logging is disabled.
    """
    if not _logging_enabled():
        return ""
    asset_id = prediction.get("asset_id") or f"asset_{int(datetime.now(timezone.utc).timestamp())}"
    safe_id = _safe_name(asset_id)
    detail_path = filename or (DETAIL_DIR / f"{safe_id}.json")
    detail_path.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(prediction)
    payload["_logged_at"] = get_utc_now()
    _atomic_write_json(detail_path, payload)

    return compute_file_hash(detail_path)

def save_audit_bundle(
    bundle_dir: Path,
    *,
    p1_bytes: bytes,
    p1_sha256: str,
    canonical_input: Dict[str, Any],
    verify_report: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Write a minimal audit bundle into `bundle_dir`:
      - p1.json            (ACJ-1 canonical bytes as-is)
      - p1.sha256          (hex)
      - canonical_input.json
      - verify_report.json (optional)
    Returns the bundle directory as string.

    SECURITY: Do not include PII in canonical_input/verify_report.
    """
    if not _logging_enabled():
        return str(bundle_dir)

    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "p1.json").write_bytes(p1_bytes)
    (bundle_dir / "p1.sha256").write_text(str(p1_sha256), encoding="utf-8")
    _atomic_write_json(bundle_dir / "canonical_input.json", canonical_input)
    if verify_report is not None:
        _atomic_write_json(bundle_dir / "verify_report.json", verify_report)
    return str(bundle_dir)
