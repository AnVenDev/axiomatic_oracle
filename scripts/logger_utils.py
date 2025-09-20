# scripts/logger_utils.py
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional, Any
import re

# Shared utils: timestamp + JSON encoder robusto a numpy
from notebooks.shared.common.utils import get_utc_now, NumpyJSONEncoder

# -----------------------------------------------------------------------------
# Paths (rispetta AI_ORACLE_LOG_DIR; fallback: notebooks/outputs/logs)
# -----------------------------------------------------------------------------
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", "notebooks/outputs"))
LOG_DIR = Path(os.getenv("AI_ORACLE_LOG_DIR", OUTPUTS_DIR / "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE_JSON = LOG_DIR / "published_assets.json"     # array
LOG_FILE_JSONL = LOG_DIR / "published_assets.jsonl"   # append-only JSONL
DETAIL_DIR = LOG_DIR / "detail_reports"
DETAIL_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Low-level helpers
# -----------------------------------------------------------------------------
def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as tmp:
        json.dump(payload, tmp, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)  # atomico su NTFS/POSIX

def _atomic_append_jsonl(payload: dict, path: Path) -> None:
    """Append atomico (O_APPEND) in formato JSONL."""
    line = json.dumps(payload, cls=NumpyJSONEncoder, ensure_ascii=False)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    with os.fdopen(fd, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def _file_sha256_stream(path: Path, chunk_size: int = 1 << 20) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def _logging_enabled() -> bool:
    # AI_ORACLE_DISABLE_API_LOG=1/true -> disabilita
    return os.getenv("AI_ORACLE_DISABLE_API_LOG", "0").lower() not in {"1", "true", "yes", "y"}

def _safe_name(name: str, *, maxlen: int = 80) -> str:
    """
    Converte una stringa in un filename sicuro: [a-zA-Z0-9._-] soltanto, tronca a maxlen.
    Rimpiazza sequenze vuote con 'item'.
    """
    base = re.sub(r"[^a-zA-Z0-9._-]", "-", str(name or "").strip())
    base = base.strip("-._") or "item"
    return base[:maxlen]

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def log_asset_publication(data: dict) -> None:
    """
    Logga una singola pubblicazione:
    - Sempre appende su JSONL (append-only, atomic)
    - Sempre aggiorna anche il file JSON array (compatibilità test)
    """
    if not _logging_enabled():
        return

    enriched = {**data, "logged_at": get_utc_now()}

    # JSONL append-only (robusto in concorrenza)
    _atomic_append_jsonl(enriched, LOG_FILE_JSONL)

    # Mantieni anche il formato array (come si aspettano i test)
    items: List[dict] = []
    if LOG_FILE_JSON.exists():
        try:
            existing = json.loads(LOG_FILE_JSON.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                items = existing
        except Exception:
            items = []
    items.append(enriched)
    _atomic_write_json(LOG_FILE_JSON, items)

def append_jsonl(record: dict, path: Optional[Path] = None) -> None:
    """
    Append generico di una riga JSONL (con timestamp automatico).
    Utile per logging di predizioni/monitoring.
    """
    if not _logging_enabled():
        return
    path = path or LOG_FILE_JSONL
    payload = {**record, "_logged_at": get_utc_now()}
    _atomic_append_jsonl(payload, path)

def save_publications_to_json(results: List[Dict], filename: Path = LOG_FILE_JSON) -> None:
    """
    Sovrascrive il file JSON (array) con una lista di risultati.
    (Manteniamo per compatibilità con flussi batch già esistenti.)
    """
    if not _logging_enabled():
        return
    enriched = [{**r, "logged_at": get_utc_now()} for r in results]
    _atomic_write_json(filename, enriched)

def compute_file_hash(path: Path) -> str:
    """SHA256 del file (streaming)."""
    return _file_sha256_stream(path)

def save_prediction_detail(prediction: dict, *, filename: Optional[Path] = None) -> str:
    """
    Salva il payload completo della predizione in logs/detail_reports/<asset_id>.json
    (o in un path custom) e ritorna l'hash SHA256 del file salvato.
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
    canonical_input: dict,
    verify_report: Optional[dict] = None,
) -> str:
    """
    Scrive un audit bundle minimale nella cartella indicata:
      - p1.json            (byte ACJ-1)
      - p1.sha256
      - canonical_input.json
      - verify_report.json (opzionale)
    Ritorna il path della cartella (stringa).
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