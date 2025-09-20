from __future__ import annotations
"""
Module: blockchain_publisher.py — Orchestrates on-chain publication for predictions.

Responsibilities:
- Save off-chain detail report and attach its hash to the response.
- Publish PoVal™ p1 as Algorand note (ACJ-1 canonical bytes; strict size guard).
- (Optional) Create 1:1 ASA whose metadata is deterministic (ACJ-1 over p1).
- Consolidated logging of publication artifacts.

NOTE:
- The legacy JSON note path (aioracle:v2) is kept only for backward compatibility.
- p1 path MUST be preferred; if p1 exceeds NOTE_MAX_BYTES, publishing fails by design.
- Explorer URL is derived by downstream callers via algorand_utils.get_tx_note_info or explorer_url.

SECURITY:
- Never include PII in on-chain data (notes/ASA metadata).
- Do not log secret material; logs include only txid/hash/ids.
"""

# =========================
# Standard library imports
# =========================
import json
import os
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

# ===========
# Local deps
# ===========
from scripts.algorand_utils import (
    AlgorandError,
    create_token_for_asset,
    explorer_url,
    publish_p1_attestation,  # PoVal: publish canonical p1 bytes
    publish_to_algorand,     # Legacy (aioracle:v2) optional
)
from scripts.canon import canonicalize_jcs
from scripts.logger_utils import (
    log_asset_publication,
    save_prediction_detail,
    save_publications_to_json,
)
from scripts.secrets_manager import get_network


# =============================================================================
# Helpers
# =============================================================================
def _valuation_k(pred: Dict[str, Any]) -> float:
    """
    Extract valuation in k€ from prediction payload.
    Supports schema v2 (metrics.valuation_k) and v1 (metrics.valuation_base_k).
    """
    m = pred.get("metrics", {})
    if "valuation_k" in m:
        return float(m["valuation_k"])
    if "valuation_base_k" in m:
        return float(m["valuation_base_k"])
    raise KeyError("Missing valuation in prediction payload (metrics.valuation_k or metrics.valuation_base_k).")


def _model_name(pred: Dict[str, Any]) -> Optional[str]:
    return (pred.get("model_meta") or {}).get("value_model_name")


def _model_hash_prefix(pred: Dict[str, Any], n: int = 16) -> str:
    h = (pred.get("model_meta") or {}).get("model_hash")
    return (h or "")[:n] if isinstance(h, str) else ""


def _schema_version(pred: Dict[str, Any]) -> str:
    return str(pred.get("schema_version") or "v2")


def _asset_type(pred: Dict[str, Any]) -> str:
    return str(pred.get("asset_type") or "property")


def _asset_id(pred: Dict[str, Any]) -> str:
    aid = pred.get("asset_id")
    if not isinstance(aid, str) or not aid:
        raise KeyError("Prediction payload missing 'asset_id'.")
    return aid


def _iso_ts(value: Any) -> Optional[str]:
    """Convert datetime/date to ISO 8601; pass through strings/None."""
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value if isinstance(value, str) else str(value)


def _build_legacy_note(pred: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build compact legacy payload (aioracle:v2). Kept for backward compatibility only.
    NOT used in the PoVal p1 flow.
    """
    return {
        "id": _asset_id(pred),
        "model": _model_name(pred),
        "val_k": _valuation_k(pred),
        "hash": _model_hash_prefix(pred, 16),
        "ts": _iso_ts(pred.get("timestamp")),
        "schema_version": _schema_version(pred),
    }


def _asa_names(pred: Dict[str, Any]) -> Tuple[str, str]:
    """Construct ASA (asset_name<=32, unit_name<=8)."""
    asset_type = _asset_type(pred)
    asset_id = _asset_id(pred)
    asset_name = f"AI_{asset_type}_{asset_id[:8]}"[:32]
    unit_name = f"V{asset_type[:4].upper()}"[:8]
    return asset_name, unit_name


def _json_default(o: Any):
    """
    Fallback for json.dumps (legacy only).
    For PoVal p1 we always use canonicalize_jcs for ASA metadata.
    """
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    try:
        import numpy as np  # type: ignore
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
    except Exception:
        pass
    return str(o)


# =============================================================================
# Public API
# =============================================================================
def publish_ai_prediction(prediction_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standard publish flow:

      1) Save off-chain **detail report** and set offchain_refs.detail_report_hash
      2) Publish **PoVal p1** in note (ACJ-1 bytes), **< NOTE_MAX_BYTES** (hard fail if exceeded)
      3) (Optional) Create 1:1 ASA; metadata = deterministic JSON (ACJ-1 over p1)
      4) Log consolidated publication info

    Env:
      PUBLISH_CREATE_ASA=true|false   # default: false (ASA optional)
      NOTE_MAX_BYTES=1024             # note byte budget (enforced by algorand_utils)
      USE_LEGACY_NOTE=false           # if true, still publish 'aioracle:v2' (discouraged)
      PUBLISH_NOTE_URL_BASE=<url>     # used ONLY in the legacy path

    SECURITY:
      - Do not include PII in published data or metadata.
      - This function returns identifiers and sizes/hashes only.
    """
    # 1) Detail report & hash
    detail_hash = save_prediction_detail(prediction_response)
    prediction_response.setdefault("offchain_refs", {})
    prediction_response["offchain_refs"]["detail_report_hash"] = detail_hash

    # 2) On-chain note
    use_legacy = os.getenv("USE_LEGACY_NOTE", "false").lower() in {"1", "true", "yes", "y"}

    if use_legacy:
        # Legacy (aioracle:v2) — kept for backward compatibility
        note = _build_legacy_note(prediction_response)
        try:
            pub = publish_to_algorand(
                note,
                fallback_url=os.getenv("PUBLISH_NOTE_URL_BASE"),
                max_note_bytes=int(os.getenv("NOTE_MAX_BYTES", "1024")),
            )
        except AlgorandError as e:
            raise RuntimeError(f"On-chain publish failed (legacy): {e}")
    else:
        # PoVal p1 path: inference_api must have attached attestation.p1
        p1 = ((prediction_response.get("attestation") or {}).get("p1")) or None
        if not isinstance(p1, dict) or p1.get("s") != "p1":
            raise RuntimeError("PoVal p1 not found in response['attestation']['p1']. Ensure inference_api built it.")
        try:
            pub = publish_p1_attestation(p1)
        except AlgorandError as e:
            raise RuntimeError(f"On-chain publish failed (p1): {e}")

    result: Dict[str, Any] = {
        "asset_id": _asset_id(prediction_response),
        "blockchain_txid": pub.get("txid"),
        "note_size": pub.get("note_size"),
        "note_sha256": pub.get("note_sha256"),
        "is_compacted": bool(pub.get("is_compacted", False)),  # always False for p1
        "confirmed_round": pub.get("confirmed_round"),
        "asa_id": None,
    }

    # 3) (Optional) ASA
    create_asa = os.getenv("PUBLISH_CREATE_ASA", "false").lower() in {"1", "true", "yes", "y"}
    if create_asa:
        try:
            asset_name, unit_name = _asa_names(prediction_response)
            tx_explorer = explorer_url(pub.get("txid"))

            if not use_legacy:
                # NOTE: Deterministic metadata — ACJ-1 over p1
                meta_payload = ((prediction_response.get("attestation") or {}).get("p1")) or {}
                metadata_json = canonicalize_jcs(meta_payload).decode("utf-8")
            else:
                meta_payload = _build_legacy_note(prediction_response)
                metadata_json = json.dumps(
                    meta_payload, separators=(",", ":"), ensure_ascii=False, default=_json_default
                )

            asa_id = create_token_for_asset(
                asset_name=asset_name,
                unit_name=unit_name,
                metadata_content=metadata_json,
                url=tx_explorer,
            )
            result["asa_id"] = asa_id
        except AlgorandError:
            # WARN: ASA creation is optional; keep publishing result without it.
            result["asa_id"] = None

    # 4) Consolidated logging (append-only)
    log_asset_publication(result)
    return result


def batch_publish_predictions(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Publish a list of predictions; always returns an array (for UI/file compatibility).
    Errors are captured per-item.
    """
    results: List[Dict[str, Any]] = []
    for i, pred in enumerate(predictions):
        try:
            results.append(publish_ai_prediction(pred))
        except Exception as e:
            aid = pred.get("asset_id", f"#{i}")
            results.append({"error": str(e), "asset_id": aid})

    # Keep a JSON array file for compatibility with existing tooling.
    save_publications_to_json(results)
    return results
