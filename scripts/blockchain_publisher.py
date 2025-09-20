# scripts/blockchain_publisher.py
from __future__ import annotations

import json
import os
from datetime import datetime, date
from typing import Dict, List, Any, Optional

from scripts.algorand_utils import (
    create_token_for_asset,
    explorer_url,
    publish_p1_attestation,   # PoVal: pubblica i byte canonici p1
    publish_to_algorand,      # Legacy (aioracle:v2) opzionale
    AlgorandError,
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
def _valuation_k(pred: dict) -> float:
    """
    Estrae la valutazione in k€ dal payload predizione.
    Supporta sia schema v2 (metrics.valuation_k) sia v1 (metrics.valuation_base_k).
    """
    m = pred.get("metrics", {})
    if "valuation_k" in m:
        return float(m["valuation_k"])
    if "valuation_base_k" in m:
        return float(m["valuation_base_k"])
    raise KeyError("Missing valuation in prediction payload (metrics.valuation_k or metrics.valuation_base_k).")


def _model_name(pred: dict) -> Optional[str]:
    return (pred.get("model_meta") or {}).get("value_model_name")


def _model_hash_prefix(pred: dict, n: int = 16) -> str:
    h = (pred.get("model_meta") or {}).get("model_hash")
    return (h or "")[:n] if isinstance(h, str) else ""


def _schema_version(pred: dict) -> str:
    return str(pred.get("schema_version") or "v2")


def _asset_type(pred: dict) -> str:
    return str(pred.get("asset_type") or "property")


def _asset_id(pred: dict) -> str:
    aid = pred.get("asset_id")
    if not isinstance(aid, str) or not aid:
        raise KeyError("Prediction payload missing 'asset_id'.")
    return aid


def _iso_ts(value: Any) -> Optional[str]:
    """Converte datetime/date in ISO 8601; se già string o None, restituisce così com'è."""
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value if isinstance(value, str) else str(value)


def _build_legacy_note(pred: dict) -> dict:
    """
    Payload 'legacy' compatto (aioracle:v2). Lo manteniamo per retro-compatibilità.
    NON è usato nel flusso PoVal p1.
    """
    return {
        "id": _asset_id(pred),
        "model": _model_name(pred),
        "val_k": _valuation_k(pred),
        "hash": _model_hash_prefix(pred, 16),
        "ts": _iso_ts(pred.get("timestamp")),
        "schema_version": _schema_version(pred),
    }


def _asa_names(pred: dict) -> tuple[str, str]:
    """
    Costruisce asset_name (<=32) e unit_name (<=8) per l'ASA.
    """
    asset_type = _asset_type(pred)
    asset_id = _asset_id(pred)
    asset_name = f"AI_{asset_type}_{asset_id[:8]}"[:32]
    unit_name = f"V{asset_type[:4].upper()}"[:8]
    return asset_name, unit_name


def _json_default(o: Any):
    """
    Fallback per json.dumps (solo per legacy).
    Con PoVal p1 usiamo sempre canonicalize_jcs per i metadata ASA.
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
def publish_ai_prediction(prediction_response: dict) -> dict:
    """
    Flusso standard:
      1) salva il **detail report** off-chain e aggiorna offchain_refs.detail_report_hash
      2) pubblica **PoVal p1** in nota (byte ACJ-1), **<1KB** (fail se oltre)
      3) (opz.) crea ASA 1:1; metadata = p1 canonico (ACJ-1)
      4) logga la pubblicazione

    Env:
      PUBLISH_CREATE_ASA=true|false      # default: false (ASA opzionale)
      NOTE_MAX_BYTES=1024             # budget nota (controllato da algorand_utils)
      USE_LEGACY_NOTE=false           # se true, pubblica ancora il formato 'aioracle:v2' (sconsigliato)
      PUBLISH_NOTE_URL_BASE=<url>     # usato SOLO nel legacy path
    """
    # 1) Detail report & hash
    detail_hash = save_prediction_detail(prediction_response)
    prediction_response.setdefault("offchain_refs", {})
    prediction_response["offchain_refs"]["detail_report_hash"] = detail_hash

    # 2) Nota on-chain
    use_legacy = os.getenv("USE_LEGACY_NOTE", "false").lower() in {"1", "true", "yes", "y"}

    if use_legacy:
        # Percorso legacy (aioracle:v2) — mantenuto per retro-compatibilità
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
        # Percorso PoVal p1: richiede che inference_api abbia allegato attestation.p1
        p1 = ((prediction_response.get("attestation") or {}).get("p1")) or None
        if not isinstance(p1, dict) or p1.get("s") != "p1":
            raise RuntimeError("PoVal p1 not found in response['attestation']['p1']. Ensure inference_api built it.")
        try:
            pub = publish_p1_attestation(p1)
        except AlgorandError as e:
            raise RuntimeError(f"On-chain publish failed (p1): {e}")

    result = {
        "asset_id": _asset_id(prediction_response),
        "blockchain_txid": pub.get("txid"),
        "note_size": pub.get("note_size"),
        "note_sha256": pub.get("note_sha256"),
        "is_compacted": bool(pub.get("is_compacted", False)),  # per p1 è sempre False
        "confirmed_round": pub.get("confirmed_round"),
        "asa_id": None,
    }

    # 3) (Opzionale) ASA
    create_asa = (os.getenv("PUBLISH_CREATE_ASA", "false").lower() in {"1", "true", "yes", "y"})
    if create_asa:
        try:
            asset_name, unit_name = _asa_names(prediction_response)
            tx_explorer = explorer_url(pub.get("txid"))

            if not use_legacy:
                meta_payload = ((prediction_response.get("attestation") or {}).get("p1")) or {}
                metadata_json = canonicalize_jcs(meta_payload).decode("utf-8")  # deterministico
            else:
                meta_payload = _build_legacy_note(prediction_response)
                metadata_json = json.dumps(meta_payload, separators=(",", ":"), ensure_ascii=False, default=_json_default)

            asa_id = create_token_for_asset(
                asset_name=asset_name,
                unit_name=unit_name,
                metadata_content=metadata_json,
                url=tx_explorer,
            )
            result["asa_id"] = asa_id
        except AlgorandError:
            result["asa_id"] = None

    # 4) Logging consolidato
    log_asset_publication(result)
    return result


def batch_publish_predictions(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for i, pred in enumerate(predictions):
        try:
            results.append(publish_ai_prediction(pred))
        except Exception as e:
            aid = pred.get("asset_id", f"#{i}")
            results.append({"error": str(e), "asset_id": aid})
    # Mantieni il file array per compat.
    save_publications_to_json(results)
    return results
