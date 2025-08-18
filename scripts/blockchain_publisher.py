# scripts/blockchain_publisher.py
from __future__ import annotations

import json
import os
from typing import Dict, List, Any, Optional

from scripts.algorand_utils import create_token_for_asset, publish_to_algorand, AlgorandError
from scripts.logger_utils import (
    log_asset_publication,
    save_prediction_detail,
    save_publications_to_json,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
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
    # meta può contenere 'model_hash' (già sha256) oppure niente
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

def _build_note_payload(pred: dict) -> dict:
    """
    Payload compatto per la nota on-chain (≤ ~1KB).
    """
    return {
        "id": _asset_id(pred),
        "model": _model_name(pred),
        "val_k": _valuation_k(pred),
        "hash": _model_hash_prefix(pred, 16),
        "ts": pred.get("timestamp"),
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


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def publish_ai_prediction(prediction_response: dict) -> dict:
    """
    - Salva il dettaglio off-chain e calcola l'hash (inserito in offchain_refs.detail_report_hash)
    - Pubblica una notarizzazione on-chain (self-transfer 0 ALGO con nota JSON compatta)
    - Crea un ASA 1:1 come “token” dell'asset (facoltativo ma attivo di default)
    - Logga il risultato (JSONL e, per compat, JSON array)

    Env opzionali:
      PUBLISH_CREATE_ASA=true|false     # default true
      PUBLISH_NOTE_URL_BASE=<url base>  # opzionale; se presente lo includiamo nel metadata ASA
    """
    # 1) Salva dettaglio e aggiorna offchain_refs
    detail_hash = save_prediction_detail(prediction_response)
    prediction_response.setdefault("offchain_refs", {})
    prediction_response["offchain_refs"]["detail_report_hash"] = detail_hash

    # 2) Costruisci payload compatto per la nota
    payload = _build_note_payload(prediction_response)

    # 3) Pubblica notarizzazione su Algorand
    #    (se la nota eccede il limite, algorand_utils userà un payload compatto con hash)
    note_url_base = os.getenv("PUBLISH_NOTE_URL_BASE") or None
    try:
        txid = publish_to_algorand(payload, fallback_url=note_url_base)
    except AlgorandError as e:
        # Fallimento della notarizzazione → solleva (niente ASA)
        raise RuntimeError(f"On-chain publish failed: {e}")

    # 4) (Opzionale) Tokenize via ASA
    create_asa = (os.getenv("PUBLISH_CREATE_ASA", "true").lower() in {"1", "true", "yes", "y"})
    asa_id: Optional[int] = None
    if create_asa:
        try:
            asset_name, unit_name = _asa_names(prediction_response)
            # Come URL, se possibile, linkiamo l'explorer al tx appena creato (TestNet Pera)
            explorer_url = f"https://testnet.explorer.perawallet.app/tx/{txid}" if txid else None
            asa_id = create_token_for_asset(
                asset_name=asset_name,
                unit_name=unit_name,
                metadata_content=json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
                url=explorer_url,
            )
        except AlgorandError as e:
            # Non blocchiamo il flusso: ASA è opzionale
            asa_id = None

    result = {
        "asset_id": _asset_id(prediction_response),
        "blockchain_txid": txid,
        "asa_id": asa_id,
    }

    # 5) Logga la pubblicazione
    # Manteniamo anche il file JSON “array” per compatibilità con /logs/published
    log_asset_publication(result, also_update_json_array=True)
    return result


def batch_publish_predictions(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for i, pred in enumerate(predictions):
        try:
            results.append(publish_ai_prediction(pred))
        except Exception as e:
            aid = pred.get("asset_id", f"#{i}")
            results.append({"error": str(e), "asset_id": aid})
    # Per compat con l’endpoint che legge logs/published_assets.json
    save_publications_to_json(results)
    return results