import json
from typing import Dict, List

from scripts.algorand_utils import create_token_for_asset, publish_to_algorand
from scripts.logger_utils import (
    log_asset_publication,
    save_prediction_detail,
    save_publications_to_json,
)


def publish_ai_prediction(prediction_response: dict) -> dict:
    """
    Publish a single AI prediction to the Algorand blockchain, creating a notarization transaction and an ASA.
    """
    # 1. Save detailed report and compute hash
    detail_hash = save_prediction_detail(prediction_response)
    prediction_response.setdefault("offchain_refs", {})
    prediction_response["offchain_refs"]["detail_report_hash"] = detail_hash

    # 2. Prepare minimal payload
    payload = {
        "id": prediction_response["asset_id"],
        "model": prediction_response["model_meta"]["value_model_name"],
        "val_k": prediction_response["metrics"]["valuation_base_k"],
        "hash": prediction_response["model_meta"].get("model_hash", "")[:16],
        "ts": prediction_response["timestamp"],
        "schema_version": prediction_response["schema_version"],
    }

    # 3: Publish notarization transaction
    txid = publish_to_algorand(payload)

    # 4: Tokenize the asset with ASA creation
    asa_id = create_token_for_asset(
        asset_name=f"AI_{prediction_response['asset_type']}_{prediction_response['asset_id'][:8]}",
        unit_name=f"V{prediction_response['asset_type'][:4].upper()}",
        metadata_content=json.dumps(payload),
        url=f"https://testnet.explorer.perawallet.app/tx/{txid}" if txid else None,
    )

    result = {
        "asset_id": prediction_response["asset_id"],
        "blockchain_txid": txid,
        "asa_id": asa_id,
    }

    print(f"asset_id: {prediction_response['asset_id']}")
    print(f"txid: {txid}")
    print(f"asa_id: {asa_id}")

    log_asset_publication(result)
    return result


def batch_publish_predictions(predictions: List[Dict]) -> List[Dict]:
    """
    Publish multiple AI predictions to the 
    Algorand blockchain in batch.
    Returns a list of result dicts with txid and ASA ID, 
    or error messages.
    """
    results = []
    for i, pred in enumerate(predictions):
        try:
            result = publish_ai_prediction(pred)
            results.append(result)
        except Exception as e:
            print(f"[❌] Failed to publish asset {pred.get('asset_id', f'#{i}')}: {e}")
            results.append({"error": str(e), "prediction": pred})

    # Save all successful publications to logs/published_assets.json
    save_publications_to_json(results)
    return results
