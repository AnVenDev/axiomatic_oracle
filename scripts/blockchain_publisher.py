from scripts.algorand_utils import publish_to_algorand, create_token_for_asset
from scripts.logger_utils import log_asset_publication, save_publications_to_json
import json
from typing import List, Dict

def publish_ai_prediction(prediction_response: dict) -> dict:
    """
    Publish a single AI prediction to the Algorand blockchain, creating a notarization transaction and an ASA.
    """
    payload = {
        "id": prediction_response["asset_id"],
        "model": prediction_response["model_meta"]["value_model_name"],
        "val_k": prediction_response["metrics"]["valuation_base_k"],
        "hash": prediction_response["model_meta"].get("model_hash", "")[:16],
        "ts": prediction_response["timestamp"],
        "schema_version": prediction_response["schema_version"]
    }

    # 1: Publish notarization transaction
    txid = publish_to_algorand(payload)

    # 2: Tokenize the asset with ASA creation
    asa_id = create_token_for_asset(
        asset_name=f"AI_{prediction_response['asset_type']}_{prediction_response['asset_id'][:8]}",
        unit_name=f"V{prediction_response['asset_type'][:4].upper()}",
        metadata_content=json.dumps(payload),
        url=f"https://testnet.explorer.perawallet.app/tx/{txid}" if txid else None
    )

    result = {
        "asset_id": prediction_response["asset_id"],
        "blockchain_txid": txid,
        "asa_id": asa_id
    }
    log_asset_publication(result)
    return result

def batch_publish_predictions(predictions: List[Dict]) -> List[Dict]:
    """
    Publish multiple AI predictions to the Algorand blockchain in batch.
    Returns a list of result dicts with txid and ASA ID, or error messages.
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