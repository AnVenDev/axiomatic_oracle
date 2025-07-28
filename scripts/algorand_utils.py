"""
algorand_utils.py

This module provides core functions for interacting with the Algorand blockchain,
specifically for:
- Publishing notarized AI predictions via self-transfer with JSON notes.
- Creating non-fungible Algorand Standard Assets (ASA) to tokenize
  real-world assets (RWAs).
- Waiting for transaction confirmations with custom error handling.
- Supports metadata hashing, dynamic URL injection, and environment-based configuration.
"""

import hashlib
import json
from typing import Any, Dict, Optional, Union

from algosdk import mnemonic, transaction
from algosdk.v2client import algod

from scripts.secrets_manager import (
    ALGORAND_ADDRESS,
    ALGORAND_MNEMONIC,
    ALGORAND_WALLET_ADDRESS,
)

# --- Safety check ---
assert ALGORAND_ADDRESS is not None, "Missing ALGORAND_ADDRESS"

SENDER_ADDR = ALGORAND_WALLET_ADDRESS

# Initialize Algod client
ALGOD_TOKEN: str = ""

client = algod.AlgodClient(ALGOD_TOKEN, ALGORAND_ADDRESS)

# Derive sender private key
SENDER_PK = mnemonic.to_private_key(ALGORAND_MNEMONIC)


# --- Custom Exception ---
class AlgorandError(Exception):
    """Custom exception for Algorand operations"""

    pass


# --- Confirm transaction ---
def wait_for_confirmation(txid: str, timeout: int = 10) -> Dict[str, Any]:
    status_raw = client.status()
    if isinstance(status_raw, bytes):
        try:
            status_info: Dict[str, Any] = json.loads(status_raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise AlgorandError("Failed to decode bytes response from status")
    elif isinstance(status_raw, dict):
        status_info = status_raw
    else:
        raise AlgorandError("Unexpected response type from status")

    last_round = status_info.get("last-round", 0)
    for _ in range(timeout):
        raw_info = client.pending_transaction_info(txid)

        if isinstance(raw_info, bytes):
            try:
                tx_info: Dict[str, Any] = json.loads(raw_info)
            except json.JSONDecodeError:
                raise AlgorandError(
                    "Invalid JSON in response from pending_transaction_info"
                )
        elif isinstance(raw_info, dict):
            tx_info = raw_info
        else:
            raise AlgorandError(
                "Unexpected response type from pending_transaction_info"
            )

        if tx_info.get("confirmed-round", 0) > 0:
            print(f"[✅] TX {txid} confirmed in round {tx_info['confirmed-round']}")
            return tx_info

        client.status_after_block(last_round + 1)
        last_round += 1

    raise AlgorandError(f"Transaction {txid} not confirmed after {timeout} rounds")


# --- Publish notarized AI prediction ---
def publish_to_algorand(note_dict: Dict[str, Any]) -> str:
    try:
        note_bytes = json.dumps(note_dict).encode("utf-8")
        params = client.suggested_params()
        txn = transaction.PaymentTxn(
            sender=SENDER_ADDR, receiver=SENDER_ADDR, amt=0, sp=params, note=note_bytes
        )
        signed_txn = txn.sign(SENDER_PK)
        txid = client.send_transaction(signed_txn)
        wait_for_confirmation(txid)
        return txid
    except Exception as e:
        raise AlgorandError(f"[❌] Algorand publish failed: {e}")


# --- Create Token (ASA) for a given asset ---
def create_token_for_asset(
    asset_name: str,
    unit_name: str,
    metadata_content: Optional[str] = None,
    url: Optional[str] = None,
) -> Optional[int]:
    try:
        params = client.suggested_params()

        # Metadata hash (must be exactly 32 bytes)
        metadata_hash = (
            hashlib.sha256(metadata_content.encode()).digest()
            if metadata_content
            else None
        )

        txn = transaction.AssetConfigTxn(
            sender=SENDER_ADDR,
            sp=params,
            total=1,
            decimals=0,
            default_frozen=False,
            unit_name=unit_name[:8],
            asset_name=asset_name[:32],
            manager=SENDER_ADDR,
            reserve=SENDER_ADDR,
            freeze=SENDER_ADDR,
            clawback=SENDER_ADDR,
            url=url or "https://example.com/aioracle/metadata.json",
            metadata_hash=metadata_hash,
        )

        signed_txn = txn.sign(SENDER_PK)
        txid = client.send_transaction(signed_txn)
        wait_for_confirmation(txid)

        ptx_raw: Union[Dict[str, Any], bytes] = client.pending_transaction_info(txid)

        # Controllo esplicito su tipo
        if isinstance(ptx_raw, bytes):
            try:
                ptx: Dict[str, Any] = json.loads(ptx_raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise AlgorandError("Invalid JSON in pending_transaction_info response")
        elif isinstance(ptx_raw, dict):
            ptx = ptx_raw
        else:
            raise AlgorandError("Unexpected type from pending_transaction_info")

        asset_id = ptx.get("asset-index")
        print(f"[✅] ASA created with ID {asset_id}")
        return asset_id if isinstance(asset_id, int) else None

    except Exception as e:
        raise AlgorandError(f"[❌] ASA creation failed: {e}")
