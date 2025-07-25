"""
algorand_utils.py

This module provides core functions for interacting with the Algorand blockchain, specifically for:
- Publishing notarized AI predictions via self-transfer with JSON notes.
- Creating non-fungible Algorand Standard Assets (ASA) to tokenize real-world assets (RWAs).
- Waiting for transaction confirmations with custom error handling.
- Supports metadata hashing, dynamic URL injection, and environment-based configuration.
"""

from algosdk import mnemonic, transaction
from algosdk.v2client import algod
from typing import Dict, Optional
import json
import hashlib
from scripts.secrets_manager import (
    ALGORAND_ADDRESS,
    ALGORAND_MNEMONIC,
    ALGORAND_WALLET_ADDRESS,
)

SENDER_ADDR = ALGORAND_WALLET_ADDRESS

# Initialize Algod client
ALGOD_TOKEN = ""
client = algod.AlgodClient(ALGOD_TOKEN, ALGORAND_ADDRESS)

# Derive sender private key
SENDER_PK = mnemonic.to_private_key(ALGORAND_MNEMONIC)


# --- Custom Exception ---
class AlgorandError(Exception):
    """Custom exception for Algorand operations"""

    pass


# --- Confirm transaction ---
def wait_for_confirmation(txid: str, timeout: int = 10):
    last_round = client.status().get("last-round")
    for _ in range(timeout):
        tx_info = client.pending_transaction_info(txid)
        if tx_info.get("confirmed-round", 0) > 0:
            print(f"[✅] TX {txid} confirmed in round {tx_info['confirmed-round']}")
            return tx_info
        client.status_after_block(last_round + 1)
        last_round += 1
    raise AlgorandError(f"Transaction {txid} not confirmed after {timeout} rounds")


# --- Publish notarized AI prediction ---
def publish_to_algorand(note_dict: Dict) -> str:
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
        metadata_hash = None
        if metadata_content:
            metadata_hash = hashlib.sha256(metadata_content.encode()).digest()

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

        ptx = client.pending_transaction_info(txid)
        asset_id = ptx.get("asset-index")
        print(f"[✅] ASA created with ID {asset_id}")
        return asset_id
    except Exception as e:
        raise AlgorandError(f"[❌] ASA creation failed: {e}")
