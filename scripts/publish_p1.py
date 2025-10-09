# scripts/publish_p1.py
"""
Smoke test Proofkit (Python): build p1 + publish su Algorand.

Prerequisiti:
- pip install -e python\axiomatic_proofkit
- pip install py-algorand-sdk
- .env nella root con:
    ALGORAND_MNEMONIC=...
    ALGORAND_NETWORK=testnet
    ALGOD_URL=https://testnet-api.algonode.cloud
"""

import os, sys, json
from pathlib import Path

# --- .env loader minimale (no dipendenze)
def load_env_from_file(env_path: Path) -> None:
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if not line or line.strip().startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            if k and (k not in os.environ):
                os.environ[k] = v
    except FileNotFoundError:
        pass

ROOT = Path(__file__).resolve().parents[1]
load_env_from_file(ROOT / ".env")

MNEMONIC   = os.getenv("ALGORAND_MNEMONIC")
NETWORK    = (os.getenv("ALGORAND_NETWORK") or "testnet").strip()
ALGOD_URL  = (os.getenv("ALGOD_URL") or ("https://mainnet-api.algonode.cloud" if NETWORK == "mainnet" else "https://testnet-api.algonode.cloud")).strip()

if not MNEMONIC:
    print("❌ ALGORAND_MNEMONIC mancante (mettilo in .env o come variabile d'ambiente).", file=sys.stderr)
    sys.exit(1)

# --- deps SDK (Python Algorand)
from algosdk import account, mnemonic, encoding
from algosdk.v2client.algod import AlgodClient
from algosdk import transaction

# --- Proofkit (nostro pacchetto)
from axiomatic_proofkit.build import build_p1, canonical_note_bytes_p1, assert_note_size_ok
from axiomatic_proofkit.publish import publish_p1

def main():
    # 1) chiavi dal mnemonic
    sk = mnemonic.to_private_key(MNEMONIC)
    addr = account.address_from_private_key(sk)

    # 2) p1 dummy
    p1 = build_p1(
        asset_tag="re:EUR",
        model_version="v2",
        model_hash_hex="",
        input_hash_hex="a" * 64,
        value_eur=550_000,
        uncertainty_low_eur=520_000,
        uncertainty_high_eur=580_000,
        timestamp_epoch=None,  # usa "now"
    )

    # 3) bytes JCS + size + sha
    note_bytes, note_sha, note_len = canonical_note_bytes_p1(p1)
    assert_note_size_ok(p1)
    print(f"P1 note size: {note_len} bytes | sha256: {note_sha}")

    # 4) signer per publish_p1 — restituisce base64 string del signed txn
    def sign(unsigned_bytes: bytes) -> str:
        tx_dict = encoding.msgpack.unpackb(unsigned_bytes)   # dict dall'unsigned msgpack
        txn = transaction.Transaction.undictify(tx_dict)     # Transaction
        stx = txn.sign(sk)                                   # SignedTransaction
        return encoding.msgpack_encode(stx)                  # base64 string

    # 5) client Algod
    algod = AlgodClient("", ALGOD_URL)

    # 6) publish
    res = publish_p1(
        p1,
        network=NETWORK,
        algod=algod,
        from_addr=addr,
        sign=sign,
        wait_rounds=4,
    )
    print("PUBLISHED:")
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
