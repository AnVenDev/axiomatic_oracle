import os
import sys
import json
import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --------------------------
# .env
# --------------------------
def load_env_from_file(env_path: Path) -> None:
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if not line or line.strip().startswith("#") or "=" not in line:
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            if k and (k not in os.environ):
                os.environ[k] = v
    except FileNotFoundError:
        pass


load_env_from_file(ROOT / ".env")

MNEMONIC = os.getenv("ALGORAND_MNEMONIC")
NETWORK = (os.getenv("ALGORAND_NETWORK") or "testnet").strip()
ALGOD_URL = (
    os.getenv("ALGOD_URL")
    or ("https://mainnet-api.algonode.cloud" if NETWORK == "mainnet"
        else "https://testnet-api.algonode.cloud")
).strip()

if not MNEMONIC:
    print("❌ ALGORAND_MNEMONIC missing.", file=sys.stderr)
    sys.exit(1)


# --------------------------
# Algorand & SDK
# --------------------------
from algosdk import account, mnemonic, encoding
from algosdk.v2client.algod import AlgodClient
from algosdk import transaction

from axiomatic_proofkit.axiomatic_proofkit.build import (
    build_p1,
    canonical_note_bytes_p1,
    assert_note_size_ok,
    build_canonical_input,
    compute_input_hash,
)
from axiomatic_proofkit.axiomatic_proofkit.publish import publish_p1


# --------------------------
# Helpers
# --------------------------
def compute_input_hash_from_golden() -> str:
    golden_dir = ROOT / "golden"
    inp_path = golden_dir / "p1_input.json"

    if not inp_path.is_file():
        return "0" * 64

    try:
        raw = json.loads(inp_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"⚠️ Unable to read golden/p1_input.json: {e}")
        return "0" * 64

    allowed_keys = list(raw.keys())
    cin = build_canonical_input(raw, allowed_keys=allowed_keys)
    ih = compute_input_hash(cin, allowed_keys=allowed_keys)
    print(f"✅ Golden input hash: {ih}")
    return ih


# --------------------------
# Main
# --------------------------
def main() -> None:
    sk = mnemonic.to_private_key(MNEMONIC)
    addr = account.address_from_private_key(sk)

    ih = compute_input_hash_from_golden()

    p1 = build_p1(
        asset_tag="re:EUR",
        model_version="v2",
        model_hash_hex="",
        input_hash_hex=ih,
        value_eur=550_000,
        uncertainty_low_eur=520_000,
        uncertainty_high_eur=580_000,
        timestamp_epoch=None,
    )

    note_bytes, note_sha, note_len = canonical_note_bytes_p1(p1)
    assert_note_size_ok(p1)
    print(f"P1 note size: {note_len} bytes | sha256: {note_sha}")

    def sign(unsigned_bytes: bytes) -> bytes:
        tx_dict = encoding.msgpack.unpackb(unsigned_bytes)
        txn = transaction.Transaction.undictify(tx_dict)
        stx = txn.sign(sk)
        stx_b64 = encoding.msgpack_encode(stx)
        return base64.b64decode(stx_b64)

    algod = AlgodClient("", ALGOD_URL)

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
