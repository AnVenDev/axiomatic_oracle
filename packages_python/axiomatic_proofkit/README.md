**Axiomatic ProofKit (Python)** is the client SDK for building and publishing **p1** attestations on Algorand for Axiomatic Oracle.

The goal is a simple, trust-minimized flow:

1. Build a canonical **p1** payload (JCS/ACJ-style JSON).
2. Publish it on-chain as the note field of a **0-ALGO self-transaction**.
3. Keep signing and key management **on the client side**.
4. Allow anyone to independently recompute and verify what was published.

This SDK does **not** depend on any Axiomatic backend.

---

## Installation

Requires **Python 3.10+**.

```bash
pip install axiomatic_proofkit
pip install py-algorand-sdk
````

---

## Exposed API

From `axiomatic_proofkit`:

* `build_p1(...)`
* `canonical_note_bytes_p1(p1)`
* `assert_note_size_ok(p1)`
* `build_canonical_input(raw_input, allowed_keys=...)`
* `compute_input_hash(canonical_input, allowed_keys=...)`
* `publish_p1(p1, from_addr=..., sign=..., network=..., algod=..., wait_rounds=...)`
* `PublishError`

These are the only functions you need for typical integrations.

---

## Quickstart: publish a p1 to Algorand TestNet

The following example mirrors the internal smoke test used to validate the SDK.

### 1. Environment

Create a `.env` file next to your script:

```env
ALGORAND_MNEMONIC=your 25-word mnemonic for a TestNet account
ALGORAND_NETWORK=testnet
# Optional: override default node
# ALGOD_URL=https://testnet-api.algonode.cloud
```

For production, use a secure signing setup (wallet, KMS, HSM, etc.).
This example uses a mnemonic **only** for demonstration.

### 2. Example script (`publish_p1_example.py`)

```python
import os
import sys
import json
import base64
from pathlib import Path

from algosdk import account, mnemonic, encoding, transaction
from algosdk.v2client.algod import AlgodClient

from axiomatic_proofkit import (
    build_p1,
    canonical_note_bytes_p1,
    assert_note_size_ok,
    build_canonical_input,
    compute_input_hash,
    publish_p1,
)

ROOT = Path(__file__).resolve().parent


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
    or (
        "https://mainnet-api.algonode.cloud"
        if NETWORK == "mainnet"
        else "https://testnet-api.algonode.cloud"
    )
).strip()

if not MNEMONIC:
    print("❌ ALGORAND_MNEMONIC missing.", file=sys.stderr)
    sys.exit(1)


def compute_input_hash_from_golden() -> str:
    """
    If golden/p1_input.json exists, compute its canonical ACJ/JCS hash.
    Otherwise return a deterministic placeholder for the smoke test.
    """
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


def main() -> None:
    # 1) Derive private key and address from mnemonic (demo only)
    sk = mnemonic.to_private_key(MNEMONIC)
    addr = account.address_from_private_key(sk)

    # 2) Compute input hash (from golden file if available)
    ih = compute_input_hash_from_golden()

    # 3) Build canonical p1
    p1 = build_p1(
        asset_tag="re:EUR",
        model_version="v2",
        model_hash_hex="",
        input_hash_hex=ih,
        value_eur=550_000,
        uncertainty_low_eur=520_000,
        uncertainty_high_eur=580_000,
        timestamp_epoch=None,  # use "now"
    )

    # 4) Inspect note size and hash
    note_bytes, note_sha, note_len = canonical_note_bytes_p1(p1)
    assert_note_size_ok(p1)
    print(f"P1 note size: {note_len} bytes | sha256: {note_sha}")

    # 5) Client-side signer: keep your keys local
    def sign(unsigned_bytes: bytes) -> bytes:
        """
        unsigned_bytes: msgpack-encoded UnsignedTransaction.
        Return: msgpack bytes of the SignedTransaction.
        """
        tx_dict = encoding.msgpack.unpackb(unsigned_bytes)
        txn = transaction.Transaction.undictify(tx_dict)
        stx = txn.sign(sk)
        stx_b64 = encoding.msgpack_encode(stx)   # base64(msgpack signed)
        return base64.b64decode(stx_b64)         # raw msgpack bytes

    # 6) Algod client (Algonode by default)
    algod = AlgodClient("", ALGOD_URL)

    # 7) Publish p1 as a 0-ALGO self-transaction note
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
```

Run:

```bash
python publish_p1_example.py
```

You should see a JSON result with:

* `txid`
* `explorer_url`
* `note_sha256`
* `note_size`
* `network`

You can open the explorer URL and inspect the on-chain note.

---

## Security notes

* The `sign` function is fully controlled by you.
* For real integrations, replace the inline mnemonic signing with:

  * wallet connectors,
  * custodial services,
  * HSM/KMS,
  * any secure signing flow.
* The SDK never sends your secrets anywhere and does not rely on Axiomatic servers.

This is an early-access SDK: feedback and issues are very welcome.