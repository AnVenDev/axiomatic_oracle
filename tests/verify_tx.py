import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env(p: Path):
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line or line.strip().startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            if k and (k not in os.environ):
                os.environ[k] = v
    except FileNotFoundError:
        pass


load_env(ROOT / ".env")


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--golden-json":
        json_path = Path(sys.argv[2])
        expect_sha = None

        for i, arg in enumerate(sys.argv[3:], start=3):
            if arg == "--expect-sha" and i + 1 < len(sys.argv):
                expect_sha = sys.argv[i + 1].strip()

        if not expect_sha:
            print("Missing --expect-sha <SHA256> for golden mode", file=sys.stderr)
            sys.exit(1)

        from axiomatic_verifier.jcs import to_jcs_bytes, sha256_hex  # type: ignore

        try:
            obj = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(json.dumps({
                "mode": "golden",
                "ok": False,
                "reason": f"cannot_read_json:{e}",
                "path": str(json_path),
            }, indent=2))
            sys.exit(1)

        b = to_jcs_bytes(obj)
        got = sha256_hex(b)

        ok = (got == expect_sha)
        print(json.dumps({
            "mode": "golden",
            "ok": ok,
            "expected_sha256": expect_sha,
            "computed_sha256": got,
            "path": str(json_path),
        }, indent=2))
        sys.exit(0 if ok else 1)

    if len(sys.argv) < 2:
        print("Usage:", file=sys.stderr)
        print("  python tests/verify_tx.py <TXID>", file=sys.stderr)
        print("  python tests/verify_tx.py --golden-json path.json --expect-sha <SHA256>", file=sys.stderr)
        sys.exit(1)

    txid = sys.argv[1].strip()
    network = (os.getenv("ALGORAND_NETWORK") or "testnet").strip()
    indexer_url = (
        os.getenv("INDEXER_URL")
        or ("https://mainnet-idx.algonode.cloud" if network == "mainnet"
            else "https://testnet-idx.algonode.cloud")
    ).strip()

    from axiomatic_verifier.axiomatic_verifier.verifier import verify_tx  # noqa

    res = verify_tx(
        txid=txid,
        network=network,
        indexer_url=indexer_url,
        max_skew_past_sec=3600,
        max_skew_future_sec=300,
    )

    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()