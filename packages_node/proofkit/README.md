`@axiomatic_oracle/proofkit` is the TypeScript/JavaScript SDK for building and publishing **p1** attestations on Algorand for Axiomatic Oracle.

The goals:

- Canonical, deterministic **p1** payloads (JCS-style JSON).
- On-chain anchoring as the note of a **0-ALGO self-transaction**.
- Client-side signing only: the SDK never handles your secrets.
- Interoperability with independent verifiers (including `@axiomatic_oracle/verifier` and the Python SDKs).

This package is ESM-first and works with Node.js (18+) and modern browsers.

---

## Installation

```bash
npm install @axiomatic_oracle/proofkit algosdk
```

Requirements:

* Node.js **18+** (or a browser with `fetch`, `TextEncoder`, `crypto.subtle`).
* `algosdk` is required by your app (listed as a dependency).

---

## Exports

From `@axiomatic_oracle/proofkit`:

* `buildP1(options)` → `p1`
* `canonicalNoteBytesP1(p1)` → `{ bytes, sha256, size }`
* `assertNoteSizeOK(p1, maxBytes?)`
* `buildCanonicalInput(record, allowedKeys, stripNone?)`
* `computeInputHash(record, allowedKeys)` → `sha256` (JCS-style)
* `publishP1(options)` → `{ txid, explorerUrl }`
* `NOTE_MAX_BYTES`
* `DEFAULT_ASSET_TAG`
* Types: `P1`, `BuildP1Opts`, `PublishOpts`, `Network`

---

## p1 structure

`buildP1` returns an object of the form:

```ts
type P1 = {
  s: "p1";                 // schema marker
  a: string;               // asset tag (e.g. "re:EUR")
  mv: string;              // model version
  mh: string;              // model hash (hex, optional)
  ih: string;              // input hash (hex, required)
  v: number;               // point estimate (e.g. value)
  u: [number, number];     // uncertainty range [low, high]
  ts: number;              // unix epoch seconds
};
```

The object is normalized and serialized using a JCS-style canonical JSON encoder for stable hashing and cross-language parity.

---

## Quickstart (Node.js): build and publish a p1

Example using a local mnemonic for TestNet (for demo only).

```ts
import "dotenv/config";
import algosdk from "algosdk";
import {
  buildP1,
  publishP1,
  buildCanonicalInput,
  computeInputHash,
} from "@axiomatic_oracle/proofkit";

async function main() {
  const mnemonic = process.env.ALGORAND_MNEMONIC || "";
  const network = (process.env.ALGORAND_NETWORK || "testnet").trim();

  if (!mnemonic) {
    throw new Error("Missing ALGORAND_MNEMONIC");
  }

  const { sk, addr } = algosdk.mnemonicToSecretKey(mnemonic);
  const from = addr.toString();

  // Optional: derive a canonical input hash from your raw input
  const rawInput = {
    property_id: "demo-123",
    country: "IT",
    value_hint: 550000,
  };
  const allowedKeys = Object.keys(rawInput);
  const canonicalInput = buildCanonicalInput(rawInput, allowedKeys);
  const inputHashHex = await computeInputHash(canonicalInput, allowedKeys);

  // Build a p1 attestation
  const p1 = buildP1({
    assetTag: "re:EUR",
    modelVersion: "v2",
    modelHashHex: "",
    inputHashHex,
    valueEUR: 550000,
    uncertaintyLowEUR: 520000,
    uncertaintyHighEUR: 580000,
    // timestampEpochSec optional (defaults to now)
  });

  // Signer: you control the keys. ProofKit only passes bytes to sign.
  const sign = async (unsignedBytes: Uint8Array): Promise<Uint8Array> => {
    const tx = algosdk.decodeUnsignedTransaction(unsignedBytes);
    const { blob } = algosdk.signTransaction(tx, sk);
    return blob; // msgpack-encoded SignedTransaction
  };

  // Publish as a 0-ALGO self-transaction with canonical note
  const res = await publishP1({
    p1,
    from,
    sign,
    network, // "testnet" or "mainnet"
    // optional: pass a custom algod client via `algod`
    // optional: use `pera` in browser environments
  });

  console.log("P1 published:", res);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
```

This will:

1. Build a canonical p1.
2. Create a 0-ALGO self-payment with the p1 note.
3. Sign using your function.
4. Submit via Algonode by default.
5. Return `{ txid, explorerUrl }`.

---

## Signing model

`publishP1` is intentionally minimal:

```ts
type PublishOpts = {
  p1: P1;
  network?: "testnet" | "mainnet";
  algod?: any; // custom Algodv2 client (optional)
  pera?: any;  // PeraWallet, in browser (optional)
  from?: string;
  sign?: (unsignedBytes: Uint8Array) => Promise<Uint8Array>;
  waitRounds?: number;
};
```

You must provide either:

* `pera`: in-browser signing flow, or
* `sign`: a function that receives the **unsigned tx bytes** and returns **signed tx bytes**.

This keeps all key management under your control.

---

## Relationship with other SDKs

* Use `@axiomatic_oracle/proofkit` to build and publish.
* Use `@axiomatic_oracle/verifier` (Node) or the Python verifier to independently validate the on-chain note.
* The canonicalization and hashing logic is aligned across languages.