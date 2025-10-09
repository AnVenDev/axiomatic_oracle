import { toJcsBytes, sha256Hex, jcsSha256 } from "../../../shared/jcs/index.js";
export const NOTE_MAX_BYTES = 1024;
export const DEFAULT_ASSET_TAG = "re:EUR";
function assertFinite(n, name) {
    if (typeof n !== "number" || !Number.isFinite(n))
        throw new Error(`${name} must be a finite number`);
}
export function buildP1(opts) {
    const { assetTag = DEFAULT_ASSET_TAG, modelVersion, modelHashHex = "", inputHashHex, valueEUR, uncertaintyLowEUR, uncertaintyHighEUR, timestampEpochSec = Math.floor(Date.now() / 1000), } = opts;
    if (uncertaintyLowEUR > uncertaintyHighEUR)
        throw new Error("uncertaintyLowEUR > uncertaintyHighEUR");
    assertFinite(valueEUR, "valueEUR");
    assertFinite(uncertaintyLowEUR, "uncertaintyLowEUR");
    assertFinite(uncertaintyHighEUR, "uncertaintyHighEUR");
    if (!modelVersion?.trim())
        throw new Error("modelVersion is required");
    if (inputHashHex.length !== 64)
        throw new Error("inputHashHex must be 64 hex chars");
    if (modelHashHex && modelHashHex.length !== 64)
        throw new Error("modelHashHex must be 64 hex chars (or empty)");
    const p1 = {
        s: "p1",
        a: String(assetTag),
        mv: String(modelVersion),
        mh: String(modelHashHex || ""),
        ih: String(inputHashHex),
        v: Number(valueEUR),
        u: [Number(uncertaintyLowEUR), Number(uncertaintyHighEUR)],
        ts: Number(timestampEpochSec),
    };
    void toJcsBytes(p1); // ACJ-1 sanity
    return p1;
}
export async function canonicalNoteBytesP1(p1) {
    if (!p1 || p1.s !== "p1")
        throw new Error("Invalid p1 object");
    const bytes = toJcsBytes(p1);
    const sha256 = await sha256Hex(bytes);
    return { bytes, sha256, size: bytes.length };
}
export async function assertNoteSizeOK(p1, max = NOTE_MAX_BYTES) {
    const { size } = await canonicalNoteBytesP1(p1);
    if (size > max)
        throw new Error(`p1 note too large: ${size} bytes (max ${max})`);
}
export async function publishP1(opts) {
    const { p1, network = "testnet", pera, algod, from, sign, waitRounds = 4 } = opts;
    // import dinamico (ESM friendly)
    const algosdkMod = await import("algosdk");
    const algosdk = algosdkMod.default ?? algosdkMod;
    await assertNoteSizeOK(p1);
    const { bytes } = await canonicalNoteBytesP1(p1);
    const client = algod ??
        new algosdk.Algodv2("", `https://node.${network === "mainnet" ? "" : "testnet."}algoexplorerapi.io`, "");
    // 1) Suggested params (cast sicuro per evitare mismatch di tipi in TS)
    const sp = await client.getTransactionParams().do();
    const suggestedParams = {
        fee: sp.fee,
        flatFee: sp.flatFee,
        firstRound: sp.firstRound,
        lastRound: sp.lastRound,
        genesisID: sp.genesisID,
        genesisHash: sp.genesisHash,
    };
    // 2) mittente
    let sender = from;
    if (!sender && pera) {
        const accounts = await pera.connect();
        sender = accounts?.[0]?.address;
    }
    if (!sender)
        throw new Error("Missing `from` or Pera wallet connection");
    // 3) Tx: usa *FromObject* (compat con la tua versione di algosdk)
    const txn = algosdk.makePaymentTxnWithSuggestedParamsFromObject({
        sender: sender,
        receiver: sender,
        amount: 0,
        note: bytes, // Uint8Array
        suggestedParams, // cast any per evitare errori TS
    });
    // 4) Firma
    const unsignedBytes = algosdk.encodeUnsignedTransaction(txn);
    let signed;
    if (pera) {
        // ✅ Pera: API corretta (niente 'signers')
        const signedBlobs = await pera.signTransactions([unsignedBytes]);
        signed = signedBlobs[0];
    }
    else if (sign) {
        signed = await sign(unsignedBytes);
    }
    else {
        throw new Error("Provide `pera` or a custom `sign` function to sign the transaction");
    }
    // 5) Invio & conferma
    const sendRes = await client.sendRawTransaction(signed).do();
    const txId = sendRes.txId;
    await algosdk.waitForConfirmation(client, txId, waitRounds);
    const explorerUrl = `https://${network === "mainnet" ? "" : "testnet."}algoexplorer.io/tx/${txId}`;
    return { txid: txId, explorerUrl };
}
export function buildCanonicalInput(rec, allowedKeys, stripNone = true) {
    const allowed = new Set(allowedKeys.map(String));
    const out = {};
    for (const [kRaw, v] of Object.entries(rec)) {
        const k = String(kRaw);
        if (!allowed.has(k))
            continue;
        if (stripNone && (v === null || typeof v === "undefined"))
            continue;
        out[k] = v;
    }
    return out;
}
export async function computeInputHash(rec, allowedKeys) {
    const cin = buildCanonicalInput(rec, allowedKeys);
    return jcsSha256(cin);
}
