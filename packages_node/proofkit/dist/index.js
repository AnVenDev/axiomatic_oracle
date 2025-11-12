import { toJcsBytes, sha256Hex, jcsSha256 } from "./jcs.js";
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
    void toJcsBytes(p1);
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
    const algosdkMod = await import("algosdk");
    const algosdk = algosdkMod.default ?? algosdkMod;
    await assertNoteSizeOK(p1);
    const { bytes: noteBytes } = await canonicalNoteBytesP1(p1);
    const client = algod ??
        new algosdk.Algodv2("", network === "mainnet"
            ? "https://mainnet-api.algonode.cloud"
            : "https://testnet-api.algonode.cloud", "");
    const sp = await client.getTransactionParams().do();
    let sender = from;
    if (!sender && pera) {
        const accounts = await pera.connect();
        sender = accounts?.[0]?.address;
    }
    if (!sender) {
        throw new Error("Missing `from` or Pera wallet connection");
    }
    if (!algosdk.isValidAddress(sender)) {
        throw new Error(`Invalid from address: ${sender}`);
    }
    const variants = [
        "senderReceiverString",
        "fromToString",
        "senderReceiverAddress",
        "fromToAddress",
    ];
    let txn = null;
    let lastErr = null;
    for (const v of variants) {
        try {
            switch (v) {
                case "senderReceiverString":
                    txn = algosdk.makePaymentTxnWithSuggestedParamsFromObject({
                        sender,
                        receiver: sender,
                        amount: 0,
                        note: noteBytes,
                        suggestedParams: sp,
                    });
                    break;
                case "fromToString":
                    txn = algosdk.makePaymentTxnWithSuggestedParamsFromObject({
                        from: sender,
                        to: sender,
                        amount: 0,
                        note: noteBytes,
                        suggestedParams: sp,
                    });
                    break;
                case "senderReceiverAddress": {
                    const A = algosdk.decodeAddress(sender);
                    txn = algosdk.makePaymentTxnWithSuggestedParamsFromObject({
                        sender: A,
                        receiver: A,
                        amount: 0,
                        note: noteBytes,
                        suggestedParams: sp,
                    });
                    break;
                }
                case "fromToAddress": {
                    const A = algosdk.decodeAddress(sender);
                    txn = algosdk.makePaymentTxnWithSuggestedParamsFromObject({
                        from: A,
                        to: A,
                        amount: 0,
                        note: noteBytes,
                        suggestedParams: sp,
                    });
                    break;
                }
            }
            if (txn) {
                break;
            }
        }
        catch (e) {
            lastErr = e;
            txn = null;
        }
    }
    if (!txn) {
        throw new Error(`Unable to construct payment txn for this algosdk version. Last error: ${lastErr?.message || String(lastErr || "unknown")}`);
    }
    const unsignedBytes = algosdk.encodeUnsignedTransaction(txn);
    let signed;
    if (pera) {
        const signedBlobs = await pera.signTransactions([unsignedBytes]);
        signed = signedBlobs[0];
    }
    else if (sign) {
        signed = await sign(unsignedBytes);
    }
    else {
        throw new Error("Provide `pera` or a custom `sign` function to sign the transaction");
    }
    const sendRes = await client.sendRawTransaction(signed).do();
    const txid = sendRes.txId || sendRes.txid;
    await algosdk.waitForConfirmation(client, txid, waitRounds);
    const explorerUrl = network === "mainnet"
        ? `https://explorer.perawallet.app/tx/${txid}`
        : `https://testnet.explorer.perawallet.app/tx/${txid}`;
    return { txid, explorerUrl };
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
