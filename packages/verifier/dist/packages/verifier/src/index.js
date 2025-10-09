import { toJcsBytes, sha256Hex, jcsSha256 } from "../../../shared/jcs/index.js";
export async function verifyTx(opts) {
    const { txid, network = "testnet", indexerUrl, maxSkewPastSec = 600, maxSkewFutureSec = 120, } = opts;
    const base = indexerUrl ?? `https://algoindexer.${network === "mainnet" ? "" : "testnet."}algoexplorerapi.io`;
    const res = await fetch(`${base}/v2/transactions/${txid}`);
    if (!res.ok)
        return { verified: false, reason: "tx_not_found", mode: "unknown" };
    const data = await res.json();
    const tx = data.transaction ?? data;
    const noteB64 = tx?.note;
    if (!noteB64)
        return { verified: false, reason: "note_missing", mode: "unknown" };
    // base64 -> bytes -> JSON
    const bytes = Uint8Array.from(atob(noteB64), (c) => c.charCodeAt(0));
    let p1;
    try {
        p1 = JSON.parse(new TextDecoder().decode(bytes));
    }
    catch {
        return { verified: false, reason: "note_not_json", mode: "unknown" };
    }
    // == Hash parity (JCS) ==
    const rebuilt = await jcsSha256(p1);
    const onchain = p1?.note_sha256 || p1?.sha256 || undefined;
    if (onchain && onchain !== rebuilt) {
        return { verified: false, reason: "onchain_hash_mismatch", mode: "p1" };
    }
    // == Time-window (ts è epoch seconds in canon.py) ==
    const nowSec = Math.floor(Date.now() / 1000);
    const ts = Number.isFinite(p1?.ts) ? Number(p1.ts) : NaN;
    if (Number.isFinite(ts)) {
        if (ts < nowSec - maxSkewPastSec || ts > nowSec + maxSkewFutureSec) {
            return { verified: false, reason: "ts_out_of_window", mode: "p1" };
        }
    }
    return {
        verified: true,
        reason: null,
        mode: "p1",
        noteSha256: onchain ?? rebuilt,
        rebuiltSha256: rebuilt,
        confirmedRound: tx["confirmed-round"],
        explorerUrl: `https://${network === "mainnet" ? "" : "testnet."}algoexplorer.io/tx/${txid}`,
        note: p1,
    };
}
// re-export util
export { toJcsBytes, sha256Hex, jcsSha256 };
