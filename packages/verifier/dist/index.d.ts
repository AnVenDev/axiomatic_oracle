import { toJcsBytes, sha256Hex, jcsSha256 } from "./jcs.js";
export type Network = "testnet" | "mainnet" | "betanet";
export interface VerifyResult {
    verified: boolean;
    reason?: string | null;
    mode: "p1" | "legacy" | "unknown";
    noteSha256?: string;
    rebuiltSha256?: string;
    confirmedRound?: number;
    explorerUrl?: string;
    note?: any;
}
export declare function verifyTx(opts: {
    txid: string;
    network?: Network;
    indexerUrl?: string;
    maxSkewPastSec?: number;
    maxSkewFutureSec?: number;
}): Promise<VerifyResult>;
export { toJcsBytes, sha256Hex, jcsSha256 };
