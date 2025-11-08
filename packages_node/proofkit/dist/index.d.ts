export declare const NOTE_MAX_BYTES = 1024;
export declare const DEFAULT_ASSET_TAG = "re:EUR";
export type Network = "testnet" | "mainnet" | "betanet";
export interface BuildP1Opts {
    assetTag?: string;
    modelVersion: string;
    modelHashHex?: string;
    inputHashHex: string;
    valueEUR: number;
    uncertaintyLowEUR: number;
    uncertaintyHighEUR: number;
    timestampEpochSec?: number;
}
export interface P1 {
    s: "p1";
    a: string;
    mv: string;
    mh: string;
    ih: string;
    v: number;
    u: [number, number];
    ts: number;
}
export declare function buildP1(opts: BuildP1Opts): P1;
export declare function canonicalNoteBytesP1(p1: P1): Promise<{
    bytes: Uint8Array;
    sha256: string;
    size: number;
}>;
export declare function assertNoteSizeOK(p1: P1, max?: number): Promise<void>;
export interface PublishOpts {
    p1: P1;
    network?: Network;
    pera?: any;
    algod?: any;
    from?: string;
    sign?: (txnBytes: Uint8Array) => Promise<Uint8Array>;
    waitRounds?: number;
}
export declare function publishP1(opts: PublishOpts): Promise<{
    txid: string;
    explorerUrl: string;
}>;
export declare function buildCanonicalInput(rec: Record<string, any>, allowedKeys: string[], stripNone?: boolean): Record<string, any>;
export declare function computeInputHash(rec: Record<string, any>, allowedKeys: string[]): Promise<string>;
