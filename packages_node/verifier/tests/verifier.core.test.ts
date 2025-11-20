import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { Buffer } from "node:buffer";
import { verifyTx } from "../src/index";

function b64Json(obj: unknown): string {
  const json = JSON.stringify(obj);
  return Buffer.from(json, "utf8").toString("base64");
}

const originalFetch = (globalThis as any).fetch;
const originalDateNow = Date.now;

describe("verifyTx core behaviour (Node verifier)", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    (globalThis as any).fetch = originalFetch;
    (Date as any).now = originalDateNow;
  });

  it("returns tx_not_found when indexer responds with non-OK status", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 404,
      json: async () => {
        throw new Error("no body");
      },
    });

    (globalThis as any).fetch = fetchMock;

    const res = await verifyTx({
      txid: "TX_NOT_FOUND",
      network: "testnet",
    });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(res.verified).toBe(false);
    expect(res.mode).toBe("unknown");
    expect(res.reason).toBe("tx_not_found:404");
    expect(res.explorerUrl).toContain(
      "testnet.explorer.perawallet.app/tx/TX_NOT_FOUND",
    );
  });

  it("returns note_missing when transaction has no note field", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        transaction: {
          id: "TX_NO_NOTE",
          "confirmed-round": 123,
        },
      }),
    });

    (globalThis as any).fetch = fetchMock;

    const res = await verifyTx({
      txid: "TX_NO_NOTE",
      network: "testnet",
    });

    expect(res.verified).toBe(false);
    expect(res.mode).toBe("unknown");
    expect(res.reason).toBe("note_missing");
  });

  it("returns note_not_json when note cannot be parsed as JSON", async () => {
    const badBase64 = Buffer.from("not-json", "utf8").toString("base64");

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        transaction: {
          id: "TX_BAD_NOTE",
          note: badBase64,
          "confirmed-round": 10,
        },
      }),
    });

    (globalThis as any).fetch = fetchMock;

    const res = await verifyTx({
      txid: "TX_BAD_NOTE",
      network: "testnet",
    });

    expect(res.verified).toBe(false);
    expect(res.mode).toBe("unknown");
    expect(res.reason).toBe("note_not_json");
  });

  it("recognizes legacy notes when 'ref' is present", async () => {
    const legacy = { ref: "legacy-valuation", schema_version: "1" };

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        transaction: {
          id: "TX_LEGACY",
          note: b64Json(legacy),
          "confirmed-round": 42,
        },
      }),
    });

    (globalThis as any).fetch = fetchMock;

    const res = await verifyTx({
      txid: "TX_LEGACY",
      network: "testnet",
    });

    expect(res.verified).toBe(true);
    expect(res.mode).toBe("legacy");
    expect(res.reason).toBeNull();
    expect(res.note).toEqual(legacy);
    expect(res.confirmedRound).toBe(42);
  });

  it("verifies a p1 note with ts in-window and no onchain hash", async () => {
    const fakeNowSec = 1_700_000_000;
    (Date as any).now = () => fakeNowSec * 1000;

    const p1 = {
      s: "p1",
      a: "re:EUR",
      mv: "model-v1",
      mh: "0".repeat(64),
      ih: "1".repeat(64),
      v: 123_456.78,
      u: [120_000, 130_000],
      ts: fakeNowSec,
    };

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        transaction: {
          id: "TX_P1_OK",
          note: b64Json(p1),
          "confirmed-round": 99,
        },
      }),
    });

    (globalThis as any).fetch = fetchMock;

    const res = await verifyTx({
      txid: "TX_P1_OK",
      network: "testnet",
      maxSkewPastSec: 600,
      maxSkewFutureSec: 600,
    });

    expect(res.verified).toBe(true);
    expect(res.mode).toBe("p1");
    expect(res.reason).toBeNull();
    expect(res.confirmedRound).toBe(99);
    expect(res.note?.s).toBe("p1");
    expect(res.noteSha256).toBe(res.rebuiltSha256);
  });

  it("returns onchain_hash_mismatch when embedded hash does not match rebuilt hash", async () => {
    const p1Bad = {
      s: "p1",
      a: "re:EUR",
      mv: "model-v1",
      mh: "0".repeat(64),
      ih: "1".repeat(64),
      v: 123_456.78,
      u: [120_000, 130_000],
      ts: 1_700_000_000,
      note_sha256: "f".repeat(64), // deliberately wrong
    };

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        transaction: {
          id: "TX_P1_BAD_HASH",
          note: b64Json(p1Bad),
          "confirmed-round": 55,
        },
      }),
    });

    (globalThis as any).fetch = fetchMock;

    const res = await verifyTx({
      txid: "TX_P1_BAD_HASH",
      network: "testnet",
    });

    expect(res.verified).toBe(false);
    expect(res.mode).toBe("p1");
    expect(res.reason).toBe("onchain_hash_mismatch");
  });

  it("returns ts_out_of_window when p1.ts is too far in the past", async () => {
    const fakeNowSec = 2_000_000_000;
    (Date as any).now = () => fakeNowSec * 1000;

    const tooOld = fakeNowSec - 10_000;
    const p1Old = {
      s: "p1",
      a: "re:EUR",
      mv: "model-v1",
      mh: "0".repeat(64),
      ih: "1".repeat(64),
      v: 123_456.78,
      u: [120_000, 130_000],
      ts: tooOld,
    };

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        transaction: {
          id: "TX_P1_OLD",
          note: b64Json(p1Old),
          "confirmed-round": 10,
        },
      }),
    });

    (globalThis as any).fetch = fetchMock;

    const res = await verifyTx({
      txid: "TX_P1_OLD",
      network: "testnet",
      maxSkewPastSec: 600,
      maxSkewFutureSec: 600,
    });

    expect(res.verified).toBe(false);
    expect(res.mode).toBe("p1");
    expect(res.reason).toBe("ts_out_of_window");
  });
});