import { describe, it, expect } from "vitest";
import { Buffer } from "node:buffer";
import {
  NOTE_MAX_BYTES,
  buildP1,
  canonicalNoteBytesP1,
  assertNoteSizeOK,
  buildCanonicalInput,
  computeInputHash,
} from "../src/index.js";
import { jcsSha256 } from "../src/jcs.js"

describe("buildP1 / canonical note / input hashing (proofkit)", () => {
  it("builds a valid p1 object with expected fields", () => {
    const p1 = buildP1({
      assetTag: "re:EUR",
      modelVersion: "model-v1",
      modelHashHex: "0".repeat(64),
      inputHashHex: "1".repeat(64),
      valueEUR: 123_456.78,
      uncertaintyLowEUR: 120_000,
      uncertaintyHighEUR: 130_000,
      timestampEpochSec: 1_700_000_000,
    });

    expect(p1.s).toBe("p1");
    expect(p1.a).toBe("re:EUR");
    expect(p1.mv).toBe("model-v1");
    expect(p1.mh).toBe("0".repeat(64));
    expect(p1.ih).toBe("1".repeat(64));
    expect(typeof p1.v).toBe("number");
    expect(p1.u).toEqual([120_000, 130_000]);
    expect(p1.ts).toBe(1_700_000_000);
  });

  it("rejects invalid hex hashes and intervals", () => {
    // bad model hash
    expect(() =>
      buildP1({
        assetTag: "re:EUR",
        modelVersion: "model-v1",
        modelHashHex: "xyz",
        inputHashHex: "1".repeat(64),
        valueEUR: 1,
        uncertaintyLowEUR: 0,
        uncertaintyHighEUR: 2,
        timestampEpochSec: 1_700_000_000,
      }),
    ).toThrow(/modelHashHex/i);

    // bad input hash
    expect(() =>
      buildP1({
        assetTag: "re:EUR",
        modelVersion: "model-v1",
        modelHashHex: "0".repeat(64),
        inputHashHex: "!",
        valueEUR: 1,
        uncertaintyLowEUR: 0,
        uncertaintyHighEUR: 2,
        timestampEpochSec: 1_700_000_000,
      }),
    ).toThrow(/inputHashHex/i);

    // invalid interval
    expect(() =>
      buildP1({
        assetTag: "re:EUR",
        modelVersion: "model-v1",
        modelHashHex: "0".repeat(64),
        inputHashHex: "1".repeat(64),
        valueEUR: 1,
        uncertaintyLowEUR: 200,
        uncertaintyHighEUR: 100,
        timestampEpochSec: 1_700_000_000,
      }),
    ).toThrow(/uncertainty/i);
  });

  it("produces stable canonical note bytes and hash", async () => {
    const base = {
      assetTag: "re:EUR",
      modelVersion: "model-v1",
      modelHashHex: "0".repeat(64),
      inputHashHex: "1".repeat(64),
      valueEUR: 123_456.78,
      uncertaintyLowEUR: 120_000,
      uncertaintyHighEUR: 130_000,
      timestampEpochSec: 1_700_000_000,
    };

    const p1a = buildP1(base);
    const p1b = buildP1({ ...base });

    const noteA = await canonicalNoteBytesP1(p1a);
    const noteB = await canonicalNoteBytesP1(p1b);

    expect(noteA.size).toBe(noteA.bytes.length);
    expect(noteB.size).toBe(noteB.bytes.length);

    expect(Buffer.from(noteA.bytes).toString("utf8")).toBe(
      Buffer.from(noteB.bytes).toString("utf8"),
    );
    expect(noteA.sha256).toBe(noteB.sha256);
  });

  it("enforces NOTE_MAX_BYTES via assertNoteSizeOK", async () => {
    const p1 = buildP1({
      assetTag: "re:EUR",
      modelVersion: "model-v1",
      modelHashHex: "0".repeat(64),
      inputHashHex: "1".repeat(64),
      valueEUR: 1,
      uncertaintyLowEUR: 0,
      uncertaintyHighEUR: 2,
      timestampEpochSec: 1_700_000_000,
    });

    // Should not throw with default NOTE_MAX_BYTES
    await expect(assertNoteSizeOK(p1)).resolves.toBeUndefined();
    expect(NOTE_MAX_BYTES).toBeGreaterThan(0);

    // Force a failure with an artificially tiny limit
    await expect(assertNoteSizeOK(p1, 1)).rejects.toThrow(/too large/i);
  });

  it("buildCanonicalInput filters and strips fields as expected", async () => {
    const rec = {
      keep: 1,
      drop: 2,
      skipNull: null,
      skipUndef: undefined as unknown,
    };

    const allowed = ["keep", "skipNull", "skipUndef"];
    const cin = buildCanonicalInput(rec, allowed);

    expect(cin).toEqual({ keep: 1 });

    const hash1 = await computeInputHash(rec, allowed);
    const hash2 = await jcsSha256(cin);

    expect(hash1).toBe(hash2);
  });
});
