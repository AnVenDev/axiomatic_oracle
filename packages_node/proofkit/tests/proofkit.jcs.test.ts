import { describe, it, expect } from "vitest";
import { toJcsBytes, jcsSha256 } from "../src/jcs.js";

const td = new TextDecoder();

/**
 * Helper to decode JCS bytes into a UTF-8 string.
 */
function jcsString(obj: unknown): string {
  return td.decode(toJcsBytes(obj));
}

describe("JCS / canonical JSON (proofkit)", () => {
  it("produces identical canonical JSON for equivalent objects", async () => {
    const obj1 = {
      s: "p1",
      a: "re:EUR",
      mv: "model-v1",
      mh: "0".repeat(64),
      ih: "1".repeat(64),
      v: 100.0,
      u: [90.0, 110.0],
      ts: 1_700_000_000,
      nested: {
        z: 3,
        a: 1,
      },
    };

    // Same logical content, different key ordering
    const obj2 = {
      nested: {
        a: 1,
        z: 3,
      },
      u: [90.0, 110.0],
      v: 100.0,
      ih: "1".repeat(64),
      mh: "0".repeat(64),
      mv: "model-v1",
      a: "re:EUR",
      s: "p1",
      ts: 1_700_000_000,
    };

    const s1 = jcsString(obj1);
    const s2 = jcsString(obj2);

    expect(s1).toBe(s2);

    const h1 = await jcsSha256(obj1);
    const h2 = await jcsSha256(obj2);

    expect(h1).toBe(h2);
  });

  it("normalizes undefined to null in canonical JSON", () => {
    const obj = { a: undefined as unknown, b: 1 };
    const s = jcsString(obj);

    // JCS-style: undefined → null
    expect(s).toBe('{"a":null,"b":1}');
  });

  it("rejects non-finite numbers", () => {
    const badValues = [NaN, Infinity, -Infinity];

    for (const x of badValues) {
      const obj = { v: x };
      expect(() => toJcsBytes(obj)).toThrow(/Non-finite float/);
    }
  });

  it("encodes bigint values as decimal strings", () => {
    const obj = { big: 42n };
    const s = jcsString(obj);

    // Key ordering is deterministic: {"big":"42"}
    expect(s).toBe('{"big":"42"}');
  });
});
