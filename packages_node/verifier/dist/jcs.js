function ensureJsonSafe(value) {
    if (typeof value === "undefined")
        return null;
    if (typeof value === "number") {
        if (!Number.isFinite(value))
            throw new Error("Non-finite float not allowed in canonical JSON (NaN/Inf).");
        return value;
    }
    if (value === null || typeof value === "string" || typeof value === "boolean")
        return value;
    if (typeof value === "bigint")
        return value.toString(10);
    if (Array.isArray(value))
        return value.map((v) => ensureJsonSafe(v));
    if (value && typeof value === "object") {
        const out = {};
        for (const k of Object.keys(value).map(String).sort())
            out[k] = ensureJsonSafe(value[k]);
        return out;
    }
    return String(value);
}
export function toJcsBytes(obj) {
    const safe = ensureJsonSafe(obj);
    const s = JSON.stringify(safe);
    return new TextEncoder().encode(s);
}
export async function sha256Hex(bytes) {
    const g = globalThis;
    const subtle = g.crypto?.subtle;
    if (subtle && typeof subtle.digest === "function") {
        const view = bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
            ? bytes.buffer
            : bytes.buffer.slice(bytes.byteOffset, bytes.byteLength + bytes.byteOffset);
        const buf = await subtle.digest("SHA-256", view);
        return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
    }
    const { createHash } = await import("node:crypto");
    const { Buffer } = await import("node:buffer");
    return createHash("sha256").update(Buffer.from(bytes)).digest("hex");
}
export async function jcsSha256(obj) {
    return sha256Hex(toJcsBytes(obj));
}
