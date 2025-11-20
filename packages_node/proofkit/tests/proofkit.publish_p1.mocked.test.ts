import { describe, it, expect, vi } from "vitest";
import {
  buildP1,
  publishP1,
} from "../src/index";

const sendRawTxSpy = vi.fn();
const getParamsSpy = vi.fn();
const encodeUnsignedTransactionSpy = vi.fn();
const waitForConfirmationSpy = vi.fn();

// Mock the "algosdk" module used internally by publishP1.
vi.mock("algosdk", () => {
  class FakeAlgod {
    token: string;
    url: string;
    port: string;

    constructor(token: string, url: string, port: string) {
      this.token = token;
      this.url = url;
      this.port = port;
    }

    getTransactionParams() {
      return {
        do: () => getParamsSpy(),
      };
    }

    sendRawTransaction(bytes: Uint8Array) {
      return {
        do: () => sendRawTxSpy(bytes),
      };
    }
  }

  return {
    default: {
      Algodv2: FakeAlgod,
      isValidAddress: (_addr: string) => true,
      decodeAddress: (_addr: string) => ({ publicKey: new Uint8Array(32) }),
      makePaymentTxnWithSuggestedParamsFromObject: (args: unknown) => ({
        ...((args as object) ?? {}),
        dummy: true,
      }),
      encodeUnsignedTransaction: (txn: unknown) => {
        encodeUnsignedTransactionSpy(txn);
        return new Uint8Array([1, 2, 3]);
      },
      waitForConfirmation: async (...args: unknown[]) => {
        waitForConfirmationSpy(...args);
        return { "confirmed-round": 123 };
      },
    },
  };
});

describe("publishP1 (proofkit)", () => {
  it("builds and sends a 0-ALGO self-payment with canonical P1 note", async () => {
    const suggestedParams = {
      fee: 1000,
      flatFee: true,
      firstRound: 1,
      lastRound: 1000,
      genesisHash: "GEN_HASH",
      genesisID: "GEN_ID",
    };

    sendRawTxSpy.mockReset();
    getParamsSpy.mockReset();
    encodeUnsignedTransactionSpy.mockReset();
    waitForConfirmationSpy.mockReset();

    getParamsSpy.mockImplementation(async () => suggestedParams);
    sendRawTxSpy.mockImplementation(
      async (_bytes: Uint8Array): Promise<{ txId: string }> => ({
        txId: "FAKE-TXID",
      }),
    );
    waitForConfirmationSpy.mockResolvedValue({ "confirmed-round": 123 });

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

    const signSpy = vi
      .fn()
      .mockImplementation(async (unsigned: Uint8Array): Promise<Uint8Array> => {
        expect(unsigned.length).toBeGreaterThan(0);
        return Uint8Array.from(unsigned);
      });

    const res = await publishP1({
      p1,
      from: "SENDER-ADDRESS",
      network: "testnet",
      sign: signSpy,
      waitRounds: 1,
      // Force publishP1 to construct its own Algodv2 instance using the mock.
      algod: undefined,
    });

    expect(res.txid).toBe("FAKE-TXID");
    expect(res.explorerUrl).toContain(
      "testnet.explorer.perawallet.app/tx/FAKE-TXID",
    );

    expect(signSpy).toHaveBeenCalledTimes(1);
    expect(getParamsSpy).toHaveBeenCalledTimes(1);
    expect(sendRawTxSpy).toHaveBeenCalledTimes(1);
    expect(encodeUnsignedTransactionSpy).toHaveBeenCalledTimes(1);
    expect(waitForConfirmationSpy).toHaveBeenCalledTimes(1);
  });
});
