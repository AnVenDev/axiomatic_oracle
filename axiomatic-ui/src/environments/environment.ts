export const environment = {
  production: false,
  apiBase: '', // vuoto: usiamo il proxy (es: /predict, /verify)
  apiKeyStorageKey: 'AI_ORACLE_API_KEY',

  // Rete Algorand di default per i link Explorer (può essere overridata da publish.network)
  defaultNetwork: 'testnet' as 'mainnet' | 'testnet' | 'betanet' | 'sandbox',
  explorers: {
    mainnet: 'https://explorer.perawallet.app/tx/',
    testnet: 'https://testnet.explorer.perawallet.app/tx/',
    betanet: null,
    sandbox: null,
  },
};
