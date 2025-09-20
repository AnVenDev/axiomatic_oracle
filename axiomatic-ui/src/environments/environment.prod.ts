export const environment = {
  production: true,
  apiBase: '',
  apiKeyStorageKey: 'AI_ORACLE_API_KEY',

  defaultNetwork: 'testnet' as 'mainnet' | 'testnet' | 'betanet' | 'sandbox',
  explorers: {
    mainnet: 'https://explorer.perawallet.app/tx/',
    testnet: 'https://testnet.explorer.perawallet.app/tx/',
    betanet: null,
    sandbox: null,
  },
};
