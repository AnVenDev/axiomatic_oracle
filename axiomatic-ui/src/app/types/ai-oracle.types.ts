export interface MetricsV2 {
  valuation_k: number;
  point_pred_k: number;
  uncertainty_k: number;
  confidence: number;
  confidence_low_k: number;
  confidence_high_k: number;
  ci_margin_k: number;
  latency_ms: number;
  ci_method?: string | null;
  n_estimators?: number | null;
}

export type Network = 'mainnet' | 'testnet' | 'betanet' | 'sandbox';

export interface AttestationP1 {
  s: 'p1';
  a: 're:EUR';
  mv: string; // model_version
  mh: string; // model_hash (sha256 hex)
  ih: string; // input_hash (sha256 hex)
  v: number; // value in EUR
  u: [number, number]; // [lo, hi] EUR
  ts: number; // epoch seconds
}

export interface ModelMetaV2 {
  value_model_version?: string | null;
  value_model_name?: string | null;
  n_features_total?: number | null;
}

export interface PublishInfo {
  status: 'success' | 'error' | 'skipped' | 'ok' | 'not_attempted';
  txid?: string;
  asa_id?: number | string | null;
  error?: string;
  network?: 'mainnet' | 'testnet' | 'betanet' | 'sandbox';
  confirmed_round?: number;
  note_size?: number;
  note_sha256?: string;
}

export interface PredictionResponseV2 {
  schema_version: 'v2' | '2.0' | string;
  asset_id: string;
  asset_type: 'property' | string;
  timestamp: string; // ISO
  metrics: MetricsV2;
  flags?: Record<string, any>;
  model_meta?: ModelMetaV2;
  value_eur?: number;
  interval_eur?: [number, number];
  model?: { name?: string; version?: string; hash?: string };
  attestation?: {
    p1?: {
      s: 'p1';
      a: 're:EUR';
      mv: string;
      mh: string; // sha256 hex
      ih: string; // sha256 hex
      v: number; // EUR
      u: [number, number]; // [lo, hi] EUR
      ts: number; // epoch seconds
    };
    p1_sha256?: string;
  };
  model_health?: Record<string, any>;
  validation?: Record<string, any>;
  explanations?: Record<string, any>;
  sanity?: Record<string, any>;
  offchain_refs?: {
    detail_report_hash?: string | null;
    sensor_batch_hash?: string | null;
  };
  cache_hit?: boolean;
  schema_validation_error?: string;
  /** legacy top-level (manteniamo per compat) */
  blockchain_txid?: string;
  asa_id?: number | string | null;
  note_sha256?: string;
  note_size?: number;
  is_compacted?: boolean;
  confirmed_round?: number;
  /** v2: publish opzionale (con extras) */
  publish?: PublishInfo;
}

export interface VerifyRequest {
  txid: string;
  prediction?: any; // opzionale: puoi passare la risposta completa per hash match
}

export interface VerifyResponse {
  txid: string;
  verified: boolean;
  mode?: 'p1';
  note_sha256?: string;
  note_size?: number;
  confirmed_round?: number;
  reason?: string | null;
  [k: string]: any;
}
