export interface PropertyRequest {
  location?: string;
  size_m2?: number;
  rooms?: number;
  bathrooms?: number;
  year_built?: number;
  floor?: number;
  building_floors?: number;
  has_elevator?: 0 | 1;
  has_garden?: 0 | 1;
  has_balcony?: 0 | 1;
  garage?: 0 | 1;
  energy_class?: string;
  humidity_level?: number;
  temperature_avg?: number;
  noise_level?: number;
  air_quality_index?: number;
  age_years?: number;
  [k: string]: any;
}

export interface PredictionResponseV2 {
  schema_version: 'v2' | string;
  asset_id: string;
  asset_type: string;
  timestamp: string;

  metrics: {
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
  };

  flags?: Record<string, any>;

  model_meta?: {
    value_model_version?: string | null;
    value_model_name?: string | null;
    n_features_total?: number | null;
    model_hash?: string | null;
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

  // — opzionali, valorizzati dopo publish
  blockchain_txid?: string;
  asa_id?: number | string | null;
  note_sha256?: string;
  note_size?: number;
  is_compacted?: boolean;
  confirmed_round?: number;

  // attestazione costruita dal backend
  attestation?: {
    p1: any;
    p1_sha256: string;
    p1_size_bytes: number;
    canonical_input?: Record<string, any>;
    input_hash?: string;
  };

  publish: {
    status: 'ok' | 'success' | 'error' | 'skipped' | 'not_attempted';
    txid?: string;
    asa_id?: number | string | null;
    network?: string | null;
    note_size?: number | null;
    note_sha256?: string | null;
    is_compacted?: boolean | null;
    confirmed_round?: number | null;
    explorer_url?: string | null;
    error?: string;
  };
}
