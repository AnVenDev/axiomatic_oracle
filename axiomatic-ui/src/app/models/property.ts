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
  // accetta extra; il backend fa canonicalizzazione/validazione
  [k: string]: any;
}

export interface PredictionResponseV2 {
  schema_version: 'v2';
  asset_id: string;
  asset_type: string;
  timestamp: string;
  metrics: {
    valuation_k: number;
    point_pred_k: number; // predizione “puntuale”
    uncertainty_k: number; // deviazione/σ in k€
    confidence: number; // es. 0.95
    confidence_low_k: number;
    confidence_high_k: number;
    ci_margin_k: number; // (high - point_pred)
    latency_ms: number;
    ci_method?: string;
    n_estimators?: number | null;
  };
  flags: {
    anomaly: boolean;
    drift_detected: boolean;
    needs_review: boolean;
    price_out_of_band?: boolean;
  };
  model_meta: {
    value_model_version?: string | null;
    value_model_name?: string | null;
    n_features_total?: number;
  };
  model_health: {
    status: string;
    model_path?: string;
    metadata_valid?: boolean;
    metrics?: Record<string, any>;
  };
  validation?: {
    ok: boolean;
    warnings?: any;
    errors?: any;
  };
  drift?: { message?: string | null };
  offchain_refs?: {
    detail_report_hash: string | null;
    sensor_batch_hash: string | null;
  };
  cache_hit?: boolean;
  schema_validation_error?: string;
  blockchain_txid?: string;
  asa_id?: string | number;
  publish: {
    status: 'skipped' | 'success' | 'error' | 'not_attempted';
    txid?: string;
    asa_id?: string | number;
    error?: string;
  };
}
