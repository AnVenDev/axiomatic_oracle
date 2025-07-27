export interface PropertyRequest {
  location: string;
  size_m2: number;
  rooms: number;
  bathrooms: number;
  year_built: number;
  floor: number;
  building_floors: number;
  has_elevator: number;
  has_garden: number;
  has_balcony: number;
  garage: number;
  energy_class: string;
  humidity_level: number;
  temperature_avg: number;
  noise_level: number;
  air_quality_index: number;
  age_years?: number;
}

export interface PredictionResponse {
  schema_version: string;
  asset_id: string;
  asset_type: string;
  timestamp: string;
  metrics: {
    valuation_base_k: number;
    uncertainty?: number;
    confidence_low_k?: number;
    confidence_high_k?: number;
    latency_ms: number;
    condition_score?: number;
    risk_score?: number;
  };
  flags: {
    anomaly: boolean;
    drift_detected?: boolean;
    needs_review: boolean;
  };
  model_meta: {
    value_model_version: string;
    value_model_name: string;
    dataset_hash?: string;
    model_hash?: string;
  };
  offchain_refs: {
    detail_report_hash: string | null;
    sensor_batch_hash: string | null;
  };
  publish?: {
    status: string;
    txid: string;
  };
  model_health?: any;
  cache_hit?: boolean;
  schema_validation_error?: string | null;
}
