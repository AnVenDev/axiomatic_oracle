# Algorand-Powered AI Oracle for Real-World Assets (Multi-RWA Ready)

A modular AI + Blockchain oracle that evaluates and monitors Real World Assets (RWA) starting with real estate and designed from day one to extend to art, logistics, agriculture, renewable energy, and industrial assets. It produces structured AI insights and (planned) publishes verifiable summaries to the Algorand blockchain.

## Purpose

This project establishes a reusable oracle layer that:

- **Estimates asset value** (starting with real estate)
- **Simulates condition / risk scoring** using interpretable rules and proxy features
- **Detects anomalies** via simple threshold rules (future: ML-based)
- **Estimates prediction confidence intervals** to assess output robustness
- **Monitors inference latency** and logs output + model metadata
- **Detects potential feature drift** using training-time statistics
- **(Planned) Publishes summaries** to Algorand TestNet/MainNet
- **(Planned) Connects to sensor/edge pipelines** (e.g., Raspberry Pi, IoT)

## Multi-RWA Vision

While the MVP focuses on real estate assets (`asset_type="property"`), the system is architected for extensibility:

| Asset Type         | Key Features (Examples)                               | Core Tasks                                     |
|--------------------|--------------------------------------------------------|------------------------------------------------|
| **property**       | size, rooms, humidity, energy_class                   | valuation, condition estimation, anomaly check |
| **art**            | medium, year, artist reputation, storage conditions    | valuation, authenticity scoring                |
| **greenhouse**     | temperature, CO₂, light, humidity, soil moisture       | yield, crop risk, anomaly detection            |
| **warehouse**      | vibration, temperature variance, occupancy             | compliance, risk, maintenance prediction       |
| **energy_asset**   | panel performance, irradiance, degradation rate        | efficiency score, maintenance alert            |
| **container**      | shock events, geo-path, temperature variation          | spoilage risk, integrity assessment            |

---

### Extensibility Principles

* **Mandatory `asset_type`** on every record
* **Unified output schema** with flexible metrics + flags
* **Model registry** keyed by asset type (`models/<asset_type>/...`)
* **Pluggable feature pipeline** (config-driven)
* **Clear separation** of data generation → preprocessing → training → inference → (future) on-chain publishing

## Dataset

### Core fields for `property` (MVP):

`asset_id`, `asset_type`, `location`, `size_m2`, `rooms`, `bathrooms`, `year_built`, `age_years`, `floor`, `building_floors`, `has_elevator`, `has_garden`, `has_balcony`, `garage`, `energy_class`, `humidity_level`, `temperature_avg`, `noise_level`, `air_quality_index`, `valuation_k`, `condition_score`, `risk_score`, `last_verified_ts`

### Future asset-specific extensions:

| Asset             | Additional Fields (planned)                          |
| ----------------- | ---------------------------------------------------- |
| **art**           | `medium`, `storage_humidity`, `authenticity_prob`    |
| **energy\_asset** | `panel_efficiency`, `irradiance`, `degradation_rate` |
| **container**     | `shock_events`, `temperature_var`, `geo_path`        |
| **greenhouse**    | `soil_moisture`, `light_index`, `co2_ppm`            |
| **warehouse**     | `vibration_level`, `occupancy_pattern`               |

*Unpopulated fields remain NaN until implemented.*

## Output Schema (Unified Draft)

This schema is used for both batch and single-sample inference. Designed to be compact, extensible, and verifiable.

```json
{
  "schema_version": "v1",
  "asset_id": "property_0102",
  "asset_type": "property",
  "timestamp": "2025-07-21T14:32:00Z",
  "metrics": {
  "valuation_base_k": 153.45,
  "uncertainty": 12.5,
  "confidence_low_k": 141.2,
  "confidence_high_k": 165.7,
  "latency_ms": 45.3,
  "condition_score": 0.81,
  "risk_score": 0.22
  },
  "flags": {
    "anomaly": false,
    "drift_detected": false,
    "needs_review": false
  },
  "model_meta": {
    "value_model_version": "v1",
    "value_model_name": "LGBMRegressor"
  },
  "offchain_refs": {
    "detail_report_hash": null,
    "sensor_batch_hash": null
  }
}
```
---

## Notable Features Implemented

### Confidence Intervals (via Prediction Variance)
Simulated prediction intervals provide a low-high estimate around the prediction.
* Method: Monte Carlo simulation using t-distribution on LGBM ensemble predictions
* Output: `uncertainty`, `confidence_low_k`, `confidence_high_k`

### Rules-Based Anomaly Detection
Simple rules catch outliers based on thresholds (e.g., extreme sizes, missing data)
* Configurable dictionary of thresholds
* Output: `flags.anomaly`

### Inference Monitoring & Logs
Every batch/single prediction logs:
* Inference time (ms)
* Model metadata
* Confidence intervals
* Anomaly & drift flags

Saved to `/data/monitoring_log.jsonl`

### Feature Drift Detection
Basic drift detection compares incoming sample stats to training-time `mean`/`std`
* Uses metadata from training notebook
* Flags via `flags.drift_detected`

### Model Performance (MVP v1, 03_train_model.ipynb)

- **MAE**: 65.27k€ (realistic for real estate)
- **RMSE**: 85.80k€
- **R²**: 0.55 (excellent given multivariate noise and synthetic data)

---

### Advanced Features (MVP v1)

- **Optuna tuning** with early stopping and robust parameter space
- **Log-transform target** for better regression stability
- **Z-score-based drift detection** against training stats
- **Partial fallback system** for model selection by asset_type/version (planned)

---

## Model Registry

### Simplified conceptual structure:

```python
MODEL_REGISTRY = {
    "property": {
        "value_regressor": "property/value_regressor_v1.joblib"
    }
}
```

**Includes**:
* Version control
* Optional hash validation
* Fallback support (future)

---

### Runtime Utilities

* Version discovery & hashing
* Metadata enrichment (`model_hash`, optional `dataset_hash`)
* Cache inspection

## Workflow Summary

1. **Generate synthetic asset records** (Notebook 01)
2. **Visualize and validate data** (Notebook 02)
3. **Train model + export pipeline & metadata** (Notebook 03)
4. **Perform prediction, log metrics** (Notebook 04)
5. **Start API** (`uvicorn scripts.inference_api:app --reload --port 8000`)
6. **Call `/predict/property`** and verify schema compliance
7. **Run E2E sanity** (`python scripts/e2e_sanity_check.py`)
9. **(Planned) Publish minimal payload to Algorand**

## Blockchain Payload (Planned)

```json
{
  "a": "property_0102",
  "t": "property",
  "ts": "2025-07-21T14:32:00Z",
  "v": 153.45,
  "c": 0.81,
  "r": 0.22,
  "an": 0
}
```

*Extended report anchored off-chain (e.g. IPFS) via hash in `detail_report_hash`.*

## Planned Enhancements

* **IsolationForest / One-Class SVM** anomaly detection
* **SHAP explainability integration**
* **SHAP explainability** integration
* **PyTEAL contract**  for on-chain validation
* **Sensor-based ingestion from Raspberry Pi**
* **Off-chain IPFS anchoring** & hash validation
* **DAO-style governance** / dispute resolution module
* **Asset-type plug-ins** (`plugins/<asset_type>/feature_builder.py`)
* **Docker image + CI pipeline**
* **Dataset & model lineage tracking** (DVC / MLflow)
* **Dynamic retraining & dataset expansion**

## Getting Started

### 1. Clone & Setup

```bash
# Clone
git clone https://github.com/AnVenDev/ai-oracle-rwa.git
cd ai-oracle-rwa

# Environment
conda create -n ai-oracle python=3.11 -y
conda activate ai-oracle

# Install core deps
pip install -r requirements.txt
# (or minimal)
# pip install pandas numpy scikit-learn joblib jupyter matplotlib seaborn

# (Optional extras)
pip install fastapi uvicorn algosdk jsonschema
```

### 2. Run notebooks in order:

1. `01_generate_dataset.ipynb`
2. `02_explore_dataset.ipynb`
3. `03_train_model.ipynb`
4. `04_infer_single_sample.ipynb`

### 3. Start API:

```bash
uvicorn scripts.inference_api:app --reload --port 8000
```

### 4. Test API:

```bash
curl -X POST http://127.0.0.1:8000/predict/property \
  -H "Content-Type: application/json" \
  -d @schemas/sample_property.json
```

### 5. Run E2E sanity:

```bash
python scripts/e2e_sanity_check.py
```

## License

MIT License (to be added).

## Disclaimer

🚨 All current data is synthetic. This is a prototype system for research on decentralized AI oracles. No predictions represent financial or investment advice.

## Contact

| Channel     | Link                                                       |
| ----------- | ---------------------------------------------------------- |
| X (Twitter) | WIP                                                        |
| GitHub      | [https://github.com/AnVenDev](https://github.com/AnVenDev) |
| Email       | [anvene.dev@gmail.com](mailto:anvene.dev@gmail.com)        |
