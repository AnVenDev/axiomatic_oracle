# Algorand-Powered AI Oracle for Real-World Assets (Multi-RWA Ready)

A modular AI + Blockchain oracle that evaluates and monitors Real World Assets (RWA) starting with real estate and designed from day one to extend to art, logistics, agriculture, renewable energy, and industrial assets. It produces structured AI insights and (planned) publishes verifiable summaries to the Algorand blockchain.

## Purpose

This project establishes a reusable oracle layer that:

* **Estimates asset value** (initial focus: property valuation)
* **Simulates condition / risk scoring** via environmental & structural indicators (to refine)
* **Detects anomalies** using statistical + ML rules (coming)
* **Publishes compact, verifiable summaries** to Algorand (TestNet → MainNet, planned)
* **Prepares a sensor / edge ingestion path** (Raspberry Pi, IoT)
* **Supports multiple asset categories** via a unified schema, model registry, and pluggable preprocessing

## Multi-RWA Vision

Phase 1 focuses on property (`asset_type="property"`), while the architecture anticipates additional asset classes:

| Asset Type (future)    | Example Features                                             | Primary Tasks                            |
| ---------------------- | ------------------------------------------------------------ | ---------------------------------------- |
| **property** (current) | size, rooms, humidity, energy\_class                         | valuation, condition, anomaly            |
| **art**                | medium, year\_created, artist\_reputation, storage\_humidity | authenticity prob., condition, valuation |
| **greenhouse**         | temperature, humidity, light, CO₂, soil\_moisture            | crop risk, yield score, anomaly          |
| **warehouse**          | vibration, temp stability, occupancy pattern                 | integrity, compliance, risk              |
| **energy\_asset**      | panel\_efficiency, irradiance, degradation\_rate             | performance index, maintenance trigger   |
| **container**          | geo\_path, temperature\_mean, shock\_events                  | spoilage risk, compliance, anomaly       |

### Extensibility Principles

* **Mandatory `asset_type`** on every record
* **Unified output schema** with flexible metrics + flags
* **Model registry** keyed by asset type (`models/<asset_type>/...`)
* **Pluggable feature pipeline** (config-driven)
* **Clear separation** of data generation → preprocessing → training → inference → (future) on-chain publishing

## Dataset (Current & Extensible Fields)

### Current property fields:

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

## Unified Output Schema (Draft)

### Property example:

```json
{
  "schema_version": "v1",
  "asset_id": "property_0102",
  "asset_type": "property",
  "timestamp": "2025-07-18T12:04:55Z",
  "metrics": {
    "valuation_base_k": 152.37,
    "condition_score": 0.82,
    "risk_score": 0.18
  },
  "flags": { "anomaly": false, "needs_review": false },
  "model_meta": {
    "value_model_version": "v1",
    "value_model_name": "RandomForestRegressor"
  },
  "offchain_refs": {
    "detail_report_hash": null,
    "sensor_batch_hash": null
  }
}
```

## Model Registry

### Simplified conceptual structure:

```python
MODEL_REGISTRY = {
    "property": {
        "value_regressor": "property/value_regressor_v1.joblib"
        # future: "anomaly_model": "property/anomaly_iforest_v0.joblib"
    },
    "art": {
        # "valuation_model": "...",
        # "authenticity_model": "..."
    }
}
```

### Runtime Utilities

* Version discovery & hashing
* Metadata enrichment (`model_hash`, optional `dataset_hash`)
* Cache inspection

## Workflow Summary

1. **Generate dataset** (Notebook 01)
2. **Explore & validate distributions** (Notebook 02)
3. **Train model + export pipeline & metadata** (Notebook 03)
4. **Single / batch inference experiments** (Notebook 04)
5. **Start API** (`uvicorn scripts.inference_api:app --reload --port 8000`)
6. **Call `/predict/property`** and verify schema compliance
7. **Run E2E sanity** (`python scripts/e2e_sanity_check.py`)
8. **(Planned) Add anomaly / condition dynamic logic**
9. **(Planned) Publish minimal payload to Algorand**

## Blockchain Publishing (Compact Payload)

```json
{
  "a": "property_0102",
  "t": "property",
  "ts": "2025-07-18T12:04:55Z",
  "v": 152.37,
  "c": 0.82,
  "r": 0.18,
  "an": 0
}
```

*Extended report anchored off-chain (e.g. IPFS) via hash in `detail_report_hash`.*

## Planned Enhancements

* **IsolationForest / One-Class SVM** anomaly detection
* **Condition & risk score refinement** (probabilistic models)
* **SHAP explainability** integration
* **PyTEAL contract** for update cadence & anomaly flagging
* **Off-chain IPFS anchoring** & hash validation
* **DAO-style governance** / dispute resolution module
* **Asset-type plug-ins** (`plugins/<asset_type>/feature_builder.py`)
* **Docker image + CI pipeline**
* **Dataset & model lineage tracking** (DVC / MLflow)

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
