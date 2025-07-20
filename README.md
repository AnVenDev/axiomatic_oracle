# Algorand-Powered AI Oracle for Real-World Assets (Multi-RWA Ready)

A modular AI + Blockchain oracle that evaluates and monitors Real World Assets (RWA) starting with real estate and designed from day one to extend to art, logistics, agriculture, renewable energy, and industrial assets. It produces structured AI insights and (planned) publishes verifiable summaries to the Algorand blockchain.

## Purpose

This project establishes a reusable oracle layer that:

- **Estimates asset value** (initial focus: property valuation)
- **Simulates condition / risk scoring** via environmental & structural indicators (to refine)
- **Detects anomalies** (planned) using statistical + ML methods
- **Publishes compact, verifiable summaries** to Algorand (TestNet → MainNet, planned)
- **Prepares a sensor / edge ingestion path** (Raspberry Pi, IoT)
- **Supports multiple asset categories** via a unified schema, model registry, and pluggable preprocessing

## Multi-RWA Vision

Phase 1 focuses on property (`asset_type="property"`), while the architecture anticipates additional asset classes:

| Asset Type (future) | Example Features | Primary Tasks |
|---------------------|------------------|---------------|
| **property** (current) | size, rooms, humidity, energy_class | valuation, condition, anomaly |
| **art** | medium, year_created, artist_reputation, storage_humidity | authenticity prob., condition, valuation |
| **greenhouse** | temperature, humidity, light, CO₂, soil_moisture | crop risk, yield score, anomaly |
| **warehouse** | vibration, temp stability, occupancy pattern | integrity, compliance, risk |
| **energy_asset** | panel_efficiency, irradiance, degradation_rate | performance index, maintenance trigger |
| **container** | geo_path, temperature_mean, shock_events | spoilage risk, compliance, anomaly |

### Extensibility Principles

- **Mandatory `asset_type`** on every record
- **Unified output schema** with flexible metrics + flags
- **Model registry** keyed by asset type (`models/<asset_type>/...`)
- **Pluggable feature pipeline** (config-driven)
- **Clear separation** of data generation → preprocessing → training → inference → (future) on-chain publishing

## Key Features (Current vs Planned)

| Feature | Status | Notes |
|---------|--------|-------|
| Synthetic property dataset (150+ rows) | ✅ | Includes asset_type, derived age_years, condition/risk placeholders |
| EDA (distribution, correlation, condition/risk) | ✅ | Notebook 02 |
| Training pipeline (OHE + RandomForest) | ✅ | Notebook 03 (saved as joblib) |
| Model metadata JSON (versioned) | ✅ | Contains metrics & feature lists |
| Inference notebook (single + batch) | ✅ | Notebook 04 |
| Unified output JSON schema | ✅ | `schemas/output_example.json` (draft with schema_version) |
| FastAPI inference API | ✅ | `/predict/property` + schema validation |
| Model registry abstraction | ✅ | `scripts/model_registry.py` |
| E2E sanity check script | ✅ | `scripts/e2e_sanity_check.py` |
| Anomaly detection module | 🔄 Planned | IsolationForest / rules (Notebook 05) |
| Condition & risk refinement logic | 🔄 Planned | Rule/ML hybrid scoring |
| Algorand publishing (Note field) | 🔄 Planned | Compact payload + TX ID |
| Angular dashboard (multi-asset) | 🔄 Planned | Filtering + detail + TX links |
| Sensor ingestion (simulated stream) | 🔮 Future | Real-time updates |
| Raspberry Pi edge device integration | 🔮 Future | Phase 3–4 |
| PyTEAL smart contract hooks | 🔮 Future | Update frequency / anomaly triggers |

## Tech Stack

| Layer | Current | Future / Optional |
|-------|---------|------------------|
| **ML / Data** | Python, Pandas, NumPy, Scikit-learn | XGBoost, LightGBM, PyTorch, SHAP |
| **Backend** | FastAPI (inference) | Auth (JWT), rate limiting |
| **Frontend** | (Planned) Angular 17 | Map view, role-based access |
| **Blockchain** | (Planned) Algorand Python SDK | PyTEAL apps, ASA metadata |
| **Storage** | CSV dataset, joblib model + JSON metadata | SQLite → PostgreSQL, IPFS anchoring |
| **Orchestration** | Notebooks + scripts | DVC, MLflow, model promotion |
| **Edge** | Simulated events | Raspberry Pi + sensors |
| **DevOps** | Git + manual runs | GitHub Actions CI, Docker image |

## Project Structure

```
algorand-ai-oracle/
├── data/                        # Datasets + prediction & API logs
├── models/
│   └── property/
│       ├── value_regressor_v1.joblib
│       └── value_regressor_v1_meta.json
├── notebooks/
│   ├── 01_generate_dataset.ipynb
│   ├── 02_explore_dataset.ipynb
│   ├── 03_train_model.ipynb
│   └── 04_infer_single_sample.ipynb
├── schemas/
│   └── output_example.json      # Unified output schema (draft)
├── scripts/
│   ├── model_registry.py
│   ├── inference_api.py
│   ├── e2e_sanity_check.py
│   ├── feature_pipeline.py      # (planned)
│   └── blockchain_publish.py    # (planned)
├── frontend/                    # (planned) Angular app
├── tests/
│   └── test_api.py
├── README.md
└── requirements.txt
```

## Dataset (Current & Extensible Fields)

### Current property fields:
`asset_id`, `asset_type`, `location`, `size_m2`, `rooms`, `bathrooms`, `year_built`, `age_years`, `floor`, `building_floors`, `has_elevator`, `has_garden`, `has_balcony`, `garage`, `energy_class`, `humidity_level`, `temperature_avg`, `noise_level`, `air_quality_index`, `valuation_k`, `condition_score`, `risk_score`, `last_verified_ts`

### Future asset-specific extensions:

| Asset | Additional Fields (planned) |
|-------|----------------------------|
| **art** | `medium`, `storage_humidity`, `authenticity_prob` |
| **energy_asset** | `panel_efficiency`, `irradiance`, `degradation_rate` |
| **container** | `shock_events`, `temperature_var`, `geo_path` |
| **greenhouse** | `soil_moisture`, `light_index`, `co2_ppm` |
| **warehouse** | `vibration_level`, `occupancy_pattern` |

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

### Future art example:

```json
{
  "schema_version": "v1",
  "asset_id": "art_0045",
  "asset_type": "art",
  "timestamp": "2025-10-02T09:11:22Z",
  "metrics": {
    "valuation_base_k": 320.5,
    "authenticity_prob": 0.94,
    "condition_score": 0.71
  },
  "flags": { "anomaly": false, "needs_review": true },
  "model_meta": {
    "auth_model_version": "cnn_v2"
  }
}
```

See `schemas/output_example.json` for the formal JSON Schema (Draft 2020-12).

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

### Runtime utilities:
- Version discovery & hashing
- Metadata enrichment (`model_hash`, optional `dataset_hash`)
- Cache inspection

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

## Planned Blockchain Publishing (Compact Payload)

### Proposed on-chain Note JSON:

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

## Getting Started

### 1. Clone & Setup

```bash
# Clone
git clone https://github.com/your-username/algorand-ai-oracle.git
cd algorand-ai-oracle

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

## Roadmap (Condensed)

| Phase | Focus | Key Actions |
|-------|-------|-------------|
| **1** | Property dataset + baseline model | Schema draft, metadata, E2E check |
| **2** | API + Algorand publishing | Compact payload + registry refinement |
| **3** | Multi-asset scaffolding | New asset dirs + config & placeholders |
| **4** | Sensor + edge integration | Streaming ingestion + anomaly triggers |
| **5** | New asset domain (e.g. art) | Authenticity model + expanded schema |

## Planned Enhancements

- **IsolationForest / One-Class SVM** anomaly detection
- **Condition & risk score refinement** (probabilistic models)
- **SHAP explainability** integration
- **PyTEAL contract** for update cadence & anomaly flagging
- **Off-chain IPFS anchoring** & hash validation
- **DAO-style governance** / dispute resolution module
- **Asset-type plug-ins** (`plugins/<asset_type>/feature_builder.py`)
- **Docker image + CI pipeline**
- **Dataset & model lineage tracking** (DVC / MLflow)

## E2E Sanity Check

Run to validate full pipeline integrity:

```bash
python scripts/e2e_sanity_check.py
```

*Checks dataset columns, model presence, local vs API predictions, schema compliance, and logging recency.*

## License

MIT License (to be added in LICENSE).

## Disclaimer

⚠️ **All current data is synthetic.** No generated valuations or risk assessments should be considered financial advice. This is an R&D project exploring verifiable AI oracles for tokenized assets on Algorand.

## Contact

| Channel | Link |
|---------|------|
| **X (Twitter)** | WIP |
| **GitHub** | [https://github.com/AnVenDev](https://github.com/AnVenDev) |
| **Email** | anvene.dev@gmail.com |

---

🤝 **If you are building in the Algorand RWA ecosystem and want to collaborate, feel free to reach out!**
