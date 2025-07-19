# Algorand-Powered AI Oracle for Real-World Assets (Multi-RWA Ready)

A modular AI + Blockchain oracle that evaluates and monitors **Real World Assets (RWA)** beginning with real estate and designed from day one to extend to **art, logistics, agriculture, renewable energy, and industrial assets**.  
It produces structured AI insights and publishes verifiable summaries to the Algorand blockchain.

---

## 1. Purpose

This project establishes a reusable oracle layer that:

- Estimates asset value (initially property valuation)
- Assesses condition / risk via environmental & structural indicators
- Detects anomalies (planned) using statistical + ML methods
- Publishes compact, verifiable summaries to Algorand (TestNet → MainNet)
- Prepares a sensor / edge ingestion path (Raspberry Pi, IoT)
- Supports *multiple asset categories* via a unified schema and model registry

---

## 2. Multi-RWA Vision

While Phase 1 focuses on **property (asset_type = "property")**, the architecture anticipates additional asset classes:

| Asset Type (future) | Example Features | Primary Tasks |
|---------------------|------------------|---------------|
| `property` (current) | size, rooms, humidity, energy_class | valuation, condition, anomaly |
| `art` | medium, year_created, artist_reputation, storage_humidity | authenticity probability, condition scoring, valuation |
| `greenhouse` | temperature, humidity, light, CO₂, soil_moisture | crop risk, yield score, anomaly |
| `warehouse` | vibration, temp stability, occupancy pattern | integrity flag, compliance, risk |
| `energy_asset` | panel_efficiency, irradiance, degradation_rate | performance index, maintenance trigger |
| `container` | geo_path, temperature_mean, shock_events | spoilage risk, compliance, anomaly |

**Design Principles for Extensibility:**

- `asset_type` field mandatory on every record
- Unified output schema with flexible `metrics` and `flags`
- Model registry keyed by asset type (e.g. `models/property/…`, `models/art/…`)
- Pluggable feature pipeline per asset type

---

## 3. Key Features (MVP Scope)

| Feature | Status | Notes |
|---------|--------|-------|
| Synthetic real estate dataset (150+ rows) | ✅ | Will add `asset_type` column |
| EDA & baseline valuation model | ✅ | RandomForest regression |
| Unified output schema (JSON draft) | Planned | To be finalized early |
| Anomaly detection module | Planned | IsolationForest / rules |
| Condition / risk scoring fields | Planned | (`condition_score`, `risk_score`) |
| FastAPI inference API | Planned | `/predict/{asset_type}` |
| Algorand publishing (Note field) | Planned | Minimal JSON payload |
| Model registry abstraction | Planned | `MODEL_REGISTRY` structure |
| Angular dashboard (multi-asset) | Planned | Asset filters & detail view |
| Sensor ingestion (simulated) | Future | JSON → API endpoint |
| Raspberry Pi edge device | Future | Phase 3–4 |
| PyTEAL smart contract hooks | Future | Update frequency & anomaly triggers |

---

## 4. Tech Stack

| Layer | Current | Future / Optional |
|------|---------|--------------------|
| ML / Data | Python, Pandas, NumPy, Scikit-learn | XGBoost, LightGBM, PyTorch (vision), SHAP |
| Backend | FastAPI (planned) | Auth (JWT), Rate limiting |
| Frontend | Angular 17 (planned) | Role-based access, Map view |
| Blockchain | Algorand Python SDK | PyTEAL, ASA metadata, AppCall contracts |
| Storage | CSV (synthetic), models in `models/` | SQLite → PostgreSQL, IPFS hashes |
| Orchestration | Notebooks → scripts | DVC, MLflow |
| Edge | Simulated data | Raspberry Pi + sensors |
| DevOps | GitHub | GitHub Actions CI, Docker |

---

## 5. Project Structure

```

algorand-ai-oracle/
├── data/                      # Datasets (synthetic + future collected)
├── models/
│   ├── property/              # Property-related models (current)
│   └── art/                   # Placeholder for future asset type
├── notebooks/                 # EDA, training, inference prototypes
├── scripts/
│   ├── feature\_pipeline.py    # (planned) Build feature frame per asset\_type
│   ├── model\_registry.py      # (planned) Paths & loading logic
│   ├── inference\_api.py       # (planned) FastAPI service
│   └── blockchain\_publish.py  # (planned) Algorand integration
├── frontend/                  # Angular app (planned)
├── schemas/
│   └── output\_example.json    # (planned) Unified output schema
├── README.md
└── requirements.txt

````

---

## 6. Dataset (Current + Future Adjustments)

**Current fields (property focus):**  
`asset_id, location, size_m2, rooms, bathrooms, year_built, floor, building_floors, has_elevator, has_garden, has_balcony, garage, energy_class, humidity_level, temperature_avg, noise_level, air_quality_index, valuation_k`

**Planned additions (multi-RWA readiness):**
- `asset_type` (string) e.g. `property`
- `condition_score` (float 0–1, placeholder initially)
- `risk_score` (float 0–1, placeholder initially)
- `last_verified_ts` (ISO timestamp)
- Asset-type specific placeholders left `NaN` until implemented

---

## 7. Unified Output Schema (Draft)

Example JSON returned by inference or published on-chain:

```json
{
  "asset_id": "asset_0102",
  "asset_type": "property",
  "timestamp": "2025-07-18T12:04:55Z",
  "metrics": {
    "valuation_base_k": 152.37,
    "condition_score": 0.82,
    "risk_score": 0.18
  },
  "flags": {
    "anomaly": false,
    "needs_review": false
  },
  "model_meta": {
    "value_model_version": "rf_v1",
    "anomaly_model_version": "iforest_v0"
  },
  "offchain_refs": {
    "detail_report_hash": null,
    "sensor_batch_hash": null
  }
}
````

For a future art asset:

```json
{
  "asset_id": "art_0045",
  "asset_type": "art",
  "timestamp": "2025-10-02T09:11:22Z",
  "metrics": {
    "valuation_base_k": 320.5,
    "authenticity_prob": 0.94,
    "condition_score": 0.71
  },
  "flags": {
    "anomaly": false,
    "needs_review": true
  },
  "model_meta": {
    "auth_model_version": "cnn_v2"
  }
}
```

---

## 8. Model Registry (Planned Structure)

```python
MODEL_REGISTRY = {
    "property": {
        "value_regressor": "models/property/regression_value_v1.pkl",
        "anomaly_model": "models/property/anomaly_iforest_v0.pkl"
    },
    "art": {
        "valuation_model": None,      # placeholder
        "authenticity_model": None
    }
}
```

A loader function will choose preprocessing + inference pipeline based on `asset_type`.

---

## 9. Workflow Summary

1. Generate / update synthetic dataset (per asset type)
2. Exploratory analysis (notebooks)
3. Train baseline models (per asset type folder)
4. Save models + encoders into structured directories
5. Inference:

   * Notebook (initial)
   * FastAPI endpoint: `POST /predict/{asset_type}`
6. Output validated against schema
7. Publish summary (compact JSON) to Algorand (Note field or ASA metadata)
8. (Future) Integrate sensor streams replacing simulated environment fields

---

## 10. Blockchain Publishing (Planned Minimal Payload)

On-chain payload kept small:

```json
{
  "a": "asset_0102",
  "t": "property",
  "ts": "2025-07-18T12:04:55Z",
  "v": 152.37,
  "c": 0.82,
  "r": 0.18,
  "an": 0
}
```

Extended report anchored off-chain (e.g., IPFS hash) → integrity assured via hash.

---

## 11. Getting Started

```bash
# Clone
git clone https://github.com/your-username/algorand-ai-oracle.git
cd algorand-ai-oracle

# Environment
conda create -n ai-oracle python=3.11 -y
conda activate ai-oracle

# Install (initial)
pip install pandas numpy scikit-learn joblib jupyter matplotlib seaborn algosdk fastapi uvicorn

# Run notebooks
jupyter notebook notebooks/
```

Run in order:

1. `01_generate_dataset.ipynb`
2. `02_explore_dataset.ipynb`
3. `03_train_model.ipynb`
4. `04_infer_single_sample.ipynb` (to be created)

---

## 12. Roadmap (Condensed)

| Phase | Focus                             | Key Multi-RWA Actions                   |
| ----- | --------------------------------- | --------------------------------------- |
| 1     | Property dataset + baseline model | Add `asset_type`, schema draft          |
| 2     | API + Algorand publishing         | Introduce registry + anomaly model      |
| 3     | Multi-asset scaffolding           | Add placeholder folders + config        |
| 4     | Sensor + hardware edge            | Real-time ingestion + on-chain triggers |
| 5     | New asset domain (e.g. art)       | New feature set + classification tasks  |

---

## 13. Planned Enhancements

* IsolationForest / One-Class SVM anomaly detection
* Condition & risk scoring calibration
* SHAP-based explanation & feature importance
* PyTEAL contract for update frequency enforcement
* Off-chain IPFS integration + on-chain hash anchoring
* DAO governance for anomaly dispute resolution
* Asset-type plug-ins (e.g. `plugins/art/feature_builder.py`)

---

## 14. License

MIT License (to be added in `LICENSE` file).

---

## 15. Disclaimer

All current data is synthetic. No real valuations or risk recommendations are to be considered authoritative. This project is an R\&D effort toward verifiable AI oracles for tokenized assets.

---

## 16. Contact

| Channel     | Link                                                                 |
| ----------- | -------------------------------------------------------------------- |
| X (Twitter) | WIP                                                                  |
| GitHub      | https://github.com/AnVenDev                                          |
| Email       | anvene.dev@gmail.com                                                 |

If you are building in the Algorand RWA ecosystem and want to collaborate, feel free to reach out.
