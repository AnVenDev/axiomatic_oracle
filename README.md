# Algorand-Powered AI Oracle for Real-World Assets (Multi-RWA Ready)

A modular AI + Blockchain oracle that evaluates and monitors **Real World Assets (RWA)** beginning with real estate and designed from day one to extend to **art, logistics, agriculture, renewable energy, and industrial assets**.  
It produces structured AI insights and publishes verifiable summaries (planned) to the Algorand blockchain.

---

## 1. Purpose

This project establishes a reusable oracle layer that:

- Estimates asset value (initial focus: property valuation)
- Assesses condition / risk via environmental & structural indicators (simulated; will evolve)
- Detects anomalies (planned) using statistical + ML methods
- Publishes compact, verifiable summaries to Algorand (TestNet → MainNet)
- Prepares a sensor / edge ingestion path (Raspberry Pi, IoT)
- Supports *multiple asset categories* via a unified schema, model registry, and pluggable preprocessing

---

## 2. Multi-RWA Vision

While Phase 1 focuses on **property (asset_type = "property")**, the architecture anticipates additional asset classes:

| Asset Type (future) | Example Features | Primary Tasks |
|---------------------|------------------|---------------|
| `property` (current) | size, rooms, humidity, energy_class | valuation, condition, anomaly |
| `art` | medium, year_created, artist_reputation, storage_humidity | authenticity prob., condition, valuation |
| `greenhouse` | temperature, humidity, light, CO₂, soil_moisture | crop risk, yield score, anomaly |
| `warehouse` | vibration, temp stability, occupancy pattern | integrity, compliance, risk |
| `energy_asset` | panel_efficiency, irradiance, degradation_rate | performance index, maintenance trigger |
| `container` | geo_path, temperature_mean, shock_events | spoilage risk, compliance, anomaly |

**Design Principles for Extensibility:**

- Mandatory `asset_type` field on every record
- Unified output schema with flexible `metrics` and `flags`
- Model registry keyed by asset type (e.g. `models/property/...`, `models/art/...`)
- Pluggable feature pipeline per asset type (config-driven)
- Separation of **data generation / preprocessing / inference / on-chain publishing**

---

## 3. Key Features (MVP Scope)

| Feature | Status | Notes |
|---------|--------|-------|
| Synthetic property dataset (150+ rows) | ✅ | Includes `asset_type`, condition/risk scores |
| EDA (distribution, correlation, condition/risk) | ✅ | Notebook 02 |
| Unified training pipeline (OHE + RF) | ✅ | Notebook 03 (saved as joblib) |
| Model metadata JSON (versioned) | ✅ | Includes features & performance |
| Inference notebook (single + batch + JSON schema) | ✅ | Notebook 04 |
| Unified output schema (draft) | ✅ | Implemented in inference; finalization pending |
| Anomaly detection module | Planned | IsolationForest / rules (Notebook 05) |
| Condition & risk refinement | Planned | Better linkage to valuation |
| FastAPI inference API | Planned | `/predict/{asset_type}` |
| Algorand publishing (Note field) | Planned | Compact payload → TX ID |
| Model registry abstraction | Planned | `scripts/model_registry.py` |
| Angular dashboard (multi-asset) | Planned | Filtering + detail + TX links |
| Sensor ingestion (simulated) | Future | JSON → API endpoint |
| Raspberry Pi edge device | Future | Phase 3–4 |
| PyTEAL smart contract hooks | Future | Update frequency / anomaly triggers |

---

## 4. Tech Stack

| Layer | Current | Future / Optional |
|-------|--------|--------------------|
| ML / Data | Python, Pandas, NumPy, Scikit-learn | XGBoost, LightGBM, PyTorch, SHAP |
| Backend | (Not yet) FastAPI planned | Auth (JWT), rate limiting |
| Frontend | (Not yet) Angular 17 planned | Map view, role-based access |
| Blockchain | Algorand Python SDK (planned) | PyTEAL, ASA metadata, AppCall |
| Storage | CSV (synthetic), `models/` pipeline & metadata | SQLite → PostgreSQL, IPFS hashes |
| Orchestration | Jupyter notebooks | DVC, MLflow |
| Edge | Simulated | Raspberry Pi + sensors |
| DevOps | Git + GitHub | GitHub Actions CI, Docker |

---

## 5. Project Structure

```

algorand-ai-oracle/
├── data/                       # Datasets + prediction logs
├── models/
│   └── property/
│       ├── value\_regressor\_v1.joblib
│       └── value\_regressor\_v1\_meta.json
├── notebooks/
│   ├── 01\_generate\_dataset.ipynb
│   ├── 02\_explore\_dataset.ipynb
│   ├── 03\_train\_model.ipynb
│   └── 04\_infer\_single\_sample.ipynb
├── scripts/
│   ├── feature\_pipeline.py      # (planned)
│   ├── model\_registry.py        # (planned)
│   ├── inference\_api.py         # (planned)
│   └── blockchain\_publish.py    # (planned)
├── frontend/                    # (planned) Angular app
├── schemas/
│   └── output\_example.json      # (planned) Finalized schema
├── README.md
└── requirements.txt

````

---

## 6. Dataset (Current + Future Adjustments)

**Current fields (property focus):**  
`asset_id, asset_type, location, size_m2, rooms, bathrooms, year_built, floor, building_floors, has_elevator, has_garden, has_balcony, garage, energy_class, humidity_level, temperature_avg, noise_level, air_quality_index, valuation_k, condition_score, risk_score, last_verified_ts, age_years`

**Future additions (per asset type):**
- `authenticity_prob` (art)
- `panel_efficiency`, `irradiance` (energy_asset)
- `shock_events`, `temperature_var` (container)
- `soil_moisture`, `light_index` (greenhouse)
- Placeholders stored as `NaN` until populated

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

Future art example:

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
        "value_regressor": "models/property/value_regressor_v1.joblib",
        "anomaly_model": "models/property/anomaly_iforest_v0.joblib"
    },
    "art": {
        "valuation_model": None,
        "authenticity_model": None
    }
}
```

A loader function will dispatch preprocessing + inference based on `asset_type`.

---

## 9. Workflow Summary

1. Generate / update synthetic dataset (per asset type)
2. Exploratory analysis (Notebook 02)
3. Train baseline pipeline (Notebook 03) → save joblib + metadata
4. Inference:

   * Notebook 04 (single & batch)
   * (Planned) FastAPI endpoint: `POST /predict/{asset_type}`
5. Validate output against unified schema
6. Publish summary (planned) to Algorand (Note field / ASA metadata)
7. (Future) Integrate sensor streams for environmental updates

---

## 10. Blockchain Publishing (Planned Minimal Payload)

Compact on-chain payload (Note field):

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

Extended / verbose report anchored off-chain (IPFS / Arweave) via hash reference.

---

## 11. Getting Started

```bash
# Clone
git clone https://github.com/your-username/algorand-ai-oracle.git
cd algorand-ai-oracle

# Environment
conda create -n ai-oracle python=3.11 -y
conda activate ai-oracle

# Install (initial minimal set)
pip install pandas numpy scikit-learn joblib jupyter matplotlib seaborn

# (Optional) For future steps
pip install fastapi uvicorn algosdk
```

Run notebooks in order:

1. `01_generate_dataset.ipynb`
2. `02_explore_dataset.ipynb`
3. `03_train_model.ipynb`
4. `04_infer_single_sample.ipynb`

---

## 12. Roadmap (Condensed)

| Phase | Focus                             | Key Multi-RWA Actions           |
| ----- | --------------------------------- | ------------------------------- |
| 1     | Property dataset + baseline model | `asset_type`, schema draft      |
| 2     | API + Algorand publishing         | Registry + anomaly model        |
| 3     | Multi-asset scaffolding           | New asset folders + configs     |
| 4     | Sensor + hardware edge            | Real-time ingestion + triggers  |
| 5     | New asset domain (e.g. art)       | Authenticity & condition models |

---

## 13. Planned Enhancements

* IsolationForest / One-Class SVM anomaly detection
* Condition & risk score linkage to valuation logic
* SHAP-based explanation & feature importance panel
* PyTEAL smart contract enforcing update windows
* Off-chain IPFS integration + on-chain hash anchoring
* DAO governance / dispute resolution module
* Asset-type plug-ins (e.g. `plugins/art/feature_builder.py`)

---

## 14. License

MIT License (to be added in `LICENSE`).

---

## 15. Disclaimer

All current data is synthetic. No real valuations or risk recommendations are authoritative. This is an R\&D project toward verifiable AI oracles for tokenized assets on Algorand.

---

## 16. Contact

| Channel     | Link                                                       |
| ----------- | ---------------------------------------------------------- |
| X (Twitter) | WIP                                                        |
| GitHub      | [https://github.com/AnVenDev](https://github.com/AnVenDev) |
| Email       | [anvene.dev@gmail.com](mailto:anvene.dev@gmail.com)        |

If you are building in the Algorand RWA ecosystem and want to collaborate, feel free to reach out.
