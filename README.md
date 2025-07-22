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

## Key Features (Current vs Planned)

| Feature                                         | Status    | Notes                                                                 |
| ----------------------------------------------- | --------- | --------------------------------------------------------------------- |
| Synthetic property dataset (150+ rows)          | ✅         | Includes asset\_type, derived age\_years, condition/risk placeholders |
| EDA (distribution, correlation, condition/risk) | ✅         | Notebook 02                                                           |
| Training pipeline (OHE + RandomForest)          | ✅         | Notebook 03 (saved as joblib)                                         |
| Cross-validation + metrics export               | ✅         | Saved in meta.json (train vs CV MAE)                                  |
| Model metadata JSON (versioned)                 | ✅         | Contains metrics, feature list, timestamp                             |
| Inference notebook (single + batch)             | ✅         | Notebook 04: validates schema, logs predictions                       |
| Unified output JSON schema                      | ✅         | `schemas/output_example.json`                                         |
| FastAPI inference API                           | ✅         | `/predict/property` endpoint, validates input + schema                |
| Model registry abstraction                      | ✅         | `scripts/model_registry.py` with fallback support                     |
| E2E sanity check script                         | ✅         | `scripts/e2e_sanity_check.py`                                         |
| Basic anomaly detection (rules)                 | ✅         | In `04_infer_single_sample` and future script                         |
| Condition & risk refinement logic               | ⏳ Planned | Rule/ML hybrid scoring                                                |
| Confidence interval on prediction               | ✅         | Bootstrap ensemble simulated                                          |
| Algorand publishing (Note field)                | ⏳ Planned | Compact payload + TX ID                                               |
| Angular dashboard (multi-asset)                 | ⏳ Planned | Filtering + detail + TX links                                         |
| Sensor ingestion (simulated stream)             | 🔮 Future | Real-time updates                                                     |
| Raspberry Pi edge device integration            | 🔮 Future | Phase 3–4                                                             |
| PyTEAL smart contract hooks                     | 🔮 Future | Update frequency / anomaly triggers                                   |

## Runtime Utilities

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

## Enhancements Ready Post-MVP

* Confidence intervals (via bootstrap prediction)
* Rules-based anomaly detection
* Inference latency monitoring
* Fallback model loading via `model_registry`
* Drift detection logic placeholder
* JSON schema validation in inference
* Batch inference support + logging to `.jsonl`
* Modular validation pipeline in progress

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
