# Algorand-Powered AI Oracle for (Multi) RWA

A modular AI + Blockchain oracle that evaluates and monitors Real World Assets (RWA), starting with real estate and built for extensibility to assets like art, logistics, agriculture, and renewable energy. It generates structured AI insights and publishes (select) summaries to the Algorand blockchain.

---

## Key Features (MVP v1)

- AI-powered valuation and metrics for real estate assets
- Unified schema with confidence intervals, drift & anomaly flags
- Model registry + metadata tracking + fallback-ready architecture
- REST API for prediction + monitoring
- Logs prediction metadata for auditability
- Mock-tested publisher for Algorand blockchain integration
- Full end-to-end sanity suite: schema, API, publish, logs
- Future: PyTEAL contracts, sensor integration, multi-asset support

---

## Purpose

This project establishes a reusable oracle layer that:

- **Estimates asset value** (starting with `property`)
- **Simulates condition & risk scoring** via interpretable logic
- **Detects anomalies** via rule-based thresholds (ML planned)
- **Computes prediction confidence intervals**
- **Tracks inference latency**, logs model metadata + flags
- **Detects drift** via z-score comparison to training stats
- **Publishes summaries** to Algorand (TestNet verified)
- **Designed for edge/sensor ingestion** (e.g. Raspberry Pi, IPFS hash)

---

## Multi-RWA Vision

Built for easy extension. Asset-specific models and pipelines plug into a shared system.

| Asset Type     | Features (examples)                        | Tasks                                    |
| -------------- | ------------------------------------------ | ---------------------------------------- |
| `property`     | size, rooms, humidity, energy_class        | valuation, anomaly, condition, risk      |
| `art`          | medium, year, artist reputation            | valuation, authenticity scoring          |
| `greenhouse`   | temperature, CO₂, light, humidity          | yield, crop risk, anomaly                |
| `warehouse`    | vibration, temperature variance, occupancy | compliance, risk, predictive maintenance |
| `energy_asset` | panel output, irradiance, degradation      | efficiency score, alerting               |
| `container`    | shocks, location, temp variation           | spoilage, transport risk                 |

---

## Architecture Principles

- **Pluggable asset_type** with specific models + schemas
- **Shared output schema**, extensible via `metrics`, `flags`, etc.
- **Model registry** with fallback support + metadata checks
- **FastAPI + JSONSchema** + LGBMRegressor (for MVP)
- **Separation** of training, prediction, logging, publishing

---

## Dataset Fields (MVP)

Core fields for `property`:

```text
location, size_m2, rooms, bathrooms, year_built, floor, building_floors,
has_elevator, has_garden, has_balcony, garage, energy_class,
humidity_level, temperature_avg, noise_level, air_quality_index
```

With derived fields:

```text
age_years, valuation_k, condition_score, risk_score
```

---

## Output Schema Example

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

## Implemented Logic

- **Prediction confidence intervals** via t-distribution (MC sim)
- **Simple anomaly detection** using field thresholds
- **Latency + model metadata logging**
- **Drift detection** based on z-score deviation
- **Robust training pipeline** with Optuna tuning
- **Batch + single prediction support**
- **Blockchain publishing tested via mocked `publish_ai_prediction`**
- **Real blockchain publishing TestNet ready** via `publish_ai_prediction` (one ASA) and `batch_publish_predictions` (multiple ASA)

---

## Test Suite

**Run all tests:**

```bash
scripts/test.bat
```

Includes:

| File                         | Purpose                                 |
| ---------------------------- | --------------------------------------- |
| `test_api.py`                | Unit test of FastAPI endpoints          |
| `test_blockchain_publish.py` | Publisher logic with mock responses     |
| `e2e_sanity_check.py`        | Full test: model load → API call → logs |

> ✔️ All tests passed with warnings resolved and schema strictness enforced.

---

## API Usage

Start local server:

```bash
scripts/start.bat
```

Example prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict/property   -H "Content-Type: application/json"   -d @schemas/sample_property.json
```

---

## Blockchain Payload (Compact)

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

Hash of full JSON report to be stored off-chain (`detail_report_hash`).

---

## Getting Started

```bash
git clone https://github.com/AnVenDev/ai-oracle-rwa.git
cd ai-oracle-rwa

conda create -n ai-oracle python=3.11 -y
conda activate ai-oracle
pip install -r requirements.txt
```

Run in order:

1. `01_generate_dataset.ipynb`
2. `02_explore_dataset.ipynb`
3. `03_train_model.ipynb`
4. `04_infer_single_sample.ipynb`

Start local server:

```bash
scripts/start.bat
```

Test everything:

```bash
scripts/test.bat
```

---

## Model Performance (v1)

- MAE: ~65k€
- RMSE: ~86k€
- R²: 0.55

_Realistic performance given multi-variable synthetic data._

---

## Roadmap (Partial)

- [x] Real estate asset valuation
- [x] Schema validation + metadata tracking
- [x] Logging + flagging
- [x] Blockchain publishing (mocked)
- [x] Real on-chain publishing
- [ ] CI/CD pipeline
- [ ] SHAP explainability
- [ ] PyTEAL contract
- [ ] Plugin support per asset type
- [ ] Sensor ingestion (e.g., Raspberry Pi)
- [ ] Model retraining & DVC support

---

## License

MIT License (to be added)

---

## Contact

| Channel | Link                                                       |
| ------- | ---------------------------------------------------------- |
| GitHub  | [https://github.com/AnVenDev](https://github.com/AnVenDev) |
| Email   | [anvene.dev@gmail.com](mailto:anvene.dev@gmail.com)        |
