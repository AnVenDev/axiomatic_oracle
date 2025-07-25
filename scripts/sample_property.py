sample_response = {
    "asset_id": "property_001",
    "asset_type": "property",
    "timestamp": "2025-07-22T18:49:57Z",
    "schema_version": "v1",
    "metrics": {"valuation_base_k": 6.011},
    "model_meta": {
        "value_model_version": "v1",
        "value_model_name": "LGBMRegressor",
        "model_hash": "fakehashpropertymodel123",
    },
}

multiple_samples = [
    {
        "asset_id": f"property_{i:03}",
        "asset_type": "property",
        "timestamp": "2025-07-22T18:{49+i%10}:57Z",
        "schema_version": "v1",
        "metrics": {"valuation_base_k": 5.5 + i * 0.1},
        "model_meta": {
            "value_model_version": "v1",
            "value_model_name": "LGBMRegressor",
            "model_hash": f"fakehash_{i}",
        },
    }
    for i in range(5)
]
