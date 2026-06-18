"""
Chạy: pytest tests/ -v
"""
import pytest
import pandas as pd
import numpy as np
from src.api.schemas import TransactionInput, PredictionResult
from src.data_processing.feature_engineer import add_features
from src.data_processing.preprocessor import scale_features

SAMPLE = {
    "Time": 0.0, "Amount": 149.62,
    **{f"V{i}": round(np.random.uniform(-3, 3), 4) for i in range(1, 29)}
}

# ── Schema tests ──────────────────────────────────────────
def test_transaction_input_valid():
    t = TransactionInput(**SAMPLE)
    assert t.Amount == 149.62
    assert t.Time == 0.0

def test_transaction_input_missing_field():
    bad = {k: v for k, v in SAMPLE.items() if k != "V1"}
    with pytest.raises(Exception):
        TransactionInput(**bad)

def test_prediction_result_low():
    r = PredictionResult(is_fraud=False, fraud_probability=0.05,
                         risk_level="LOW", model_used="RandomForest")
    assert not r.is_fraud
    assert r.risk_level == "LOW"

def test_prediction_result_high():
    r = PredictionResult(is_fraud=True, fraud_probability=0.91,
                         risk_level="HIGH", model_used="XGBoost")
    assert r.is_fraud
    assert r.fraud_probability > 0.5

# ── Feature engineering tests ─────────────────────────────
def test_add_features_columns():
    df = pd.DataFrame([{**SAMPLE, "Class": 0}])
    out = add_features(df)
    for col in ["log_amount", "is_tiny_amount", "is_large_amount",
                 "v1_v2_interact", "time_period"]:
        assert col in out.columns, f"Missing column: {col}"

def test_log_amount_positive():
    df = pd.DataFrame([{**SAMPLE, "Amount": 100.0, "Class": 0}])
    out = add_features(df)
    assert out["log_amount"].iloc[0] == pytest.approx(np.log1p(100.0))

def test_tiny_amount_flag():
    df = pd.DataFrame([{**SAMPLE, "Amount": 0.5, "Class": 0}])
    out = add_features(df)
    assert out["is_tiny_amount"].iloc[0] == 1

# ── Preprocessor tests ────────────────────────────────────
def test_scale_features_shape():
    import pandas as pd
    cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
    X = pd.DataFrame(np.random.randn(100, 30), columns=cols)
    X_train, X_test, scaler = scale_features(X[:80], X[80:])
    assert X_train.shape == (80, 30)
    assert X_test.shape  == (20, 30)
