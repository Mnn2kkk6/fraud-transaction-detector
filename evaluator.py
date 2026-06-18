import numpy as np
import pandas as pd
import joblib
from src.api.schemas import TransactionInput, PredictionResult
from src.utils.config import MODEL_PATH, THRESHOLD
from src.utils.logger import get_logger

logger = get_logger(__name__)

_model = None
_scaler = None

def load_model():
    global _model, _scaler
    _model = joblib.load(MODEL_PATH)
    _scaler = joblib.load(MODEL_PATH.replace(".pkl", "_scaler.pkl"))
    logger.info("Model & scaler loaded into memory")

def get_risk_level(prob: float) -> str:
    if prob < 0.3:
        return "LOW"
    elif prob < 0.6:
        return "MEDIUM"
    return "HIGH"

def predict_transaction(data: TransactionInput) -> PredictionResult:
    if _model is None:
        load_model()

    df = pd.DataFrame([data.dict()])
    df[["Amount", "Time"]] = _scaler.transform(df[["Amount", "Time"]])

    prob = float(_model.predict_proba(df)[:, 1][0])
    is_fraud = prob >= THRESHOLD

    return PredictionResult(
        is_fraud=is_fraud,
        fraud_probability=round(prob, 4),
        risk_level=get_risk_level(prob),
        model_used=type(_model).__name__
    )
