"""
Script train model. Chạy một lần trước khi start server.
  python train.py --model rf      # RandomForest (mặc định)
  python train.py --model xgb     # XGBoost
"""
import argparse
import joblib
import os
from src.data_processing.loader import load_data, split_data
from src.data_processing.preprocessor import scale_features, apply_smote
from src.models.random_forest import FraudRandomForest
from src.models.xgboost_model import FraudXGBoost
from src.models.evaluator import evaluate
from src.utils.config import MODEL_PATH
from src.utils.logger import get_logger

logger = get_logger("train")

def main(model_type: str):
    # 1. Load & split
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # 2. Scale
    X_train, X_test, scaler = scale_features(X_train, X_test)

    # 3. SMOTE (chỉ apply trên train)
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # 4. Train
    if model_type == "xgb":
        model = FraudXGBoost()
    else:
        model = FraudRandomForest()

    model.train(X_train_res, y_train_res)

    # 5. Evaluate
    metrics = evaluate(model, X_test, y_test)
    logger.info(f"Metrics: {metrics}")

    # 6. Save model + scaler
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    scaler_path = MODEL_PATH.replace(".pkl", "_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved → {scaler_path}")
    logger.info("✅ Done! Giờ chạy: uvicorn src.api.main:app --reload")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rf", "xgb"], default="rf")
    args = parser.parse_args()
    main(args.model)
