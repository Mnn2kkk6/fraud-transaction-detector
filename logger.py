import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from src.utils.logger import get_logger

logger = get_logger(__name__)

def scale_features(X_train, X_test):
    """Scale Amount và Time, các cột V1-V28 đã được PCA."""
    scaler = StandardScaler()
    cols_to_scale = ["Amount", "Time"]

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    logger.info("Scaled Amount and Time columns")
    return X_train, X_test, scaler

def apply_smote(X_train, y_train):
    """Xử lý imbalanced data bằng SMOTE."""
    logger.info(f"Before SMOTE — Fraud: {y_train.sum()} / Normal: {(y_train==0).sum()}")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE  — Fraud: {y_res.sum()} / Normal: {(y_res==0).sum()}")
    return X_res, y_res
