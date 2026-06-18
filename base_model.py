import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, f1_score
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

def evaluate(model, X_test, y_test, threshold: float = 0.5) -> dict:
    proba = model.predict_proba(X_test)
    preds = (proba >= threshold).astype(int)

    metrics = {
        "roc_auc":          round(roc_auc_score(y_test, proba), 4),
        "pr_auc":           round(average_precision_score(y_test, proba), 4),
        "f1_fraud":         round(f1_score(y_test, preds, pos_label=1), 4),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "report":           classification_report(y_test, preds, output_dict=True),
    }

    logger.info(f"ROC-AUC: {metrics['roc_auc']} | PR-AUC: {metrics['pr_auc']} | F1-Fraud: {metrics['f1_fraud']}")
    return metrics
