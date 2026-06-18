from xgboost import XGBClassifier
from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

class FraudXGBoost(BaseModel):
    def __init__(self):
        super().__init__("XGBoost")
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=577,   # ~ratio normal/fraud trong Kaggle dataset
            use_label_encoder=False,
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1
        )

    def train(self, X_train, y_train):
        logger.info(f"Training {self.name}...")
        self.model.fit(X_train, y_train, verbose=False)
        logger.info("Training complete")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
