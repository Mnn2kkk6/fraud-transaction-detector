from sklearn.ensemble import RandomForestClassifier
from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

class FraudRandomForest(BaseModel):
    def __init__(self, n_estimators=100, max_depth=10):
        super().__init__("RandomForest")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )

    def train(self, X_train, y_train):
        logger.info(f"Training {self.name}...")
        self.model.fit(X_train, y_train)
        logger.info("Training complete")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
