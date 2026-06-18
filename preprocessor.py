from abc import ABC, abstractmethod
import joblib
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

class BaseModel(ABC):
    def __init__(self, name: str):
        self.name = name
        self.model = None

    @abstractmethod
    def train(self, X_train, y_train): ...

    @abstractmethod
    def predict(self, X): ...

    @abstractmethod
    def predict_proba(self, X): ...

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Model saved → {path}")

    def load(self, path: str):
        self.model = joblib.load(path)
        logger.info(f"Model loaded ← {path}")
        return self
