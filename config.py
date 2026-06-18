import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.config import DATA_PATH, TEST_SIZE, RANDOM_STATE
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df

def split_data(df: pd.DataFrame):
    X = df.drop(columns=["Class"])
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,} | Fraud rate: {y.mean():.4%}")
    return X_train, X_test, y_train, y_test
