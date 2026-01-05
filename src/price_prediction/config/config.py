from pathlib import Path

from sklearn.dummy import DummyRegressor


RANDOM_STATE = 147

PROJECT_NAME = "price-prediction"
PACKAGE_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PACKAGE_ROOT / "data"
MODEL_PATH = PACKAGE_ROOT / "model"
METRICS_PATH = PACKAGE_ROOT / "metrics"

RAW_FILE = "" # Name of raw CSV dataset
TRAIN_FILE = "train.csv"
VALIDATION_FILE = "validation.csv"
TEST_FILE = "test.csv"

TARGET = "" # Used for supervised learning tasks

NUMERICAL_FEATURES = []

CATEGORICAL_FEATURES = [] # No implied ordering

ORDINAL_FEATURES = [] # Implied ordering (e.g., High, Medium, Low)

BINARY_FEATURES = [] # Binary (0/1) features

# Temp variable used in train.py --> Adapt to ML project
BASELINE_MODEL = DummyRegressor(strategy="median")
