from pathlib import Path

from sklearn.dummy import DummyRegressor


# Global config
PROJECT_NAME = "price-prediction"
MLFLOW_TRACKING_URI = "http://localhost:5000"
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
RANDOM_STATE = 147

# Path config
ARTIFACTS_DIR = PACKAGE_ROOT / "artifacts"
DATA_DIR = PACKAGE_ROOT / "data"
MODEL_DIR = PACKAGE_ROOT / "model"
METRICS_DIR = PACKAGE_ROOT / "metrics"

# Data ingestion config
RAW_FILE = "car_price_prediction.csv"
TRAIN_FILE = "train.csv"
VAL_FILE = "validation.csv"
TEST_FILE = "test.csv"
TRAIN_TEST_SPLIT = 0.2
TRAIN_VAL_SPLIT = 0.2
TARGET = "Price"

# Data pipeline config
NUM_FEATURES = ["Prod. year", "Cylinders", "Airbags"]
CAT_FEATURES = [
    "Doors", "Drive wheels", "Gear box type",
    "Category", "Color", "Fuel type"
]
BIN_FEATURES = ["Leather interior", "Wheel"]
ENC_FEATURES = [
    "Levy", "Manufacturer", "Engine volume", "Mileage"
] # Need custom encoding

# Pipeline versioning config
PIPELINE_STAGE = "preprocessing"
PIPELINE_VERSION = "v1"
FEATURE_SCHEMA_VERSION = "v1"
DATA_VERSION = "dvc_v2"
DATA_COMMIT_HASH = "b4b6bb2b"
CODE_COMMIT_HASH = "692fb90a"
TARGET_TRANSFORM = "log"
MILEAGE_TRANSFORM = "log1p"
LEVY_TRANSFORM = "none"
ENGINE_VOLUME_SPLIT = "yes"
PIPELINE_STATUS = "locked"

# Temp variable used in train.py --> Adapt to ML project
BASELINE_MODEL = DummyRegressor(strategy="median")
