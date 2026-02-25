from pathlib import Path

from sklearn.dummy import DummyRegressor


# Global config
PROJECT_NAME = "price-prediction"
MLFLOW_TRACKING_URI = "http://localhost:5000"
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
RANDOM_STATE = 147

# Path config
DATA_PATH = PACKAGE_ROOT / "data"
MODEL_PATH = PACKAGE_ROOT / "model"
METRICS_PATH = PACKAGE_ROOT / "metrics"

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

# Temp variable used in train.py --> Adapt to ML project
BASELINE_MODEL = DummyRegressor(strategy="median")
