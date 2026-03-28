import os
from pathlib import Path

from .config import config


def evaluate(model_path, data_path):
    _ = model_path
    _ = data_path


def main():
    data_path = Path(config.DATA_DIR) / "prepared" / config.TEST_FILE
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "Test dataset not found. Please run ingest.py before evaluating."
        )
    model_path = Path(config.MODEL_DIR) / "model.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Trained model not found. Please run train.py before evaluating."
        )

    evaluate(model_path, data_path)


if __name__ == "__main__":
    main()
