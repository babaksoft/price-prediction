import os
from pathlib import Path

import joblib
import pandas as pd

from .config import config


def make_prediction(input_data):
    model_path = Path(config.MODEL_DIR) / "model.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Trained model not found. Please run train.py before predicting."
        )
    model = joblib.load(model_path)
    data = pd.DataFrame(input_data)

    prediction = model.predict(data)
    result = {"prediction": prediction}
    return result


def predict(data_path):
    df_test = pd.read_csv(data_path)
    input_data = df_test[0:1]
    return make_prediction(input_data)


def main():
    data_path = Path(config.DATA_DIR) / "prepared" / config.TEST_FILE
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "Test dataset not found. Please run ingest.py before predicting."
        )

    prediction = predict(data_path)
    print(prediction)


if __name__ == "__main__":
    main()
