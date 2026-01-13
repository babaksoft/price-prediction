import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

from .config import config


def ingest(raw_path, to_dir):
    rs = config.RANDOM_STATE
    stratify = "Price bin"
    bin_percentiles = [50, 90]

    data = pd.read_csv(raw_path)
    percentiles = np.percentile(data[config.TARGET], bin_percentiles)
    data[stratify] = pd.cut(
        data[config.TARGET],
        bins=[0, percentiles[0], percentiles[1], np.inf],
        labels=[1, 2, 3])
    df_train, df_test = train_test_split(
        data, test_size=config.TEST_SPLIT,
        stratify=data[stratify], random_state=rs)
    df_train, df_val = train_test_split(
        df_train, test_size=config.VAL_SPLIT,
        stratify=df_train[stratify], random_state=rs)

    df_train = df_train.drop([stratify], axis=1)
    df_val = df_val.drop([stratify], axis=1)
    df_test = df_test.drop([stratify], axis=1)

    df_train.to_csv(to_dir / config.TRAIN_FILE, header=True, index=False)
    df_val.to_csv(to_dir / config.VAL_FILE, header=True, index=False)
    df_test.to_csv(to_dir / config.TEST_FILE, header=True, index=False)

    # Prepare MLflow experiment
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Data Ingestion")
    with mlflow.start_run(run_name="main"):
        split_params = {
        "random_state": config.RANDOM_STATE,
        "price_bin_percentiles": bin_percentiles,
        "train_test_split": config.TEST_SPLIT,
        "train_val_split": config.VAL_SPLIT
        }
        mlflow.log_params(split_params)

def main():
    raw_path = Path(config.DATA_PATH) / "raw" / config.RAW_FILE
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            "Raw dataset not found. You may need to reinstall this package."
        )

    to_dir = Path(config.DATA_PATH) / "prepared"
    if not os.path.exists(to_dir):
        os.mkdir(to_dir)
    if os.path.exists(to_dir / config.TRAIN_FILE) or \
        os.path.exists(to_dir / config.VAL_FILE) or \
        os.path.exists(to_dir / config.TEST_FILE):
        print("[INFO] Dataset is already ingested.")
        return

    ingest(raw_path, to_dir)
    print("[INFO] Raw dataset successfully ingested.")


if __name__ == '__main__':
    main()
