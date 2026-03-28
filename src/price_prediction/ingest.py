import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

from .config import config


def fix_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    with mlflow.start_run(run_name="Drop duplicates") as run:
        mlflow.set_tag("run_id", run.info.run_id)
        dup_count = df.duplicated().sum()
        metrics = {
            "raw_size": len(df),
            "duplicate_count": dup_count,
            "duplicate_percent": round(100.0 * dup_count / len(df), 2),
            "cleaned_size": len(df) - dup_count,
        }
        mlflow.log_metrics(metrics)
        mlflow.end_run()

        return df.drop_duplicates()


def fix_target_noise(df: pd.DataFrame) -> pd.DataFrame:
    """
    To improve predictive signal, all data points with Price < 800 will be dropped.
    """

    min_price = 800
    with mlflow.start_run(run_name="Fix target noise") as run:
        mlflow.set_tag("run_id", run.info.run_id)
        df_clean = df.loc[df[config.TARGET] >= min_price, :]
        noise_count = len(df) - len(df_clean)
        metrics = {
            "min_target_value": min_price,
            "raw_size": len(df),
            "noisy_target_count": noise_count,
            "noisy_target_percent": round(100.0 * noise_count / len(df), 2),
            "cleaned_size": len(df_clean),
        }
        mlflow.log_metrics(metrics)
        mlflow.end_run()

        return df_clean


def fix_target_conflict(data: pd.DataFrame) -> pd.DataFrame:
    df: pd.DataFrame = data.copy()
    features = list(df.columns.drop(config.TARGET))

    # Step 1 : Find feature groups with >1 unique label
    grouped = df.groupby(features)[config.TARGET].nunique()
    conflicting_groups = grouped[grouped > 1]

    # Step 2 : Extract all conflicting rows
    conflict_keys = conflicting_groups.index
    df_conflict = pd.DataFrame(list(conflict_keys), columns=features)
    df.merge(df_conflict, on=features, how="inner")

    # Step 3 : Remove conflicting rows
    df_clean = df.merge(df_conflict, on=features, how="left", indicator=True)
    df_clean = df_clean[df_clean["_merge"] == "left_only"].drop(columns="_merge")

    with mlflow.start_run(run_name="Fix target conflict") as run:
        mlflow.set_tag("run_id", run.info.run_id)
        conflict_count = len(data) - len(df_clean)
        metrics = {
            "raw_size": len(data),
            "conflict_target_count": conflict_count,
            "conflict_target_percent": round(100.0 * conflict_count / len(data), 2),
            "cleaned_size": len(df_clean),
        }
        mlflow.log_metrics(metrics)
        mlflow.end_run()

    return df_clean


def ingest(raw_path, to_dir):
    """
    Data ingestion consists of the following steps:
    1. Drop ID column
    2. Remove duplicate rows
    3. Remove rows with noisy targets (Price < 800)
    4. Remove duplicate rows with conflicting target values
    5. Perform stratified splits using 5 equally sized bins
    6. Drop temporary bin column before saving splits
    """

    rs = config.RANDOM_STATE
    bin_label = "Price bin"
    bin_percentiles = [20, 40, 60, 80]

    # Prepare MLflow experiment
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Data Ingestion")

    df = pd.read_csv(raw_path)
    df = df.drop(["ID"], axis=1)
    df = fix_duplicates(df)
    df = fix_target_noise(df)
    df = fix_target_conflict(df)

    percentiles = np.percentile(df[config.TARGET], bin_percentiles)
    df[bin_label] = pd.cut(
        df[config.TARGET],
        bins=[
            0,
            percentiles[0],
            percentiles[1],
            percentiles[2],
            percentiles[3],
            np.inf,
        ],
        labels=[1, 2, 3, 4, 5],
    )

    df_train, df_test = train_test_split(
        df, test_size=config.TRAIN_TEST_SPLIT, stratify=df[bin_label], random_state=rs
    )
    df_train, df_val = train_test_split(
        df_train,
        test_size=config.TRAIN_VAL_SPLIT,
        stratify=df_train[bin_label],
        random_state=rs,
    )

    df_train = df_train.drop([bin_label], axis=1)
    df_val = df_val.drop([bin_label], axis=1)
    df_test = df_test.drop([bin_label], axis=1)

    df_train.to_csv(to_dir / config.TRAIN_FILE, header=True, index=False)
    df_val.to_csv(to_dir / config.VAL_FILE, header=True, index=False)
    df_test.to_csv(to_dir / config.TEST_FILE, header=True, index=False)

    with mlflow.start_run(run_name="Split data (stratified)") as run:
        mlflow.set_tag("run_id", run.info.run_id)
        split_params = {
            "dataset_size": len(df),
            "train_test_split": config.TRAIN_TEST_SPLIT,
            "train_val_split": config.TRAIN_VAL_SPLIT,
            "train_size": len(df_train),
            "val_size": len(df_val),
            "test_size": len(df_test),
            "random_state": config.RANDOM_STATE,
        }
        mlflow.log_metrics(split_params)
        mlflow.end_run()


def main():
    raw_path = Path(config.DATA_DIR) / "raw" / config.RAW_FILE
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            "Raw dataset not found. You may need to reinstall this package."
        )

    to_dir = Path(config.DATA_DIR) / "prepared"
    if not os.path.exists(to_dir):
        os.mkdir(to_dir)
    if (
        os.path.exists(to_dir / config.TRAIN_FILE)
        or os.path.exists(to_dir / config.VAL_FILE)
        or os.path.exists(to_dir / config.TEST_FILE)
    ):
        print("[INFO] Dataset is already ingested.")
        return

    ingest(raw_path, to_dir)
    print("[INFO] Raw dataset successfully ingested.")


if __name__ == "__main__":
    main()
