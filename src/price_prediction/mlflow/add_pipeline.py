import os
import json

import pandas as pd
import mlflow
import joblib

from ..config import config
from ..pipeline import build_pipeline


def set_version_tags():
    tags = {
        "stage": config.PIPELINE_STAGE,
        "pipeline_version": config.PIPELINE_VERSION,
        "feature_schema_version": config.FEATURE_SCHEMA_VERSION,
        "data_version": config.DATA_VERSION,
        "data_commit_hash": config.DATA_COMMIT_HASH,
        "code_commit_hash": config.CODE_COMMIT_HASH,
        "target_transform": config.TARGET_TRANSFORM,
        "mileage_transform": config.MILEAGE_TRANSFORM,
        "levy_transform": config.LEVY_TRANSFORM,
        "engine_volume_split": config.ENGINE_VOLUME_SPLIT,
        "status": config.PIPELINE_STATUS,
    }
    mlflow.set_tags(tags)


def set_version_params():
    df = pd.read_csv(config.DATA_DIR / "raw" / config.RAW_FILE)
    df_train = pd.read_csv(config.DATA_DIR / "prepared" / config.TRAIN_FILE)
    df_val = pd.read_csv(config.DATA_DIR / "prepared" / config.VAL_FILE)
    df_test = pd.read_csv(config.DATA_DIR / "prepared" / config.TEST_FILE)
    params = {
        "raw_size": len(df),
        "cleaned_size": len(df_train) + len(df_val) + len(df_test),
        "train_size": len(df_train),
        "val_size": len(df_val),
        "test_size": len(df_test),
    }
    mlflow.log_params(params)


def set_version_artifacts(pipeline):
    root_dir = config.ARTIFACTS_DIR / f"pipeline_{config.PIPELINE_VERSION}"
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    path = root_dir / f"pipeline_{config.PIPELINE_VERSION}.joblib"
    joblib.dump(pipeline, path)
    mlflow.log_artifact(path)

    path = root_dir / f"features_{config.PIPELINE_VERSION}.txt"
    features = list(pipeline.get_feature_names_out())
    with open(path, "w") as file:
        file.write("\n".join(features))
    mlflow.log_artifact(path)

    path = root_dir / f"features_{config.PIPELINE_VERSION}.json"
    with open(path, "w") as file:
        json.dump(features, file)
    mlflow.log_artifact(path)

    path = root_dir / f"pipeline_{config.PIPELINE_VERSION}.md"  # Manually created
    mlflow.log_artifact(path)


def mlflow_register():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name="Preprocessing")
    with mlflow.start_run(run_name=f"Pipeline {config.PIPELINE_VERSION}") as run:
        mlflow.set_tag("run_id", run.info.run_id)
        set_version_tags()
        set_version_params()

        train_path = config.DATA_DIR / "prepared" / config.TRAIN_FILE
        df_train = pd.read_csv(train_path)
        x_train = df_train.drop(config.TARGET, axis=1)
        y_train = df_train[config.TARGET]
        pipeline = build_pipeline().fit(x_train, y_train)
        set_version_artifacts(pipeline)


if __name__ == "__main__":
    mlflow_register()
