import os
from typing import Any
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from .config import config
from .pipeline import build_pipeline


def get_data(split_name: str = "train"):
    files = {
        "train": config.TRAIN_FILE,
        "validation": config.VAL_FILE,
        "test": config.TEST_FILE
    }
    name = split_name.lower()
    if name in files:
        file = files[name]
    else:
        file = config.TRAIN_FILE

    path = Path(config.DATA_DIR) / "prepared" / file
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Dataset not found. Please run ingest.py first.")

    df = pd.read_csv(path)
    x = df.drop(config.TARGET, axis=1)
    y = np.log(df[config.TARGET])
    return x, y


def evaluate_model(run_name, model, params: dict[str, Any] = None):
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("run_id", run.info.run_id)

        transformer = build_pipeline()
        pipeline = Pipeline([
            ("transformer", transformer),
            ("estimator", model)
        ])

        x, y = get_data()
        cv = KFold(
            n_splits=10, shuffle=True,
            random_state=config.RANDOM_STATE
        )

        start = datetime.now()
        scoring = "neg_mean_absolute_error"
        scores = -cross_val_score(
            pipeline, x, y, scoring=scoring, cv=cv, n_jobs=-1)
        end = datetime.now()

        metrics = {
            "cv_score_mean": round(scores.mean(), 4),
            "cv_score_std": round(scores.std(), 4)
        }

        cv_params = {
            "model_type": type(model).__name__
        }
        if params:
            cv_params.update({
                f"model_{key}": value for key, value in params.items()
            })
        cv_params.update({
            "cv_scoring": scoring,
            "cv_splits": 10,
            "cv_shuffle": True,
            "cv_random_state": config.RANDOM_STATE,
            "cv_duration": str(end - start)
        })

        mlflow.log_metrics(metrics)
        mlflow.log_params(cv_params)
        mlflow.end_run()
        return metrics


def feature_target_split(csv_path):
    data = pd.read_csv(csv_path)
    x = data.drop(config.TARGET, axis=1)
    y = data[config.TARGET]
    return x,y
