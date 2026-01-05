import os
from pathlib import Path

from sklearn.pipeline import make_pipeline

from .config import config
from .pipeline import pipeline
from .utils import feature_target_split


def train(data_path):
    x_train, y_train = feature_target_split(data_path)
    full_pipeline = make_pipeline(
        pipeline,
        config.BASELINE_MODEL
    )
    full_pipeline.fit(x_train, y_train)


def main():
    data_path = Path(config.DATA_PATH) / "prepared" / config.TRAIN_FILE
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "Train dataset not found. Please run ingest.py before training."
        )

    train(data_path)


if __name__ == "__main__":
    main()
