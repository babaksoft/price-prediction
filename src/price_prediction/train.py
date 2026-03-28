import os
from pathlib import Path

from .config import config


def train(data_path):
    _ = data_path
    pass


def main():
    data_path = Path(config.DATA_DIR) / "prepared" / config.TRAIN_FILE
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "Train dataset not found. Please run ingest.py before training."
        )

    train(data_path)


if __name__ == "__main__":
    main()
