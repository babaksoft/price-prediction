import pandas as pd

from .config import config


def feature_target_split(csv_path):
    data = pd.read_csv(csv_path)
    x = data.drop(config.TARGET, axis=1)
    y = data[config.TARGET]
    return x,y
