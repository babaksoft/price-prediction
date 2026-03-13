
import mlflow
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from ..config import config
from ..utils import evaluate_model


def evaluate_baseline():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Baseline")

    params = { "random_state": config.RANDOM_STATE }
    l2_reg = Ridge(**params)
    evaluate_model("Ridge", l2_reg, params=params)

    rf_reg = RandomForestRegressor(**params)
    evaluate_model("RF", rf_reg, params=params)


if __name__ == "__main__":
    evaluate_baseline()
