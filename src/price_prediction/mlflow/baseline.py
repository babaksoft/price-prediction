
import mlflow
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from ..config import config
from ..utils import evaluate_model


def evaluate_baseline():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Baseline")

    l2_reg = Ridge(random_state=config.RANDOM_STATE)
    evaluate_model("Ridge", l2_reg)

    rf_reg = RandomForestRegressor(
        n_jobs=-1, random_state=config.RANDOM_STATE
    )
    evaluate_model("RF", rf_reg)


if __name__ == "__main__":
    evaluate_baseline()
