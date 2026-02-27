
import mlflow
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from ..config import config
from ..utils import evaluate_model


def evaluate_pipeline():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Baseline v1")

    lr_clf = Ridge(random_state=config.RANDOM_STATE)
    evaluate_model(lr_clf, run_name="Ridge")

    rf_clf = RandomForestRegressor(
        n_jobs=-1, random_state=config.RANDOM_STATE
    )
    evaluate_model(rf_clf, run_name="RF")


if __name__ == "__main__":
    evaluate_pipeline()
