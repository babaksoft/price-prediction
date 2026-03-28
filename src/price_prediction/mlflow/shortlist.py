import mlflow
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm

from ..config import config
from ..utils import evaluate_model


def train_candidate_models():
    rs = config.RANDOM_STATE
    agg_metrics = []
    model_names = ["RF", "HGB", "XGB", "LGBM", "KNN"]
    models = [
        RandomForestRegressor(random_state=rs),
        HistGradientBoostingRegressor(random_state=rs),
        XGBRegressor(random_state=rs),
        LGBMRegressor(random_state=rs),
        KNeighborsRegressor(),
    ]

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment("Model Shortlisting")
    for name, model in tqdm(zip(model_names, models)):
        print(f"\nEvaluating '{name}' model...")
        params = {"random_state": rs} if name != "KNN" else None
        metrics = evaluate_model(name, model, params)
        agg_metrics.append(pd.Series(metrics, name=name))

    path = config.METRICS_DIR / "sl_metrics.csv"
    df_metrics = pd.DataFrame(agg_metrics, index=model_names)
    df_metrics = df_metrics.reset_index().rename(columns={"index": "model"})
    df_metrics.to_csv(path, index=False, header=True)

    with mlflow.start_run(
        run_name="Performance", experiment_id=experiment.experiment_id
    ) as run:
        mlflow.set_tag("run_id", run.info.run_id)
        mlflow.log_artifact(path)
        mlflow.end_run()


if __name__ == "__main__":
    train_candidate_models()
