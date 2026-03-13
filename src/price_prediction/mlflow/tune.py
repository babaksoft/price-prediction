from datetime import datetime

import numpy as np
import mlflow
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestRegressor, HistGradientBoostingRegressor)

from ..config import config
from ..pipeline import build_pipeline
from ..utils import get_data


def tune_model(name, model, x, y, param_grid):
    rs = config.RANDOM_STATE
    scoring = "neg_mean_absolute_error"
    params = {
        "model_type": type(model).__name__,
        "model_random_state": model.random_state,
        "param_grid": param_grid,
        "scoring": scoring
    }

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tag("run_id", run.info.run_id)
        transform = build_pipeline()
        pipeline = Pipeline([
            ("transformer", transform),
            ("estimator", model)
        ])

        cv = KFold(n_splits=10, shuffle=True, random_state=rs)
        params.update({
            "cv_n_splits": cv.n_splits,
            "cv_shuffle": cv.shuffle,
            "cv_random_state": cv.random_state,
        })

        start = datetime.now()
        search = GridSearchCV(
            pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=-1
        )
        search.fit(x, y)
        end = datetime.now()

        mae_mean = -search.best_score_
        mae_std = search.cv_results_["std_test_score"][search.best_index_]
        params.update({
            "best_params": search.best_params_,
            "cv_results": search.cv_results_,
            "tuning_duration": str(end - start)
        })
        metrics = {
            "best_score_mean": round(mae_mean, 4),
            "best_score_std": round(mae_std, 4),
            "best_mae_percent": round(100 * (np.exp(mae_mean) - 1), 2),
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.end_run()

    print(type(model).__name__)
    print("-" * (len(type(model).__name__) + 2))
    print(f"Best params :\n{params['best_params']}\n")
    print(f"MAE mean : {metrics['best_score_mean']}")
    print(f"MAE std : {metrics['best_score_std']}")
    print(f"MAPE : {metrics['best_mae_percent']}")
    print("Tuning time :", {params['tuning_duration']})


def tune_hyperparameters():
    rs = config.RANDOM_STATE

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Hyperparameter Tuning")
    x_train, y_train = get_data()

    ## Tune first model shortlisted for tuning (Hist-GB)
    hgb_reg = HistGradientBoostingRegressor(
        early_stopping=True,
        validation_fraction=0.2,
        random_state=rs)
    param_grid = {
        "estimator__max_depth": [8, 12, 16],
        "estimator__max_iter": [120, 150, 200],
        "estimator__max_features": [0.8, 0.9, 1.0],
        "estimator__l2_regularization": [1.0, 5.0, 10.0],
        "estimator__learning_rate": [0.05, 0.1]
    }
    tune_model("HGB", hgb_reg, x_train, y_train, param_grid)

    ## Tune second model shortlisted for tuning (LR)
    rf_reg = RandomForestRegressor(random_state=rs)
    param_grid = {
        "estimator__n_estimators": [150, 250, 400],
        "estimator__max_depth": [8, 12, 16],
        "estimator__max_features": [0.6, 0.8, 1.0],
        "estimator__max_samples": [0.7, 0.85, 1.0],
    }
    tune_model("RF", rf_reg, x_train, y_train, param_grid)


if __name__ == "__main__":
    tune_hyperparameters()
