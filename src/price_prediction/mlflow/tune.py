from datetime import datetime

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

        params.update({
            "best_params": search.best_params_,
            "best_score": -round(search.best_score_, 4),
            "tuning_duration": str(end - start)
        })

        mlflow.log_params(params)
        mlflow.end_run()

    print(type(model).__name__)
    print("-" * (len(type(model).__name__) + 2))
    print(f"Best params :\n{search.best_params_}\n")
    print(f"MAE : {-round(search.best_score_, 4)}")
    print("Tuning time :", str(end-start))


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
        "estimator__max_depth": [10, 12, 15],
        "estimator__max_iter": [120, 150, 180],
        "estimator__max_features": [0.8, 0.85, 0.9],
        "estimator__l2_regularization": [2.0, 5.0, 10.0]
    }
    tune_model("HGB", hgb_reg, x_train, y_train, param_grid)

    ## Tune second model shortlisted for tuning (LR)
    rf_reg = RandomForestRegressor(random_state=rs)
    param_grid = {
        "estimator__n_estimators": [150, 200, 250],
        "estimator__max_depth": [8, 10, 12, 15],
        "estimator__max_features": [0.7, 0.8, 0.9],
        "estimator__max_samples": [0.7, 0.8, 0.9],
    }
    tune_model("RF", rf_reg, x_train, y_train, param_grid)


if __name__ == "__main__":
    tune_hyperparameters()
