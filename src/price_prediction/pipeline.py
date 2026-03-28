import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    RobustScaler,
    FunctionTransformer,
)

from .config import config


def get_values(x):
    if isinstance(x, (pd.DataFrame, pd.Series)):
        values = x.values.flatten()
    else:
        values = x.flatten()
    return values


def levy_transform(x):
    values = get_values(x)
    return np.array(
        [np.nan if levy == "-" else float(levy) for levy in values]
    ).reshape(-1, 1)


def mileage_transform(x):
    values = get_values(x)
    return np.array([float(mileage.replace(" km", "")) for mileage in values]).reshape(
        -1, 1
    )


def engine_volume_transform(x):
    values = get_values(x)
    volume_turbo = []
    for engine_volume in values:
        turbo = int(engine_volume.endswith(" Turbo"))
        volume = float(engine_volume.replace(" Turbo", ""))
        volume_turbo.append([volume, turbo])
    return np.array(volume_turbo)


def engine_volume_features(transform: FunctionTransformer, input_features=None):
    _ = transform
    _ = input_features
    return ["Engine volume", "Turbo"]


def cast_to_int(x):
    return x.astype(np.int64)


def build_pipeline() -> ColumnTransformer:
    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    high_cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", TargetEncoder(target_type="continuous")),
            ("scaler", RobustScaler()),
        ]
    )
    num_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    bin_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
        ]
    )

    # Custom transforms
    levy_pipeline = Pipeline(
        [
            (
                "transformer",
                FunctionTransformer(
                    func=levy_transform, feature_names_out="one-to-one"
                ),
            ),
            ("imputer", SimpleImputer(strategy="median")),
            (
                "log_scaler",
                FunctionTransformer(func=np.log1p, feature_names_out="one-to-one"),
            ),
            ("scaler", RobustScaler()),
        ]
    )
    mileage_pipeline = Pipeline(
        [
            (
                "transformer",
                FunctionTransformer(
                    func=mileage_transform, feature_names_out="one-to-one"
                ),
            ),
            ("imputer", SimpleImputer(strategy="median")),
            (
                "log_scaler",
                FunctionTransformer(func=np.log1p, feature_names_out="one-to-one"),
            ),
            ("scaler", RobustScaler()),
        ]
    )
    engine_volume_num_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    turbo_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "cast",
                FunctionTransformer(func=cast_to_int, feature_names_out="one-to-one"),
            ),
        ]
    )
    engine_volume_splitter = FunctionTransformer(
        func=engine_volume_transform, feature_names_out=engine_volume_features
    )
    engine_volume_pipeline = Pipeline(
        [
            ("split", engine_volume_splitter),
            (
                "postprocess",
                ColumnTransformer(
                    transformers=[
                        ("engine", engine_volume_num_pipeline, [0]),
                        ("turbo", turbo_pipeline, [1]),
                    ],
                    remainder="drop",
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("cat", cat_pipeline, config.CAT_FEATURES),
            ("high_cat", high_cat_pipeline, config.HIGH_CAT_FEATURES),
            ("num", num_pipeline, config.NUM_FEATURES),
            ("bin", bin_pipeline, config.BIN_FEATURES),
            ("levy", levy_pipeline, ["Levy"]),
            ("mileage", mileage_pipeline, ["Mileage"]),
            ("volume", engine_volume_pipeline, ["Engine volume"]),
        ]
    )


def pipeline_smoke_test():
    train_path = config.DATA_DIR / "prepared" / config.TRAIN_FILE
    df = pd.read_csv(train_path)
    x = df.drop(config.TARGET, axis=1)
    y = df[config.TARGET]
    pipeline = build_pipeline()
    _ = pipeline.fit_transform(x, y)
    features = pipeline.get_feature_names_out()
    print(f"{len(features)} Output features :\n{features}")


if __name__ == "__main__":
    pipeline_smoke_test()
