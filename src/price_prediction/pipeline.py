import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, FunctionTransformer)

from .config import config
from .config.country import get_country_map


def get_values(x):
    if isinstance(x, (pd.DataFrame, pd.Series)):
        values = x.values.flatten()
    else:
        values = x.flatten()
    return values


def manufacturer_transform(x):
    country_map = get_country_map()
    countries = []
    for m in x:
        key = m[0] if isinstance(m, (list, np.ndarray)) else m
        country = country_map[key] if key in country_map else "Unknown"
        countries.append(country)
    return np.array(countries).reshape(-1, 1)


def levy_transform(x):
    values = get_values(x)
    return np.array([
        np.nan if levy == "-" else float(levy) for levy in values
    ]).reshape(-1, 1)


def mileage_transform(x):
    values = get_values(x)
    return np.array([
        float(mileage.replace(" km", "")) for mileage in values
    ]).reshape(-1, 1)


def engine_volume_transform(x):
    values = get_values(x)
    volume_turbo = []
    for engine_volume in values:
        turbo = int(engine_volume.endswith(" Turbo"))
        volume = float(engine_volume.replace(" Turbo", ""))
        volume_turbo.append([volume, turbo])
    return np.array(volume_turbo)


def engine_volume_features(
        transform: FunctionTransformer, input_features=None
):
    _ = transform
    _ = input_features
    return ["Engine volume", "Turbo"]


def cast_to_int(x):
    return x.astype(np.int64)


def build_pipeline() -> ColumnTransformer:
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    bin_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            drop="if_binary"
        ))
    ])

    # Custom transforms
    levy_pipeline = Pipeline([
        ("transformer", FunctionTransformer(
            func=levy_transform,
            feature_names_out="one-to-one"
        )),
        ("imputer", SimpleImputer(strategy="median")),
        ("log_scaler", FunctionTransformer(
            func=np.log1p,
            feature_names_out="one-to-one"
        )),
        ("scaler", StandardScaler())
    ])
    mileage_pipeline = Pipeline([
        ("transformer", FunctionTransformer(
            func=mileage_transform,
            feature_names_out="one-to-one"
        )),
        ("imputer", SimpleImputer(strategy="median")),
        ("log_scaler", FunctionTransformer(
            func=np.log1p,
            feature_names_out="one-to-one"
        )),
        ("scaler", StandardScaler())
    ])
    manufacturer_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("mapper", FunctionTransformer(
            func=manufacturer_transform,
            feature_names_out="one-to-one"
        )),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    engine_volume_num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    turbo_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("cast", FunctionTransformer(
            func=cast_to_int,
            feature_names_out="one-to-one"
        )),
    ])
    engine_volume_splitter = FunctionTransformer(
        func=engine_volume_transform,
        feature_names_out=engine_volume_features
    )
    engine_volume_pipeline = Pipeline([
        ("split", engine_volume_splitter),
        ("postprocess", ColumnTransformer(
            transformers=[
                ("engine_volume", engine_volume_num_pipeline, [0]),
                ("turbo", turbo_pipeline, [1])
            ],
            remainder="drop")
        )
    ])

    return ColumnTransformer([
        ("categorical", cat_pipeline, config.CAT_FEATURES),
        ("numerical", num_pipeline, config.NUM_FEATURES),
        ("binary", bin_pipeline, config.BIN_FEATURES),
        ("levy", levy_pipeline, ["Levy"]),
        ("mileage", mileage_pipeline, ["Mileage"]),
        ("country", manufacturer_pipeline, ["Manufacturer"]),
        ("engine_volume", engine_volume_pipeline, ["Engine volume"])
    ])


def pipeline_smoke_test():
    train_path = config.DATA_DIR / "prepared" / config.TRAIN_FILE
    df = pd.read_csv(train_path)
    x = df.drop(config.TARGET, axis=1)
    pipeline = build_pipeline()
    _ = pipeline.fit_transform(x)
    features = pipeline.get_feature_names_out()
    print(f"{len(features)} Output features :\n{features}")


if __name__ == "__main__":
    pipeline_smoke_test()
