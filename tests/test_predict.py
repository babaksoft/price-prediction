import pandas as pd
import pytest

from price_prediction.config import config
from price_prediction.predict import make_prediction


@pytest.fixture
def single_prediction():
    """ This function will predict the result for a single record"""
    data_path = config.DATA_PATH / "prepared" / config.TEST_FILE
    df_test = pd.read_csv(data_path)
    single_test = df_test[0:1]
    result = make_prediction(single_test)
    return result

def test_single_prediction_not_none(single_prediction):
    """ This function will check if result of prediction is not None"""
    assert single_prediction is not None

def test_single_prediction_dtype(single_prediction):
    """ This function will check if data type of prediction is float i.e. number """
    assert isinstance(single_prediction.get("prediction")[0], float)

def test_single_prediction_output(single_prediction):
    """ This function will check if prediction is a positive value """
    assert single_prediction.get("prediction")[0] > 0.0
