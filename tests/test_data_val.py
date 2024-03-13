import pytest
from price_alchemy import config, data_loading, data_validation
import pandas as pd

def test_data_source():

    assert config.GCP_URL=='https://storage.googleapis.com/price_alchemy/Data/data.csv'

def test_data_loading():
    
    assert isinstance(data_loading.load_data_gcp(config.GCP_URL), pd.DataFrame)

@pytest.mark.xfail
def test_data_loading2():

    assert isinstance(data_loading.load_data_sql('WRONG_PASSWORD'), pd.DataFrame)

def test_data_validation():

    df= data_loading.load_data_gcp(config.GCP_URL)

    # validate the schema using pandera
    result_pandera=data_validation.pandera_validate(df)

    # run additional tests using great expectations
    result_gx= data_validation.great_exp_validate(df)

    assert result_pandera==True
    assert result_gx==True