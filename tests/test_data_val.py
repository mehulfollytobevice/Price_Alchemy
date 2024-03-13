from price_alchemy import config

def test_data_source():

    assert config.GCP_URL=='https://storage.googleapis.com/price_alchemy/Data/data.csv'