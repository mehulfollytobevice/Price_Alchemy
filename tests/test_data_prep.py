import pytest
import numpy as np
from price_alchemy import config, data_loading, data_preprocessing
import pandas as pd


# note:
# parameterize this function for additional testing in future
# test function to split hierarchical category
def test_split_cat():

    result= data_preprocessing.split_cat('Electronics/Computers & Tablets/iPad/Tablet/eBook Access')

    assert result == ['Electronics', 'Computers & Tablets', 'iPad/Tablet/eBook Access']

# test data preprocessing 
def test_data_preprocessing():

    # load the data
    df= data_loading.load_data_gcp(config.GCP_URL)
    sample= df.iloc[:100, :]

    # validate that the preprocessing function works
    text_prep= config.TEXT_PREP_OPTS['spacy']
    col_trans= 'tfidf_chargram'

    X,y= data_preprocessing.preprocessing_pipe(sample, text_prep, config.COL_TRANS_OPTS[col_trans])

    # test conditions
    assert X.shape[0]<=sample.shape[0] # some rows might be dropped because of null values so less than equal to is used 
    assert y.shape[0]<=sample.shape[0]
    assert X.shape[0]== y.shape[0]
    assert X.shape[1]>sample.shape[1]


