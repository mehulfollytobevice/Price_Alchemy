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


# test text preprocessing function v1
def test_text_prep_func_v1():

    input_sentence= ["this is a test sentence to check if the function works"]

    out= data_preprocessing.text_preprocess_v1(input_sentence)

    assert  out[0] == "test sentenc check function work"

# test text preprocessing function v2
def test_text_prep_func_v2():

    input_sentence= ["this is a test sentence to check if the text preprocessing from spacy works"]

    out= data_preprocessing.text_preprocess_v2(input_sentence)

    assert out[0]== "<s> this be a test sentence to check if the text preprocesse from spacy work </s>"


# test data sampling function
def test_data_sampling():

    # load the data
    df= data_loading.load_data_gcp(config.GCP_URL)
    sample= data_preprocessing.sample_df(df, sample_size=1000)

    assert df.shape[0] > sample.shape[0]
    assert df.shape[1] == sample.shape[1]
    assert sample.shape[0]==1000


