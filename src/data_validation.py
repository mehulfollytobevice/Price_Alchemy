# import libraries
import logging
import uuid
import os
import pandas as pd
import config

# data validation libraries
import pandera as pa
import great_expectations as gx
import matplotlib.pyplot as plt
from great_expectations.checkpoint import Checkpoint
import warnings
warnings.filterwarnings('ignore')


def load_data(gcp_url):
    
    try:

        df = pd.read_csv(gcp_url)
        return df 
    
    except:
        logging.error('Data not available')

def great_exp_validate(df):
    
    context=gx.get_context()

    # add data sources
    DS_NAME='mercari_data_file'
    datasource=context.sources.add_pandas(name=DS_NAME)

    # adding csv asset
    asset_name='mercari_data'
    asset= datasource.add_dataframe_asset(asset_name)

    # build batch request
    batch_request= asset.build_batch_request(dataframe= df)

    context.add_or_update_expectation_suite('mercari_expectation_suite')

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name='mercari_expectation_suite',
    )

    # column should exist
    validator.expect_column_to_exist("name")
    validator.expect_column_to_exist("price")
    validator.expect_column_to_exist("item_condition_id")
    validator.expect_column_to_exist("category_name")
    validator.expect_column_to_exist("brand_name")
    validator.expect_column_to_exist("shipping")
    validator.expect_column_to_exist("item_description")
    validator.expect_column_to_exist("created_at")
    validator.expect_column_to_exist("last_updated_at")

    # not null expectations
    validator.expect_column_values_to_not_be_null("name")
    validator.expect_column_values_to_not_be_null("price")
    validator.expect_column_values_to_not_be_null("item_condition_id")
    validator.expect_column_values_to_not_be_null("shipping")
    validator.expect_column_values_to_not_be_null("created_at")
    validator.expect_column_values_to_not_be_null("last_updated_at")
    validator.expect_column_values_to_not_be_null("item_description", mostly=0.95)
    validator.expect_column_values_to_not_be_null("category_name", mostly=.95)
    validator.expect_column_values_to_not_be_null("brand_name", mostly=0.5)

    # value expectations
    validator.expect_column_max_to_be_between(
        "price", min_value=1000, max_value=2500)

    validator.expect_column_distinct_values_to_be_in_set(
            "shipping",
            [0,1])

    validator.expect_column_distinct_values_to_be_in_set(
            "item_condition_id",
            [1,2,3,4,5])

    #  distribution expectations
    validator.expect_column_stdev_to_be_between(
    'price', min_value=30, max_value=50)

    validator.expect_column_mean_to_be_between(
    'price', min_value=20, max_value=30)

    validator.expect_column_value_z_scores_to_be_less_than(
    'price', threshold=3, mostly=.9, double_sided=False)

    # regex expectations
    # should not be urls
    validator.expect_column_values_to_not_match_regex(
    'name', regex='https?:\/\/.*[\r\n]*')

    validator.expect_column_values_to_not_match_regex(
    'brand_name', regex='https?:\/\/.*[\r\n]*')

    validator.expect_column_values_to_not_match_regex(
    'category_name', regex='https?:\/\/.*[\r\n]*')

    # date columns
    validator.expect_column_values_to_be_dateutil_parseable('created_at')
    validator.expect_column_values_to_be_dateutil_parseable('last_updated_at')

    validator.save_expectation_suite(discard_failed_expectations=False)

    checkpoint = context.add_or_update_checkpoint(
        name="checkpoint_v4",
        validator=validator,
    )

    checkpoint_result = checkpoint.run()

    if checkpoint_result["success"] is True:
        return True

    else:
        logging.error('Great expectation data validation checks failed.')
        return False

def pandera_validate(df):

    # define a schema for the data frame 
    schema= pa.DataFrameSchema({
        'train_id':pa.Column(int),
        'name': pa.Column(str), 
        'item_condition_id': pa.Column(int, checks=pa.Check.isin([1,2,3,4,5])),
        'category_name':pa.Column(str, nullable=True),
        'brand_name':pa.Column(str, nullable=True),
        'price':pa.Column(float, checks=pa.Check(lambda x: x>=0)), 
        'shipping':pa.Column(int, checks=pa.Check.isin([0,1])),
        'item_description':pa.Column(str, nullable=True),
        'created_at':pa.Column(str),
        'last_updated_at':pa.Column(str)}, 
        unique=['train_id','name'])

    try:
        schema(df)
        return True
    except:
        logging.error('Pandera data validation checks failed.')
        return False

if __name__=="__main__":

    try:

        # Get the parent directory of the current directory (src)
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

        # Define the logs directory path
        logs_dir = os.path.join(parent_dir, 'logs')

        # Create the logs directory if it doesn't exist
        os.makedirs(logs_dir, exist_ok=True)

        # generate unique id
        new_uuid = uuid.uuid4()

        # Configure logging to log to a file in the logs directory
        logging.basicConfig(filename=os.path.join(logs_dir, f'logfile_{new_uuid}.log'), level=logging.INFO)
        
        # load the data 
        logging.debug('Reading csv data')
        df= load_data(config.GCP_URL)

        # validate the schema using pandera
        logging.debug('Validating data using pandera.')
        result_pandera=pandera_validate(df)

        # print(f'Passed pandera data validation check: {result_pandera}')
        logging.info(f'Data validation check result -  pandera : {result_pandera} \n')

        # run additional tests using great expectations
        logging.debug('Validating data using great expectations.')
        result_gx= great_exp_validate(df)

        # print(f'Passed great expectations data validation check: {result_gx}')
        logging.info(f'Data validation check result -  GX : {result_gx}')
    
    except:
        logging.critical('Data validation pipeline failed')