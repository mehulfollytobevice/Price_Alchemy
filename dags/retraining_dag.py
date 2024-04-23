#  import essentials
from airflow import DAG
from airflow import configuration as conf
from airflow.models.baseoperator import chain
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.db import provide_session
from airflow.models import XCom
from datetime import datetime, timedelta
from price_alchemy import config, model_dispatcher, train, data_loading, data_preprocessing, data_validation
import pickle 
import os
import pandas as pd
import glob
from datetime import datetime
import logging

now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

#Define Github repo, owner and endpoint
github_repo = "mehulfollytobevice/Price_Alchemy"
owner, repo = github_repo.split("/")
endpoint = f"repos/{owner}/{repo}/issues"


#Enable pickle support for XCom, allowing data to be passed between tasks
conf.set('core', 'enable_xcom_pickling', 'True')

# Default arguments for DAG
default_args = {
    'owner': owner,
    'start_date': datetime.now(),
    'retries': 0 # NUmber of attempts in case of failure
}

def logging_function(train_data, test_data):

    task_logger = logging.getLogger("airflow.task")

    train_cats= set(train_data['category_name'].values.tolist())
    test_cats= set(test_data['category_name'].values.tolist())
    new_cats= list (test_cats - train_cats)

    # with default airflow logging settings, DEBUG logs are ignored
    task_logger.info(f"Size of training data:{train_data.shape}")
    task_logger.info(f"Size of new data:{test_data.shape}")
    task_logger.info(f"Number of total categories in training data: {train_data.category_name.nunique()}")
    task_logger.info(f"Number of total categories in new data: {test_data.category_name.nunique()}")
    task_logger.info(f"New product categories: {new_cats}")
    task_logger.info(f"Number of new product categories: {len(new_cats)}")
    task_logger.info(f"Number of brands in training data: {train_data.brand_name.nunique()}")
    task_logger.info(f"Number of brands in new data: {test_data.brand_name.nunique()}")

    # Using the Task flow API to push to XCom by returning a value
    return True

# load preprocessed pickle file
def load_preprocessed_data(filename):

    # what's the output path
    output_path = os.path.join( os.getcwd() ,"working_data" , filename)

    # load the pickle data
    data= pickle.load(open(output_path, 'rb'))

    return data['X'], data['y']

# utility function to dump trained model into a pickle file
def dump_trained_model(model, filename):

    filename= filename +".pickle" 

    # what's the output path
    output_path = os.path.join( os.getcwd() ,"working_data" ,"models" , filename)

    # dump the pickle data
    pickle.dump(model , open(output_path, 'wb'))

# read the most recent file 
def read_most_recent_file(folder, pattern):
    # Get list of files matching the pattern
    files = glob.glob(os.path.join(folder, pattern))
    # Sort files by modification time (most recent first)
    files.sort(key=os.path.getmtime, reverse=True)
    if files:
        most_recent_file = files[0]
        
    return most_recent_file

# load the training data from the pickle file
def load_train_data(folder, pattern):

    # what's the output path
    folder= os.path.join( os.getcwd() ,"working_data" , folder)
    filename= read_most_recent_file(folder, pattern)
    output_path = os.path.join( os.getcwd() ,"working_data" , filename)

    # load the pickle data
    data= pickle.load(open(output_path, 'rb'))

    return data['X']

# utility function to train the model
def airflow_train(model_name, folder, pattern):

    folder= os.path.join( os.getcwd() ,"working_data" , folder)
    filename= read_most_recent_file(folder, pattern)

    # load data from pickle file
    X, y= load_preprocessed_data(filename)

    # get model from dispatch
    model= model_dispatcher.models[model_name]

    # train model
    trained_model, metrics= train.train_model(X, y.values, model)

    # save the trained model
    model_name= model_name + "_" + f"{timestamp}"
    dump_trained_model(trained_model, model_name)

    return trained_model, metrics

def category_match(train_data, test_data):

    train_cats= set(train_data['category_name'].values.tolist())

    test_cats= set(test_data['category_name'].values.tolist())

    if test_cats.issubset(train_cats)  or train_cats == test_cats:
        return True
    else:
        return False

def retrain_decision(ti):

    valid_checks= ti.xcom_pull(task_ids=['cat_match_task','data_validation_gx_task'])

    if False in valid_checks:

        return 'retrain_data_task'
    
    return 'no_train_task'

def concatenate_df(train_df, test_df):

    return pd.concat([train_df, test_df],ignore_index=True)

# Create DAG instance
with DAG('Re-Training_pipeline', 
           default_args=default_args,
           description='This dag runs the end-to-end model training pipeline. ',
           schedule_interval='@daily',
           catchup=False          
) as dag:


    # load training data from the pickle file
    load_train_data_task = PythonOperator(
        task_id = 'load_train_data_task',
        python_callable=load_train_data,
        execution_timeout=timedelta(minutes= 2),
        op_kwargs={"folder": "data", "pattern":"train_data_*"}
    )

    # Task to load all data present in the database, calls the 'load_data' Python function
    load_test_data_task = PythonOperator(
        task_id = 'load_test_data_task',
        python_callable=data_loading.load_data_gcp,
        execution_timeout=timedelta(minutes= 2),
        op_kwargs={"gcp_url":config.GCP_URL}
    )

    # sample a part of the dataframe
    train_data_sample_task = PythonOperator(
        task_id='train_data_sample_task',
        python_callable=data_preprocessing.sample_df,
        op_kwargs={"df":load_train_data_task.output,"sample_size":config.TEST_SAMPLE_SIZE},
    )

    # sample a part of the dataframe for testing as new data
    test_data_sample_task = PythonOperator(
        task_id='test_data_sample_task',
        python_callable=data_preprocessing.sample_df,
        op_kwargs={"df":load_test_data_task.output,"sample_size":config.TEST_SAMPLE_SIZE, "random_state":37},
    )

    # check for category match 
    cat_match_task=PythonOperator(
        task_id='cat_match_task',
        python_callable=category_match,
        op_kwargs={"train_data":train_data_sample_task.output,"test_data":test_data_sample_task.output},
    )

    # check for pandera schema of test data
    data_validation_gx_task = PythonOperator(
        task_id='data_validation_gx_task',
        python_callable=data_validation.great_exp_validate,
        op_kwargs={"df":test_data_sample_task.output}
    )

    # check for great expectations tests on test data
    data_validation_pandera_task = PythonOperator(
        task_id='data_validation_pandera_task',
        python_callable=data_validation.pandera_validate,
        op_kwargs={"df":test_data_sample_task.output}
    )

    # branch operator to see if any if these fail [category check or great expectations]
    retrain_decision_task= BranchPythonOperator(
        task_id='retrain_decision_task',
        python_callable= retrain_decision
    )

    no_train_task= BashOperator(
        task_id="no_train_task",
        bash_command="echo 'No retraining required' "
    )

    # logging function
    logging_task = PythonOperator(
        task_id='logging_task',
        python_callable=logging_function,
        op_args=[train_data_sample_task.output, test_data_sample_task.output]
        )
    
    # concatenate with training data 
    retrain_data_task= PythonOperator(
        task_id='retrain_data_task',
        python_callable=concatenate_df,
        op_kwargs={"train_df":train_data_sample_task.output, 'test_df': test_data_sample_task.output},
    )
 
    # Task to perform data preprocessing, depends on 'retrain_data_task'
    data_preprocessing_task_A = PythonOperator(
        task_id='data_preprocessing_task_A',
        python_callable=data_preprocessing.preprocessing_pipe,
        op_kwargs={"df":retrain_data_task.output,
                    "text_prep_func":config.TEXT_PREP_OPTS['spacy'],
                    "column_trans":config.COL_TRANS_OPTS['tfidf_concat'],
                    "if_airflow":True,
                    "filename": f"preprocessors/tidf_concat_preprocessor_{timestamp}.pickle"},
        do_xcom_push= True
    )

    data_preprocessing_task_B = PythonOperator(
        task_id='data_preprocessing_task_B',
        python_callable=data_preprocessing.preprocessing_pipe,
        op_kwargs={"df":retrain_data_task.output,
                    "text_prep_func":config.TEXT_PREP_OPTS['spacy'],
                    "column_trans":config.COL_TRANS_OPTS['tfidf_ngram'],
                    "if_airflow":True,
                    "filename": f"preprocessors/tidf_ngram_preprocessor_{timestamp}.pickle"},
        do_xcom_push= True
    )

    data_preprocessing_task_C = PythonOperator(
        task_id='data_preprocessing_task_C',
        python_callable=data_preprocessing.preprocessing_pipe,
        op_kwargs={"df":retrain_data_task.output,
                    "text_prep_func":config.TEXT_PREP_OPTS['spacy'],
                    "column_trans":config.COL_TRANS_OPTS['tfidf_chargram'],
                    "if_airflow":True,
                    "filename": f"preprocessors/tidf_chargram_preprocessor_{timestamp}.pickle"},
        do_xcom_push= True
    )

    # Task to save the data 
    data_saving_task_A = PythonOperator(
        task_id='data_saving_task_A',
        python_callable=data_preprocessing.dump_preprocessed_data,
        op_args=[data_preprocessing_task_A.output,  f"model_data/tfidf_concat_data_sm_{timestamp}.pkl", True], 
        do_xcom_push= False
    )

    data_saving_task_B = PythonOperator(
        task_id='data_saving_task_B',
        python_callable=data_preprocessing.dump_preprocessed_data,
        op_args=[data_preprocessing_task_B.output, f"model_data/tfidf_ngram_data_sm_{timestamp}.pkl", True], 
        do_xcom_push= False
    )

    data_saving_task_C = PythonOperator(
        task_id='data_saving_task_C',
        python_callable=data_preprocessing.dump_preprocessed_data,
        op_args=[data_preprocessing_task_C.output, f"model_data/tfidf_chargram_data_sm_{timestamp}.pkl", True], 
        do_xcom_push= False
    )

    @provide_session
    def cleanup_xcom(session=None, **context):
        dag = context["dag"]
        dag_id = dag._dag_id 
        # It will delete all xcom of the dag_id
        session.query(XCom).filter(XCom.dag_id == dag_id).delete()

    clean_xcom = PythonOperator(
        task_id="clean_xcom",
        python_callable = cleanup_xcom,
        provide_context=True, 
        )

    # retrain models 
    model_train_task_A= PythonOperator(
        task_id= 'model_train_task_A',
        python_callable=airflow_train,
        op_args=['mlp_tfidf',  "model_data" , "tfidf_concat_*"],
        do_xcom_push= False
    )

    model_train_task_B= PythonOperator(
        task_id= 'model_train_task_B',
        python_callable=airflow_train,
        op_args=['mlp_chargram',  "model_data" , "tfidf_ngram_*"],
        do_xcom_push= False
    )

    model_train_task_C= PythonOperator(
        task_id= 'model_train_task_C',
        python_callable=airflow_train,
        op_args=['mlp_ngram',  "model_data" , "tfidf_chargram_*"],
        do_xcom_push= False
    )

    # airflow task sequence
    load_train_data_task >> train_data_sample_task
    load_test_data_task >> test_data_sample_task
    test_data_sample_task >> [data_validation_pandera_task,data_validation_gx_task, cat_match_task]
    [data_validation_gx_task, cat_match_task] >> retrain_decision_task
    retrain_decision_task >> no_train_task
    retrain_decision_task >> retrain_data_task 
    retrain_data_task >> logging_task >> [data_preprocessing_task_A, data_preprocessing_task_B, data_preprocessing_task_C]
    data_preprocessing_task_A >> data_saving_task_A 
    data_preprocessing_task_B >> data_saving_task_B 
    data_preprocessing_task_C >> data_saving_task_C 
    [data_saving_task_A, data_saving_task_B, data_saving_task_C] >> clean_xcom >> model_train_task_A >> model_train_task_B >> model_train_task_C




    

