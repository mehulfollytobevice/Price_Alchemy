#  import essentials
from airflow import DAG
from airflow import configuration as conf
from airflow.operators.python_operator import PythonOperator
import os
import glob
from datetime import datetime, timedelta
from price_alchemy import config, data_loading, data_preprocessing, data_validation
import logging

#Define Github repo, owner and endpoint
github_repo = "mehulfollytobevice/Price_Alchemy"
owner, repo = github_repo.split("/")
endpoint = f"repos/{owner}/{repo}/issues"

now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")


#Enable pickle support for XCom, allowing data to be passed between tasks
conf.set('core', 'enable_xcom_pickling', 'True')

# Default arguments for DAG
default_args = {
    'owner': owner,
    'start_date': datetime.now(),
    'retries': 0 # NUmber of attempts in case of failure
}


# Define a function to separate data outputs
def separate_data_outputs(**kwargs):
    ti = kwargs['ti']
    X, y = ti.xcom_pull(task_ids='data_preprocessing_task')
    return X, y

def logging_function(df, sample_df):

    task_logger = logging.getLogger("airflow.task")

    # with default airflow logging settings, DEBUG logs are ignored
    task_logger.info("Starting Debugger Logs")
    task_logger.info(f"Loaded data size: {df.shape}")
    task_logger.info(f"Sampled data size: {sample_df.shape}")
    task_logger.info(f"Number of categories in loaded data: {df.category_name.nunique()}")
    task_logger.info(f"Number of brands in loaded data: {df.brand_name.nunique()}")

    # Using the Task flow API to push to XCom by returning a value
    return True


# Create DAG instance
with DAG('Data_Pipeline', 
           default_args=default_args,
           description='This dag runs the end-to-end data pipeline to load, validate, preprocess and save the data.',
           schedule_interval=None,
           catchup=False          
) as dag:


    # Task to load data, calls the 'load_data' Python function
    load_data_task = PythonOperator(
        task_id = 'load_data_task',
        python_callable=data_loading.load_data_gcp,
        execution_timeout=timedelta(minutes= 2),
        op_kwargs={"gcp_url":config.GCP_URL}
    )

    # sample a part of the dataframe
    data_sampling_task = PythonOperator(
        task_id='data_sampling_task',
        python_callable=data_preprocessing.sample_df,
        op_kwargs={"df":load_data_task.output,"sample_size":config.TEST_SAMPLE_SIZE},
    )

    # logging function
    logging_task = PythonOperator(
        task_id='logging_task',
        python_callable=logging_function,
        op_args=[load_data_task.output, data_sampling_task.output]
        )

    # Task to perform data validation
    data_validation_gx_task = PythonOperator(
        task_id='data_validation_gx_task',
        python_callable=data_validation.great_exp_validate,
        op_kwargs={"df":data_sampling_task.output}
    )

    data_validation_pandera_task = PythonOperator(
        task_id='data_validation_pandera_task',
        python_callable=data_validation.pandera_validate,
        op_kwargs={"df":data_sampling_task.output}
    )
    
    # Task to perform data preprocessing, depends on 'retrain_data_task'
    data_preprocessing_task_A = PythonOperator(
        task_id='data_preprocessing_task_A',
        python_callable=data_preprocessing.preprocessing_pipe,
        op_kwargs={"df":data_sampling_task.output,
                    "text_prep_func":config.TEXT_PREP_OPTS['spacy'],
                    "column_trans":config.COL_TRANS_OPTS['tfidf_concat'],
                    "if_airflow":True,
                    "filename": f"preprocessors/tidf_concat_preprocessor_{timestamp}.pickle"},
        do_xcom_push= True
    )

    data_preprocessing_task_B = PythonOperator(
        task_id='data_preprocessing_task_B',
        python_callable=data_preprocessing.preprocessing_pipe,
        op_kwargs={"df":data_sampling_task.output,
                    "text_prep_func":config.TEXT_PREP_OPTS['spacy'],
                    "column_trans":config.COL_TRANS_OPTS['tfidf_ngram'],
                    "if_airflow":True,
                    "filename": f"preprocessors/tidf_ngram_preprocessor_{timestamp}.pickle"},
        do_xcom_push= True
    )

    data_preprocessing_task_C = PythonOperator(
        task_id='data_preprocessing_task_C',
        python_callable=data_preprocessing.preprocessing_pipe,
        op_kwargs={"df":data_sampling_task.output,
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
        op_args=[data_preprocessing_task_A.output, f"model_data/tfidf_concat_data_sm_{timestamp}.pkl", True], 
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

    data_dump_task= PythonOperator(
        task_id='data_dump_task',
        python_callable=data_preprocessing.dump_preprocessed_data,
        op_args=[[load_data_task.output, data_sampling_task.output] , f"data/train_data_{timestamp}.pkl", True], 
        do_xcom_push= False
    )

    # Set task dependencies
    load_data_task >> data_sampling_task >> logging_task >> [data_validation_pandera_task, data_validation_gx_task] 
    [data_validation_pandera_task, data_validation_gx_task]>> data_preprocessing_task_A >> data_saving_task_A
    [data_validation_pandera_task, data_validation_gx_task]>> data_preprocessing_task_B >> data_saving_task_B
    [data_validation_pandera_task, data_validation_gx_task]>> data_preprocessing_task_C >> data_saving_task_C
    [data_saving_task_A, data_saving_task_B, data_saving_task_C ] >> data_dump_task
    
