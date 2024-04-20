#  import essentials
from airflow import DAG
from airflow import configuration as conf
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from price_alchemy import config, data_loading, data_preprocessing, data_validation

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


# Define a function to separate data outputs
def separate_data_outputs(**kwargs):
    ti = kwargs['ti']
    X, y = ti.xcom_pull(task_ids='data_preprocessing_task')
    return X, y

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

    # Task to perform data validation
    data_validation_gx_task = PythonOperator(
        task_id='data_validation_gx_task',
        python_callable=data_validation.great_exp_validate,
        op_kwargs={"df":load_data_task.output}
    )

    data_validation_pandera_task = PythonOperator(
        task_id='data_validation_pandera_task',
        python_callable=data_validation.pandera_validate,
        op_kwargs={"df":load_data_task.output}
    )

    # sample a part of the dataframe
    data_sampling_task = PythonOperator(
        task_id='data_sampling_task',
        python_callable=data_preprocessing.sample_df,
        op_kwargs={"df":load_data_task.output,"sample_size":config.TEST_SAMPLE_SIZE},
    )

    # Task to perform data preprocessing, depends on 'load_data_task'
    data_preprocessing_task = PythonOperator(
        task_id='data_preprocessing_task',
        python_callable=data_preprocessing.preprocessing_pipe,
        op_kwargs={"df":data_sampling_task.output,"text_prep_func":config.TEXT_PREP_OPTS['nltk'], "column_trans":config.COL_TRANS_OPTS['tfidf_concat']},
        do_xcom_push= True
    )

    # Task to execute the 'separate_data_outputs' function
    separate_data_outputs_task = PythonOperator(
        task_id='separate_data_outputs_task',
        python_callable=separate_data_outputs,
        provide_context=True
        )

    # Task to save the preprocessed data
    data_saving_task = PythonOperator(
        task_id='data_saving_task',
        python_callable=data_preprocessing.dump_preprocessed_data,
        op_args=[separate_data_outputs_task.output, "airflow_preprocessed.pkl", True], 
        do_xcom_push= True
    )


    # Set task dependencies
    load_data_task  >> [data_validation_pandera_task, data_validation_gx_task , data_sampling_task] >> data_preprocessing_task >> separate_data_outputs_task >> data_saving_task
