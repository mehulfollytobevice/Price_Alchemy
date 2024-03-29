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

    # Task to perform data preprocessing, depends on 'load_data_task'
    data_preprocessing_task = PythonOperator(
        task_id='data_preprocessing_task',
        python_callable=data_preprocessing.data_prep_v1,
        op_kwargs={"df":load_data_task.output},
        do_xcom_push= False
    )


    # Set task dependencies
    load_data_task >> data_validation_pandera_task >> data_validation_gx_task >> data_preprocessing_task 
