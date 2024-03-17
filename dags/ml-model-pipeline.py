from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime

default_args = {
    'owner': 'aryan',
    'start_date': datetime(2024, 3, 12),
    'email': ['aryandeore98@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
    'max_active_runs': 1,
    'schedule_interval': '*/10 * * * *', # every 10 minutes
}

dag = DAG('dummy_operator_example', default_args=default_args, schedule_interval='@daily')

data_validation = DummyOperator(task_id='data_validation', dag=dag)

model_training = DummyOperator(task_id='model_training', dag=dag)
model_evaluation = DummyOperator(task_id='model_evaluation', dag=dag)

update_metrics_dashboard = DummyOperator(task_id='update_metrics_dashboard', dag=dag)

data_validation >> model_training >> model_evaluation >> update_metrics_dashboard