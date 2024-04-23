# from airflow import DAG
# from airflow.operators.bash import BashOperator
# import datetime

# with DAG(
#     dag_id='hello_world_dag',
#     start_date=datetime.datetime(2023, 10, 5),
#     tags=['example', 'python'],
# ) as dag:
#     task_1 = BashOperator(
#     task_id='hello_world_task1',
#     bash_command="echo 'Hello World!'"
#     )

#     task_2 = BashOperator(
#     task_id='hello_world_task2',
#     bash_command="echo 'Hello World! Now I am triggered after the first task!'",
#     )

#     task_1 >> task_2 # We use >> the syntax to signify downstream tasks


from airflow.models import DAG
from airflow.utils.db import provide_session
from airflow.models import XCom
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago

with DAG(dag_id="cleanup_xcom_demo", schedule_interval=None, start_date=days_ago(2)) as dag:
    # cleanup_xcom
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
        # dag=dag
    )
    
    start  = DummyOperator(task_id="start")
    end = DummyOperator(task_id="end", trigger_rule="none_failed")
    
    start >> clean_xcom >> end