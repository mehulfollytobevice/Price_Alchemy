import logging
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.google.cloud.operators.sql import CloudSqlExecuteQueryOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 3, 12),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def migrate_records():
    logging.info("Starting to migrate records from holdout_table to mercari_table...")
    query = """
        INSERT INTO mercari_table (train_id, name, item_condition_id, category_name, brand_name, price, shipping, item_description, created_at, last_updated_at)
        SELECT train_id, name, item_condition_id, category_name, brand_name, price, shipping, item_description, NOW(), NOW()
        FROM holdout_table
        WHERE is_migrated = 0
        ORDER BY id
        LIMIT 10
    """
    logging.info("Executing SQL query to migrate records...")
    return query

def update_migration_status():
    logging.info("Updating migration status of migrated records...")
    query = """
        UPDATE holdout_table
        SET is_migrated = 1
        WHERE id IN (
            SELECT id
            FROM holdout_table
            WHERE is_migrated = 0
            ORDER BY id
            LIMIT 10
        )
    """
    logging.info("Executing SQL query to update migration status...")
    return query

with DAG('mercari_migration_dag', default_args=default_args, schedule_interval='@daily') as dag:

    migrate_records_task = CloudSqlExecuteQueryOperator(
        task_id='migrate_records',
        sql=migrate_records,
        instance='your_instance',
        database='your_database',
        gcp_conn_id='google_cloud_default',
    )

    update_migration_status_task = CloudSqlExecuteQueryOperator(
        task_id='update_migration_status',
        sql=update_migration_status,
        instance='your_instance',
        database='your_database',
        gcp_conn_id='google_cloud_default',
    )

    migrate_records_task >> update_migration_status_task
