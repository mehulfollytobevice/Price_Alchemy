import logging
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import mysql.connector
from mysql.connector import Error
import pendulum


default_args = {
    'owner': 'aryan',
    # 'start_date': datetime(2024, 3, 12),
    'start_date': pendulum.datetime(2024, 3, 12, tz="EST"),
    'email': ['aryandeore98@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
    'max_active_runs': 1,
    'catchup': False,
    'schedule_interval': '*/10 * * * *', # every 10 minutes
}

def migrate_records():

    try:
        connection = mysql.connector.connect(
            user='root', 
            password='Mlops@Group8', 
            host='34.30.80.103',
            database='mercari_db'
        )

        if connection.is_connected():
            print("Connection to MySQL database established")

            cursor_obj = connection.cursor()
            
            query_migrate = """
                INSERT INTO testing_product (train_id, name, item_condition_id, category_name, brand_name, price, shipping, item_description, created_at, last_updated_at)
                SELECT train_id, name, item_condition_id, category_name, brand_name, price, shipping, item_description, NOW(), NOW()
                FROM testing_holdout
                WHERE is_migrated = 0
                ORDER BY id
                LIMIT 10
            """
            cursor_obj.execute(query_migrate)
            cursor_obj.close()
            connection.commit()
            print("Query executed successfully")

    except Error as e:
        print("Error while connecting to MySQL", e)

    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor_obj.close()
            connection.close()
            print("MySQL connection is closed")

def update_migration_status():
    logging.info("Updating migration status of migrated records...")
    try:
        connection = mysql.connector.connect(
            user='root', 
            password='Mlops@Group8', 
            host='34.30.80.103',
            database='mercari_db')
        
        cursor_task2 = connection.cursor()
        query_migration_status = """
            UPDATE testing_holdout h
            JOIN (
                SELECT id
                FROM testing_holdout
                WHERE is_migrated = 0
                ORDER BY id
                LIMIT 10
            ) AS subquery ON h.id = subquery.id
            SET h.is_migrated = 1
        """
        cursor_task2.execute(query_migration_status)
        connection.commit()
        cursor_task2.close()
        connection.close()
        logging.info("Migration status updated successfully.")
    except Exception as e:
        logging.error(f"Error occurred while updating migration status: {str(e)}")

with DAG('mercari_migration_dag_gpt', default_args=default_args, schedule_interval='*/2 * * * *') as dag:
    
    migrate_records_task = PythonOperator(task_id='migrate_records',
                                          python_callable=migrate_records
                                          )

    update_migration_status_task = PythonOperator(task_id='update_migration_status',
                                                  python_callable=update_migration_status
                                                  )

    migrate_records_task >> update_migration_status_task
