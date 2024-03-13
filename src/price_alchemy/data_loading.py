# import libraries
import pandas as pd
import sqlite3
from google.cloud import storage 
from datetime import datetime
import mysql.connector
from typing import Iterable, Any
from cred import MYSQL_PASSWORD
import logging

def load_data_sql():

    connection = mysql.connector.connect(
        user='root', 
        password=MYSQL_PASSWORD, 
        host='34.30.80.103',
        database='mercari_db')

    cursor = connection.cursor()

    # Execute your query
    query1 = "SELECT * FROM product_listing"
    cursor.execute(query1)

    # Convert to Iterable[Any]
    iterable_obj: Iterable[Any]

    if isinstance(cursor.description, list) and all(isinstance(item, tuple) for item in cursor.description):
        iterable_obj = cursor.description  # If obj is already an iterable (list of tuples)
    else:
        iterable_obj = [cursor.description]  # Wrap obj in a list to make it iterable
    
    # Fetch column names
    cd_list= list(iterable_obj)
    column_names = [i[0] for i in cd_list]

    # Create a DataFrame from the query result
    frame = pd.DataFrame(cursor.fetchall(), columns=column_names)

    # Close cursor and connection
    cursor.close()
    connection.close()

    return frame


def load_data_gcp(gcp_url):
    
    try:

        df = pd.read_csv(gcp_url)
        return df 
    
    except:
        logging.error('Data not available')

