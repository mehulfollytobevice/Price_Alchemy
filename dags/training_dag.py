#  import essentials
from airflow import DAG
from airflow import configuration as conf
from airflow.models.baseoperator import chain
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from price_alchemy import config, model_dispatcher, train
import pickle 
import os
import glob
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

# Create DAG instance
with DAG('Training_pipeline', 
           default_args=default_args,
           description='This dag runs the end-to-end model training pipeline. ',
           schedule_interval='@weekly',
           catchup=False          
) as dag:


    # Task to perform model training
    model_train_task_A= PythonOperator(
        task_id= 'model_train_task_A',
        python_callable=airflow_train,
        op_args=['mlp_tfidf', "model_data" , "tfidf_concat_*"],
        do_xcom_push= False
    )

    model_train_task_B= PythonOperator(
        task_id= 'model_train_task_B',
        python_callable=airflow_train,
        op_args=['mlp_chargram', "model_data", "tfidf_chargram_*"],
        do_xcom_push= False
    )

    model_train_task_C= PythonOperator(
        task_id= 'model_train_task_C',
        python_callable=airflow_train,
        op_args=['mlp_ngram', "model_data" , "tfidf_ngram_*"],
        do_xcom_push= False
    )

    # Set task dependencies
    model_train_task_A >> model_train_task_B >> model_train_task_C 
