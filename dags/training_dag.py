#  import essentials
from airflow import DAG
from airflow import configuration as conf
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from price_alchemy import config, model_dispatcher, train
import pickle 
import os

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

# utility function to train the model
def airflow_train(data, model_name):

    # seperate data 
    X, y= data

    # get model from dispatch
    model= model_dispatcher.models[model_name]

    # train model
    trained_model, metrics= train.train_model(X, y.values, model)

    return trained_model, metrics

# utility function to dump trained model into a pickle file
def dump_trained_model(model, filename):

    # what's the output path
    output_path = os.path.join( os.getcwd() ,"working_data" , filename)

    # dump the pickle data
    pickle.dump(model , open(output_path, 'wb'))

# Create DAG instance
with DAG('Training_pipeline', 
           default_args=default_args,
           description='This dag runs the end-to-end model training pipeline. ',
           schedule_interval='@weekly',
           catchup=False          
) as dag:


    # Task to load data, calls the 'load_data' Python function
    load_data_task = PythonOperator(
        task_id = 'load_data_task',
        python_callable=load_preprocessed_data,
        execution_timeout=timedelta(minutes= 2),
        op_kwargs={"filename":"tfidf_concat_data_sm.pickle"}
    )

    # Task to perform model training
    model_train_task= PythonOperator(
        task_id= 'model_train_task',
        python_callable=airflow_train,
        op_args=[load_data_task.output,'mlp']
    )

    # Task to save model 
    save_model_task= PythonOperator(
        task_id= 'save_model_task',
        python_callable= dump_trained_model,
        op_args=[model_train_task.output,"mlp.pickle"]
    )

    # data_validation_pandera_task = PythonOperator(
    #     task_id='data_validation_pandera_task',
    #     python_callable=data_validation.pandera_validate,
    #     op_kwargs={"df":load_data_task.output}
    # )

    # # sample a part of the dataframe
    # data_sampling_task = PythonOperator(
    #     task_id='data_sampling_task',
    #     python_callable=data_preprocessing.sample_df,
    #     op_kwargs={"df":load_data_task.output,"sample_size":config.TEST_SAMPLE_SIZE},
    # )

    # # Task to perform data preprocessing, depends on 'load_data_task'
    # data_preprocessing_task = PythonOperator(
    #     task_id='data_preprocessing_task',
    #     python_callable=data_preprocessing.preprocessing_pipe,
    #     op_kwargs={"df":data_sampling_task.output,"text_prep_func":config.TEXT_PREP_OPTS['nltk'], "column_trans":config.COL_TRANS_OPTS['tfidf_concat']},
    #     do_xcom_push= True
    # )

    # # Task to execute the 'separate_data_outputs' function
    # separate_data_outputs_task = PythonOperator(
    #     task_id='separate_data_outputs_task',
    #     python_callable=separate_data_outputs,
    #     provide_context=True
    #     )

    # # Task to save the preprocessed data
    # data_saving_task = PythonOperator(
    #     task_id='data_saving_task',
    #     python_callable=data_preprocessing.dump_preprocessed_data,
    #     op_args=[separate_data_outputs_task.output, "airflow_preprocessed.pkl"], 
    #     do_xcom_push= True
    # )


    # Set task dependencies
    load_data_task >> model_train_task >> save_model_task
