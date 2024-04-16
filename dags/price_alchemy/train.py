# this file contains code for training a model
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix
import logging
import time
import price_alchemy.data_preprocessing as dp
import price_alchemy.config as cfg
import price_alchemy.logging_setup as ls
import price_alchemy.data_loading as dl
import price_alchemy.model_dispatcher as md
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error

def train_model(X, y, model, num_folds=5):

    if isinstance(X, (csr_matrix, csc_matrix)):
        
        # Convert sparse matrix to numpy array
        X = X.toarray()

    # kfold cross validation
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # metrics scores
    mse_scores=[]
    rmse_scores=[]
    r_squared_scores=[]
    rmsle_scores=[]

    for train_index, test_index in kf.split(X):

        # Split data into training and test sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Create and train the model  
        model.fit(X_train, y_train)
        
        # Make predictions on the test set
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, a_min=1e-6, a_max=None) 
        
        # Calculate mean squared error for the test set
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r_squared = r2_score(y_test, y_pred)
        rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))

        # add metrics to list
        mse_scores.append(mse)
        rmse_scores.append(rmse)
        r_squared_scores.append(r_squared)
        rmsle_scores.append(rmsle)


    # average metric scores
    average_mse = np.mean(mse_scores)
    average_rmse= np.mean(rmse_scores)
    average_r2= np.mean(r_squared_scores)
    average_rmsle= np.mean(rmsle_scores)

    metrics={"mse":average_mse,
            "rmse":average_rmse,
            "r_2":average_r2,
            "rmsle":average_rmsle
            }

    return model, metrics


if __name__=="__main__":

    # setup log configuration
    ls.log_setup()
    logging.info('RUNNING DATA PREPROCESSING')

    # load the data from GCP SQL table
    logging.info('Reading data')
    try:
        df= dl.load_data_sql('MYSQL_PASSWORD')
    except:
        df= dl.load_data_gcp(cfg.GCP_URL)

    # sample the dataset if it's larger than 10000
    if df.shape[0]>10000:
        logging.info('Sampling data since df_size > 10000.')
        df_sampled= dp.sample_df(df, sample_size=20000)
        sample= True
    else:
        df_sampled= df.copy()

    # preprocess the data 
    logging.info('Preprocessing started')
    start= time.time()
    X,y= dp.preprocessing_pipe( df_sampled, cfg.TEXT_PREP_OPTS['nltk'], cfg.COL_TRANS_OPTS['tfidf'])
    end= time.time()
    logging.info(f'Preprocessing complete. Total time taken:{end-start} seconds')
    logging.info(f'Preprocessed X shape:{X.shape}, y shape:{y.shape}')

    # train the model
    mdl= md.models['huber']
    model,metrics= train_model(X, y.values, model=mdl)
    logging.info(f"Model parameters:{mdl.get_params()} ")
    logging.info(f"Model metrics on the validation set:{metrics}")