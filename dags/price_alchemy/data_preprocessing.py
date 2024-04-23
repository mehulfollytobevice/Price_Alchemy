# import libraries
import pandas as pd
import numpy as np
import logging
import time
import os
from pathlib import Path
import pickle
from price_alchemy import logging_setup, data_loading, config

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import spacy
import re
from tqdm import tqdm

# HELPER FUNCTIONS

def split_cat(category_str):

    # split by '/'
    l=category_str.split('/')

    # return list
    l_ret= l[:2]
    
    # rest of the categories
    rest= l[2:]
    rest_cat= '/'.join(rest)

    # add rest of categories as one category
    l_ret.append(rest_cat)
           
    return l_ret

def process_name(data):

    corpus=[]
    
    for i in tqdm(data):
   
        dis=i.lower() # converting into lower case
        corpus.append(dis)

    return corpus

# text preprocessing function version 1
def text_preprocess_v1(data):
    
    corpus=[]
    ps=PorterStemmer()
    
    for i in tqdm(data):
        
        try:
            dis=re.sub(r'https?:\/\/.*[\r\n]*', '',i)  #removing hyperlinks
            dis=re.sub(r'http?:\/\/.*[\r\n]*', '',dis) #removing hyperlinks
            #dis=re.sub(r'\@[a-zA-Z0-9]\w+'," ",dis) # removing account mentions actually decreases the accuracy of the model 
            dis=re.sub('[^a-zA-Z]'," ",dis) #removing punctuation marks and numbers
            dis=dis.lower() # converting into lower case
            dis_list=dis.split() # splitting 
            dis_list=[ps.stem(word) for word in dis_list if not word in set(stopwords.words("english"))]  #stemming the words to trim down the number of words
            dis=' '.join(dis_list)
            corpus.append(dis)
            
        except:
            corpus.append(f"<BAD> {i}")
    
    return corpus

# text preprocessing version 2
def text_preprocess_v2(text) :

    # define list that will contain preprocessed text
    preprocessed= []

    # load the preprocessing pipeline
    nlp = spacy.load("en_core_web_sm")

    # pass data through the pipeline
    docs= nlp.pipe(text)
    #  ,n_process=4 )

    # apply rules on the data 
    for doc in docs:

        p=[]
        for tok in doc:

            # token should not be a digit
            if not tok.is_digit:

                if tok.is_sent_start:
                    p.append('<s>')
                    p.append(tok.lemma_)
                elif tok.is_sent_end:
                    if not tok.is_punct:
                        p.append(tok.lemma_)
                    p.append('</s>')
                else:

                    # should not be a punct mark
                    if not tok.is_punct:
                        p.append(tok.lemma_)
                    
            # if sentence starts with a digit
            else:
                if tok.is_sent_start:
                    p.append('<s>')
        
        # lower case all the words to avoid confusion
        p= [i.lower() for i in p]
        p_str=' '.join(p)
        preprocessed.append(p_str)

    return preprocessed


# sampling function
def sample_df(df: pd.DataFrame, sample_size: int, random_state: int = 42, replace: bool =  False):
    """
    This function is used to sample the dataset if len(df)>1000. 
    Since, preprocessing and training on a large dataset is quite expensive
    : param df:
    : param sample_size: number of instances in the returned sample
    : param random_state: random_state set to generate sample
    : param replacement: should samples be replaced back 
    : return:  df_sample, sampled dataset
    """
    
    df_sample= df.sample(n=sample_size, random_state=random_state,replace=replace)

    return df_sample

# function to perform basic data manipulation and preprocessing
def data_manipulation(df):

    # Preprocessing steps
    # 1. Remove rows with missing values in the 'price' column
    df['category_name'].replace('', np.nan, inplace=True)
    m_df=df.dropna(subset=['price','category_name'])

    # 2. Convert 'price' to numeric
    m_df['price'] = pd.to_numeric(m_df['price'], errors='coerce')

    # 3. Remove rows with price <= 0
    m_df = m_df[m_df['price'] > 0]

    # 4. Convert 'shipping' to categorical
    m_df['shipping'] = m_df['shipping'].astype('category')

    # 5. Convert 'item_condition_id' to categorical
    m_df['item_condition_id'] = m_df['item_condition_id'].astype('category')

    # 6. Drop created and updated at
    try:
        m_df = m_df.drop(columns=['created_at', 'last_updated_at'])
    except:
        pass

    # 7. fill null text values
    m_df['brand_name']=m_df['brand_name'].fillna('Not known')
    m_df['name']=m_df['name'].fillna('No name')
    m_df['item_description']=m_df['item_description'].fillna('No description yet')

    # 8. split hierarchical category into sub-categories
    m_df['category_split']= m_df['category_name'].apply(lambda x: split_cat(x))

    # category 
    m_df['parent_category']=m_df['category_split'].apply(lambda x: x[0])
    m_df['child_category']=m_df['category_split'].apply(lambda x: x[1])
    m_df['grandchild_category']=m_df['category_split'].apply(lambda x: x[2])

    # Concatenate the two columns
    m_df['text'] = m_df['name'].str.cat(m_df['item_description'], sep=' ')

    # # 9. select the columns
    # m_df=m_df[['name','item_condition_id','brand_name',
    #         'parent_category','child_category','grandchild_category',
    #         'shipping','item_description','price']]
    
    return m_df

# function to apply input column transform
def feature_transform(df, column_trans):

    # independent and dependent variable
    X=df.drop(columns=['price'])
    y=df['price']

    X= column_trans.fit_transform(X)

    return X, y

# MAIN PREPROCESSING FUNCTION
def preprocessing_pipe(df,text_prep_func, column_trans, if_airflow= False, filename= ''):

    # basic preprocessing on the data
    df= data_manipulation(df)

    # select the columns
    df=df[['item_condition_id','brand_name',
            'parent_category','child_category','grandchild_category',
            'shipping','text','price']]

    # preprocess text columns
    if text_prep_func=="version_1":
        process_text= text_preprocess_v1
    
    elif text_prep_func=="version_2":
        process_text= text_preprocess_v2

    # process the name column
    # raw_text= df['name'].to_list()
    # data_final= process_text(raw_text)
    # df['name']= data_final

    # # process item_description column
    # raw_text= df['item_description'].to_list()
    # data_final= process_text(raw_text)
    # df['item_description']= data_final

    # process the text column
    raw_text= df['text'].to_list()
    data_final= process_text(raw_text)
    df['text']= data_final

    # independent and dependent variable
    X=df.drop(columns=['price'])
    y=df['price']

    # transform the data using column transformer
    X= column_trans.fit_transform(X)

    # save the data
    if if_airflow:
        output_path = os.path.join(os.getcwd(), "working_data" , filename)
        
        data_req={'text_preprocessor': text_prep_func , "column_transformer": column_trans}

        # Save the trained model to a file
        pickle.dump(data_req , open(output_path, 'wb'))

    return X, y

# function to save the data
def dump_preprocessed_data(data,filename, if_airflow=False):
    
    # seperate X and y 
    X, y = data

    data={
        'X':X,
        'y':y
    }

    if if_airflow:

        # define the output path 
        output_path = os.path.join( os.getcwd(), "working_data" , filename)
    
    else:
        # what's the file path
        output_path = os.path.join( config.DATA_DIR , filename)

    # Save the trained model to a file
    pickle.dump(data , open(output_path, 'wb'))


if __name__=="__main__":

    # setup log configuration
    logging_setup.log_setup()
    logging.info('RUNNING DATA PREPROCESSING')

    # load the data from GCP SQL table
    logging.info('Reading data')
    try:
        df= data_loading.load_data_sql('MYSQL_PASSWORD')
    except:
        df= data_loading.load_data_gcp(config.GCP_URL)

    # sample the dataset if it's larger than 10000
    if df.shape[0]>10000:
        logging.info('Sampling data since df_size > 10000.')
        df_sampled= sample_df(df, sample_size=200)
        sample= True
    else:
        df_sampled= df.copy()

    # preprocess the data 
    logging.info('Preprocessing started')
    start= time.time()
    X,y= preprocessing_pipe( df_sampled, config.TEXT_PREP_OPTS['nltk'], config.COL_TRANS_OPTS['tfidf_concat'])
    end= time.time()
    logging.info(f'Preprocessing complete. Total time taken:{end-start} seconds')
    logging.info(f'Preprocessed X shape:{X.shape}, y shape:{y.shape}')

    # save data
    # filename= config.PREPROCESSED_DATA
    # if sample: 
    #     filename=  f"sample_" + filename
    
    # logging.info('Saving data.')
    # dump_preprocessed_data(X, y.values, filename)



