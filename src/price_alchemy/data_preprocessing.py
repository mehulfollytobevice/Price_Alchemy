# import libraries
import pandas as pd
import numpy as np
import sqlite3
from google.cloud import storage 
from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from tqdm import tqdm
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import mysql.connector
from cred import MYSQL_PASSWORD

def load_data():

    connection = mysql.connector.connect(
        user='root', 
        password=MYSQL_PASSWORD, 
        host='34.30.80.103',
        database='mercari_db')

    cursor = connection.cursor()

    # Execute your query
    query1 = "SELECT * FROM product_listing"
    cursor.execute(query1)

    # Fetch column names
    column_names = [i[0] for i in cursor.description]

    # Create a DataFrame from the query result
    frame = pd.DataFrame(cursor.fetchall(), columns=column_names)

    # Close cursor and connection
    cursor.close()
    connection.close()

    return frame

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

def preprocess(data):
    
    corpus=[]
    ps=PorterStemmer()
    
    for i in tqdm(data):
        
        try:
            dis=re.sub(r'https?:\/\/.*[\r\n]*', '',i)  #removing hyperlinks
            dis=re.sub(r'http?:\/\/.*[\r\n]*', '',dis) #removing hyperlinks
            #dis=re.sub(r'\@[a-zA-Z0-9]\w+'," ",dis) # removing account mentions actually decreases the accuracy of the model 
            dis=re.sub('[^a-zA-Z]'," ",dis) #removing punctuation marks and numbers
            dis=dis.lower() # converting into lower case
            dis=dis.split() # splitting 
            dis=[ps.stem(word) for word in dis if not word in set(stopwords.words("english"))]  #stemming the words to trim down the number of words
            dis=' '.join(dis)
            corpus.append(dis)
            
        except:
            corpus.append(f"<BAD> {i}")
    
    return corpus

# MAIN PREPROCESSING FUNCTION

def data_prep_v1(df):

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

    # 9. select the columns
    m_df=m_df[['name','item_condition_id','brand_name',
            'parent_category','child_category','grandchild_category',
            'shipping','item_description','price']]

    # 10. process text columns
    # process name column
    raw_text= m_df['name'].to_list()
    data_final= process_name(raw_text)
    m_df['name']= data_final

    # process item_description column
    raw_text= m_df['item_description'].to_list()
    data_final= preprocess(raw_text)
    m_df['item_description']= data_final

    # 11. Apply column transformer and preprocessing methods
    # apply column transformer
    column_trans = ColumnTransformer([('categories', OrdinalEncoder(dtype='int'),['brand_name','parent_category', 'child_category', 'grandchild_category']),
                ('name', CountVectorizer(max_features=10000), 'name'),
                ('item_desc',TfidfVectorizer(max_features=10000),'item_description')
                ],
                remainder='passthrough',
                verbose_feature_names_out=True)
    

    # independent and dependent variable
    X=m_df.drop(columns=['price'])
    y=m_df['price']

    X= column_trans.fit_transform(X)

    return X, y


if __name__=="__main__":

    print('Loading data')
    # load the data from GCP SQL table
    df= load_data()
    print('Data loaded')
    print(df.head())

    print('Preprocess data')
    # preprocess the data 
    X,y= data_prep_v1(df)
    print('Data preprocessed')



