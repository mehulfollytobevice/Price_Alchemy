#  this is the configuration file, all the settings like model hyper parameters, storage links are stored here
from price_alchemy import data_preprocessing as dp
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# this is GCP bucket URL where the data is saved
GCP_URL= 'https://storage.googleapis.com/price_alchemy/Data/data.csv'

# name of the preprocessed data pickle file
PREPROCESSED_DATA='preprocessed_data.pickle'

# number of samples to be used while preprocessing the data
NUM_SAMPLES=20000

# preprocessing settings
COL_TRANS_OPTS={
    "tfidf": ColumnTransformer([('categories', OrdinalEncoder(dtype='int'),['brand_name','parent_category', 'child_category', 'grandchild_category']),
                ('name', TfidfVectorizer(max_features=10000), 'name'),
                ('item_desc',TfidfVectorizer(max_features=10000),'item_description')
                ],
                remainder='passthrough',
                verbose_feature_names_out=True) ,
                
    "word_vector": ColumnTransformer([('categories', OrdinalEncoder(dtype='int'),['parent_category', 'child_category', 'grandchild_category']),
                ('text', dp.WordVectorTransformer(), 'item_description'),
                ('name',dp.WordVectorTransformer(),'name')],
                remainder='drop',
                verbose_feature_names_out=True)
}

TEXT_PREP_OPTS={
    "nltk": dp.text_preprocess_v1 ,
    "spacy": dp.text_preprocess_v2 
}
