#  this is the configuration file, all the settings like model hyper parameters, storage links are stored here
import numpy as np
import spacy
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

# this is GCP bucket URL where the data is saved
GCP_URL= 'https://storage.googleapis.com/price_alchemy/Data/data.csv'

# name of the preprocessed data pickle file
PREPROCESSED_DATA='preprocessed_data.pickle'
DATA_DIR=r"/Users/mehuljain/Documents/course_related/ML_Ops/project/Price_Alchemy/data"

# number of samples to be used while preprocessing the data
NUM_SAMPLES=20000

# test sample size
TEST_SAMPLE_SIZE=2000

# preprocessing settings
# function to convert text to vectors
class WordVectorTransformer(TransformerMixin,BaseEstimator):
    def __init__(self, model="en_core_web_lg"):
        self.model = model

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        nlp = spacy.load(self.model)
        return np.concatenate([nlp(doc).vector.reshape((1,-1)) for doc in X])


COL_TRANS_OPTS={
    
    "tfidf_concat":ColumnTransformer([('categories', OneHotEncoder(dtype='int'),['brand_name','parent_category', 'child_category', 'grandchild_category']),
                ('vectorizer', TfidfVectorizer(max_features=10000), 'text'),
                ],
                remainder='passthrough',
                verbose_feature_names_out=True),

    'tfidf_full':  ColumnTransformer([
                ('categories', OneHotEncoder(dtype='int'),['brand_name','parent_category', 'child_category', 'grandchild_category']),
                ('vectorizer', TfidfVectorizer(), 'text'),
                ],
                remainder='passthrough',
                verbose_feature_names_out=True),
    
    "tfidf_bigram":ColumnTransformer([('categories', OneHotEncoder(dtype='int'),['brand_name','parent_category', 'child_category', 'grandchild_category']),
                ('vectorizer', TfidfVectorizer(ngram_range=(2,2), max_features=10000), 'text'),
                ],
                remainder='passthrough',
                verbose_feature_names_out=True),

    "tfidf_ngram":ColumnTransformer([('categories', OneHotEncoder(dtype='int'),['brand_name','parent_category', 'child_category', 'grandchild_category']),
                ('vectorizer', TfidfVectorizer(ngram_range=(1,3), max_features=20000), 'text'),
                ],
                remainder='passthrough',
                verbose_feature_names_out=True),

    "tfidf_chargram": ColumnTransformer([('categories', OneHotEncoder(dtype='int'),['brand_name','parent_category', 'child_category', 'grandchild_category']),
                ('vectorizer', TfidfVectorizer(analyzer='char',ngram_range=(3,8), max_features=20000), 'text'),
                ],
                remainder='passthrough',
                verbose_feature_names_out=True),

    "word_vector": ColumnTransformer([('categories', OneHotEncoder(dtype='int'),['brand_name','parent_category', 'child_category', 'grandchild_category']),
                ('vectorizer', WordVectorTransformer(), 'text')],
                remainder='passthrough',
                verbose_feature_names_out=True),
}

TEXT_PREP_OPTS={
    "nltk": "version_1" ,
    "spacy": "version_2"
}

