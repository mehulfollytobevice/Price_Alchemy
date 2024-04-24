from fastapi import FastAPI, HTTPException
import os
import joblib
import numpy as np
from pydantic import BaseModel, conint, constr, PositiveFloat
import pandas as pd
from typing import Optional, List
import pickle
import sklearn
import logging
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import spacy
import re
from tqdm import tqdm
import glob

# read the most recent file 
def read_most_recent_file(folder, pattern):
    # Get list of files matching the pattern
    files = glob.glob(os.path.join(folder, pattern))
    # Sort files by modification time (most recent first)
    files.sort(key=os.path.getmtime, reverse=True)
    if files:
        most_recent_file = files[0]
        
    return most_recent_file

# load the models 
output_path = os.path.join( os.getcwd() ,"working_data", "models")
output_path = read_most_recent_file(output_path, "mlp_tfidf*")
with open(output_path, "rb") as f:
    # Load the pickled object
    m1 = pickle.load(f)

output_path = os.path.join( os.getcwd() ,"working_data", "models")
output_path = read_most_recent_file(output_path, "mlp_ngram*")
with open(output_path, "rb") as f:
    # Load the pickled object
    m2 = pickle.load(f)

output_path = os.path.join( os.getcwd() ,"working_data", "models")
output_path = read_most_recent_file(output_path, "mlp_chargram*")
with open(output_path, "rb") as f:
    # Load the pickled object
    m3 = pickle.load(f)


# load the preprocessors
output_path = os.path.join( os.getcwd() ,"working_data", "preprocessors")
output_path = read_most_recent_file(output_path, "tidf_concat*")
with open(output_path, "rb") as f:
    # Load the pickled object
    d1 = pickle.load(f)

output_path = os.path.join( os.getcwd() ,"working_data", "preprocessors")
output_path = read_most_recent_file(output_path, "tidf_ngram*")
with open(output_path, "rb") as f:
    # Load the pickled object
    d2 = pickle.load(f)

output_path = os.path.join( os.getcwd() ,"working_data", "preprocessors")
output_path = read_most_recent_file(output_path, "tidf_chargram*")
with open(output_path, "rb") as f:
    # Load the pickled object
    d3 = pickle.load(f)



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


# apply preprocessing and get prediction
def predict_out(df,model,preprocessor):

    tp=preprocessor['text_preprocessor']
    ct= preprocessor["column_transformer"]

    if tp=="version_1":
        process_text= text_preprocess_v1
    
    else:
        process_text= text_preprocess_v2

    # process the text column
    raw_text= df['text'].to_list()
    data_final= process_text(raw_text)
    df['text']= data_final

    # # transform
    X= ct.transform(df)

    # # get output
    pred= model.predict(X)

    return pred[0]


class InputData(BaseModel):
    train_id: int
    name: str
    item_condition_id: conint(ge=1, le=5)
    parent_category:str
    child_category:str 
    grandchild_category:str 
    brand_name: Optional[str]
    shipping: conint(ge=0, le=1)
    item_description: Optional[str]
    created_at: str
    last_updated_at: str


app = FastAPI()


@app.post("/predict/")
async def predict(data: InputData):
    try:
        input_data = data
        # Remove non-numeric columns and created_at, last_updated_at
        features = {key: [value] for key, value in input_data.dict().items() if key not in ["created_at", "last_updated_at"]}
        
        # Convert to array and reshape
        features_df= pd.DataFrame(data= features)

        # Some basic data manipulation
        features_df=features_df.fillna(0)
        features_df['text'] = features_df['name'].str.cat(features_df['item_description'], sep=' ')
        features_df=features_df[['item_condition_id','brand_name',
            'parent_category','child_category','grandchild_category',
            'shipping','text']]

        
        # Apply preprocessing
        p1= predict_out(features_df,m1,d1)
        p2= predict_out(features_df,m2,d2)
        p3= predict_out(features_df,m3,d3)

        # Make prediction using average
        prediction = sum([p1,p2,p3])/3
        
        return {"ensemble_prediction": prediction,
                "model1_prediction":p1,
                "model2_prediction":p2,
                "model3_prediction":p3
                }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/model_details")
async def model_details():
    try:
        # Get model details
        model_info = {"model1":{
            "model_type": str(type(m1)),
            "model_details": str(m1),
            "preprocessor": str(d1)
            },

            "model2":{
            "model_type": str(type(m2)),
            "model_details": str(m2),
            "preprocessor": str(d2)
            },
            "model3":{
            "model_type": str(type(m3)),
            "model_details": str(m3),
            "preprocessor": str(d3)
            }
            
        }
        
        return model_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
