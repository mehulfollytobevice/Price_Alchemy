# Price Alchemy: 
> ### End-to-End Price Prediction System

![Tests](https://github.com/mehulfollytobevice/Price_Alchemy/actions/workflows/tests.yml/badge.svg)

- This repository contains the code for **"Price Alchemy Project"**.
- Using this project, we can intelligently suggest optimal price points for a diverse array of products listed on Mercari.
- This project focuses on the application of MLOps and fundamental ML algorithms to accomplish the above mentioned task.
  
## 📝 Description
- Given the complexity and scale of online marketplaces today, accurately pricing products can be a daunting task for sellers. This project aims to develop an algorithm capable of automatically suggesting appropriate prices for products listed on Mercari, Japan's largest community-powered shopping app. The challenge underscores the difficulty of product pricing, which can vary significantly based on minute details, brand names, seasonal trends, and product specifications.
- Our approach for creating a price suggestion system is to develop an ensemble model that combines a NLP based model with a tabular model to predict the price of an item taking in multiple variables based on item category, brand, description, etc.
- In this project we have used the **Mercari Price Suggestion Challenge** to train our model. 

## ⏳ Dataset
The dataset provided consists of information on product listings, designed for a predictive modeling. Here's an overview based on the provided information:

1. **Data Format**: The dataset is split into two files, `train.tsv` and `test.tsv`, both in a tab-delimited format.

2. **Data Fields**:
   - `train_id` or `test_id`: Unique identifiers for each listing.
   - `name`: The title of the listing. Text resembling prices has been removed to prevent data leakage.
   - `item_condition_id`: Indicates the condition of the item as provided by the seller.
   - `category_name`: Specifies the category of the listing.
   - `brand_name`: Denotes the brand of the item, if available.
   - `price`: The target variable representing the sale price of the item in USD. Present only in the training dataset.
   - `shipping`: Binary variable indicating whether the shipping fee is paid by the seller (1) or the buyer (0).
   - `item_description`: Provides a full description of the item. Text resembling prices has been removed to prevent data leakage.

3. **Target Variable**: The objective of the competition is to predict the sale price (`price`) of the listings using the provided features.

4. **Features**: Features available for prediction include the listing's title (`name`), item condition (`item_condition_id`), category (`category_name`), brand (`brand_name`), shipping information (`shipping`), and item description (`item_description`).

5. **Evaluation**: Model performance is likely assessed using metrics such as Root Mean Squared Log Error (RMSLE) 

6. **Leakage Prevention**: Measures have been taken to remove text resembling prices from both the listing title and item description to prevent the model from inadvertently learning from the target variable during training.

## 📝 Description of files:

- __data/:__ This directory contains subdirectories for raw and processed data. Raw data files (train.tsv and test.tsv) are stored in the raw/ subdirectory, while processed data files (train.csv and test.csv) are stored in the processed/ subdirectory. Intermediate data files generated during data preprocessing can be stored in the interim/ subdirectory if necessary.

- __notebooks/:__ Jupyter notebooks for exploratory data analysis, data preprocessing, model development, and evaluation are stored in this directory.

- __dags/:__ This directory contains the source code for Airflow DAGs that aim to automate various aspects of the machine learning pipeline developed in the project. Furthermore, it contains the source code for the `price_alchemy` module which provides classes and functions for managing different parts of the ML workflow, including data loading, data validation and data preprocessing, modeling, and model evaluation.

- __tests/:__ This directory contains unit tests, integration tests, or any other tests relevant to the project. 

- __models/:__ Saved model files generated during model training are stored in this directory.

- __requirements.txt:__ A text file listing the Python package dependencies required to run the project.

- __README.md:__ Documentation providing an overview of the project, instructions for setup and usage, and any additional information relevant to users and contributors.

## :hammer_and_wrench: Requirements
* Python 3.5+
* tensorflow
* MLFlow
* ELK
* DVC
* Flask API
* Airflow
* pillow<7
* packaging
* ipywidgets==7.5.1
* Linux

## Contributors <img src="https://raw.githubusercontent.com/TheDudeThatCode/TheDudeThatCode/master/Assets/Developer.gif" width=35 height=25> 
-	Aryan Deore
-	Bishal Agrawal 
-	Mehul Jain 
-	Priyanka Dipak Gujar
-	Rakesh Rathod

