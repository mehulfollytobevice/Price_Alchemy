# Price Alchemy - Predicting suggested prices for sellers:

- This repository contains the code for **"Price Alchemy Project"**.
- Using this project, we can intelligently suggest optimal price points for a diverse array of products listed on Mercari.
- This project focuses on the application of MLOps and fundamental ML algorithms to accomplish the above mentioned task.
  
## 📝 Description
- Given the complexity and scale of online marketplaces today, accurately pricing products can be a daunting task for sellers. This project aims to develop an algorithm capable of automatically suggesting appropriate prices for products listed on Mercari, Japan's largest community-powered shopping app. The challenge underscores the difficulty of product pricing, which can vary significantly based on minute details, brand names, seasonal trends, and product specifications.
- Our approach for creating a price suggestion system is to develop an ensemble model that combines a NLP based model with a tabular model to predict the price of an item taking in multiple variables based on item category, brand, description, etc.
- In this project we have used the **Mercari Price Suggestion Challenge** to train our model. 

## ⏳ Dataset
- There are different versions of the Flickr Dataset. We use the Flickr8K dataset which has 8,000 images that are each paired with five different captions which provide clear descriptions of the salient entities and events.
- Download from here: https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
- Download the text descriptions: https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
- Download the dataset and place it in the main directory.

## 📝 Description of files:

- <b>main.py -</b> This is the main file of the project. It contains the code to train the image captioning model and save it into the '/data/' directory.
- <b>image_captioner.ipynb -</b> This file contains the code for our image captioning application. The application is built using ipywidgets and Voila. This notebook can be directly deployed onto Binder where it gets converted into an interactive application. For more information, see the next section.
- <b>requirements.txt -</b> This file contains the dependencies required to run the image captioning application (image_captioner.ipynb).

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

