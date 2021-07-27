# Disaster-Response-Pipelines

### Table of Contents

1. [Project Motivation](#motivation)
2. [File Descriptions](#files)
3. [How to Run](#run)


## Project Motivation <a name="motivation"></a>

In this project, it was carried out based on data from Figure Eight. 
The main objective was build a model for an API that classifies disaster real messages, the model was a Random Forest Classifier.
Through a Machine Learning Pipeline the messages events was categorized in order to sent to an appropriate disaster agency.

In addition, there is a web app where messages can get classification in several categories.


## File Descriptions <a name="files"></a>

1. app - This folder contains the "run.py" file that builds the web app and another "templates" folder that contains the html files where the web app is programmed.

2. data - This folder contains two csv files and python file "process_data.py". 

- The file "disaster_messages.csv" has the information of the text messages.
- The file "disaster_categories.csv" has the classification labels of the messages. 
- The python file is where the information is extract, transform, clean and load in a SQLite database. 

3. models - This folder contains a python file "train_classifier.py" where is created a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories. Then the model was export to a pickle file.


## How to Run <a name="run"></a>

In order to develop the project you must run the following commands in the project's root directory:

- To run ETL pipeline that cleans data and stores in database:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
- To run ML pipeline that trains classifier and saves:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        
 - To run the web app run the following command in the app's directory:
         `python run.py`
  
 - Finally go to http://0.0.0.0:3001/



