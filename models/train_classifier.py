import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.tokenize import  word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import PorterStemmer
import pickle


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    
    '''
    Loads the table from Response located in the SQLite database
    
    Input:
    
    database_filepath: filepath to the SQLite database
    
    Output:
    
    X: data of texts
    Y: labels of the categories
    category_names: list of categories names
        
    '''
    
    # Load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Responses_', engine)
    
    #Assign values to X, Y and category names
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns.values

    return X, Y, category_names


def tokenize(text):
    
    '''
    This function first sets all the text in lower case, second removes punctuation, third tokenizes the text,
    then remove stop words and finally does lemmatization
    
    Input:
    
    text: string of text
    
    Output:
    
    words: final list of tokenizes and clean words
    
    '''
    
    # Lower case
    text = text.lower()
    
    # Removes puntuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    
    # Tokenizes
    words = word_tokenize(text)
    
    # Removes stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    
    # Lemmatizes
    words = [WordNetLemmatizer().lemmatize(w) for w in words]

    return words



def build_model():
    
    '''
    Builds the pipeline of the model
    
    Input:
    
    NA
    
    Output:
    
    cv: grid search of pipeline
    
    '''
    
    # Build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    # Grid of parameters
    parameters = {
        'clf__estimator__max_depth': [50, 100, 200],
        'clf__estimator__n_estimators': [2, 5, 10]}

    # Grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    
    return cv 



def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
    Evaluates model on test data
    
    Input:
    
    model: fitted model
    X_test: test features (messages)
    Y_test: test labels
    category_names: names of the categories
        
    Output:
    
    Print results for each category
    
    '''
    
    # Calculates predictions
    y_prediction_test = model.predict(X_test)
    
    print(classification_report(Y_test.values, y_prediction_test, target_names=category_names))


def save_model(model, model_filepath):
    
    '''
    Save model as pickle file
    
    Input:
        
    model: trained model
    model_filepath: path and file name to stor pickle   
    
    Return:
    
    NA
    
    '''
    
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()