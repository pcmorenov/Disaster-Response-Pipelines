import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    '''
    First imports data from cvs filepaths and then merges them into a single dataframe
    
    Input:    
    messages_filepath: filepath of messages cvs
    categories_filepath: filepath of categories cvs
        
    Output:    
    df: pandas dataframe that contains the two cvs merged
    
    '''
    
    #Loads data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #Merges datasets
    df = messages.merge(categories, on="id")
    
    return df


def clean_data(df):
    
    '''
    Cleans the dataframe for the ML pipeline      
    
    Input:
    
    df: dirty pandas dataframe of messages and categories 
    
    Output:
    
    df: clean pandas dataframe
    
    '''

    # Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand=True)
    
    # Select the first row of the categories dataframe
    frow = categories.iloc[0]
    
    # Use this row to extract a list of new column names for categories.
    category_names = frow.apply(lambda x: x[:-2])
    
    # Rename the columns of `categories`
    categories.columns = category_names
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # 2 will be converted to 0
    categories = categories.replace(2, 0)
    
    # The child alone column is nos used
    categories.drop("child_alone", axis=1, inplace=True)
    
    # Drop the original categories column from `df`
    # Concatenate the original dataframe with the new `categories` dataframe
    df.drop("categories", axis=1, inplace=True)
    df = pd.concat([df,categories], axis=1)

    # Drop duplicates
    df=df.drop_duplicates()

    return df


def save_data(df, database_filename):
    
    '''
    Saves the cleaned data in SQlite database
    
    Input:
    df: cleaned pandas dataframe with data of messages and categories
    database_filename: name of the output database
    
    Output:
    NA
    
    '''
    
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('Responses_', engine, index=False, if_exists='replace')  
    
    pass


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()