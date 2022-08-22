import sys
import pandas as pd
import numpy as np


def load_data(messages_filepath, categories_filepath):
    
    """
    Load two input csv files, load them in Pandas dataframes, and merge them in a single dataframe
    
    INPUT
        messages_filepath --> The path to the 'message.csv' file
        categories_filepath --> The path to the 'categories.csv' file
        
    OUTPUT
        df --> Pandas Dataframe created by merging the dataframes from the 'message.csv' and 'categories.csv' files
    """
    
    # Load the message and categories .csv files in two Pandas Dataframes
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge the two dataframes into one, by using the common column "id"
    df = pd.merge(messages, categories, on='id', how='outer')
    
    return df


def clean_data(df):
    
    """
    Take a Pandas Dataframe in input and performs data wrangling operations:
        - Split hte dataframe's column 'categories' into separate category columns
        - Convert category values to just numbers 0 or 1
        - Replace categories column in df with new category columns
        - Remove duplicates
    
    INPUT
        df --> Pandas Dataframe
        
    OUTPUT 
        df --> cleaned Pandas Dataframe
    
    """
    
    # The df column "categories" contains in each row, 36 emergency categories seperated by a semicolon
    
    # Created dataframe of the 36 individual category columns: 
    categories = df.categories.str.split(";", expand = True)
    
    # Select the first row of the categories dataframe
    # use this row to extract a list of new column names for categories.
    # rename the columns of `categories`
    row = categories.iloc[0].tolist()
    category_colnames = [col.replace('-1', '').replace('-0', '') for col in row]
    categories.columns = category_colnames
    
    # Iterate through the category columns in df to keep only the last character of each string (the 1 or 0).
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # Drop the value of the column "related" = 2
    categories = categories[categories.related != 2]
        
    # Drop the categories column from the df dataframe since it is no longer needed.
    df.drop('categories', inplace=True, axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filepath):
    
    """
    Save the clean dataset into an sqlite database
    
    INPUT
        df --> Pandas Dataframe to load in the database
        database_filename --> name of the database
        
    """
    
    from sqlalchemy import create_engine
    
    # Create the engine with the SQLAlchemy library 
    engine = create_engine('sqlite:///' + database_filepath)
    
    # Load df in the database
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')  


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
