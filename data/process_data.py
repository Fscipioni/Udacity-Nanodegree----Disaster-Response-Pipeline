import sys


def load_data(messages_filepath, categories_filepath):
    
    # Load the message and categories .csv files in two Pandas Dataframes
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge the two dataframes into one, by using the common column "id"
    df = pd.merge(messages, categories, on='id', how='outer')
    
    return df


def clean_data(df):
    
    # The df column "categories" contains in each row, 36 emergency categories seperated by a semicolon
    
    # create a dataframe of the 36 individual category columns 
    categories = df.categories.str.split(";", expand = True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0].tolist()
    
    # use this row to extract a list of new column names for categories.
    category_colnames = [col.replace('-1', '').replace('-0', '') for col in row]
    
    # rename the columns of `categories`
    categories.columns = category_colnames


def save_data(df, database_filename):
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