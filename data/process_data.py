import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Input:
    messages_filepath: Filepath for the csv file containing messages. [string]
    categories_filepath: Filepath for the csv file containing categories. [string]

    Output:
    df: Dataframe containing messages and respective categories. [dataframe]

    Description:
    Loads datasets and merges into one dataframe.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, how='left', on='id')
    return df

def clean_data(df):
    """
    Input:
    df: Dataframe containing messages and respective categories. [dataframe]

    Output:
    df: Dataframe containing cleaned messages and categories. [dataframe]

    Description:
    Cleans dataframe with removing unneccesary columns, duplicates and text artifacts.
    """
    # expand categories to one column per category
    categories = df['categories'].astype(str).str.split(';',expand=True)

    # extract column names with slicing upto second to last character of each string
    categories.columns = [x[:-2] for x in categories.iloc[0]]
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype('int64')
        # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df.drop_duplicates(subset=['message', 'original'], inplace=True)
    # change category values to boolean. This will also convert values higher than 1 to "True"
    df.loc[:, ~df.columns.isin(['id', 'message', 'original', 'genre'])] = df.loc[:, ~df.columns.isin(['id', 'message', 'original', 'genre'])].astype('bool')
    return df

def save_data(df, database_filename):
    """
    Input:
    df: Dataframe containing cleaned messages and categories. [dataframe]
    database_filename: Filepath for the output database. [string]
    
    Output:
    None

    Description:
    Saves the cleaned data to database
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster', engine, index=False, if_exists='replace')  


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