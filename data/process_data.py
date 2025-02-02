import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    # load and merge datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories)

    return df


def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(";", expand=True)

    # extract and assign column labels
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2]).tolist()
    categories.columns = category_colnames

    # extract number from category values
    for col in category_colnames:
        categories[col] = categories[col].str.replace(col, "").str.replace("-", "")
        categories[col] = categories[col].astype(int)

    # replace original categories column
    df.drop("categories", axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # Drop the column with the original message as all analyses will be based on the English version.
    df.drop("original", axis=1, inplace=True)

    # remove duplicate rows and rows containing undefined values
    df.drop(index=df[(df[category_colnames].values != 0) & (df[category_colnames].values != 1)].index,
            inplace=True)

    return df


def save_data(df, database_filename):
    engine = create_engine("sqlite:///" + database_filename)
    df.to_sql("messages", engine, index=False, if_exists="replace")


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
