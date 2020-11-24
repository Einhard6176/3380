import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import os.path
from pathlib import Path

directory = '/media/einhard/Seagate Expansion Drive/3380_data/data/Raw data/'

#######################################################################################
                                # Books processing
#######################################################################################

def process_df(df):
    print('Adding author')
    df['author'] = df['authors'].apply(lambda x: x[0].get('author_id'))
    print('Adding shelf 1')
    df['shelf_1'] = df['popular_shelves'].apply(lambda x: x[0].get('name'))
    print('Adding shelf 2')
    df['shelf_2'] = df['popular_shelves'].apply(lambda x: x[1].get('name'))
    return df

def english_books(df):
    print('Dropping languages')
    df = df[df.language_code.isin(['en', 'eng', 'en-US', 'en-GB'])]
    return df

def convert_book_dtypes(df):
    print('Convert data types')
    df['average_rating'] = df['average_rating'].astype('float32')
    df['ratings_count'] = df['ratings_count'].astype('int32')
    df['text_reviews_count'] = df['text_reviews_count'].astype('int32')
    return df

def remove_low_ratings(df, min_reviews=10):
    print('Filtering by reviews')
    df = df[df['ratings_count'] > min_reviews]
    return df

def clean_books(df):
    keep_cols = ['title','author','average_rating','ratings_count','text_reviews_count','description','shelf_1','book_id']
    df = df[keep_cols]
    df.description.fillna('', inplace=True)
    df.author.fillna('', inplace=True)
    df.shelf_1.fillna('', inplace=True)
    # df.shelf_2.fillna('', inplace=True)
    df.dropna(inplace=True)
    return df


def get_books(file_name='goodreads_books.json.gz', directory=directory, bite=10000, nrows=2000000):
    print('Function start')
    data = pd.read_json((os.path.join(directory, file_name)), lines=True,
                                                              chunksize=bite,
                                                              nrows=nrows,
                                                              compression='gzip')
    df = []
    keep_cols = ['title','author','average_rating','ratings_count','text_reviews_count','description','shelf_1', 'shelf_2', 'book_id', 'work_id']
    print('Loop start')
    for chunk in tqdm(data):
        filter_df = pd.DataFrame(chunk)
        print(f'DF length at loop start: {len(df)}')
        try:
            filter_df = english_books(filter_df)
            filter_df = convert_book_dtypes(filter_df)
            filter_df = remove_low_ratings(filter_df)
            filter_df = process_df(filter_df)
            filter_df = filter_df[keep_cols]
            print('append')
            df.append(filter_df)
            print(f'\n DF length at try end: {len(df)}')
            print('End loop')
        except Exception as e:
            print('\n======EXCEPTION=====')
            print(e)
            print('Except append')
            df.append(filter_df)
            print(f'\n Except DF length: {len(df)}')
            print('=====EXCEPTION END=====')
            continue
    print('Concatenating')
    print(len(df))
    df_concat = pd.concat(df)
    return df_concat


#######################################################################################
                                # Reviews processing
#######################################################################################

def convert_reviews_dtypes(df, dtype='int32'):
    columns = ['rating']
    df[columns] = df[columns].astype(dtype)
    return df

def get_reviews(books_df, file_name='goodreads_reviews_dedup.json.gz', directory=directory, bite=10000, nrows=15000000):
    print('Function start')
    data = pd.read_json((directory + file_name), lines=True,
                                                              chunksize=bite,
                                                              nrows=nrows,
                                                              compression='gzip',
                                                              convert_axes=True,
                                                              convert_dates=True)
    drop_cols = ['date_updated','read_at','started_at']
    df = []
    for chunk in tqdm(data):
        filter_df = pd.DataFrame(chunk).drop(drop_cols, axis=1)
        filter_df = filter_df[filter_df.book_id.isin(books_df)]
        filter_df = convert_reviews_dtypes(filter_df)
        df.append(filter_df)
        del filter_df
    print('Concatenating')
    print(len(df))
    df_concat = pd.concat(df)
    return df_concat


#############################################################################################
