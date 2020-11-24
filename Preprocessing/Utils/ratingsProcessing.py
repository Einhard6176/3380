import pandas as pd
import numpy as np

def filter_ratings(df=ratings, min_book_ratings_count = 10, min_users_rating_count = 5):
    # Filtering sparse books
    filter_books = (df['book_id'].value_counts() > min_book_ratings_count)
    filter_books = filter_books[filter_books].index.tolist()

    # Filtering sparse users
    filter_users = (df['user_id'].value_counts() > min_users_rating_count)
    filter_users = filter_users[filter_users].index.tolist()

    # Filtering ratings data frame
    filtered_reviews = df[(df['book_id'].isin(filter_books)) & (df['user_id'].isin(filter_users))]
    del filter_books, filter_users
    return filtered_reviews

def create_train_split(df=filtered_reviews, test_size=100000):
    df_train = df[:-test_size]
    df_test = [-test_size:]
    return df_train, df_test
