import pandas as pd
import numpy as np
import re

def filter_books(books_df, min_text_reviews= 300):
    books_df.description.dropna(inplace=True)
    books_df = books_df[books_df.text_reviews_count > min_text_reviews]
    return books_df

def filter_reviews(reviews_df, books_df, min_review_length=30, max_review_length=90):
    reviews_df.review_text.dropna(inplace=True)
    reviews_df = reviews_df[reviews_df.book_id.isin(books_df.book_id)]
    reviews_df = reviews_df[(reviews_df.review_text.str.len() < max_review_length) & (reviews_df.review_text.str.len() > min_review_length)]

    # Remove reviews that specify rating in body of text
    reviews_df = reviews_df[~reviews_df.review_text.str.contains(r'\d\sstar', case=False, regex=True, na=False)]
    reviews_df = reviews_df[~reviews_df.review_text.str.contains(r'\d\/\d', case=False, regex=True, na=False)]
    reviews_df = reviews_df[~reviews_df.review_text.str.contains(r'\d\.\d', case=False, regex=True, na=False)]
    reviews_df = reviews_df[~reviews_df.review_text.str.contains(r'\d\S+\D', case=False, regex=True, na=False)]
    reviews_df = reviews_df[~reviews_df.review_text.str.contains(r'spoiler', case=False, regex=True, na=False)]

    # Remove reviews that contain link to other websites
    reviews_df = reviews_df[~reviews_df.review_text.str.contains(r'http\S+', case=False, regex=True, na=False)]
    reviews_df = reviews_df[~reviews_df.review_text.str.contains(r'www\S+', case=False, regex=True, na=False)]
    return reviews_df
