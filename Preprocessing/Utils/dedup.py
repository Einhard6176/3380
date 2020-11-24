import numpy as np
import pandas as pd

def remove_dups(books, reviews, sort_column='weighted_score', drop_column='title' ):
    '''
    Sorts values by highest weighted score and returns dataframe with lower scored duplicates removed.
    Then applies filtering for reviews based on books left over in main books dataframe

    Args:
    books = DataFrame with book metadata
    reviews = DataFrame with user, book_id, and review_counts
    '''
    books = books.sort_values(sort_column, ascending=False).drop_duplicates(drop_column, keep='first')
    reviews = reviews[reviews.book_id.isin(books.book_id)]
    return books, reviews
