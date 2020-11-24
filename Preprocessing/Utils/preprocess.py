import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import os.path
from pathlib import Path

# custom functions:
from preprocessFunctions import get_reviews, get_reviews, clean_books, get_books


directory = '/home/einhard/embeddingData/'
books_path = Path(directory + 'books_processed.csv')
reviews_path = Path(directory + 'reviewsEmbedding.csv')

if books_path.is_file():
    print('Books path found!')
    print('Loading books_df')
    books_df = pd.read_csv(books_path).drop('Unnamed: 0', axis=1)
    print(f'books_df loaded! \n books length: {len(books_df)}')
else:
    print('Books path not found, processing')
    books_df = get_books()
    books_df = clean_books(books_df)
    print('Processing complete, saving..')
    books_df.to_csv(directory + 'books_processed.csv')
    print('Saving success')

books_df = books_df.book_id

if reviews_path.is_file():
    print('Reviews path found!')
    reviews_df = pd.read_csv(directory + 'reviews_processed.csv')
else:
    print('Reviews path not found, processing')
    reviews_df = get_reviews(directory=directory, file_name='goodreads_reviews_dedup.json.gz', bite=10000, books_df=books_df, nrows=None)
    print('Processing complete, saving')
    reviews_df.to_csv(directory + 'reviewsEmbedding.csv')
    print('Saving success!')