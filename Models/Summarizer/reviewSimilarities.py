import pandas as pd
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import joblib
from loader import data_loader

import warnings;
warnings.filterwarnings('ignore')


datapath = '/media/einhard/Seagate Expansion Drive/3380_data/data/'

print('Loading USE...')
embed = hub.load('/media/einhard/Seagate Expansion Drive/3380_data/tensorflow_hub/universal-sentence-encoder_4')
print('Success!')

print('Loading word embeddings')
sentence_array = joblib.load('/media/einhard/Seagate Expansion Drive/3380_data/3380-Book-Recommendations/Models/Summarizer/reviewEmbeddings.pkl')

print('Loading reviews and books')
books, reviews = data_loader(datapath, 'filtered_books.csv', 'filtered_reviews.csv')

def find_reviews(query,reviews=reviews, n_results=10):
    # Create vector from query and compare with global embedding
    sentence = [query]
    sentence_vector = np.array(embed(sentence))
    inner_product = np.inner(sentence_vector, sentence_array)[0]

    # Find sentences with highest inner products
    top_n_sentences = pd.Series(inner_product).nlargest(n_results+1)
    top_n_indices = top_n_sentences.index.tolist()
    top_n_list = list(reviews.review_text.iloc[top_n_indices][1:])

    print(f'Input sentence: "{query}"\n')
    print(f'{n_results} most semantically similar reviews: \n\n')
    print(*top_n_list, sep='\n\n')
    return top_n_indices

def find_books(query, reviews=reviews, books=books):
    top_n_indices = find_reviews(query)
    return books[books.book_id.isin(reviews.iloc[top_ten_indices].book_id.tolist())][['title', 'name','description', 'weighted_score']]

find_books("I've been waiting for a hard sci-fi novel for a long time")
