import numpy as np
import pandas as pd

# To process embeddings
import tensorflow as tf
import tensorflow_hub as hub

# To silence warnings from TensorFlow
import os
import logging
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# To load saved embeddings
import joblib

# To create webapp
import streamlit as st
import psutil

#######################################################################################
                            # Load functions
#######################################################################################

@st.cache
def load_embeddings():
    # Path to USE
    embed = hub.load('/media/einhard/Seagate Expansion Drive/3380_data/data/tensorflow_hub/universal-sentence-encoder_4')

    # Load pre-trained sentence arrays
    sentence_array = joblib.load('/media/einhard/Seagate Expansion Drive/3380_data/data/Models/reviewEmbeddings.pkl')
    descriptions_array = joblib.load('/media/einhard/Seagate Expansion Drive/3380_data/data/Models/descriptionEmbeddings.pkl')
    return embed, sentence_array, descriptions_array

@st.cache
def data_loader(datapath, books_file, reviews_file):
    books = pd.read_csv(datapath + books_file).drop('Unnamed: 0', axis=1)
    reviews = pd.read_csv(datapath + reviews_file).drop('Unnamed: 0', axis=1)
    return books, reviews


                    # Return recommendations based on reviews
@st.cache
def find_reviews(query,reviews, n_results=5):
    # Create vector from query and compare with global embedding
    sentence = [query]
    sentence_vector = np.array(embed(sentence))
    inner_product = np.inner(sentence_vector, sentence_array)[0]

    # Find sentences with highest inner products
    top_n_sentences = pd.Series(inner_product).nlargest(n_results+1)
    top_n_indices = top_n_sentences.index.tolist()
    top_n_list = list(reviews.review_text.iloc[top_n_indices][1:])

    #print(f'Input sentence: "{query}"\n')
    #print(f'{n_results} most semantically similar reviews: \n\n')
    #print(*top_n_list, sep='\n\n')
    #print(top_n_indices)
    return top_n_indices

@st.cache
def find_books(query, reviews, books, n_results=5):
    top_n_indices = find_reviews(query, reviews, n_results)
    return books[books.book_id.isin(reviews.iloc[top_n_indices].book_id.tolist())][['title', 'name','description', 'weighted_score']].fillna('').reset_index().drop('index', axis=1)


                    # Return recommendations based on descriptions

@st.cache
def find_description(query, books, n_results=10):
    # Create vector from query and compare with global embedding
    sentence = [query]
    sentence_vector = np.array(embed(sentence))
    inner_product = np.inner(sentence_vector, descriptions_array)[0]

    # Find sentences with highest inner products
    top_n_sentences = pd.Series(inner_product).nlargest(n_results)
    top_n_indices = top_n_sentences.index.tolist()
    top_n_list = list(books.description.iloc[top_n_indices][1:])

    #print(f'Input sentence: "{query}"\n')
    #print(f'{n_results} most semantically similar book descriptions: \n\n')
    #print(*top_n_list, sep='\n\n')
    return top_n_indices

@st.cache
def find_books_description(query, reviews, books):
    top_n_indices = find_description(query)
    return books[books.book_id.isin(books.iloc[top_n_indices].book_id.tolist())][['title', 'name','description', 'weighted_score']].fillna('')


                    # Return recommendations based on reviews

@st.cache
def load_sentences(input_books):
    '''
    Function to load and embed a book's sentences
    '''
    # Copy sentence column to new variable
    sentences = input_books['review_text']

    # Vectorize sentences
    sentence_vectors = embed(sentences)

    return sentences, sentence_vectors

#######################################################################################
                            # Load variables and data
#######################################################################################

# Path to books and reviews DataFrames
datapath = '/media/einhard/Seagate Expansion Drive/3380_data/data/Filtered books/'

# Books and reviews file names and loading
books_file = 'filtered_books.csv'
reviews_file = 'filtered_reviews.csv'

books, reviews = data_loader(datapath, books_file, reviews_file)
embed, sentence_array, descriptions_array = load_embeddings()


#######################################################################################
                                # Web App
#######################################################################################

st.title('3380')

st.sidebar.markdown(
    '''
    *Read books, not reviews!*
    '''
)
n_results = st.sidebar.slider('Select how many results to show',
                                1, 50, value=10, step=1)
sentence = st.text_input('Input your sentence here')
if sentence:
    st.dataframe(find_books(sentence, reviews=reviews, books=books, n_results=n_results-1))
    if psutil.virtual_memory()[2] > 50:
        st.write('Clearing cache to give you the best results. Hang on!')
        st.caching.clear_cache()