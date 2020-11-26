import numpy as np
import pandas as pd

# To process embeddings
import tensorflow as tf
import tensorflow_hub as hub

# To create sentence clusters
from sklearn.cluster import KMeans

# To silence warnings from TensorFlow
import os
import logging
import warnings;
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# To load saved embeddings
import joblib

# To create webapp
import psutil
import streamlit as st
from streamlit import caching


# To sort final recommendation list
from collections import Counter


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


def find_books(query, reviews, books, n_results=5):
    top_n_indices = find_reviews(query, reviews, n_results)
    return books[books.book_id.isin(reviews.iloc[top_n_indices].book_id.tolist())][['title', 'name','description', 'weighted_score', 'book_id']].fillna('').reset_index().drop('index', axis=1)


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
    return books[books.book_id.isin(books.iloc[top_n_indices].book_id.tolist())][['title', 'name','description', 'weighted_score', 'book_id']].fillna('')

@st.cache
def show_recommendations(query, reviews, books, n_results=5):
    top_n_indices = find_reviews(query, reviews, n_results)
    book_recommends = find_books(query, reviews, books, n_results)
    book_recommends['for_url'] = book_recommends['book_id'].astype(str) + '.' + book_recommends['title'].replace(r'\(.*$', '', regex = True)
    return top_n_indices, book_recommends

# To find book clusters

@st.cache
def embedSentences(book_title):
    sentences = reviews_for_cluster[reviews_for_cluster.book_id.isin(books[books.title.isin([book_title])].book_id.tolist())]['review_text']
    sentence_vectors = embed(sentences)
    return sentences, sentence_vectors


def findClusters(sentences, sentence_vectors, k, n_results):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(sentence_vectors)
    clusters = pd.DataFrame()
    for i in range(k):
        centre = kmeans.cluster_centers_[i]
        ips = np.inner(centre, sentence_vectors)
        idx = pd.Series(ips).nlargest(n_results).index
        clusteredSentences = list(sentences.iloc[idx])
        clusters.append(sentences.iloc[idx])

        st.write(f'**Cluster #{i+1} sentences:**\n')
        for sent in clusteredSentences:
            '\t'
            st.write(sent)
            st.write('\n')


#######################################################################################
                            # Load variables and data
#######################################################################################

# Path to books and reviews DataFrames
datapath = '/media/einhard/Seagate Expansion Drive/3380_data/data/Filtered books/'

# Books and reviews file names and loading
books_file = 'filtered_books.csv'
reviews_file = 'filtered_reviews.csv'
reviews_for_cluster = pd.read_csv('/media/einhard/Seagate Expansion Drive/3380_data/data/Processed/reviews_for_cluster.csv')

books, reviews = data_loader(datapath, books_file, reviews_file)
embed, sentence_array, descriptions_array = load_embeddings()


#######################################################################################
                                # Web App
#######################################################################################

'''# 3380 Books'''

st.sidebar.markdown(
    '''
    *Read books, not reviews!*
    '''
)
n_results = st.sidebar.slider('Select how many results to show',
                                1, 50, value=10, step=1)
links = st.sidebar.checkbox('Show Goodreads links.')
sentence = st.text_input('Input')
if sentence:
    '''## Book recommendations based on your input sentence:'''
    ''' _(In no particular order)_'''
    #st.dataframe(find_books(sentence, reviews=reviews, books=books, n_results=n_results-1))
    top_n_indices, book_recommends = show_recommendations(sentence, reviews=reviews, books=books, n_results=n_results-1)
    common_titles = []

    for idx, i in enumerate(reviews.iloc[top_n_indices].index):
        '**---**'

        book_title = book_recommends[book_recommends.book_id == (reviews[reviews.index == i].book_id.tolist()[0])].title.tolist()[0]
        common_titles.append(book_title)

        f'**Book title:**', book_title

        f'**Author:**', book_recommends[book_recommends.book_id == (reviews[reviews.index == i].book_id.tolist()[0])].name.tolist()[0]

        f'**Weighted Score:**', str(round(books[books.book_id.isin(reviews[reviews.index == i].book_id.tolist())].weighted_score.tolist()[0], 2)), '/ 5'

        f'**Recommended because somebody wrote:**', reviews[reviews.index == i].review_text.tolist()[0]

        button = st.button(label='Load review clusters for this book?', key=idx)
        if button:
            n_clusters = st.sidebar.slider('Select how many clusters to create',
                                    3, 10, value=8, step=1)
            n_sentences = st.sidebar.slider('Select how many sentences per cluster to show',
                                    1, 10, value=5, step=1)
            sentences, sentence_vectors = embedSentences(book_title)
            findClusters(sentences, sentence_vectors, k=n_clusters, n_results=n_sentences)

        if links:
            good_reads_link = 'https://www.goodreads.com/book/show/' + book_recommends[book_recommends.book_id == (reviews[reviews.index == i].book_id.tolist()[0])].for_url.tolist()[0].replace('\s', '\\')
            good_reads_link

    # code for sidebar most common tables. Shows table only if there are repeats in the main results
    most_common = Counter(common_titles).most_common(n_results)
    sidetable = pd.DataFrame(most_common, index=[x + 1 for x in range(len(most_common))]).rename(columns={0:'Title', 1:'No. of appearences ----->'})

    if sidetable.iloc[0]['No. of appearences ----->'] > 1:
        st.sidebar.write('Books that show up more than once given your input:')
        st.sidebar.table(sidetable[sidetable['No. of appearences ----->'] > 1])

if psutil.virtual_memory()[2] > 70:
    caching.clear_cache()