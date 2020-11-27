import numpy as np
import pandas as pd

# To process embeddings
import tensorflow_hub as hub

# To create sentence clusters
from sklearn.cluster import KMeans

# To load saved embeddings
import joblib

# To match strings
import re

# To create webapp
import psutil
import streamlit as st
st.set_page_config(layout="wide")
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
    books = pd.read_csv(datapath + books_file).drop('Unnamed: 0', axis=1).fillna('')
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


def findClusters(sentences, sentence_vectors, book_title, k, n_results):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(sentence_vectors)
    for i in range(k):
        centre = kmeans.cluster_centers_[i]
        ips = np.inner(centre, sentence_vectors)
        idx = pd.Series(ips).nlargest(n_results).index
        clusteredSentences = list(sentences.iloc[idx])

        '---'
        st.write(f'**Cluster #{i+1} reviews:** \n\n*Book title: {book_title}*')
        for sent in clusteredSentences:
            st.write(sent)

## UI and interactions

def display_results(idx, i, book_recommends, common_titles):

    '**---**'
    book_title = book_recommends[book_recommends.book_id == (reviews[reviews.   index == i].book_id.tolist()[0])].title.tolist()[0]
    common_titles.append(book_title)

    f'**Book title:**', book_title

    f'**Author:**', book_recommends[book_recommends.book_id == (reviews[reviews.    index == i].book_id.tolist()[0])].name.tolist()[0]

    f'**Weighted Score:**', str(round(books[books.book_id.isin(reviews[reviews. index == i].book_id.tolist())].weighted_score.tolist()[0], 2)), '/ 5'

    f'**Recommended because somebody wrote:**', reviews[reviews.index == i].    review_text.tolist()[0]

    return book_title


def Recommendations(sentence, reviews, books, n_results, n_clusters, n_cluster_reviews):
    with results:
        '''## Book recommendations based on your input sentence:'''
        ''' _(In no particular order)_'''
        top_n_indices, book_recommends = show_recommendations(sentence, reviews=reviews,    books=books, n_results=n_results-1)
        common_titles = []
        for idx, i in enumerate(reviews.iloc[top_n_indices].index):
            '**---**'
            book_title = display_results(idx,i, book_recommends, common_titles)

            button = st.button(label='Load review clusters for this book?', key=idx)
            if button:
                with clusters:
                    sentences, sentence_vectors = embedSentences(book_title)
                    findClusters(sentences, sentence_vectors,book_title, k=n_clusters, n_results=5)
            if links:
                good_reads_link = goodreadsURL + book_recommends[book_recommends.book_id == (reviews[reviews.index == i].book_id.tolist() [0])].for_url.tolist()[0].replace(r'\s', '\\')
                good_reads_link
    # code for sidebar most common tables. Shows table only if there are repeats in the main results
    most_common = Counter(common_titles).most_common(n_results)
    sidetable = pd.DataFrame(most_common, index=[x + 1 for x in range(len(most_common))]).rename(columns={0:'Title', 1:'No. of appearences ----->'})
    if sidetable.iloc[0]['No. of appearences ----->'] > 1:
        st.sidebar.write('Books that show up more than once given your input:')
        st.sidebar.table(sidetable[sidetable['No. of appearences ----->'] > 1])

# Search by book title
def searchBookTitles(sentence, reviews, books, n_clusters, n_cluster_reviews):
    book_title = books[books.title.str.contains(sentence, case=False)].title.tolist()
    with results:
        for idx, bookTitle in enumerate(book_title):
            info = books[books.title == bookTitle]
            '**---**'
            '**Book title:**', info.title.tolist()[0]
            '**Author:**', info.name.tolist()[0]
            '**Weighted Score**', str(round(info.weighted_score.tolist()[0], 2)), '/5'

            showClusters = st.button(label='Show review clusters for this book?', key=idx)
            if showClusters:
                with clusters:
                    sentences, sentence_vectors = embedSentences(bookTitle)
                    findClusters(sentences, sentence_vectors, bookTitle, k=n_clusters, n_results=n_cluster_reviews)
            if links:
                good_reads_link = goodreadsURL + '.'.join([info.book_id.astype(str).tolist()[0], info.title.replace(r'\(.*$', '', regex = True).tolist()[0]])
                good_reads_link

def searchAuthorNames(sentence, reviews, books, n_clusters, n_cluster_reviews):
    author_name = books[books.name.str.contains(sentence, case=False)].name.tolist()
    with results:
        for idx, authorName in enumerate(author_name):
            info = books[books.name == authorName]
            '**---**'
            '**Book title:**', info.title.tolist()[0]
            '**Author:**', info.name.tolist()[0]
            '**Weighted Score**', str(round(info.weighted_score.tolist()[0], 2)), '/5'

            showClusters = st.button(label='Show review clusters for this book?', key=idx)
            if showClusters:
                with clusters:
                    sentences, sentence_vectors = embedSentences(bookTitle)
                    findClusters(sentences, sentence_vectors, bookTitle, k=n_clusters, n_results=n_cluster_reviews)
            if links:
                good_reads_link = goodreadsURL + '.'.join([info.book_id.astype(str).tolist()[0], info.title.replace(r'\(.*$', '', regex = True).tolist()[0]])
                good_reads_link



#######################################################################################
                            # Load variables and data
#######################################################################################

# Path to books and reviews DataFrames
datapath = '/media/einhard/Seagate Expansion Drive/3380_data/data/Filtered books/'

# Books and reviews file names and loading
books_file = 'clean_filtered_books.csv'
reviews_file = 'clean_filtered_reviews.csv'
reviews_for_cluster = pd.read_csv('/media/einhard/Seagate Expansion Drive/3380_data/data/Processed/reviews_for_cluster.csv')

# Base URL for goodreads
goodreadsURL = 'https://www.goodreads.com/book/show/'

books, reviews = data_loader(datapath, books_file, reviews_file)
embed, sentence_array, descriptions_array = load_embeddings()


#######################################################################################
                                # Web App
#######################################################################################

'''# 3380 Books'''

st.sidebar.markdown(
    '''
    # 3380 Books
    *Read books, not reviews!*
    '''
)


links = st.sidebar.checkbox('Show Goodreads links.')
n_clusters = st.sidebar.slider('Select how many review clusters to generate for a book',
                                2,10,value=8,step=1)
n_cluster_reviews = st.sidebar.slider('Select how many reviews to show per cluster',
                                1,10,value=3,step=1)


sentence = st.text_input('Input')
results, clusters = st.beta_columns(2)

if re.match(r'title: ', sentence):
    sentence = sentence.replace('title: ', '')
    searchBookTitles(sentence,
                    reviews=reviews,
                    books=books,
                    n_clusters=n_clusters,
                    n_cluster_reviews=n_cluster_reviews)

elif re.match(r'author: ', sentence):
    sentence = sentence.replace('author: ', '')
    searchAuthorNames(sentence,
                    reviews=reviews,
                    books=books,
                    n_clusters=n_clusters,
                    n_cluster_reviews=n_cluster_reviews)


elif sentence or re.match(r'review: ', sentence):
    sentence = sentence.replace('review: ', '')
    n_results = st.sidebar.slider('Select how many book results to show',
                                1, 25, value=10, step=1)
    Recommendations(sentence,
                    reviews=reviews,
                    books=books,
                    n_results=n_results,
                    n_clusters=n_clusters,
                    n_cluster_reviews=n_cluster_reviews)


if psutil.virtual_memory()[2] > 70:
    st.write('Clearing cache to make sure things continue to run smoothly. Hang on!')
    caching.clear_cache()