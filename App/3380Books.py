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
                            # Functions
#######################################################################################

#################### Data Loading  Functions ####################

@st.cache
def dataLoader(datapath, books_file, reviews_file, reviewsAll_file):
    '''
    Loads DataFrames with books and review information
    '''
    books = pd.read_csv(datapath + books_file).drop('Unnamed: 0', axis=1).fillna('')
    reviews = pd.read_csv(datapath + reviews_file).drop('Unnamed: 0', axis=1)
    reviewsAll = pd.read_csv(datapath + reviewsAll_file).drop('Unnamed: 0', axis=1)
    return books, reviews, reviewsAll

@st.cache
def loadEmbeddings():
    '''
    Loads pre-trained sentence and review arrays
    '''
    # Path to USE
    embed = hub.load('/media/einhard/Seagate Expansion Drive/3380_data/data/tensorflow_hub/universal-sentence-encoder_4')

    # Load pre-trained sentence arrays
    ## Reviews array is a set of embeddings trained on review lengths of < 90 characters
    reviews_array = joblib.load('/media/einhard/Seagate Expansion Drive/3380_data/data/Models/reviewEmbeddings.pkl')
    ## Descriptions array is a set of embeddings trained on all book descriptions
    descriptions_array = joblib.load('/media/einhard/Seagate Expansion Drive/3380_data/data/Models/descriptionEmbeddings.pkl')

    return embed, reviews_array, descriptions_array

#################### Basic Clustering Functionality ####################

@st.cache
def embedInputs(books_df, review_df, search_param, review_max_len, searchTitle=True):
    '''
    Converts input reviews into USE arrays. Returns vectorized reviews for the book that was
    passed as bookTitle.

    Args:
        search_param = List of book titles or author names whose reviews we want to embed. For authors, see 'searchTitle' argument
        books_df = DataFrame with book_id information
        review_df = DataFrame with book_id and review text
        searchTitle = If True, will search for book_id based on title of a book. If False, it will look for author names to find book_id.
    '''
    if searchTitle:
        #Finds book_id from title of book
        input_book_id = books_df[books_df.title.isin([search_param])].book_id.tolist()
    else:
        # Finds book_id from author name
        input_book_id = books_df[books_df.name.isin([search_param])].book_id.tolist()

    # Finds reviews for specified book
    input_sentences = review_df[review_df.book_id.isin(input_book_id)].review_text

    # Filters review length
    input_sentences = input_sentences[input_sentences.str.len() <= review_max_len]

    # Converts reviews into 512-dimensional arrays
    input_vectors = embed(input_sentences)

    # Returns reviews and vectorized reviews for a particular book
    return input_sentences, input_vectors

# Pass "sentence_array" as input_vectors to compare with user input???????
@st.cache
def getClusters(input_vectors, n_clusters):
    '''
    Creates KMeans instance and fits model.

    The for nested for loop is used to display the returned sentences on Streamlit.

    Args:
        input_sentences =  Sentences to compare
        n_clusters = How many clusters to generate
    '''
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, algorithm='full')
    return kmeans.fit(input_vectors)


def showClusters(input_sentences, input_vectors, authorTitle, n_clusters, n_results, model, searchTitle=True):
    '''
    This function will find theme clusters in the reviews of a particular book or set of sentences.
    Uses cluster centers to find semantically similar sentences to the input vectors.

    The nested for loop is used to display the returned sentences on Streamlit.

    Args:
        input_sentences =  Sentences to compare
        input_vectors = USE Array generated by embedding input sentences
        authorTitle = Title of book in question, or name of author --> Used to display header only
        n_clusters = How many clusters to generate
        n_results = How many sentences to display per n_cluster
        model = The model used to create the clusters.
    '''
    if searchTitle:
        # Displays which book's reviews are being clustered
        st.header(f'Opinion clusters about *{authorTitle}*')
    else:
        st.header(f'Opinion clusters about {authorTitle}\'s books')

    # Iterates through centroids to and computes inner products to find nlargest
    for i in range(n_clusters):
        centre = model.cluster_centers_[i]
        inner_product = np.inner(centre, input_vectors)
        indices = pd.Series(inner_product).nlargest(n_results).index
        clusteredInputs = list(input_sentences.iloc[indices])

        # Prints reviews that are closest to centroid
        st.markdown('---')
        st.write(f'**Cluster #{i+1}**')
        for sentence in clusteredInputs:
            st.write(sentence)



#################### Specific Clustering Functionality ####################

# Search by book title (used with the prefix 'title:' in input)
def searchBookTitles(input_text, reviews, books, n_clusters, n_cluster_reviews):
    pass


#################### App UI and Interactions ####################


def showInfo(iterator, n_clusters, n_results,n_books, review_max_len):
    with results:
        for idx, i in enumerate(iterator[:n_books]):
            try:
                info = books[books.title == i]
                '**---**'
                '**Book title:**', info.title.tolist()[0]
                '**Author:**', info.name.tolist()[0]
                '**Weighted Score**', str(round(info.weighted_score.tolist()[0], 2)), '/ 5'

                showReviewClusters = st.button(label='Show opinion clusters for this book?', key=idx)
                showAuthorClusters = st.button(label='Show opinion clusters for this author?', key=idx+100)
            except IndexError:
                break

            if showReviewClusters:
                with clusters:
                    try:
                        input_sentences, input_vectors = embedInputs(books,
                                                                    reviewsAll,
                                                                    search_param=info.title.tolist()[0],
                                                                    review_max_len=review_max_len,
                                                                    searchTitle=True)
                        model = getClusters(input_vectors=input_vectors,
                                            n_clusters=n_clusters)
                        showClusters(input_sentences=input_sentences,
                                    input_vectors=input_vectors,
                                    authorTitle = info.title.tolist()[0],
                                    n_clusters=n_clusters,
                                    n_results=n_results,
                                    model=model,
                                    searchTitle=True)
                    except ValueError:
                        st.warning(f"It looks like this book doesn't have enough reviews to generate {n_clusters} distinct themes. Try decreasing how many themes you look for!")
                        continue
            if showAuthorClusters:
                with clusters:
                    try:
                        input_sentences, input_vectors = embedInputs(books,
                                                                    reviewsAll,
                                                                    search_param=info.name.tolist()[0],
                                                                    review_max_len=review_max_len,
                                                                    searchTitle=False)
                        model = getClusters(input_vectors=input_vectors,
                                            n_clusters=n_clusters)
                        showClusters(input_sentences=input_sentences,
                                    input_vectors=input_vectors,
                                    authorTitle = info.name.tolist()[0],
                                    n_clusters=n_clusters,
                                    n_results=n_results,
                                    model=model,
                                    searchTitle=False)
                    except ValueError:
                        st.warning(f"It looks like this author's books don't have enough reviews to generate {n_clusters} distinct themes. Try decreasing how many themes you look for!")
                        continue
            if goodreadsLink:
                good_reads_link = goodreadsURL + info.book_id.astype(str).tolist()[0]
                st.write(f'*Goodreads Link: {good_reads_link}*')
#######################################################################################
                            # Load variables and data
#######################################################################################

# Paths to books and reviews DataFrames
datapath = '/media/einhard/Seagate Expansion Drive/3380_data/data/Filtered books/'
books_file = 'clean_filtered_books.csv'
reviews_file = 'clean_filtered_reviews.csv'
reviewsAll_file = 'reviews_for_cluster.csv'

# Loading DataFrames
books, reviews, reviewsAll = dataLoader(datapath, books_file, reviews_file, reviewsAll_file)

# Loadding pre-trained embeddings and embedder for input sentences
embed, reviews_array, descriptions_array = loadEmbeddings()

# Setting base URL for Goodreads
goodreadsURL = 'https://www.goodreads.com/book/show/'


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

# Creating options dropdown menu in sidebar
options = st.sidebar.beta_expander('Options')
with options:
    goodreadsLink = st.checkbox('Show Goodreads links.')
    n_clusters = st.slider('Select how many themes to search for in the reviews',
                                2,10,value=3,step=1)
    n_results = st.slider('Select how many reviews to show per theme',
                                1,10,value=3,step=1)
    n_books = st.slider('Select how many book results to show',
                                1, 25, value=10, step=1)
    review_max_len = st.slider('Select maximum review length each theme group',
                                50, 350, value=80, step=10)

# Asking for user input
input_text = st.text_input('Try specifying `author:` or `title:` for more specific results')

# Creating columns for book results on the left, review clusters on the right
results, clusters = st.beta_columns(2)

# Welcome message
if not input_text:
    '''
        ### How this works:
        In the search bar above, you can type in any sentence that describes a book you'd like to read.

        The machine learning algorithm will then look through a database of books and reviews to find the most appropriate recommendations for you based on how other people review the books they read (to be more specific, it will find the most semantically similar reviews).

        If you want to explore further, you can generate thematically linked clusters for any book. On the sidebar, you can adjust how many opinion themes are generated for a particular book, as well as the number of reviews per theme and a couple of other options.

        ### Where is the data from?

        All data for this project comes from the [UCSD Book Graph dataset](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home?authuser=0).
    '''

# Title specific book searches
elif re.match(r'title: ', input_text):
    input_text = input_text.replace('title: ', '')
    book_title = books[books.title.str.contains(input_text, case=False)].title.tolist()
    showInfo(iterator=book_title,
            n_clusters=n_clusters,
            n_results=n_results,
            n_books=n_books,
            review_max_len=review_max_len)


# Author specific searches
elif re.match(r'author: ', input_text):
    input_text = input_text.replace('author: ', '')
    author_name = books[books.name.str.contains(input_text, case=False)].title.tolist()
    showInfo(iterator=author_name,
            n_clusters=n_clusters,
            n_results=n_results,
            n_books=n_books,
            review_max_len=review_max_len)

# Description specific searches
elif re.match(r'description: ', input_text):
    input_text = input_text.replace('description: ', '')




if psutil.virtual_memory()[2] > 60:
    st.write('Clearing cache to make sure things continue to run smoothly. Hang on!')
    caching.clear_cache()