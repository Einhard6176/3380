import numpy as np
import pandas as pd

# To process embeddings
import tensorflow_hub as hub
from langdetect import detect
from pathlib import Path
from nltk.tokenize import sent_tokenize

# To create recommendations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# To create sentence clusters
from sklearn.cluster import KMeans

# To load saved embeddings
import joblib

# To match strings
import re

# To create webapp
import psutil
import streamlit as st
st.set_page_config(page_title='3380 Books',
                  layout="wide",
                  page_icon= ':books:')
from streamlit import caching


# To sort final recommendation list
from collections import Counter

#######################################################################################
                            # Functions
#######################################################################################

#################### Data Loading  Functions ####################

@st.cache
def dataLoader(datapath, books_file, reviewsAll_file):
    '''
    Loads DataFrames with books and review information

    Args:
        datapath = Path/to/directory/with/data
        books_file = name of CSV with book metadata
        reviewsALL_file = name of CSV with filtered reviews --> Generally, this is the file that will be used in the app.
    '''
    books = pd.read_csv(datapath + books_file).drop('Unnamed: 0', axis=1).fillna('')
    reviewsAll = pd.read_csv(datapath + reviewsAll_file).drop('Unnamed: 0', axis=1)
    return books, reviewsAll

@st.cache
def loadEmbeddings(datapath):
    '''
    Loads pre-trained sentence and review arrays

    Args:
        datapath = Path/to/directory/with/data
    '''
    # Path to USE
    embed = hub.load(datapath + 'tensorflow_hub/universal-sentence-encoder_4')

    # Load pre-trained sentence arrays
    ## Reviews array is a set of embeddings trained on review lengths of < 90 characters
    #reviews_array = joblib.load(datapath + 'Models/reviewEmbeddings.pkl')
    ## Descriptions array is a set of embeddings trained on all book descriptions
    #descriptions_array = joblib.load(datapath + 'Models/descriptionEmbeddings.pkl')

    return embed # reviews_array, descriptions_array

#################### Basic Clustering Functionality ####################

@st.cache(allow_output_mutation=True)
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
        author_name = 'VariousAuthors/'
    else:
        # Finds book_id from author name
        input_book_id = books_df[books_df.name.isin([search_param])].book_id.tolist()
        author_name = books_df[books_df.name.isin([search_param])].name.iloc[0]

    # Finds reviews for specified book
    # input_sentences = review_df[review_df.book_id.isin(input_book_id)].review_text
    input_sentences = cleanAndTokenize(review_df[review_df.book_id.isin(input_book_id)], tokenizedData, searchTitle=searchTitle, author=author_name).review_text

    # Filters review length
    input_sentences = input_sentences[input_sentences.str.len() <= review_max_len]

    # Converts reviews into 512-dimensional arrays
    input_vectors = embed(input_sentences)

    # Returns reviews and vectorized reviews for a particular book
    return input_sentences, input_vectors

# Pass "sentence_array" as input_vectors to compare with user input???????
@st.cache(allow_output_mutation=True)
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

        # Writes reviews that are closest to centroid
        st.markdown('---')
        collapseCluster = st.beta_expander(f'Opinion cluster #{i+1}', expanded=True)
        with collapseCluster:
            for sentence in clusteredInputs:
                st.write(sentence)


#################### Tokenizing and saving data for embedding ####################

@st.cache(allow_output_mutation=True)
def clean_reviews(df):
    '''
    Copyright (c) 2020 Willie Costello
    '''
    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Define spoiler marker & remove from all reviews
    spoiler_str_ucsd = '\*\* spoiler alert \*\* \n'
    df['review_text'] = df['review_text'].str.replace(spoiler_str_ucsd, '')

    # Replace all new line characters
    df['review_text'] = df['review_text'].str.replace('\n', ' ')

    # Append space to all sentence end characters
    df['review_text'] = df['review_text'].str.replace('.', '. ').replace('!', '! ').replace('?', '? ')

    # Initialize dataframe to store English-language reviews
    reviews_df = pd.DataFrame()

    # Loop through each row in dataframe
    for i in range(len(df)):

        # Save review to variable
        review = df.iloc[i]['review_text']

        # Check if review is English
        if detect(review) == 'en':
            # If so, add row to English-language dataframe
            reviews_df = reviews_df.append(df.iloc[i, :])

    reviews_df.book_id = reviews_df.book_id.astype(int)
    return reviews_df

@st.cache(allow_output_mutation=True)
def make_sentences(reviews_df):
    '''
    Copyright (c) 2020 Willie Costello
    '''
    # Initialize dataframe to store review sentences, and counter
    sentences_df = pd.DataFrame()

    # Loop through each review
    for i in range(len(reviews_df)):

        # Save row and review to variables
        row = reviews_df.iloc[i]
        review = row.loc['review_text']

        # Tokenize review into sentences
        sentences = sent_tokenize(review)

        # Loop through each sentence in list of tokenized sentences
        for sentence in sentences:
            # Add row for sentence to sentences dataframe
            new_row = row.copy()
            new_row.at['review_text'] = sentence
            sentences_df = sentences_df.append(new_row, ignore_index=True)

    sentences_df = sentences_df[(sentences_df.review_text.str.len() >= 20) & (sentences_df.review_text.str.len() <= 350)]
    return sentences_df

@st.cache(allow_output_mutation=True)
def cleanAndTokenize(df, filepath, searchTitle, author):
    '''
    Helper function that first checks if a CSV file has already been generated for that book or author.
    If not, runs another cleaning pass on the set of reviews and tokenizes text into sentences before
    saving a CSV file to speed up recalls for the same book alter.

    Returns DataFrame with tokenized reviews for a particular author or book.

    Args:
        df          = dataframe to tokenize. It will save and load based on the first book_id if searching titles, or based on the author passed as an argument.
        filepath    = Path to output folder. If if a file exits, it will look in this path for the CSV files. Subdirectories must be made for ./book_id/ and for ./author/
        searchTitle = Specify whether the reviews for an individual book should be tokenized or the all reviews for an author.
        author      = If searchTitle=False, the "author" argument is used for saving and loading files. Note that it has no effect on the tokenization itself.
    '''

    if searchTitle:
        if Path(filepath + 'app_data/book_id/' + df.book_id.iloc[1].astype(str) + '.csv').is_file():
            sentences_df = pd.read_csv(filepath + 'app_data/book_id/' + df.book_id.iloc[1].astype(str) + '.csv').drop('Unnamed: 0', axis=1)
        else:
            reviews_df = clean_reviews(df)
            sentences_df =  make_sentences(reviews_df)
            sentences_df.to_csv(filepath + 'app_data/book_id/' + df.book_id.iloc[1].astype(str) + '.csv')
        sentences_df.book_id = sentences_df.book_id.astype(int)
        return sentences_df

    else:
        if Path(filepath + 'app_data/author/' + author + '.csv').is_file():
            sentences_df = pd.read_csv(filepath + 'app_data/author/' + author + '.csv').drop('Unnamed: 0', axis=1)
        else:
            reviews_df = clean_reviews(df)
            sentences_df =  make_sentences(reviews_df)
            sentences_df.to_csv(filepath + 'app_data/author/' + author + '.csv')
        sentences_df.book_id = sentences_df.book_id.astype(int)

        return sentences_df



#################### Searching based on description or review ####################

def findSimilarity(input_text, df, searchDescription):
    pass

def searchBookTitles(input_text, reviews, books, n_clusters, n_cluster_reviews):
    pass

# Basic TF-IDF cosine similarity engine
@st.cache(allow_output_mutation=True)
def createSimilarities(books_df):
    '''
    Creates a similarity matrix for book recommendations based on description.
    Returns similarity matrix and mapping for book-finding

    Args:
        books_df = DataFrame with books and descriptions
    '''
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 5), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(books['description'])
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    mapping = pd.Series(books.index,index = books['title'])
    return cosine_similarities, mapping

@st.cache(allow_output_mutation=True)
def bookRecommendation(book_title, mapping, cosine_similarities, n_books):
    '''
    Function to match input book with recommended books using cosine similarity matrix.

    Args:
        book_title          = Input book title. Recommendations will be made based on this.
        mapping             = Mapping of cosine similarity matrix to book indices
        cosine_similarities = Similarity matrix for content based recommendation
        n_books             = How many books to recommend
    '''
    book_index = mapping[book_title]
    similarity_score = list(enumerate(cosine_similarities[book_index]))
    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    similarity_score = similarity_score[1:n_books+1]
    book_indices = [i[0] for i in similarity_score]
    return (books['title'].iloc[book_indices])

##### Experimental functionality, WIP #####

#@st.cache(allow_output_mutation=True)
#def findSemanticallySimilarReviews(query,reviews, books, sentence_array,  n_books):
#    # Create vector from query and compare with global embedding
#    sentence = [query]
#    sentence_vector = np.array(embed(sentence))
#    inner_product = np.inner(sentence_vector, sentence_array)[0]
#
#    # Find sentences with highest inner products
#    top_n_sentences = pd.Series(inner_product).nlargest(n_books+1)
#    top_n_indices = top_n_sentences.index.tolist()
#    book_titles = books[books.book_id.isin(reviews.iloc[top_n_indices].book_id.tolist())].title.tolist()
#
#    return book_titles, reviews.iloc[top_n_indices].index



#################### App UI and Interactions ####################


def showInfo(iterator, n_clusters, n_results,n_books, review_max_len=350):
    with results:
        for idx, i in enumerate(iterator[:n_books]):
            try:
                info = books[books.title == i]
                '**---**'
                '**Book title:**', f'*{info.title.tolist()[0]}*'
                '**Author:**', info.name.tolist()[0]
                '**Weighted Score**', str(round(info.weighted_score.tolist()[0], 2)), '/ 5'

                showDescription = st.beta_expander(label='Show description?')
                showReviewClusters = st.button(label='Show opinion clusters for this book?', key=idx)
                showAuthorClusters = st.button(label='Show opinion clusters for this author?', key=idx+100)
            #    showSimilarBooks = st.button(label='Show similar books?', key=idx+555)
            except IndexError:
                break
            with showDescription:
                st.write(info.description.tolist()[0])

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
                        st.warning(f"It looks like this book doesn't have enough reviews to generate {n_clusters} distinct clusters. Try decreasing how many clusters you look for!")
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
                        st.warning(f"It looks like this author's books don't have enough reviews to generate {n_clusters} distinct clusters. Try decreasing how many clusters you look for!")
                        continue

            ##### Experimental functionality, WIP #####

            #if showSimilarBooks:
            #    showInfo(iterator=info.title.tolist()[0],
            #            n_clusters=n_clusters,
            #            n_results=n_results,
            #            n_books=n_books,
            #            review_max_len=review_max_len)

            if goodreadsLink:
                good_reads_link = goodreadsURL + info.book_id.astype(str).tolist()[0]
                st.write(f'*Goodreads Link: {good_reads_link}*')


#######################################################################################
                            # Load variables and data
#######################################################################################

# Paths to books and reviews DataFrames
datapath = '/media/einhard/Seagate Expansion Drive/3380_data/data/'

# Stores tokenized reviews so they only need to be processed the first time that particular book is called
tokenizedData = '/media/einhard/Seagate Expansion Drive/3380_data/data/'
books_file = 'Filtered books/clean_filtered_books.csv'
reviewsAll_file = 'Filtered books/reviews_for_cluster.csv'

# Loading DataFrames
books, reviewsAll = dataLoader(datapath, books_file, reviewsAll_file)

# Loadding pre-trained embeddings and embedder for input sentences
embed = loadEmbeddings(datapath) # , reviews_array, descriptions_array

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
options = st.sidebar.beta_expander('Options', expanded=True )
with options:
    goodreadsLink = st.checkbox('Show Goodreads links.')
    n_books = st.slider('Select how many book results to show',
                            1, 25, value=10, step=1)
    n_clusters = st.slider('Select how many clusters to search for in the reviews',
                                2,10,value=8,step=1)
    n_results = st.slider('Select how many reviews to show per cluster',
                                1,15,value=8,step=1)
    review_max_len = st.slider('Select maximum review length for each cluster calculation',
                                 50, 350, value=350, step=10)

helpFAQ = st.sidebar.beta_expander('Help')
with helpFAQ:
    '''
    **What is an opinion cluster?**\n
    An opinion cluster is a set of reviews that display semantic textual similarities. The algorithm takes a review and analyses its grammatical structure, word choice, etc. and coverts into into a mathematical vector. It then does the same for all other reviews, and finds which reviews are close to each other in a 512-dimensional space. If you're curious, here's an [explanation with some images](https://amitness.com/2020/06/universal-sentence-encoder/_) \n

    **Select how many clusters:** \n
    This slider controls how many review clusters the algorithm will create.\n
    **Select how many reviews:** \n
    This slider controls the maxmimum items the algorithm will pull from *each* cluster.\n
    **Select maximum review length:** \n
    The maximum character count per review to be passed into the clustering algorithm.

    **Weighted Score:** \n
    The weighted score is a formula used to minimize the effect of high-score low-quantity reviews for a particular book. That is, books with very few reviews and high scores will not be trusted as much as books with many reviews: for example, consider two books with 5-star ratings; if one of them has ten 5-star ratings, and the other has five hundred, it is more likely that the latter will be more accurate.
    '''

st.sidebar.markdown('''
---
*Created by [Adnan Valdes](https://www.linkedin.com/in/adnanvaldes/)*\n
*Find the project on [GitHub](https://github.com/Einhard6176/3380)*
'''
)

# Asking for user input
input_text = st.text_input('Try specifying `author:` or `title: ` if you want more specific results')

# Creating columns for book results on the left, review clusters on the right
results, clusters = st.beta_columns(2)

# Welcome message
if not input_text:
    '''
    ### Some useful commands:

    You can type in a title or partial title and the search algorithm will return some book recommendations. \n

    You can also try any of the following prefixes to fine tune your search:\n
    `author: `      - will search database for specific authors (`author: Frank Herbert`)\n
    `title: `       - will search database for specific titles (`title: Dune`)\n
    `description: ` - will search database for books that match description (`description: Set on the desert planet Arrakis`)\n
    ### How this works:

    In the search bar above, type the title of a book you're interested in. The algorithm will then come back with some book recommendations based on the description of the book you entered. If you're looking for a particular author, use the `author: ` prefix, or the `title: ` prefix if you want a particular book. You can type partial names or titles (i.e. you can search for "author: Adrian" and the app will return all authors that have "Adrian" in their name).

    Once you have a list of books, you can load review clusters that describe that particular book. You can also load review clusters for an author - in this case, the machine learning algorithm will look through *all* reviews associated with that author, instead of just a particular book.

    On the sidebar, you can adjust how many opinion clusters are generated for a particular search, as well as the number of reviews per clusters, and a couple of other options.
    '''
    about = st.beta_expander('About')
    with about:
        '''
        ## About

        3380 Books was created as part of my final data science project at [Lighthouse Labs, Vancouver](https://www.lighthouselabs.ca/).

        The inspiration for the app came from Goodread's yearly reading [challenge](https://www.goodreads.com/challenges/show/11621-2020-reading-challenge). The name, 3380, comes from a simple calculation: if you were born in 1993, like I was, your [average global life expectancy at birth](https://data.worldbank.org/indicator/SP.DYN.LE00.IN) would be around 65 years. If you committed to reading one book every single week from the day you were born till the day you die, you'd read 52 books per year, or about 3,380 books in your lifetime.

        To put that in perspective,  there were [45,860 working authors](https://www.statista.com/topics/1177/book-market/) in 2019 and over 1.5 million self-published books in 2018 *in the US alone*. As of late 2019, the Google Books project had scanned over [40 million books in over 400 languages](https://www.blog.google/products/search/15-years-google-books/). This means that with 3380 books to read in a lifetime, we will still only be able to read less than 0.01% of extant books (and every year more and more books are being published). It is therefore increasingly important to find a way to filter information; but, crucially, to not rely *only* on other users - in the way that Netflix suggests TV shows, for example. Rather, it
        behooves us to take an active part in choosing the content we want to consume.
        '''

    acknowledgements = st.beta_expander('Acknowledgements')
    with acknowledgements:
        '''
        ## Acknowledgements

        I'd like to start by thanking Menging Wan, Julian McAuley, Rishabh Misra and Ndapa Nakashole for creating and publishing the [UCSD Book Graph](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home)
        database. All the book data for this app comes from their work. If you're interested, check out their papers: [*Item Recommendation on Monotonic Behaviour Chains*](https://github.com/MengtingWan/mengtingwan.github.io/raw/master/paperrecsys18_mwan.pdf), and [*Fine-Grained Spoiler Detection from Large-Scale Review Corpora*](https://www.aclweb.org/anthology/P19-1248/).

        Additionally, I want to thank [Willie Costello](https://williecostello.com/), whose BetterReads algorithm laid the foundation for my work.

        I would also like to thank all the staff at Lighthouse Labs for helping us through this journey.

        Most of all, I'd like to thank my classmates, without whom I surely would not have made it this far. In particular, for their immense support and unlimited camaraderie, I'd like to acknowledge:
        * [Atlas Kazemian](https://www.linkedin.com/in/atlas-kazemian-874b11100/)
        * [Olivia Kim](https://github.com/yjik122)
        * [Elliot Lupini](https://www.linkedin.com/in/elliot-lupini-8824681b1/)
        * [Lane Clark](http://lclark.ca/)
        * [Henri Vandersleyen](https://www.linkedin.com/in/henri-vandersleyen-a25a8312b/)
        '''

# Author specific searches
elif re.match(r'author:', input_text):
    input_text = input_text.replace('author:', '').strip()
    author_name = books[books.name.str.contains(input_text, case=False)].sort_values('weighted_score', ascending=False).title.tolist()
    showInfo(iterator=author_name,
            n_clusters=n_clusters,
            n_results=n_results,
            n_books=n_books,
            review_max_len=review_max_len)

# Title book searches
elif re.match(r'title:', input_text):
    input_text = input_text.replace('title:', '').strip()
    book_title = books[books.title.str.contains(input_text, case=False)].sort_values('weighted_score', ascending=False).title.tolist()
    showInfo(iterator=book_title,
            n_clusters=n_clusters,
            n_results=n_results,
            n_books=n_books,
            review_max_len=review_max_len)

# Description specific searches
elif re.match(r'description: ', input_text):
    input_text = input_text.replace('description:', '').strip()
    book_title = books[books.description.str.contains(input_text, case=False)].sort_values('weighted_score', ascending=False).title.tolist()
    showInfo(iterator=book_title,
        n_clusters=n_clusters,
        n_results=n_results,
        n_books=n_books,
        review_max_len=review_max_len)

# Show all books in database
elif re.match(r'list: all', input_text):
    st.table(books[['title', 'name', 'weighted_score']].rename(columns={'name':'author', 'weighted_score':'score'}))

    listAll()

##### Experimental functionality, WIP #####

# # Experimental book recommendations based on review text
# elif re.match(r'beta: ', input_text):
#     input_text = input_text.replace('beta: ', '')
#     book_title, review_index = findSemanticallySimilarReviews(query=input_text,
#                                                reviews=reviewsAll,
#                                                books=books,
#                                                n_books=n_books,
#                                                sentence_array=reviews_array)
#     with clusters:
#         for idx, i in enumerate(reviews.iloc[review_index].index):
#             reviews[reviews.index == i].review_text.tolist()[0]
#     showInfo(iterator=book_title,
#         n_clusters=n_clusters,
#         n_results=n_results,
#         n_books=n_books,
#         review_max_len=review_max_len)

elif input_text:
    try:
        book_title = books[books.title.str.contains(input_text, case=False)].sort_values('weighted_score', ascending=False).title.tolist()[0]
        with results:
            st.markdown(f'## Book recommendations based on *{book_title}*')
        cosine_similarities, mapping = createSimilarities(books)
        book_recommends = bookRecommendation(book_title=book_title,
                                            mapping=mapping,
                                            cosine_similarities=cosine_similarities,
                                            n_books=n_books)
        showInfo(iterator=book_recommends,
                 n_clusters=n_clusters,
                 n_results=n_results,
                 n_books=n_books,
                 review_max_len=review_max_len)
    except IndexError:
        st.warning('Sorry, it looks like this book is not in our database.')

if psutil.virtual_memory()[2] > 60:
    st.write('Clearing cache to make sure things continue to run smoothly. Hang on!')
    caching.clear_cache()