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
def data_loader(datapath, books_file, reviews_file):
    books = pd.read_csv(datapath + books_file).drop('Unnamed: 0', axis=1).fillna('')
    reviews = pd.read_csv(datapath + reviews_file).drop('Unnamed: 0', axis=1)
    return books, reviews

@st.cache
def load_embeddings():
    # Path to USE
    embed = hub.load('/media/einhard/Seagate Expansion Drive/3380_data/data/tensorflow_hub/universal-sentence-encoder_4')

    # Load pre-trained sentence arrays
    sentence_array = joblib.load('/media/einhard/Seagate Expansion Drive/3380_data/data/Models/reviewEmbeddings.pkl')
    descriptions_array = joblib.load('/media/einhard/Seagate Expansion Drive/3380_data/data/Models/descriptionEmbeddings.pkl')
    return embed, sentence_array, descriptions_array

