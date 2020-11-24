import numpy as np
import pandas as pd

# To process embeddings
import tensorflow as tf
import tensorflow_hub as hub

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
import streamlit

# Custom libraries
from loader import data_loader
from summarize import find_books, find_reviews, find_description, find_books_description

#######################################################################################
                                # Load variables
#######################################################################################

# Path to books and reviews DataFrames
datapath = '/media/einhard/Seagate Expansion Drive/3380_data/data/Filtered books/'

# Path to USE
embed = hub.load('/media/einhard/Seagate Expansion Drive/3380_data/data/tensorflow_hub/universal-sentence-encoder_4')

# Load pre-trained sentence arrays
sentence_array = joblib.load('/media/einhard/Seagate Expansion Drive/3380_data/data/Models/reviewEmbeddings.pkl')
descriptions_array = joblib.load('/media/einhard/Seagate Expansion Drive/3380_data/data/Models/descriptionEmbeddings.pkl')

