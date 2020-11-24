import pandas as pd
import numpy as np

def data_loader(datapath, books_file, reviews_file):
    books = pd.read_csv(path + books_file).drop('Unnamed: 0', axis=1)
    reviews = pd.read_csv(path + books_file).drop('Unnamed: 0', axis=1)
    return books, reviews