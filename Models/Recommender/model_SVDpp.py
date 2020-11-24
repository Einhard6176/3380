import os
import pandas as pd
import numpy as np
import surprise as sp
from surprise.model_selection import GridSearchCV, train_test_split
from surprise.accuracy import rmse
from surprise import dump

print('Losading reviews')
reviews_df = pd.read_csv('/media/einhard/Seagate Expansion Drive/3380_data/data/user_book_rating.csv').drop('Unnamed: 0', axis=1)
print('Success!')

print('Creating Dataset')
data = sp.Dataset.load_from_df(reviews_df, sp.Reader())
del reviews_df
print('Success!')

train, test = train_test_split(data, test_size=.25)
del data

algo = SVDpp()
alg = SVDpp()
output = alg.fit(train)
predictions = alg.test(test)


print('Saving algorithm')
# Dump algorithm and reload it.
file_name = os.path.expanduser('/media/einhard/Seagate Expansion Drive/3380_data/3380-Book-Recommendations/Models/model_SVDpp.base')
dump.dump(file_name, algo=algo)
print('Success!')
