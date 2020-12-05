import pandas as pd
import numpy as np


#######################################################################################
                    # Return recommendations based on reviews
#######################################################################################
def find_reviews(query,reviews, n_results=5):
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
    print(top_n_indices)
    return top_n_indices

def find_books(query, reviews, books, n_results=5):
    top_n_indices = find_reviews(query, reviews, n_results)
    return books[books.book_id.isin(reviews.iloc[top_n_indices].book_id.tolist())][['title', 'name','description', 'weighted_score']].fillna('')

#######################################################################################
                    # Return recommendations based on descriptions
#######################################################################################

def find_description(query, books, n_results=10):
    # Create vector from query and compare with global embedding
    sentence = [query]
    sentence_vector = np.array(embed(sentence))
    inner_product = np.inner(sentence_vector, descriptions_array)[0]

    # Find sentences with highest inner products
    top_n_sentences = pd.Series(inner_product).nlargest(n_results+1)
    top_n_indices = top_n_sentences.index.tolist()
    top_n_list = list(books.description.iloc[top_n_indices][1:])

    print(f'Input sentence: "{query}"\n')
    print(f'{n_results} most semantically similar book descriptions: \n\n')
    print(*top_n_list, sep='\n\n')
    return top_n_indices

def find_books_description(query, reviews, books):
    top_n_indices = find_description(query)
    return books[books.book_id.isin(books.iloc[top_n_indices].book_id.tolist())][['title', 'name','description', 'weighted_score']].fillna('')
