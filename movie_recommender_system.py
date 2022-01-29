import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

credits.rename(index = str, columns = {"movie_id":"id"},inplace = True)

merged = movies.merge(credits, on ="id")
merged = merged.fillna(" ")

merged.drop(["title_x","production_countries","homepage","status"],axis = 1,inplace = True)

merged.rename(index = str,columns = {"title_y":"title"},inplace = True)

tfidf = TfidfVectorizer(min_df=3,  max_features=None,
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3),
            stop_words = 'english')

tfidf_matrix = tfidf.fit_transform(merged["overview"])

sig_kernel = sigmoid_kernel(tfidf_matrix,tfidf_matrix)

indices = pd.Series(merged.index, index=merged['title']).drop_duplicates()

def recommend(title, sig=sig_kernel):
    idx = int(indices[title])
    sig_scores = list(enumerate(sig[idx]))
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    sig_scores = sig_scores[1:11]
    movie_indices = [i[0] for i in sig_scores]
    return merged['title'].iloc[movie_indices]
