# %% [markdown]
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import os

if os.path.exists("movielens-1m/movies.dat"):
    movie_dir = "movielens-1m/movies.dat"
elif os.path.exists("../movielens-1m/movies.dat"):
    movie_dir = "../movielens-1m/movies.dat"
else:
    movie_dir = "../../movielens-1m/movies.dat"

movies = pd.read_csv(movie_dir, sep='::', engine='python', 
                        names=['movieId', 'title', 'genres'], encoding='ISO-8859-1')

if os.path.exists("movielens_movies_with_descriptions.csv"):
    movielens_dir = "movielens_movies_with_descriptions.csv"
elif os.path.exists("../movielens_movies_with_descriptions.csv"):
    movielens_dir = "../movielens_movies_with_descriptions.csv"
else:
    movielens_dir = "../../movielens_movies_with_descriptions.csv"

movies_with_des = pd.read_csv(movielens_dir, sep=',')

if os.path.exists("movielens-1m/users.dat"):
    users_dir = "movielens-1m/users.dat"
elif os.path.exists("../movielens-1m/users.dat"):
    users_dir = "../movielens-1m/users.dat"
else:
    users_dir = "../../movielens-1m/users.dat"

users = pd.read_csv(users_dir, sep='::', engine='python',
                    names=['userId', 'gender', 'age', 'occupation', 'zip-code'], encoding='ISO-8859-1')

if os.path.exists("movielens-1m/ratings.dat"):
    ratings_dir = "movielens-1m/ratings.dat"
elif os.path.exists("../movielens-1m/ratings.dat"):
    ratings_dir = "../movielens-1m/ratings.dat"
else:
    ratings_dir = "../../movielens-1m/ratings.dat"

ratings = pd.read_csv(ratings_dir,
                      sep='::', engine='python', 
                      names=['userId', 'movieId', 'rating', 'timestamp'], encoding='ISO-8859-1')

# %%
ratings.drop(['timestamp'], axis=1, inplace=True)
users.drop(['zip-code'], axis=1, inplace=True)

# %%

movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')

movies['title'] = movies['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()


movies['title'] = movies['title'].str.replace(r'\s+', ' ', regex=True).str.strip()


def fix_title_regex(title):
    return re.sub(r"^(.*), (The|A|An|L'|Le)( \(.+\))?$", r'\2 \1\3', title)
movies['title'] = movies['title'].apply(fix_title_regex)

user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Ma trận tương đồng giữa item (dựa trên rating vector)
item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, 
                                   index=user_item_matrix.columns, 
                                   columns=user_item_matrix.columns)

# %% [markdown]
# <span style='color:#007ACC; font-size:15pt; font-weight:bold'>Recommender</span>

# %%
# Hàm gợi ý các phim tương tự (CF truyền thống)
def item_based_recommend(movie_id, top_n=5):
    if movie_id not in item_similarity_df.columns:
        return f"Movie ID {movie_id} not exists in the system."
    
    original_movie = movies[movies['movieId'] == movie_id][['title', 'year']].values
    original_title, original_year = original_movie[0]

    similar_scores = item_similarity_df[movie_id].sort_values(ascending=False)
    similar_movies = similar_scores.iloc[1:top_n+1].index

    recommended_movies = movies[movies['movieId'].isin(similar_movies)][['movieId', 'title', 'year']]
    recommended_movies['similarity'] = recommended_movies['movieId'].apply(lambda x: similar_scores[x])
    recommended_movies = recommended_movies.sort_values(by='similarity', ascending=False).reset_index(drop=True)

    print(f"Top {top_n} similarity movies with Movie ID {movie_id}: \n{original_title} ({original_year}):")
    return recommended_movies

print(item_based_recommend(45))
print(item_based_recommend(60))
print(item_based_recommend(75))