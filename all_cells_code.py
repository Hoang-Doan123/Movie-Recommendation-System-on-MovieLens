import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# pd.set_option('display.max_rows', None) # This code will display all of the dataframe
# pd.reset_option('all') # Reset to default

movies = pd.read_csv('movielens-1m/movies.dat', sep='::', engine='python', 
                        names=['movieId', 'title', 'genres'], encoding='ISO-8859-1')

movies_with_des = pd.read_csv('movielens_movies_with_descriptions.csv', sep=',')

users = pd.read_csv('movielens-1m/users.dat', sep='::', engine='python',
                    names=['userId', 'gender', 'age', 'occupation', 'zip-code'], encoding='ISO-8859-1')

ratings = pd.read_csv('movielens-1m/ratings.dat',
                      sep='::', engine='python', 
                      names=['userId', 'movieId', 'rating', 'timestamp'], encoding='ISO-8859-1')

ratings.drop(['timestamp'], axis=1, inplace=True)
users.drop(['zip-code'], axis=1, inplace=True)


movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')

movies['title'] = movies['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()


movies['title'] = movies['title'].str.replace(r'\s+', ' ', regex=True).str.strip()


def fix_title_regex(title):
    return re.sub(r"^(.*), (The|A|An|L'|Le)( \(.+\))?$", r'\2 \1\3', title)
movies['title'] = movies['title'].apply(fix_title_regex)

# Pivot table: users x movies
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Ma trận tương đồng giữa item (dựa trên rating vector)
item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, 
                                   index=user_item_matrix.columns, 
                                   columns=user_item_matrix.columns)

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

item_based_recommend(1, 5)

from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
user_item_train = train_data.pivot(index='userId', columns='movieId', values='rating').fillna(0)

item_similarity = cosine_similarity(user_item_train.T)
item_similarity_df = pd.DataFrame(item_similarity, 
                                   index=user_item_train.columns, 
                                   columns=user_item_train.columns)

def predict_rating(user_id, movie_id, k=5):
    if movie_id not in item_similarity_df.columns or user_id not in user_item_train.index:
        return np.nan
    
    user_ratings = user_item_train.loc[user_id]
    rated_movies = user_ratings[user_ratings > 0].index

    if len(rated_movies) == 0:
        return np.nan

    similarities = item_similarity_df.loc[movie_id, rated_movies]
    ratings = user_ratings[rated_movies]

    top_k = similarities.sort_values(ascending=False)[:k]
    if top_k.sum() == 0:
        return np.nan

    return np.dot(top_k, ratings[top_k.index]) / top_k.sum()

from sklearn.metrics import mean_squared_error, mean_absolute_error

true_ratings = []
pred_ratings = []

for _, row in test_data.iterrows():
    pred = predict_rating(row['userId'], row['movieId'])
    if not np.isnan(pred):
        true_ratings.append(row['rating'])
        pred_ratings.append(pred)

print("RMSE:", mean_squared_error(true_ratings, pred_ratings, squared=False))
print("MAE:", mean_absolute_error(true_ratings, pred_ratings))


def recommend_top_k(user_id, k=10):
    if user_id not in user_item_train.index:
        return []

    user_ratings = user_item_train.loc[user_id]
    rated_movies = user_ratings[user_ratings > 0].index

    scores = {}

    for movie in user_item_train.columns:
        if movie in rated_movies:
            continue  # Bỏ qua phim đã xem

        pred = predict_rating(user_id, movie)
        if not np.isnan(pred):
            scores[movie] = pred

    top_k_movies = sorted(scores, key=scores.get, reverse=True)[:k]
    return top_k_movies


# Tạo dict: userId → danh sách phim họ thích trong test
test_like = test_data[test_data['rating'] >= 4].groupby('userId')['movieId'].apply(set).to_dict()

def precision_recall_f1_at_k(k=10):
    precisions = []
    recalls = []

    user_list = list(set(test_like.keys()) & set(user_item_train.index))

    for user_id in user_list:
        true_movies = test_like[user_id]
        pred_movies = set(recommend_top_k(user_id, k))

        if not pred_movies:
            continue

        true_positives = len(true_movies & pred_movies)

        precision = true_positives / len(pred_movies)
        recall = true_positives / len(true_movies) if true_movies else 0

        precisions.append(precision)
        recalls.append(recall)

    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall + 1e-8)

    print(f"Precision@{k}: {avg_precision:.4f}")
    print(f"Recall@{k}: {avg_recall:.4f}")
    print(f"F1@{k}: {f1:.4f}")


precision_recall_f1_at_k(k=5)
precision_recall_f1_at_k(k=10)
