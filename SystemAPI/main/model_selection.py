from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from load_data import data, tfidf_matrix

knn = NearestNeighbors(n_neighbors=10, metric="cosine", algorithm="auto")
knn.fit(tfidf_matrix)

consine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def knn_recommendations(movie_id, n = 5):
    # Get the index of the movie that matches the id.
    movie_idx = data[data['movieId'] == movie_id].index[0]

    movie_vector = tfidf_matrix[movie_idx].reshape(1, -1)
    distances, indices = knn.kneighbors(movie_vector)
    
    return [(data.iloc[i]['title'], data.iloc[i]["genres"], 1-d) for i, d in zip(indices[0][1:], distances[0][1:])]

print(knn_recommendations(3))