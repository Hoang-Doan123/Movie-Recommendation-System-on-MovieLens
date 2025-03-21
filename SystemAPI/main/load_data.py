import nltk
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download("punkt", download_dir="E:/nltk_data")
nltk.download("stopwords", download_dir="E:/nltk_data")
nltk.download("wordnet", download_dir="E:/nltk_data")

data = pd.read_csv("E:/Python Tests/API Tests/Django_/MovieRecommendationSystem/movielens_movies_with_descriptions.csv")

def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text)

    text = text.lower()

    text = re.sub(r'\s+', ' ', text).strip()

    return text

data["cleaned_description"] = data["description"].apply(clean_text)

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    tokens = word_tokenize(text)

    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return " ".join(tokens)

data["processed_description"] = data["cleaned_description"].apply(preprocess_text)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf.fit_transform(data["processed_description"])

tfidf_data = pd.DataFrame(
    tfidf_matrix.toarray(), 
    columns=tfidf.get_feature_names_out(),
    index=data["movieId"]
)

genres_df = pd.DataFrame(data["genres"].str.get_dummies())

current_year = 2025

data["year_factor"] = (current_year - data["year"]) / 100

from scipy import sparse
genre_matrix = sparse.csr_matrix(genres_df.values)
year_matrix = sparse.csr_matrix(data[['year_factor']].values)

combined_features = sparse.hstack([
    tfidf_matrix * 0.6, 
    genre_matrix * 0.3, 
    year_matrix * 0.1])