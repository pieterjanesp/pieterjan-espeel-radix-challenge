"""
API endpoints of the hiring challenge.

This module contains API endpoints for training a model to predict movie
genres based on their synopsis and for predicting genres of given movies.
"""

from io import BytesIO
import joblib
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from stop_words import get_stop_words
from fastapi import FastAPI, File

app = FastAPI()
stop_words = get_stop_words('en')

@app.post("/genres/train")
async def train(file: bytes = File(...)) -> None:
    """Train a model to predict movie genres.

    Train a predictive model using provided training data to predict
    movie genres based on their synopsis.

    Parameters:
    file : bytes
        CSV file with at least 'synopsis' and 'genres' columns.

    Returns:
    dict
        Confirmation message.
    """
    df = pd.read_csv(BytesIO(file))
    df['genres'] = df['genres'].apply(lambda x: x.split())
    
    # Preprocessing Section of Synopsis
    df['synopsis'] = df['synopsis'].apply(lambda x: clean_text(x))
    df['synopsis'] = df['synopsis'].apply(lambda x: remove_stopwords(x))
    print(df['synopsis'])
    mlb = MultiLabelBinarizer()
    transformed_genres = mlb.fit_transform(df['genres'])

    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000,
                                  max_df=0.1,
                                  min_df=0.001,
                                  ngram_range=(1,3))),
        ('clf', MultiOutputClassifier(MultinomialNB()))
    ])

    # param_grid_tfidf = {
    #     'tfidf__max_df': [0.08, 0.1, 0.12],
    #     'tfidf__min_df': [0.01, 0.03],
    #     'tfidf__ngram_range': [(1, 1), (1, 2)]
    # }

    # grid_search = GridSearchCV(model_pipeline, param_grid_tfidf, scoring='accuracy', cv=3)
    # grid_search.fit(df['synopsis'], transformed_genres)

    # best_model_pipeline = grid_search.best_estimator_

    model_pipeline.fit(df['synopsis'], transformed_genres)

    joblib.dump(model_pipeline, "model_pipeline.joblib")
    joblib.dump(mlb, "mlb.joblib")

    return {"detail": "Model trained and saved."}


@app.post("/genres/predict")
async def predict(file: bytes = File(...)) -> dict:
    """Predict the genres of movies.

    Predict movie genres based on their synopsis using a pre-trained model.

    Parameters:
    file : bytes
        CSV file with at least 'synopsis' and 'movie_id' columns.

    Returns:
    dict
        Movie ID mapped to predicted genres.
    """
    # Load model and label binarizer
    try:
        model_pipeline = joblib.load("model_pipeline.joblib")
        mlb = joblib.load("mlb.joblib")
    except FileNotFoundError:
        return {"detail": "Model not found. Train the model first.", "error": True}
    
    df = pd.read_csv(BytesIO(file))

    predictions_proba = model_pipeline.predict_proba(df['synopsis'])
    predictions_proba_np = np.array(predictions_proba)

    num_classifiers, num_data_elems, _ = predictions_proba_np.shape
    predictions_probability_matrix = np.zeros((num_data_elems, num_classifiers))

    for i in range(num_data_elems):
        for j in range(num_classifiers):
            predictions_probability_matrix[i, j] = predictions_proba_np[j, i, 0]

    response = {}
    top_genres_index = np.argsort(predictions_probability_matrix, axis=1)[:, :5]

    for movie_idx, movie_id in enumerate(df['movie_id']):
        top_genres = mlb.classes_[top_genres_index[movie_idx]].tolist()
        response[str(movie_id)] = {i: genre for i, genre in enumerate(top_genres)}

    return response

def clean_text(text):
    # Remove everything except letters in alphabet
    text = re.sub("[^a-zA-Z]", " ", text)
    # Strip away extra whitespacing
    text = " ".join(text.split())
    # Convert text to lower case
    text = text.lower()
    return text

def remove_stopwords(text):
    text_no_stopwords = [w for w in text.split() if not w in stop_words]
    return ' '.join(text_no_stopwords)