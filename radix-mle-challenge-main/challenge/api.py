"""
API endpoints of the hiring challenge.

This module contains API endpoints for training a model to predict movie
genres based on their synopsis and for predicting genres of given movies.
"""

import re
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from stop_words import get_stop_words

app = FastAPI()
stop_words = get_stop_words('en')


@app.post("/genres/train")
async def train(file: bytes = File(...)) -> None:
    """Train a predictive model using provided training data to predict movie genres based on their synopsis."""
    # Create Dataframe from training data
    df = pd.read_csv(BytesIO(file))
    df['genres'] = df['genres'].apply(lambda x: x.split())
    # Preprocessing of synopsis: All stopwords are removed
    df['synopsis'] = df['synopsis'].apply(lambda x: clean_text(x))
    df['synopsis'] = df['synopsis'].apply(lambda x: remove_stopwords(x))
    # Binarization of possible movie genres
    mlb = MultiLabelBinarizer()
    transformed_genres = mlb.fit_transform(df['genres'])
    """

    Generate Pipeline of Operations. Start with tokenization of input synopses. A MultiOutput Classifier is used
    which uses logistic regression as base estimator
    """
    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, max_df=0.8, min_df=0.001, ngram_range=(1, 3))),
        ('clf', MultiOutputClassifier(LogisticRegression(C=10, max_iter=1000)))
    ])
    # Train the model using the machine learning pipeline defined above
    model_pipeline.fit(df['synopsis'], transformed_genres)
    # Save the model and the binarized genres
    joblib.dump(model_pipeline, "model_pipeline.joblib")
    joblib.dump(mlb, "mlb.joblib")


@app.post("/genres/predict")
async def predict(file: bytes = File(...)) -> dict:
    """Predict movie genres based on their synopsis using a pre-trained model."""
    # Try to load the trained model
    try:
        model_pipeline = joblib.load("model_pipeline.joblib")
        mlb = joblib.load("mlb.joblib")
    except FileNotFoundError:
        return {"detail": "Model not found. Train the model first.",
                "error": True}

    df = pd.read_csv(BytesIO(file))

    # Preprocessing of input synopses
    df['synopsis'] = df['synopsis'].apply(lambda x: clean_text(x))
    df['synopsis'] = df['synopsis'].apply(lambda x: remove_stopwords(x))
    # Apply the trained model to determine probabilities of genres
    predictions_proba = model_pipeline.predict_proba(df['synopsis'])
    predictions_proba_np = np.array(predictions_proba)
    # Reshape predictions tensor into matrix
    num_classifiers, num_data_elems, _ = predictions_proba_np.shape
    predictions_probability_matrix = np.zeros((num_data_elems, num_classifiers))

    for i in range(num_data_elems):
        for j in range(num_classifiers):
            predictions_probability_matrix[i, j] = predictions_proba_np[j, i, 0]
    # Determine top 5 highest probability genres
    response = {}
    top_genres_index = np.argsort(
        predictions_probability_matrix, axis=1)[:, :5]
    # Build output dictionary
    for movie_idx, movie_id in enumerate(df['movie_id']):
        top_genres = mlb.classes_[top_genres_index[movie_idx]].tolist()
        response[str(movie_id)] = {i: genre for i, genre in enumerate(top_genres)}

    return response


def clean_text(text: str) -> str:
    """Clean text by keeping only letters and converting to lower case."""
    # Remove everything except letters in alphabet
    text = re.sub("[^a-zA-Z]", " ", text)
    # Strip away extra whitespacing
    text = " ".join(text.split())
    # Convert text to lower case
    text = text.lower()
    return str(text)


def remove_stopwords(text: str) -> str:
    """Remove stopwords from a given text."""
    text_no_stopwords = [w for w in text.split() if w not in stop_words]
    return str(' '.join(text_no_stopwords))
