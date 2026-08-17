import os
import pickle
import pandas as pd

from src.data.data_preprocessing import preprocess_comment


def load_model():

    model_path = "lgbm_model.pkl"

    with open(
        model_path,
        "rb"
    ) as file:

        model = pickle.load(file)

    return model


def load_vectorizer():

    vectorizer_path = "tfidf_vectorizer.pkl"

    with open(
        vectorizer_path,
        "rb"
    ) as file:

        vectorizer = pickle.load(file)

    return vectorizer


def predict_sentiment(comments):

    model = load_model()

    vectorizer = load_vectorizer()

    # Preprocess comments
    cleaned_comments = []

    for comment in comments:

        cleaned_comment = preprocess_comment(
            comment
        )

        cleaned_comments.append(
            cleaned_comment
        )

    # Convert comments into TF-IDF
    X = vectorizer.transform(
        cleaned_comments
    )

    # Predict sentiment
    predictions = model.predict(X)

    # Count each sentiment
    sentiment_counts = {}

    for prediction in predictions:

        prediction = str(prediction)

        if prediction not in sentiment_counts:

            sentiment_counts[prediction] = 0

        sentiment_counts[prediction] += 1

    total_comments = len(predictions)

    sentiment_percentages = {}

    for sentiment, count in sentiment_counts.items():

        percentage = (
            count / total_comments
        ) * 100

        sentiment_percentages[sentiment] = round(
            percentage,
            2
        )

    return {
        "total_comments": total_comments,
        "sentiment_counts": sentiment_counts,
        "sentiment_percentages": sentiment_percentages
    }


if __name__ == "__main__":

    comments = [
        "This video is amazing!",
        "I really enjoyed this video.",
        "This video is terrible."
    ]

    result = predict_sentiment(
        comments
    )

    print(
        result
    )