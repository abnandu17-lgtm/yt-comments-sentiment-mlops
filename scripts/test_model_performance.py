import os
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score


def test_model_performance():

    model_path = "lgbm_model.pkl"
    vectorizer_path = "tfidf_vectorizer.pkl"
    test_data_path = "data/interim/test_processed.csv"

    # Check files
    assert os.path.exists(
        model_path
    ), "Model file does not exist"

    assert os.path.exists(
        vectorizer_path
    ), "Vectorizer file does not exist"

    assert os.path.exists(
        test_data_path
    ), "Test data does not exist"

    # Load model
    with open(
        model_path,
        "rb"
    ) as file:

        model = pickle.load(file)

    # Load vectorizer
    with open(
        vectorizer_path,
        "rb"
    ) as file:

        vectorizer = pickle.load(file)

    # Load test data
    test_data = pd.read_csv(
        test_data_path
    )

    test_data.fillna(
        "",
        inplace=True
    )

    # Prepare test data
    X_test = test_data[
        "clean_comment"
    ]

    y_test = test_data[
        "category"
    ]

    # Convert text into TF-IDF
    X_test_tfidf = vectorizer.transform(
        X_test
    )

    # Make predictions
    y_pred = model.predict(
        X_test_tfidf
    )

    # Calculate accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print(
        f"Model accuracy: {accuracy:.4f}"
    )

    # Minimum performance requirement
    assert accuracy >= 0.60, (
        "Model performance is below "
        "the required accuracy"
    )

    print(
        "Model performance test passed"
    )


if __name__ == "__main__":
    test_model_performance()