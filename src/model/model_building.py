import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
import lightgbm as lgb

from sklearn.feature_extraction.text import TfidfVectorizer


# logging configuration
logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("model_building_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:

    try:

        with open(params_path, "r") as file:
            params = yaml.safe_load(file)

        logger.debug(
            "Parameters retrieved from %s",
            params_path
        )

        return params

    except FileNotFoundError:

        logger.error(
            "File not found: %s",
            params_path
        )

        raise

    except yaml.YAMLError as e:

        logger.error(
            "YAML error: %s",
            e
        )

        raise

    except Exception as e:

        logger.error(
            "Unexpected error: %s",
            e
        )

        raise


def load_data(file_path: str) -> pd.DataFrame:

    try:

        df = pd.read_csv(file_path)

        # Fill missing values
        df.fillna("", inplace=True)

        logger.debug(
            "Data loaded and NaNs filled from %s",
            file_path
        )

        return df

    except pd.errors.ParserError as e:

        logger.error(
            "Failed to parse the CSV file: %s",
            e
        )

        raise

    except Exception as e:

        logger.error(
            "Unexpected error occurred while loading data: %s",
            e
        )

        raise


def apply_tfidf(
    train_data: pd.DataFrame,
    max_features: int,
    ngram_range: tuple
) -> tuple:

    try:

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range
        )

        X_train = train_data["clean_comment"].values
        y_train = train_data["category"].values

        # Apply TF-IDF
        X_train_tfidf = vectorizer.fit_transform(
            X_train
        )

        logger.debug(
            "TF-IDF transformation complete. Train shape: %s",
            X_train_tfidf.shape
        )

        # Save TF-IDF vectorizer
        with open(
            os.path.join(
                get_root_directory(),
                "tfidf_vectorizer.pkl"
            ),
            "wb"
        ) as f:

            pickle.dump(
                vectorizer,
                f
            )

        logger.debug(
            "TF-IDF applied successfully"
        )

        return X_train_tfidf, y_train

    except Exception as e:

        logger.error(
            "Error during TF-IDF transformation: %s",
            e
        )

        raise


def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    learning_rate: float,
    max_depth: int,
    n_estimators: int
) -> lgb.LGBMClassifier:

    try:

        model = lgb.LGBMClassifier(

            objective="multiclass",

            num_class=3,

            metric="multi_logloss",

            is_unbalance=True,

            class_weight="balanced",

            reg_alpha=0.1,

            reg_lambda=0.1,

            learning_rate=learning_rate,

            max_depth=max_depth,

            n_estimators=n_estimators
        )

        model.fit(
            X_train,
            y_train
        )

        logger.debug(
            "LightGBM model training completed"
        )

        return model

    except Exception as e:

        logger.error(
            "Error during LightGBM model training: %s",
            e
        )

        raise


def save_model(
    model,
    file_path: str
) -> None:

    try:

        with open(
            file_path,
            "wb"
        ) as file:

            pickle.dump(
                model,
                file
            )

        logger.debug(
            "Model saved to %s",
            file_path
        )

    except Exception as e:

        logger.error(
            "Error occurred while saving model: %s",
            e
        )

        raise


def get_root_directory() -> str:

    current_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    root_directory = os.path.abspath(
        os.path.join(
            current_dir,
            "../../"
        )
    )

    return root_directory


def main():

    try:

        # Get project root directory
        root_dir = get_root_directory()

        # Load parameters
        params = load_params(
            os.path.join(
                root_dir,
                "params.yaml"
            )
        )

        max_features = params[
            "model_building"
        ]["max_features"]

        ngram_range = tuple(
            params[
                "model_building"
            ]["ngram_range"]
        )

        learning_rate = params[
            "model_building"
        ]["learning_rate"]

        max_depth = params[
            "model_building"
        ]["max_depth"]

        n_estimators = params[
            "model_building"
        ]["n_estimators"]

        # Load processed training data
        train_data = load_data(
            os.path.join(
                root_dir,
                "data/interim/train_processed.csv"
            )
        )

        # Apply TF-IDF
        X_train_tfidf, y_train = apply_tfidf(
            train_data,
            max_features,
            ngram_range
        )

        # Train LightGBM
        model = train_lgbm(
            X_train_tfidf,
            y_train,
            learning_rate,
            max_depth,
            n_estimators
        )

        # Save trained model
        save_model(
            model,
            os.path.join(
                root_dir,
                "lgbm_model.pkl"
            )
        )

    except Exception as e:

        logger.error(
            "Failed to complete model building process: %s",
            e
        )

        print(
            f"Error: {e}"
        )


if __name__ == "__main__":
    main()