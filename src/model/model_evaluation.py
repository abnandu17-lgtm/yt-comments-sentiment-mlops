import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
import mlflow
import mlflow.lightgbm
import json

from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from mlflow.models import infer_signature


# logging configuration
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(
    "model_evaluation_errors.log"
)
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path):

    try:

        df = pd.read_csv(file_path)

        df.fillna("", inplace=True)

        logger.debug(
            "Data loaded from %s",
            file_path
        )

        return df

    except Exception as e:

        logger.error(
            "Error loading data: %s",
            e
        )

        raise


def load_model(model_path):

    try:

        with open(model_path, "rb") as file:

            model = pickle.load(file)

        logger.debug(
            "Model loaded successfully"
        )

        return model

    except Exception as e:

        logger.error(
            "Error loading model: %s",
            e
        )

        raise


def load_vectorizer(vectorizer_path):

    try:

        with open(
            vectorizer_path,
            "rb"
        ) as file:

            vectorizer = pickle.load(file)

        logger.debug(
            "Vectorizer loaded successfully"
        )

        return vectorizer

    except Exception as e:

        logger.error(
            "Error loading vectorizer: %s",
            e
        )

        raise


def load_params(params_path):

    try:

        with open(
            params_path,
            "r"
        ) as file:

            params = yaml.safe_load(file)

        logger.debug(
            "Parameters loaded successfully"
        )

        return params

    except Exception as e:

        logger.error(
            "Error loading parameters: %s",
            e
        )

        raise


def evaluate_model(
    model,
    X_test,
    y_test
):

    try:

        y_pred = model.predict(X_test)

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True
        )

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        logger.debug(
            "Model evaluation completed"
        )

        return report, cm

    except Exception as e:

        logger.error(
            "Error during model evaluation: %s",
            e
        )

        raise


def save_confusion_matrix(
    cm,
    file_path
):

    try:

        plt.figure(
            figsize=(8, 6)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues"
        )

        plt.title(
            "Confusion Matrix"
        )

        plt.xlabel(
            "Predicted"
        )

        plt.ylabel(
            "Actual"
        )

        plt.savefig(
            file_path
        )

        plt.close()

        logger.debug(
            "Confusion matrix saved successfully"
        )

    except Exception as e:

        logger.error(
            "Error saving confusion matrix: %s",
            e
        )

        raise


def save_model_info(
    run_id,
    model_path,
    file_path
):

    try:

        model_info = {
            "run_id": run_id,
            "model_path": model_path
        }

        with open(
            file_path,
            "w"
        ) as file:

            json.dump(
                model_info,
                file,
                indent=4
            )

        logger.debug(
            "Model information saved successfully"
        )

    except Exception as e:

        logger.error(
            "Error saving model information: %s",
            e
        )

        raise


def main():

    try:

        root_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../../"
            )
        )

        # Local MLflow database
        mlflow.set_tracking_uri(
            "sqlite:///mlflow.db"
        )

        mlflow.set_experiment(
            "sentiment-analysis"
        )

        with mlflow.start_run() as run:

            # Load parameters
            params = load_params(
                os.path.join(
                    root_dir,
                    "params.yaml"
                )
            )

            model_params = params[
                "model_building"
            ]

            # Log model parameters
            mlflow.log_params(
                model_params
            )

            # Load model
            model = load_model(
                os.path.join(
                    root_dir,
                    "lgbm_model.pkl"
                )
            )

            # Load TF-IDF vectorizer
            vectorizer = load_vectorizer(
                os.path.join(
                    root_dir,
                    "tfidf_vectorizer.pkl"
                )
            )

            # Load test data
            test_data = load_data(
                os.path.join(
                    root_dir,
                    "data",
                    "interim",
                    "test_processed.csv"
                )
            )

            # Get test comments
            X_test = test_data[
                "clean_comment"
            ].values

            # Get actual labels
            y_test = test_data[
                "category"
            ].values

            # Transform comments using TF-IDF
            X_test_tfidf = vectorizer.transform(
                X_test
            )

            logger.info(
                "Test data transformed using TF-IDF"
            )

            # Evaluate model
            report, cm = evaluate_model(
                model,
                X_test_tfidf,
                y_test
            )

            # Log metrics
            for label, metrics in report.items():

                if isinstance(
                    metrics,
                    dict
                ):

                    mlflow.log_metric(
                        f"{label}_precision",
                        metrics["precision"]
                    )

                    mlflow.log_metric(
                        f"{label}_recall",
                        metrics["recall"]
                    )

                    mlflow.log_metric(
                        f"{label}_f1_score",
                        metrics["f1-score"]
                    )

            # Save confusion matrix
            confusion_matrix_path = os.path.join(
                root_dir,
                "confusion_matrix.png"
            )

            save_confusion_matrix(
                cm,
                confusion_matrix_path
            )

            # Log confusion matrix
            mlflow.log_artifact(
                confusion_matrix_path
            )

            # Create model signature
            input_example = X_test_tfidf[:5]

            prediction_example = model.predict(
                input_example
            )

            signature = infer_signature(
                input_example,
                prediction_example
            )

            # Log LightGBM model
            mlflow.lightgbm.log_model(
                model,
                "lgbm_model",
                signature=signature
            )

            # Log TF-IDF vectorizer
            mlflow.log_artifact(
                os.path.join(
                    root_dir,
                    "tfidf_vectorizer.pkl"
                )
            )

            # Save experiment information
            save_model_info(
                run.info.run_id,
                "lgbm_model",
                os.path.join(
                    root_dir,
                    "experiment_info.json"
                )
            )

            # Add tags
            mlflow.set_tag(
                "model",
                "LightGBM"
            )

            mlflow.set_tag(
                "task",
                "sentiment-analysis"
            )

            mlflow.set_tag(
                "dataset",
                "YouTube comments"
            )

            logger.info(
                "Model evaluation completed successfully"
            )

            print(
                "\nModel evaluation completed successfully!"
            )

            print(
                f"MLflow Run ID: {run.info.run_id}"
            )

    except Exception as e:

        logger.error(
            "Model evaluation failed: %s",
            e
        )

        print(
            f"Error: {e}"
        )


if __name__ == "__main__":
    main()