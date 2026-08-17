import json
import mlflow
import logging
import os


# Set up MLflow tracking URI
mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)


# logging configuration
logger = logging.getLogger("model_registration")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(
    "model_registration_errors.log"
)
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model_info(file_path):

    try:

        with open(
            file_path,
            "r"
        ) as file:

            model_info = json.load(file)

        logger.debug(
            "Model info loaded from %s",
            file_path
        )

        return model_info

    except FileNotFoundError:

        logger.error(
            "File not found: %s",
            file_path
        )

        raise

    except Exception as e:

        logger.error(
            "Error loading model info: %s",
            e
        )

        raise


def register_model(
    model_name,
    model_info
):

    try:

        # Create model URI
        model_uri = (
            f"runs:/{model_info['run_id']}/"
            f"{model_info['model_path']}"
        )

        # Register model
        model_version = mlflow.register_model(
            model_uri,
            model_name
        )

        # Create MLflow client
        client = mlflow.tracking.MlflowClient()

        # Move model to Staging
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        logger.debug(
            "Model %s version %s registered "
            "and moved to Staging",
            model_name,
            model_version.version
        )

        print(
            "Model registered successfully!"
        )

        print(
            f"Model name: {model_name}"
        )

        print(
            f"Model version: {model_version.version}"
        )

        print(
            "Model stage: Staging"
        )

    except Exception as e:

        logger.error(
            "Error during model registration: %s",
            e
        )

        raise


def main():

    try:

        # Project root directory
        root_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../../"
            )
        )

        # Path to experiment information
        model_info_path = os.path.join(
            root_dir,
            "experiment_info.json"
        )

        # Load model information
        model_info = load_model_info(
            model_info_path
        )

        # Model name
        model_name = "sentiment_model"

        # Register model
        register_model(
            model_name,
            model_info
        )

    except Exception as e:

        logger.error(
            "Failed to complete model registration: %s",
            e
        )

        print(
            f"Error: {e}"
        )


if __name__ == "__main__":
    main()