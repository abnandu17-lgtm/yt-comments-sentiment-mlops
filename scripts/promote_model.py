import mlflow
import logging


# MLflow tracking URI
mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)


# Logging configuration
logger = logging.getLogger("model_promotion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(
    "model_promotion_errors.log"
)
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def promote_model():

    try:

        model_name = "sentiment_model"

        client = mlflow.tracking.MlflowClient()

        # Get the model in Staging
        staging_versions = client.get_latest_versions(
            model_name,
            stages=["Staging"]
        )

        if not staging_versions:

            raise Exception(
                "No model found in Staging"
            )

        staging_version = staging_versions[0].version

        # Get current Production models
        production_versions = client.get_latest_versions(
            model_name,
            stages=["Production"]
        )

        # Archive old Production models
        for version in production_versions:

            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage="Archived"
            )

        # Promote Staging model
        client.transition_model_version_stage(
            name=model_name,
            version=staging_version,
            stage="Production"
        )

        print(
            "Model promoted successfully!"
        )

        print(
            f"Model name: {model_name}"
        )

        print(
            f"Model version: {staging_version}"
        )

        print(
            "Model stage: Production"
        )

    except Exception as e:

        logger.error(
            "Error during model promotion: %s",
            e
        )

        print(
            f"Error: {e}"
        )


if __name__ == "__main__":
    promote_model()