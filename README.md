# YouTube Comments Sentiment Analysis

A Machine Learning project that analyzes YouTube comments and predicts their sentiment.

## Goal

To learn and implement a simple end-to-end MLOps workflow for a Machine Learning project.

## MLOps Workflow

Data → Model → MLflow → DVC → Testing → Model Promotion → Docker → CI/CD

## Technologies Used

Python | Machine Learning | MLflow | DVC | Docker | GitHub | GitHub Actions | Docker Hub | AWS EC2

## Project Structure

- `src/` – Source code
- `scripts/` – Testing and model promotion
- `app.py` – Streamlit application
- `Dockerfile` – Docker configuration
- `dvc.yaml` – DVC pipeline
- `.github/workflows/` – CI/CD workflow

## ML Model

The project uses a Machine Learning model to analyze YouTube comments and predict their sentiment.

## MLOps

MLflow is used for experiment tracking, DVC is used for data and pipeline management, and testing and model promotion are performed before deployment.

## Docker

The application is containerized using Docker and the image is stored on Docker Hub.

## CI/CD

GitHub Actions automatically builds the Docker image and pushes it to Docker Hub whenever changes are pushed to the `master` branch.

## Deployment

The Dockerized application is deployed on AWS EC2 and tested successfully.

## Result

The application accepts a YouTube video URL, analyzes its comments, and displays the sentiment results.
