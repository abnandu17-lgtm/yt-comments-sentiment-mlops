import os
import pickle


def test_load_model():

    model_path = "lgbm_model.pkl"

    assert os.path.exists(
        model_path
    ), "Model file does not exist"

    with open(
        model_path,
        "rb"
    ) as file:

        model = pickle.load(file)

    assert model is not None

    print("Model loaded successfully")


if __name__ == "__main__":
    test_load_model()