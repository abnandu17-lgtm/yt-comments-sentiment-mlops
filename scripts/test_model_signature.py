import pickle
import os
import numpy as np


def test_model_signature():

    model_path = "lgbm_model.pkl"

    assert os.path.exists(
        model_path
    ), "Model file does not exist"

    with open(
        model_path,
        "rb"
    ) as file:

        model = pickle.load(file)

    # Create sample TF-IDF-like input
    sample_input = np.zeros(
        (1, 10000)
    )

    prediction = model.predict(
        sample_input
    )

    assert prediction is not None

    assert len(prediction) == 1

    print(
        "Model signature test passed"
    )


if __name__ == "__main__":
    test_model_signature()