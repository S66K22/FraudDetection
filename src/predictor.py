import numpy as np

from src.model_loader import load_model


def predict(model_name: str, data: list[list[float]]) -> list[int]:
    model = load_model(model_name)

    X = np.asarray(data)

    predictions = model.predict(X)

    return predictions.astype(int).tolist()
