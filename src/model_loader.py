import joblib


MODEL_PATHS = {
    "logistic regression": "models/logistic_regression.pkl",
    "decision tree": "models/decision_tree.pkl",
    "random forest": "models/random_forest.pkl",
    "knn": "models/knn.pkl",
}


_models = {}


def load_model(model_name: str):
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unknown model: {model_name}")

    if model_name not in _models:
        _models[model_name] = joblib.load(MODEL_PATHS[model_name])

    return _models[model_name]