import argparse
import json

import joblib
import torch

MODEL_FILES = {
    "logistic regression": {
        "model": "models/logistic_regression.pkl",
        "params": "models/logistic_regression_params.json",
    },
    "decision tree": {
        "model": "models/decision_tree.pkl",
        "params": "models/decision_tree_params.json",
    },
    # "random forest": {
    #     "model": "models/random_forest.pkl",
    #     "params": "models/random_forest_params.json",
    # },
    "knn": {
        "model": "models/knn.pkl",
        "params": "models/knn_params.json",
    },
    "deep model": {
        "model": "models/deep_model.pt",
        "params": "models/deep_model_params.json",
    },
}


def load_model(model_name):
    files = MODEL_FILES[model_name]

    # Load model
    if model_name == "deep model":
        model = torch.load(
            files["model"],
            weights_only=False,
        )
    else:
        model = joblib.load(files["model"])

    return model


def main():
    parser = argparse.ArgumentParser(description="Fraud Detection Model")

    parser.add_argument(
        "--model",
        choices=MODEL_FILES.keys(),
        help="Model to load",
    )

    args = parser.parse_args()

    # Interactive model selection
    if args.model is None:
        models = list(MODEL_FILES.keys())

        print("\nAvailable models:")
        for i, model in enumerate(models, start=1):
            print(f"{i}. {model}")

        choice = int(input("\nWhich model do you want? "))

        if not 1 <= choice <= len(models):
            parser.error("Invalid model selection.")

        model_name = models[choice - 1]

    else:
        model_name = args.model

    # Load model and parameters
    model, params = load_model(model_name)

    print(f"\nModel: {model_name}")
    print(f"Best parameters: {params}")

    return model, params


if __name__ == "__main__":
    model, params = main()
