import argparse
import json

import joblib
import pandas as pd
import torch

from src import (
    create_isolation_score,
    create_model,
    preprocess_time_column,
    scale_dataset,
)


def load_deep_model():
    model = create_model(32, [128, 128])
    state_dict = torch.load("models/fraud_model_final.pth", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_input_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    return pd.DataFrame([data])


def load_scaler(pathname):
    return joblib.load(pathname)


def deep_predict(model, inputs):
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        return torch.sigmoid(logits)


def main():

    threshold = 0.5

    # Load model and preprocessing objects only once
    model = load_deep_model()
    time_scaler = load_scaler("models/time_scaler.pkl")
    isolation_forest = load_scaler("models/isolation_forest.pkl")
    time_iso_scaler = load_scaler("models/time_iso_scaler.pkl")

    print("Fraud detection model is ready.")
    print("Enter JSON file path, or 'q' to quit.")

    while True:

        input_path = input("\nInput JSON file: ").strip()

        if input_path.lower() in {"q", "quit", "exit"}:
            print("Exiting...")
            break

        try:
            # Load Dataset
            X = load_input_data(input_path)

            # Preprocess to create iso time dataset
            X_time_scaled_, _ = preprocess_time_column(X, time_scaler)

            _, X_time_scaled_iso_score_, _ = create_isolation_score(
                None, X_time_scaled_, isolation_forest
            )

            _, X_time_scaled_iso_score = scale_dataset(
                time_iso_scaler, None, X_time_scaled_iso_score_
            )

            # Make predictions

            probs = deep_predict(model, torch.from_numpy(X_time_scaled_iso_score).float())

            class_id = (probs >= threshold).long()
            class_id = class_id.item()
            probs = probs.item()

            prediction1 = "Fraud" if class_id == 1 else "Non-Fraud"

            result_dict = {
                "prediction": prediction1,
                "class_id": class_id,
                "probability": probs,
                "threshold": threshold,
                "status": "success",
            }

            print("\nPrediction using time + isolation features:")
            print(result_dict)

        except Exception as e:
            print(f"\nPrediction failed: {e}")
            print(
                {
                    "prediction": 0,
                    "class_id": -1,
                    "probability": -1,
                    "threshold": threshold,
                    "status": "failure",
                }
            )


if __name__ == "__main__":
    main()
