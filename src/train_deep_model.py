import logging

import numpy as np
import torch

import train

# Read dataset
data = np.load("data/preprocessed_data.npz")
X_train_scaled = data["X_train_scaled"]
X_train_time_scaled_iso_score = data["X_train_time_scaled_iso_score"]
y_train = data["y_train"]

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

train.deep_model_loop(X_train_scaled, y_train, logging, device)
