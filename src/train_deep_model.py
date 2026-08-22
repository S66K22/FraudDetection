import logging

import numpy as np
import torch

import train

# logging.basicConfig(
#     filename="./reports/deep.log",
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     force=True,
# )

# Read dataset
data = np.load("data/preprocessed_data.npz")
X_train_scaled = data["X_train_scaled"]
X_test_scaled = data["X_test_scaled"]
X_train_time_scaled_iso_score = data["X_train_time_scaled_iso_score"]
X_test_time_scaled_iso_score = data["X_test_time_scaled_iso_score"]
y_train = data["y_train"]
y_test = data["y_test"]

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# logging.info(
#     "=======================Training X_train_scaled============================="
# )
# fold_results = train.loop(
#     X_train_scaled,
#     y_train,
#     [[32, 32], [64, 32], [64, 64], [128, 64], [128, 128]],
#     logging,
#     device,
#     50,
# )
# logging.info(
#     "\n\n=======================Training X_train_time_scaled_iso_score=============================\n\n"
# )
# fold_results = train.loop(
#     X_train_time_scaled_iso_score,
#     y_train,
#     [[32, 32], [64, 32], [64, 64], [128, 64], [128, 128]],
#     logging,
#     device,
#     50,
# )

logging.basicConfig(
    filename="./reports/deep_eval.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)

logging.info(
    "=======================Training X_train_scaled on full dataset============================="
)

hidden_layer = [128, 128]

model, history, test_metrics = train.train_deep_model_on_full_train(
    X_train=X_train_scaled,
    y_train=y_train,
    X_test=X_test_scaled,
    y_test=y_test,
    hidden_layer=[128, 128],
    device=device,
    logging=logging,
    n_epochs=20,
    model_path="models/fraud_model_final.pth",
)

cm = train.plot_confusion_matrix(
    model=model,
    X_test=X_test_scaled,
    y_test=y_test,
    device=device,
    path_to_save="reports/deep_model1.png",
    threshold=0.5,
)

logging.info(
    "\n\n=======================Training X_train_time_scaled_iso_score on full dataset=============================\n\n"
)

model, history, test_metrics = train.train_deep_model_on_full_train(
    X_train=X_train_time_scaled_iso_score,
    y_train=y_train,
    X_test=X_test_time_scaled_iso_score,
    y_test=y_test,
    hidden_layer=[128, 128],
    device=device,
    logging=logging,
    n_epochs=20,
    model_path="models/fraud_model_final.pth",
)

cm = train.plot_confusion_matrix(
    model=model,
    X_test=X_test_time_scaled_iso_score,
    y_test=y_test,
    device=device,
    path_to_save="reports/deep_model2.png",
    threshold=0.5,
)
