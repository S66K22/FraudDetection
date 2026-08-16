import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import train

# Set logging config
logging.basicConfig(
    filename="./reports/rf.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)

# Read dataset
data = np.load("data/preprocessed_data.npz")
X_train_scaled = data["X_train_scaled"]
X_train_time_scaled_iso_score = data["X_train_time_scaled_iso_score"]
y_train = data["y_train"]

# Train Random Forest
rf_param_grid = {
    "n_estimators": [100, 200, 400],
    "criterion": ["gini", "entropy"],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "max_features": ["sqrt", "log2", None],
    "class_weight": [
        None,
        "balanced",
        "balanced_subsample",
        {0: 1, 1: 2},
        {0: 1, 1: 5},
        {0: 1, 1: 10},
        {0: 1, 1: 20},
    ],
    "bootstrap": [True, False],
}

rf_clf_grid_search_1 = train.train(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    "rf1",
    rf_param_grid,
    train.scoring,
    train.refit,
    logging,
    X_train_scaled,
    y_train,
    "./models/best_rf_clf1",
    cv=5,
)

train.plot_cm(
    rf_clf_grid_search_1.best_estimator_,
    X_train_time_scaled_iso_score,
    y_train,
    "reports/rf_cm1.png",
)

rf_clf_grid_search_2 = train.train(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    "rf2",
    rf_param_grid,
    train.scoring,
    train.refit,
    logging,
    X_train_time_scaled_iso_score,
    y_train,
    "./models/best_rf_clf2",
    cv=5,
)

train.plot_cm(
    rf_clf_grid_search_2.best_estimator_,
    X_train_time_scaled_iso_score,
    y_train,
    "reports/rf_cm2.png",
)
