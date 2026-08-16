import logging

import numpy as np
from sklearn.tree import DecisionTreeClassifier

import train

# Set logging config
logging.basicConfig(
    filename="./reports/dt.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)

# Read dataset
data = np.load("data/preprocessed_data.npz")
X_train_scaled = data["X_train_scaled"]
X_test_scaled = data["X_test_scaled"]
X_train_time_scaled_iso_score = data["X_train_time_scaled_iso_score"]
X_test_time_scaled_iso_score = data["X_test_time_scaled_iso_score"]
y_train = data["y_train"]
y_test = data["y_test"]

# Train Decision Tree Classifier
dt_param_grid = {
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
    "max_features": [None, "sqrt"],
    "class_weight": [
        None,
        "balanced",
        {0: 1, 1: 2},
        {0: 1, 1: 5},
        {0: 1, 1: 10},
        {0: 1, 1: 20},
    ]
}

dt_clf_grid_search_1 = train.train(
    DecisionTreeClassifier(random_state=42),
    "dt1",
    dt_param_grid,
    train.scoring,
    train.refit,
    logging,
    X_train_scaled,
    y_train,
    "./models/best_dt_clf1",
    cv=5,
)

train.plot_cm(
    dt_clf_grid_search_1.best_estimator_,
    X_test_scaled,
    y_test,
    "reports/dt_cm1.png",
)

train.calc_metric(logging, dt_clf_grid_search_1.best_estimator_, X_test_scaled, y_test)


dt_clf_grid_search_2 = train.train(
    DecisionTreeClassifier(random_state=42),
    "dt2",
    dt_param_grid,
    train.scoring,
    train.refit,
    logging,
    X_train_time_scaled_iso_score,
    y_train,
    "./models/best_dt_clf2",
    cv=5,
)

train.plot_cm(
    dt_clf_grid_search_2.best_estimator_,
    X_test_time_scaled_iso_score,
    y_test,
    "reports/dt_cm2.png",
)

train.calc_metric(logging, dt_clf_grid_search_2.best_estimator_, X_test_time_scaled_iso_score, y_test)
