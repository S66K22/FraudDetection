import logging

import numpy as np
from sklearn.linear_model import LogisticRegression

import train

# Set logging config
logging.basicConfig(
    filename="./reports/log_regr.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)

# Train logistic regression with grid search cv
log_regr_param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "class_weight": [
        None,
        "balanced",
        {0: 1, 1: 2},
        {0: 1, 1: 5},
        {0: 1, 1: 10},
        {0: 1, 1: 20},
    ],
}

data = np.load("data/preprocessed_data.npz")
X_train_scaled = data["X_train_scaled"]
X_test_scaled = data["X_test_scaled"]
X_train_time_scaled_iso_score = data["X_train_time_scaled_iso_score"]
X_test_time_scaled_iso_score = data["X_test_time_scaled_iso_score"]
y_train = data["y_train"]
y_test = data["y_test"]

log_reg_grid_search_1 = train.train(
    LogisticRegression(random_state=42),
    "log_regr1",
    log_regr_param_grid,
    train.scoring,
    train.refit,
    logging,
    X_train_scaled,
    y_train,
    "models/best_log_regr1",
    cv=5,
)

train.plot_cm(
    log_reg_grid_search_1.best_estimator_,
    X_test_scaled,
    y_test,
    "reports/log_regr_cm1.png",
)

train.calc_metric(logging, log_reg_grid_search_1.best_estimator_, X_test_scaled, y_test)

log_reg_grid_search_2 = train.train(
    LogisticRegression(random_state=42),
    "log_regr2",
    log_regr_param_grid,
    train.scoring,
    train.refit,
    logging,
    X_train_time_scaled_iso_score,
    y_train,
    "./models/best_log_regr2",
    cv=5,
)

train.plot_cm(
    log_reg_grid_search_2.best_estimator_,
    X_test_time_scaled_iso_score,
    y_test,
    "reports/log_regr_cm2.png",
)

train.calc_metric(
    logging, log_reg_grid_search_2.best_estimator_, X_test_time_scaled_iso_score, y_test
)
