import logging

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

import train

# Set logging config
logging.basicConfig(
    filename="./reports/knn.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)

# Train KNN classifier with grid search cv
knn_clf_param_grid = {
    "n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"],
    "p": [1, 2],
}

# Read dataset
data = np.load("data/preprocessed_data.npz")
X_train_scaled = data["X_train_scaled"]
X_test_scaled = data["X_test_scaled"]
X_train_time_scaled_iso_score = data["X_train_time_scaled_iso_score"]
X_test_time_scaled_iso_score = data["X_test_time_scaled_iso_score"]
y_train = data["y_train"]
y_test = data["y_test"]

knn_clf_grid_search_1 = train.train(
    KNeighborsClassifier(),
    "knn1",
    knn_clf_param_grid,
    train.scoring,
    train.refit,
    logging,
    X_train_scaled,
    y_train,
    "./models/best_knn_clf1",
    cv=5,
)

train.plot_cm_from_estimator(
    knn_clf_grid_search_1.best_estimator_,
    X_test_scaled,
    y_test,
    "reports/knn_cm1.png",
)

train.calc_metric(logging, knn_clf_grid_search_1.best_estimator_, X_test_scaled, y_test)

knn_clf_grid_search_2 = train.train(
    KNeighborsClassifier(),
    "knn2",
    knn_clf_param_grid,
    train.scoring,
    train.refit,
    logging,
    X_train_time_scaled_iso_score,
    y_train,
    "./models/best_knn_clf2",
    cv=5,
)

train.plot_cm_from_estimator(
    knn_clf_grid_search_2.best_estimator_,
    X_test_time_scaled_iso_score,
    y_test,
    "reports/knn_cm2.png",
)

train.calc_metric(
    logging, knn_clf_grid_search_2.best_estimator_, X_test_time_scaled_iso_score, y_test
)
