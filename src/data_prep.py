# Import Libraries
import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set logging config
logging.basicConfig(
    filename="./reports/data_cleaning.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)


# Read Dataset and Log Infos
def read_dataset(path):
    return pd.read_csv(path)


def log_data_info(df, logging):
    logging.info(f"df.shape = {df.shape}")
    logging.info(f"Number of Rows: {df.shape[0]}")
    logging.info(f"Number of Columns [Features + Target]: {df.shape[1]}")

    logging.info("DataFrame info:\n")
    logging.info(df.info())

    logging.info("Counts of each class")
    logging.info(df["Class"].value_counts())

    value_counts = df["Class"].value_counts()
    fraud_ratio = value_counts[1] / (value_counts[0] + value_counts[1])
    logging.info(f"Percent of fraud data: %{(fraud_ratio * 100):.2f}")

    columns = df.columns
    logging.info(f"Feature Names: {columns[:-1]}")

    if df.isna().sum().sum() == 0:
        logging.info("There is no missing value")
    else:
        logging.info("There are some missing values")


def remove_duplicates(df):
    df.drop_duplicates(inplace=True)


def check_non_numeric_columns(df):
    return df.select_dtypes(include="object").columns.any()


def create_hist_image(df, path_to_save):
    df.hist(figsize=(16, 12), bins=50, edgecolor="black")

    plt.tight_layout()
    plt.savefig(path_to_save, dpi=300, bbox_inches="tight")
    plt.close()


def split_dataset(df, test_size=0.2):
    X = df.drop(columns="Class")
    y = df["Class"]
    return train_test_split(X, y, stratify=y, random_state=42, test_size=test_size)


def preprocess_time_column(X, time_scaler=None, mode1=75_000, mode2=150_000, gamma=0.1):
    X_copy = X.copy()

    if time_scaler is None:
        time_scaler = StandardScaler()
        time_scaler.fit(X_copy[["Time"]])

    # Scale Time
    time_scaled = time_scaler.transform(X_copy[["Time"]])

    mode1_scaled = time_scaler.transform([[mode1]])
    mode2_scaled = time_scaler.transform([[mode2]])

    X_copy["time_sim_1"] = rbf_kernel(time_scaled, mode1_scaled, gamma=gamma).ravel()

    X_copy["time_sim_2"] = rbf_kernel(time_scaled, mode2_scaled, gamma=gamma).ravel()

    X_copy = X_copy.drop("Time", axis=1)

    return X_copy, time_scaler


def plot_similariy_hist(X, path_to_save):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(X["time_sim_1"])
    axes[0].set_title("Time Distribution")
    axes[0].set_xlabel("time_sim_1")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(X["time_sim_2"])
    axes[1].set_title("Time Distribution")
    axes[1].set_xlabel("time_sim_2")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.legend()
    plt.savefig(path_to_save, dpi=300, bbox_inches="tight")
    plt.close()


def scale_dataset(scaler, X_train, X_test):
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def create_isolation_score(X_train, X_test):
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()

    isolation_forest = IsolationForest(random_state=42)

    outlier_pred = isolation_forest.fit(X_train_copy)
    X_train_copy["anomaly_score"] = isolation_forest.decision_function(X_train_copy)
    X_test_copy["anomaly_score"] = isolation_forest.decision_function(X_test_copy)

    return (
        X_train_copy,
        X_test_copy,
        isolation_forest,
    )


if __name__ == "__main__":
    # Read dataset
    df = read_dataset("data/creditcard.csv")
    # Log Dataframe Information
    log_data_info(df, logging)

    # Log Duplicate Data Info
    logging.info(
        f"From {df.shape[0]} samples, there are {df.duplicated(keep=False).sum()} duplicates."
    )

    # Remove Duplicate Data
    remove_duplicates(df)

    # Log Duplicate Info After Removing Duplicates
    logging.info(
        f"From {df.shape[0]} samples, there are {df.duplicated(keep=False).sum()} duplicates."
    )

    # Check existance of non-numeric columns
    if check_non_numeric_columns(df):
        logging.info("There are not numeric columns")
    else:
        logging.info("All columns are numeric")

    # Log description of df
    logging.info(df.describe())
    # Create histogram image of dataset
    create_hist_image(df, "reports/dataset_hist.png")

    # Split Dataset
    X_train, X_test, y_train, y_test = split_dataset(df)

    # Preprocess time column and use rbf similarity
    X_train_time_scaled_, scaler = preprocess_time_column(X_train)
    X_test_time_scaled_, _ = preprocess_time_column(X_test)
    joblib.dump(scaler, "models/time_scaler.pkl")

    # Plot similarity histograms
    plot_similariy_hist(X_train_time_scaled_, "reports/time_similarity_train.png")
    plot_similariy_hist(X_test_time_scaled_, "reports/time_similarity_test.png")

    # Add Isolation Score
    X_train_time_scaled_iso_score_, X_test_time_scaled_iso_score_, isolation_forest = (
        create_isolation_score(X_train_time_scaled_, X_test_time_scaled_)
    )
    joblib.dump(isolation_forest, "models/isolation_forest.pkl")

    # Create Two Scaled Dataset
    # One with no feature engineering
    scaler = StandardScaler()
    X_train_scaled, X_test_scaled = scale_dataset(scaler, X_train, X_test)
    joblib.dump(scaler, "models/standard_scaler.pkl")

    # One with feature engineering
    scaler = StandardScaler()
    X_train_time_scaled_iso_score, X_test_time_scaled_iso_score = scale_dataset(
        scaler, X_train_time_scaled_iso_score_, X_test_time_scaled_iso_score_
    )
    joblib.dump(scaler, "models/time_iso_scaler.pkl")

    # Save preprocessed dataset
    np.savez(
        "data/preprocessed_data.npz",
        X_train_scaled=X_train_scaled,
        X_train_time_scaled_iso_score=X_train_time_scaled_iso_score,
        X_test_scaled=X_test_scaled,
        X_test_time_scaled_iso_score=X_test_time_scaled_iso_score,
        y_train=y_train,
        y_test=y_test,
    )
