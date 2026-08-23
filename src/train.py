import copy
import json

import joblib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchmetrics
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import MetricCollection

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}

refit = "pr_auc"


def plot_cm_from_estimator(model, X, y, path_to_save):
    ConfusionMatrixDisplay.from_estimator(model, X, y, cmap="Blues")

    plt.title("Confusion Matrix - Best Model")
    plt.savefig(path_to_save, dpi=300, bbox_inches="tight")
    plt.close()


def train(
    model,
    model_name,
    param_grid,
    scoring,
    refit,
    logging,
    X_train,
    y_train,
    path_to_save,
    cv=5,
):

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        refit=refit,
        cv=cv_strategy,
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    results = grid_search.cv_results_
    i = grid_search.best_index_

    logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")

    logging.info(f"Accuracy : {results['mean_test_accuracy'][i]:.4f}")
    logging.info(f"Precision: {results['mean_test_precision'][i]:.4f}")
    logging.info(f"Recall   : {results['mean_test_recall'][i]:.4f}")
    logging.info(f"F1       : {results['mean_test_f1'][i]:.4f}")
    logging.info(f"ROC-AUC  : {results['mean_test_roc_auc'][i]:.4f}")
    logging.info(f"PR-AUC   : {results['mean_test_pr_auc'][i]:.4f}")

    joblib.dump(grid_search.best_estimator_, path_to_save + ".pkl")

    with open(path_to_save + "_params.json", "w") as f:
        json.dump(grid_search.best_params_, f, indent=4)

    return grid_search


class FraudDetectionModel(nn.Module):
    def __init__(self, in_features, hidden_dims):
        super().__init__()
        layers = []
        input_features = in_features

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_features, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Sigmoid())
            input_features = hidden_dim

        layers.append(nn.Linear(input_features, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)


def create_model(in_features, hidden_dims):
    return FraudDetectionModel(in_features, hidden_dims)


def evaluate_tm(model, device, data_loader, metric):
    model.eval()
    metric.reset()
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            target = y_batch.long().view_as(y_pred).squeeze(-1)
            probs = torch.sigmoid(y_pred).squeeze(-1)
            metric.update(probs, target)
    return metric.compute()


def train_deep_model(
    logging,
    model,
    device,
    optimizer,
    loss_fn,
    metric,
    train_loader,
    valid_loader,
    n_epochs,
    patience=2,
    factor=0.5,
    epoch_callback=None,
):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=patience,
        factor=factor,
    )

    history = {
        "train_losses": [],
        "train_metrics": [],
        "valid_metrics": [],
    }

    best_valid_pr_auc = -float("inf")
    best_model_state = None
    best_epoch = -1

    for epoch in range(n_epochs):
        # -------------------------
        # Training
        # -------------------------

        model.train()
        metric.reset()

        total_loss = 0.0

        if epoch_callback is not None:
            epoch_callback(model, epoch)

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float().view(-1, 1)

            optimizer.zero_grad()

            y_pred = model(X_batch)

            loss = loss_fn(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            target = y_batch.long().view_as(y_pred).squeeze(-1)

            probs = torch.sigmoid(y_pred).squeeze(-1)

            metric.update(probs, target)

        train_loss = total_loss / len(train_loader)

        train_metrics = {name: value.item() for name, value in metric.compute().items()}

        # -------------------------
        # Validation
        # -------------------------

        valid_metrics = evaluate_tm(
            model=model,
            device=device,
            data_loader=valid_loader,
            metric=metric,
        )

        valid_metrics = {name: value.item() for name, value in valid_metrics.items()}

        # -------------------------
        # Save best model
        # -------------------------

        valid_pr_auc = valid_metrics["pr_auc"]

        if valid_pr_auc > best_valid_pr_auc:
            best_valid_pr_auc = valid_pr_auc
            best_epoch = epoch

            best_model_state = copy.deepcopy(model.state_dict())

            logging.info(
                f"Epoch {epoch + 1}: "
                f"new best validation PR-AUC = "
                f"{best_valid_pr_auc:.4f}"
            )

        # -------------------------
        # Scheduler
        # -------------------------

        scheduler.step(valid_pr_auc)

        # -------------------------
        # History
        # -------------------------

        history["train_losses"].append(train_loss)
        history["train_metrics"].append(train_metrics)
        history["valid_metrics"].append(valid_metrics)

        logging.info(
            f"Epoch {epoch + 1}/{n_epochs}, "
            f"loss={train_loss:.4f}, "
            f"train_acc={train_metrics['accuracy']:.4f}, "
            f"train_precision={train_metrics['precision']:.4f}, "
            f"train_recall={train_metrics['recall']:.4f}, "
            f"train_f1={train_metrics['f1']:.4f}, "
            f"train_roc_auc={train_metrics['roc_auc']:.4f}, "
            f"train_pr_auc={train_metrics['pr_auc']:.4f}, "
            f"valid_acc={valid_metrics['accuracy']:.4f}, "
            f"valid_precision={valid_metrics['precision']:.4f}, "
            f"valid_recall={valid_metrics['recall']:.4f}, "
            f"valid_f1={valid_metrics['f1']:.4f}, "
            f"valid_roc_auc={valid_metrics['roc_auc']:.4f}, "
            f"valid_pr_auc={valid_metrics['pr_auc']:.4f}"
        )

    # -------------------------
    # Restore best model
    # -------------------------

    model.load_state_dict(best_model_state)

    return history, best_valid_pr_auc, best_epoch


def deep_model_cv(X, y, hidden_layer, model_index, logging, device, n_epochs):
    model_path = ""

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.int32,
    )

    dataset = TensorDataset(
        X_tensor,
        y_tensor,
    )

    n_splits = 5

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    fold_results = []

    for fold, (train_idx, valid_idx) in enumerate(
        skf.split(X, y),
        start=1,
    ):
        logging.info(f"\n========== Fold {fold}/{n_splits} ==========")

        # -------------------------
        # Datasets
        # -------------------------

        train_dataset = torch.utils.data.Subset(
            dataset,
            train_idx,
        )

        valid_dataset = torch.utils.data.Subset(
            dataset,
            valid_idx,
        )

        # -------------------------
        # DataLoaders
        # -------------------------

        train_loader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True,
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=64,
            shuffle=False,
        )

        # -------------------------
        # New model for this fold
        # -------------------------

        model = FraudDetectionModel(
            in_features=X.shape[1],
            hidden_dims=hidden_layer,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
        )

        loss_fn = torch.nn.BCEWithLogitsLoss()

        # -------------------------
        # Metrics
        # -------------------------

        metrics = MetricCollection(
            {
                "accuracy": torchmetrics.Accuracy(task="binary"),
                "precision": torchmetrics.Precision(task="binary"),
                "recall": torchmetrics.Recall(task="binary"),
                "f1": torchmetrics.F1Score(task="binary"),
                "roc_auc": torchmetrics.AUROC(task="binary"),
                "pr_auc": torchmetrics.AveragePrecision(task="binary"),
            }
        ).to(device)

        # -------------------------
        # Train
        # -------------------------

        history, best_valid_pr_auc, best_epoch = train_deep_model(
            logging=logging,
            model=model,
            device=device,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metric=metrics,
            train_loader=train_loader,
            valid_loader=valid_loader,
            n_epochs=n_epochs,
        )

        # The model has already been
        # restored to the best epoch.

        best_scores = history["valid_metrics"][best_epoch]

        fold_results.append(best_scores)

        # -------------------------
        # Save best model
        # -------------------------

        model_path = f"models/fraud_model_index_{model_index}_fold_{fold}.pth"

        torch.save(
            model.state_dict(),
            model_path,
        )

        # -------------------------
        # Logging
        # -------------------------

        logging.info(f"Best epoch: {best_epoch + 1}")

        logging.info(f"Best validation PR-AUC: {best_valid_pr_auc:.4f}")

        logging.info(f"Fold {fold} results:")

        for name, value in best_scores.items():
            logging.info(f"{name}: {value:.4f}")

        logging.info(f"Model saved to {model_path}")

    return fold_results


def loop(X, y, hidden_layers, logging, device, n_epochs):
    results = []
    for i, hidden_layer in enumerate(hidden_layers):
        logging.info(f"================== Training Model {i} ==================")
        results.append(deep_model_cv(X, y, hidden_layer, i, logging, device, n_epochs))
    return results


def calc_metric(logging, model, X, y):
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    else:
        y_score = model.decision_function(X)

    logging.info("*********************Test Metrics*********************")
    logging.info(f"Accuracy : {accuracy_score(y, y_pred):.4f}")
    logging.info(f"Precision: {precision_score(y, y_pred):.4f}")
    logging.info(f"Recall   : {recall_score(y, y_pred):.4f}")
    logging.info(f"F1       : {f1_score(y, y_pred):.4f}")
    logging.info(f"ROC-AUC  : {roc_auc_score(y, y_score):.4f}")
    logging.info(f"PR-AUC   : {average_precision_score(y, y_score):.4f}")


def train_deep_model_final(
    logging,
    model,
    device,
    optimizer,
    loss_fn,
    train_loader,
    n_epochs,
):
    """Train the final model on the complete training dataset. There is no validation set in this stage. The number of epochs should be determined from cross-validation."""
    history = {
        "train_losses": [],
        "train_metrics": [],
    }
    metrics = torchmetrics.MetricCollection(
        {
            "accuracy": torchmetrics.Accuracy(task="binary"),
            "precision": torchmetrics.Precision(task="binary"),
            "recall": torchmetrics.Recall(task="binary"),
            "f1": torchmetrics.F1Score(task="binary"),
            "roc_auc": torchmetrics.AUROC(task="binary"),
            "pr_auc": torchmetrics.AveragePrecision(task="binary"),
        }
    ).to(device)

    for epoch in range(n_epochs):
        model.train()
        metrics.reset()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float().view(-1, 1)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            target = y_batch.long().view_as(y_pred).squeeze(-1)
            probs = torch.sigmoid(y_pred).squeeze(-1)
            metrics.update(probs, target)

        train_loss = total_loss / len(train_loader)
        train_metrics = {
            name: value.item() for name, value in metrics.compute().items()
        }
        history["train_losses"].append(train_loss)
        history["train_metrics"].append(train_metrics)
        logging.info(
            f"Epoch {epoch + 1}/{n_epochs}, "
            f"loss={train_loss:.4f}, "
            f"accuracy={train_metrics['accuracy']:.4f}, "
            f"precision={train_metrics['precision']:.4f}, "
            f"recall={train_metrics['recall']:.4f}, "
            f"f1={train_metrics['f1']:.4f}, "
            f"roc_auc={train_metrics['roc_auc']:.4f}, "
            f"pr_auc={train_metrics['pr_auc']:.4f}"
        )

    return history


def evaluate_deep_model(
    model,
    device,
    test_loader,
):
    metrics = torchmetrics.MetricCollection(
        {
            "accuracy": torchmetrics.Accuracy(task="binary"),
            "precision": torchmetrics.Precision(task="binary"),
            "recall": torchmetrics.Recall(task="binary"),
            "f1": torchmetrics.F1Score(task="binary"),
            "roc_auc": torchmetrics.AUROC(task="binary"),
            "pr_auc": torchmetrics.AveragePrecision(task="binary"),
        }
    ).to(device)

    test_metrics = evaluate_tm(
        model=model,
        device=device,
        data_loader=test_loader,
        metric=metrics,
    )
    return {name: value.item() for name, value in test_metrics.items()}


def train_deep_model_on_full_train(
    X_train,
    y_train,
    X_test,
    y_test,
    hidden_layer,
    device,
    logging,
    n_epochs,
    model_path,
):
    """Train a new model on the complete training dataset and evaluate it on the untouched test dataset."""
    # --------------------------------------------------
    # Convert data to tensors
    # --------------------------------------------------
    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32,
    )
    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.int32,
    )
    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32,
    )

    y_test_tensor = torch.tensor(
        y_test,
        dtype=torch.int32,
    )
    # --------------------------------------------------
    # Datasets
    # --------------------------------------------------
    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor,
    )

    test_dataset = TensorDataset(
        X_test_tensor,
        y_test_tensor,
    )
    # --------------------------------------------------
    # DataLoaders
    # --------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
    )
    # --------------------------------------------------
    # Create a NEW model
    # --------------------------------------------------
    model = FraudDetectionModel(
        in_features=X_train.shape[1],
        hidden_dims=hidden_layer,
    ).to(device)
    # --------------------------------------------------
    # Optimizer and loss
    # --------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    # --------------------------------------------------
    # Train on ALL training data
    # --------------------------------------------------
    logging.info("================ FINAL MODEL TRAINING ================")
    logging.info(f"Architecture: {hidden_layer}")

    logging.info(f"Training samples: {len(train_dataset)}")
    logging.info(f"Epochs: {n_epochs}")
    history = train_deep_model_final(
        logging=logging,
        model=model,
        device=device,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        n_epochs=n_epochs,
    )

    # --------------------------------------------------
    # Save final model
    # --------------------------------------------------
    torch.save(
        model.state_dict(),
        model_path,
    )
    logging.info(f"Final model saved to {model_path}")
    # --------------------------------------------------
    # Evaluate on TEST set
    # -------------------------------------------------
    logging.info("================ FINAL TEST EVALUATION ================")
    test_metrics = evaluate_deep_model(
        model=model,
        device=device,
        test_loader=test_loader,
    )
    for name, value in test_metrics.items():
        logging.info(f"test_{name}={value:.4f}")

    return model, history, test_metrics


import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion_matrix(
    model,
    X_test,
    y_test,
    device,
    path_to_save,
    threshold=0.5,
):
    model.eval()

    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32,
    ).to(device)

    with torch.no_grad():
        logits = model(X_test_tensor)
        probs = torch.sigmoid(logits).squeeze(-1)

    # Convert probabilities to binary predictions
    y_pred = (probs >= threshold).long().cpu().numpy()

    y_true = np.asarray(y_test).astype(int)

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-Fraud", "Fraud"],
    )

    disp.plot()
    plt.title(f"Confusion Matrix (Threshold = {threshold:.2f})")
    plt.tight_layout()
    plt.savefig(path_to_save, dpi=300, bbox_inches="tight")
    plt.close()

    return cm
