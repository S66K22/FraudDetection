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


def plot_cm(model, X, y, path_to_save):
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
            metric.update(y_pred, y_batch)
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

    best_val_roc_auc = -float("inf")
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

            metric.update(y_pred, y_batch)

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

        val_roc_auc = valid_metrics["roc_auc"]

        if val_roc_auc > best_val_roc_auc:

            best_val_roc_auc = val_roc_auc
            best_epoch = epoch

            best_model_state = copy.deepcopy(model.state_dict())

            logging.info(
                f"Epoch {epoch + 1}: "
                f"new best validation ROC-AUC = "
                f"{best_val_roc_auc:.4f}"
            )

        # -------------------------
        # Scheduler
        # -------------------------

        scheduler.step(val_roc_auc)

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

    return history, best_val_roc_auc, best_epoch


def deep_model_loop(X, y, logging, device):

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.float32,
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
            hidden_dims=[64, 32],
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

        history, best_val_roc_auc, best_epoch = train_deep_model(
            logging=logging,
            model=model,
            device=device,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metric=metrics,
            train_loader=train_loader,
            valid_loader=valid_loader,
            n_epochs=20,
        )

        # The model has already been
        # restored to the best epoch.

        best_scores = history["valid_metrics"][best_epoch]

        fold_results.append(best_scores)

        # -------------------------
        # Save best model
        # -------------------------

        model_path = f"fraud_model_fold_{fold}.pth"

        torch.save(
            model.state_dict(),
            model_path,
        )

        # -------------------------
        # Logging
        # -------------------------

        logging.info(f"Best epoch: {best_epoch + 1}")

        logging.info(f"Best validation ROC-AUC: " f"{best_val_roc_auc:.4f}")

        logging.info(f"Fold {fold} results:")

        for name, value in best_scores.items():
            logging.info(f"{name}: {value:.4f}")

        logging.info(f"Model saved to {model_path}")

    return fold_results


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
