"""
modelling_tuning.py
===================
Pelatihan model Credit Risk Prediction tingkat Advanced:
- Hyperparameter Tuning (RandomizedSearchCV)
- Manual logging MLflow (metrics, params, artifacts)
- Tracking tersimpan ke DagsHub
- Artefak visual: ROC-AUC Curve + Feature Importance Plot

Penggunaan:
    python modelling_tuning.py

Prasyarat:
    - Set environment variables DAGSHUB_TOKEN, atau login via CLI.
    - Dataset tersedia di credit_risk_preprocessing/.
"""

import logging
import os
import sys

import dagshub
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    classification_report,
)
from sklearn.model_selection import RandomizedSearchCV

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Konfigurasi ─────────────────────────────────────────────────────────────
DAGSHUB_OWNER  = "Adrian463588"
DAGSHUB_REPO   = "CreditRiskPrediction"
EXPERIMENT_NAME = "credit-risk-hyperparameter-tuning"
DATA_DIR        = "credit_risk_preprocessing"
TRAIN_FILE      = os.path.join(DATA_DIR, "credit_risk_train.csv")
TEST_FILE       = os.path.join(DATA_DIR, "credit_risk_test.csv")
TARGET_COL      = "loan_status"
ARTIFACT_DIR    = "artifacts"

# Hyperparameter search space dengan regulasi ketat (mencegah overfitting)
PARAM_DIST = {
    "n_estimators":       [100, 200, 300],
    "max_depth":          [8, 12, 16],
    "min_samples_split":  [5, 10],
    "min_samples_leaf":   [2, 4, 6],
    "max_features":       ["sqrt", "log2"],
}


# ─── Utility Functions ────────────────────────────────────────────────────────

def load_data(train_path: str, test_path: str):
    """Memuat dataset training dan testing dari CSV."""
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di '{DATA_DIR}'")

    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]
    X_test  = test.drop(columns=[TARGET_COL])
    y_test  = test[TARGET_COL]

    logger.info("Data dimuat — train: %s | test: %s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test


def compute_metrics(model, X_test, y_test) -> dict:
    """Hitung seluruh metrik evaluasi model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        "_y_pred":   y_pred,
        "_y_prob":   y_prob,
    }


def plot_roc_auc_curve(y_test, y_prob, save_path: str) -> None:
    """Buat dan simpan ROC-AUC Curve sebagai artefak visual."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score   = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2ECC71", lw=2.5,
             label=f"ROC Curve (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5,
             linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2ECC71")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC-AUC Curve — Credit Risk Prediction", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("✅ ROC-AUC Curve disimpan: %s", save_path)


def plot_feature_importance(model, feature_names, save_path: str, top_n: int = 20) -> None:
    """Buat dan simpan Feature Importance Plot sebagai artefak visual."""
    importances  = model.best_estimator_.feature_importances_
    indices      = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_values   = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))
    bars = ax.barh(range(top_n), top_values[::-1], color=colors[::-1], edgecolor="black", alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=10)
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances — Credit Risk Model",
                  fontsize=14, fontweight="bold")
    ax.axvline(x=np.mean(top_values), color="red", linestyle="--",
                linewidth=1.5, label=f"Mean Importance")
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("✅ Feature Importance Plot disimpan: %s", save_path)


# ─── Pipeline Training ────────────────────────────────────────────────────────

def train() -> None:
    """Menjalankan hyperparameter tuning + logging manual ke DagsHub MLflow."""
    if "DAGSHUB_TOKEN" in os.environ and "DAGSHUB_USER_TOKEN" not in os.environ:
        os.environ["DAGSHUB_USER_TOKEN"] = os.environ["DAGSHUB_TOKEN"]

    # Init DagsHub → Set MLflow Tracking URI ke DagsHub server
    dagshub.init(
        repo_owner=DAGSHUB_OWNER,
        repo_name=DAGSHUB_REPO,
        mlflow=True,
    )

    mlflow.set_experiment(EXPERIMENT_NAME)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data(TRAIN_FILE, TEST_FILE)
    feature_names = list(X_train.columns)

    logger.info("Memulai Hyperparameter Tuning (RandomizedSearchCV)...")

    with mlflow.start_run(run_name="rf-hyperparameter-tuning") as run:
        # ── Hyperparameter Tuning ──
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=PARAM_DIST,
            n_iter=30,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )
        search.fit(X_train, y_train)

        best_params = search.best_params_
        logger.info("Best params: %s", best_params)

        # ── Manual Logging: Parameters ──
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("n_iter_search", 30)
        mlflow.log_param("scoring", "roc_auc")

        # ── Evaluasi ──
        metrics = compute_metrics(search, X_test, y_test)
        y_pred  = metrics.pop("_y_pred")
        y_prob  = metrics.pop("_y_prob")

        # ── Manual Logging: Metrics ──
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        logger.info("Metrik evaluasi:")
        for k, v in metrics.items():
            logger.info("   %s: %.4f", k, v)

        report = classification_report(
            y_test, y_pred, target_names=["Non-Default", "Default"]
        )
        logger.info("\n%s", report)

        # ── Artefak 1: ROC-AUC Curve ──
        roc_path = os.path.join(ARTIFACT_DIR, "roc_auc_curve.png")
        plot_roc_auc_curve(y_test, y_prob, roc_path)
        mlflow.log_artifact(roc_path, artifact_path="plots")

        # ── Artefak 2: Feature Importance Plot ──
        fi_path = os.path.join(ARTIFACT_DIR, "feature_importance.png")
        plot_feature_importance(search, feature_names, fi_path)
        mlflow.log_artifact(fi_path, artifact_path="plots")

        # ── Log Model ──
        mlflow.sklearn.log_model(
            search.best_estimator_,
            artifact_path="credit-risk-model",
            registered_model_name="CreditRiskRandomForest",
        )

        logger.info("✅ Training selesai | Run ID: %s", run.info.run_id)
        logger.info("   Buka DagsHub MLflow UI untuk melihat hasil tracking.")


if __name__ == "__main__":
    try:
        train()
    except Exception as exc:
        logger.error("Training gagal: %s", exc)
        sys.exit(1)
