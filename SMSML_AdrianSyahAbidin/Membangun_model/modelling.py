"""
modelling.py
============
Pelatihan model dasar Credit Risk Prediction menggunakan MLflow autologging.
Menggunakan RandomForestClassifier tanpa hyperparameter tuning manual.

Penggunaan:
    python modelling.py

Catatan:
    - Script ini menggunakan MLflow autolog (Kriteria 2 Basic).
    - Untuk tracking ke DagsHub dengan tuning, gunakan modelling_tuning.py.
"""

import logging
import os
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Konfigurasi ─────────────────────────────────────────────────────────────
DATA_DIR    = "credit_risk_preprocessing"
TRAIN_FILE  = os.path.join(DATA_DIR, "credit_risk_train.csv")
TEST_FILE   = os.path.join(DATA_DIR, "credit_risk_test.csv")
TARGET_COL  = "loan_status"
EXPERIMENT  = "credit-risk-basic"


def load_data(train_path: str, test_path: str):
    """Memuat dataset training dan testing."""
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]
    X_test  = test.drop(columns=[TARGET_COL])
    y_test  = test[TARGET_COL]

    logger.info("Data dimuat: train=%s | test=%s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluasi model dan kembalikan dictionary metrik."""
    y_pred = model.predict(X_test)
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1_score":  f1_score(y_test, y_pred, zero_division=0),
    }


def train() -> None:
    """Menjalankan training model dengan MLflow autolog."""
    if not os.path.exists(TRAIN_FILE) or not os.path.exists(TEST_FILE):
        logger.error(
            "Dataset tidak ditemukan di '%s'. Jalankan preprocessing terlebih dahulu.",
            DATA_DIR,
        )
        sys.exit(1)

    X_train, X_test, y_train, y_test = load_data(TRAIN_FILE, TEST_FILE)

    mlflow.set_experiment(EXPERIMENT)
    mlflow.sklearn.autolog(log_models=True, log_datasets=False)

    logger.info("Memulai training dengan MLflow autolog...")

    with mlflow.start_run(run_name="random-forest-basic"):
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)
        logger.info("Hasil evaluasi model:")
        for name, value in metrics.items():
            logger.info("   %s: %.4f", name, value)

        report = classification_report(
            y_test, model.predict(X_test),
            target_names=["Non-Default", "Default"]
        )
        logger.info("\n%s", report)

    logger.info("✅ Training selesai. Buka MLflow UI: mlflow ui")


if __name__ == "__main__":
    train()
