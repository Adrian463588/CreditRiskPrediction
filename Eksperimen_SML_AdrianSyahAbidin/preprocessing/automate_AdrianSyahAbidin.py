"""
automate_AdrianSyahAbidin.py
============================
Script otomatisasi preprocessing dataset Credit Risk.
Mengikuti prinsip DRY (Don't Repeat Yourself) dan Clean Code.

Penggunaan:
    python automate_AdrianSyahAbidin.py --input <path_csv> --output-dir <dir>

Contoh:
    python automate_AdrianSyahAbidin.py \
        --input ../credit_risk_raw/credit_risk_dataset.csv \
        --output-dir credit_risk_preprocessing
"""

import argparse
import logging
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─── Konfigurasi Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Konstanta Konfigurasi ────────────────────────────────────────────────────
TARGET_COL = "loan_status"
TEST_SIZE = 0.2
RANDOM_STATE = 42

NUMERIC_IMPUTE_COLS = ["person_emp_length", "loan_int_rate"]

LABEL_ENCODE_MAPPING = {
    "loan_grade": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6},
    "cb_person_default_on_file": {"N": 0, "Y": 1},
}

OHE_COLS = ["person_home_ownership", "loan_intent"]

SCALE_COLS = [
    "person_income",
    "loan_amnt",
    "person_age",
    "person_emp_length",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "debt_to_income_ratio",
    "income_per_year_employed",
]


# ─── Fungsi Preprocessing ─────────────────────────────────────────────────────

def load_dataset(file_path: str) -> pd.DataFrame:
    """Memuat dataset dari file CSV ke DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")
    df = pd.read_csv(file_path)
    logger.info("Dataset dimuat: %s | Shape: %s", file_path, df.shape)
    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Imputasi missing values dengan nilai median untuk kolom numerik."""
    df = df.copy()
    for col in NUMERIC_IMPUTE_COLS:
        if col not in df.columns:
            logger.warning("Kolom '%s' tidak ditemukan, dilewati.", col)
            continue
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.info("Imputasi '%s': %d nilai → median=%.4f", col, n_missing, median_val)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Membuat fitur turunan yang relevan secara bisnis."""
    df = df.copy()
    df["debt_to_income_ratio"] = df["loan_amnt"] / (df["person_income"] + 1)
    df["income_per_year_employed"] = df["person_income"] / (df["person_emp_length"] + 1)
    logger.info("Fitur baru dibuat: debt_to_income_ratio, income_per_year_employed")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Menerapkan Label Encoding dan One-Hot Encoding pada variabel kategorikal."""
    df = df.copy()

    # Label Encoding untuk variabel ordinal dan binary
    for col, mapping in LABEL_ENCODE_MAPPING.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            logger.info("Label Encoded: '%s'", col)

    # One-Hot Encoding untuk variabel nominal
    existing_ohe_cols = [c for c in OHE_COLS if c in df.columns]
    if existing_ohe_cols:
        df = pd.get_dummies(df, columns=existing_ohe_cols, drop_first=False, dtype=int)
        logger.info("One-Hot Encoded: %s | Shape baru: %s", existing_ohe_cols, df.shape)

    return df


def scale_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Standarisasi fitur numerik menggunakan StandardScaler."""
    df = df.copy()
    existing_scale_cols = [c for c in SCALE_COLS if c in df.columns]

    if not existing_scale_cols:
        logger.warning("Tidak ada kolom untuk di-scale.")
        return df, None

    scaler = StandardScaler()
    df[existing_scale_cols] = scaler.fit_transform(df[existing_scale_cols])
    logger.info("StandardScaler diterapkan pada %d kolom.", len(existing_scale_cols))
    return df, scaler


def split_dataset(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset menjadi training dan testing set dengan stratifikasi."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train.values

    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values

    logger.info(
        "Split selesai: train=%s | test=%s | default_rate_train=%.1f%%",
        train_df.shape,
        test_df.shape,
        y_train.mean() * 100,
    )
    return train_df, test_df


def save_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str) -> None:
    """Simpan dataset hasil preprocessing ke direktori output."""
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "credit_risk_train.csv")
    test_path = os.path.join(output_dir, "credit_risk_test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info("Disimpan: %s (%d baris)", train_path, len(train_df))
    logger.info("Disimpan: %s (%d baris)", test_path, len(test_df))


# ─── Pipeline Utama ───────────────────────────────────────────────────────────

def run_preprocessing_pipeline(input_path: str, output_dir: str) -> None:
    """Menjalankan seluruh pipeline preprocessing dari raw data ke dataset siap latih."""
    logger.info("═" * 60)
    logger.info("MULAI: Pipeline Preprocessing Credit Risk")
    logger.info("═" * 60)

    # Step 1: Load
    df = load_dataset(input_path)

    # Step 2: Impute missing values
    df = impute_missing_values(df)

    # Step 3: Feature engineering
    df = engineer_features(df)

    # Step 4: Encode categorical
    df = encode_categorical(df)

    # Step 5: Scale features
    df, _ = scale_features(df)

    # Step 6: Split
    train_df, test_df = split_dataset(df)

    # Step 7: Save
    save_datasets(train_df, test_df, output_dir)

    logger.info("═" * 60)
    logger.info("SELESAI: Dataset siap untuk pelatihan model.")
    logger.info("Fitur akhir (%d kolom): %s", len(train_df.columns), list(train_df.columns))
    logger.info("═" * 60)


# ─── Entry Point ──────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Automate preprocessing pipeline untuk Credit Risk Dataset."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join("..", "credit_risk_raw", "credit_risk_dataset.csv"),
        help="Path ke file CSV raw dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="credit_risk_preprocessing",
        help="Direktori output untuk menyimpan dataset yang sudah diproses.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        run_preprocessing_pipeline(
            input_path=args.input,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        logger.error("Pipeline gagal: %s", exc)
        sys.exit(1)
