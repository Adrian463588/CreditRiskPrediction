"""
generate_dataset.py
===================
Script untuk menghasilkan synthetic Credit Risk Dataset yang representatif.
Digunakan untuk GitHub Actions CI (menghindari ketergantungan download Kaggle API).
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def generate_credit_risk_dataset(n_samples: int = 32581, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic credit risk dataset mencerminkan distribusi dataset Kaggle asli."""
    rng = np.random.RandomState(random_state)

    home_ownership = rng.choice(
        ["RENT", "MORTGAGE", "OWN", "OTHER"],
        n_samples,
        p=[0.50, 0.41, 0.07, 0.02],
    )
    loan_intent = rng.choice(
        ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"],
        n_samples,
    )
    loan_grade = rng.choice(
        ["A", "B", "C", "D", "E", "F", "G"],
        n_samples,
        p=[0.27, 0.29, 0.20, 0.14, 0.07, 0.02, 0.01],
    )
    cb_default = rng.choice(["N", "Y"], n_samples, p=[0.82, 0.18])

    person_age    = rng.randint(20, 80, n_samples)
    person_income = rng.lognormal(mean=10.7, sigma=0.7, size=n_samples).astype(int).clip(4000, 6_000_000)
    loan_amnt     = rng.randint(500, 35_000, n_samples)
    loan_int_rate = rng.uniform(5.42, 23.22, n_samples).round(2)
    emp_length    = rng.uniform(0, 41, n_samples).round(1)
    cred_hist_len = rng.randint(2, 30, n_samples)
    loan_pct_inc  = (loan_amnt / (person_income + 1)).round(2)

    # Simulasi probabilitas default dengan korelasi realistis
    grade_risk    = np.array([0, 1, 2, 3, 4, 5, 6])[[ord(g) - ord("A") for g in loan_grade]]
    default_prob  = (
        0.12
        + 0.04 * grade_risk / 6
        + 0.10 * (cb_default == "Y").astype(float)
        + 0.05 * (loan_pct_inc > 0.3).astype(float)
    ).clip(0, 1)
    loan_status = (rng.uniform(0, 1, n_samples) < default_prob).astype(int)

    # Inject missing values
    emp_length_arr    = emp_length.copy()
    loan_int_rate_arr = loan_int_rate.copy()
    emp_length_arr[rng.random(n_samples)    < 0.03] = np.nan
    loan_int_rate_arr[rng.random(n_samples) < 0.09] = np.nan

    df = pd.DataFrame({
        "person_age":                person_age,
        "person_income":              person_income,
        "person_home_ownership":      home_ownership,
        "person_emp_length":          emp_length_arr,
        "loan_intent":                loan_intent,
        "loan_grade":                 loan_grade,
        "loan_amnt":                  loan_amnt,
        "loan_int_rate":              loan_int_rate_arr,
        "loan_percent_income":        loan_pct_inc,
        "cb_person_default_on_file":  cb_default,
        "cb_person_cred_hist_length": cred_hist_len,
        "loan_status":                loan_status,
    })

    logger.info(
        "Dataset dihasilkan: %d baris | Default rate: %.1f%%",
        len(df),
        df["loan_status"].mean() * 100,
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic credit risk dataset.")
    parser.add_argument("--output", type=str, default="credit_risk_dataset.csv")
    parser.add_argument("--n-samples", type=int, default=32581)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    df = generate_credit_risk_dataset(n_samples=args.n_samples)
    df.to_csv(args.output, index=False)
    logger.info("Dataset disimpan ke: %s", args.output)
