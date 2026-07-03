"""
7.Inference.py
==============
Script simulasi traffic untuk Credit Risk Prediction API.
Mensimulasikan berbagai skenario pengajuan pinjaman untuk:
    1. Mengisi metrik Prometheus dengan data realistis
    2. Memicu alert "Loan Approval Rate Critically Low" (< 10%) pada mode krisis

Cara menjalankan:
    # Mode normal (default)
    python 7.Inference.py

    # Mode krisis — simulasi drift untuk trigger alert approval rate
    python 7.Inference.py --mode crisis

    # Custom jumlah request dan interval
    python 7.Inference.py --requests 500 --interval 0.2
"""

import argparse
import logging
import random
import sys
import time
from typing import Dict, Any

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Konfigurasi ─────────────────────────────────────────────────────────────
API_BASE_URL   = "http://localhost:8000"
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"
HEALTH_ENDPOINT  = f"{API_BASE_URL}/health"


# ─── Profil Nasabah ───────────────────────────────────────────────────────────

NORMAL_APPLICANTS = [
    # Profil low-risk (high income, moderate loan)
    {"person_income": 85000, "person_age": 35, "loan_amnt": 10000, "loan_int_rate": 8.5,
     "person_emp_length": 8.0, "loan_percent_income": 0.12, "cb_person_cred_hist_length": 10},
    {"person_income": 65000, "person_age": 42, "loan_amnt": 5000, "loan_int_rate": 7.2,
     "person_emp_length": 12.0, "loan_percent_income": 0.08, "cb_person_cred_hist_length": 15},
    {"person_income": 120000, "person_age": 38, "loan_amnt": 25000, "loan_int_rate": 9.0,
     "person_emp_length": 6.0, "loan_percent_income": 0.21, "cb_person_cred_hist_length": 8},
    {"person_income": 55000, "person_age": 29, "loan_amnt": 8000, "loan_int_rate": 11.0,
     "person_emp_length": 3.0, "loan_percent_income": 0.15, "cb_person_cred_hist_length": 5},
    {"person_income": 95000, "person_age": 45, "loan_amnt": 15000, "loan_int_rate": 8.0,
     "person_emp_length": 15.0, "loan_percent_income": 0.16, "cb_person_cred_hist_length": 20},
]

CRISIS_APPLICANTS = [
    # Profil high-risk (low income, large loan, high interest)
    {"person_income": 8000, "person_age": 22, "loan_amnt": 30000, "loan_int_rate": 22.0,
     "person_emp_length": 0.5, "loan_percent_income": 0.75, "cb_person_cred_hist_length": 1},
    {"person_income": 5000, "person_age": 19, "loan_amnt": 25000, "loan_int_rate": 23.0,
     "person_emp_length": 0.0, "loan_percent_income": 0.90, "cb_person_cred_hist_length": 0},
    {"person_income": 12000, "person_age": 24, "loan_amnt": 28000, "loan_int_rate": 21.5,
     "person_emp_length": 1.0, "loan_percent_income": 0.80, "cb_person_cred_hist_length": 2},
    {"person_income": 9500, "person_age": 21, "loan_amnt": 32000, "loan_int_rate": 20.0,
     "person_emp_length": 0.0, "loan_percent_income": 0.85, "cb_person_cred_hist_length": 1},
]


# ─── Utility Functions ────────────────────────────────────────────────────────

def check_api_health(max_retries: int = 10) -> bool:
    """Cek apakah API aktif sebelum mengirim traffic."""
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(HEALTH_ENDPOINT, timeout=5)
            if response.status_code == 200:
                logger.info("✅ API aktif dan siap menerima request")
                return True
        except requests.ConnectionError:
            logger.warning("Attempt %d/%d: API belum siap...", attempt, max_retries)
            time.sleep(2)
    return False


def send_prediction(applicant_data: Dict[str, Any]) -> Dict[str, Any]:
    """Kirim satu request prediksi ke API dan kembalikan response."""
    try:
        response = requests.post(PREDICT_ENDPOINT, json=applicant_data, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        logger.error("Request gagal: %s", exc)
        return {}


def add_noise(applicant: Dict[str, Any], noise_factor: float = 0.1) -> Dict[str, Any]:
    """Tambahkan variasi kecil pada data nasabah untuk simulasi yang lebih realistis."""
    noisy = applicant.copy()
    noisy["person_income"] *= random.uniform(1 - noise_factor, 1 + noise_factor)
    noisy["loan_amnt"]     *= random.uniform(1 - noise_factor, 1 + noise_factor)
    noisy["person_age"]     = max(18, noisy["person_age"] + random.randint(-3, 3))
    return noisy


# ─── Mode Simulasi ────────────────────────────────────────────────────────────

def run_normal_simulation(n_requests: int, interval: float) -> None:
    """Simulasi traffic normal dengan campuran nasabah berisiko rendah dan sedang."""
    logger.info("═" * 60)
    logger.info("MODE NORMAL: Simulasi traffic pengajuan pinjaman")
    logger.info("Jumlah request: %d | Interval: %.2fs", n_requests, interval)
    logger.info("═" * 60)

    approved = rejected = errors = 0

    for i in range(1, n_requests + 1):
        applicant = add_noise(random.choice(NORMAL_APPLICANTS))
        result    = send_prediction(applicant)

        if result:
            pred = result.get("prediction", -1)
            conf = result.get("confidence_score", 0)
            if pred == 0:
                approved += 1
            elif pred == 1:
                rejected += 1
        else:
            errors += 1

        if i % 50 == 0 or i == n_requests:
            total = approved + rejected
            rate  = approved / total if total > 0 else 0
            logger.info(
                "Progress %d/%d | Approved: %d | Rejected: %d | Rate: %.1f%%",
                i, n_requests, approved, rejected, rate * 100,
            )

        time.sleep(interval)

    logger.info("═" * 60)
    logger.info("SELESAI | Approval Rate: %.1f%% | Errors: %d", approved / (approved + rejected + 1e-9) * 100, errors)


def run_crisis_simulation(n_requests: int, interval: float) -> None:
    """
    Simulasi skenario krisis: kirim nasabah berisiko sangat tinggi.
    Dirancang untuk memicu alert 'LoanApprovalRateCriticallyLow' (< 10%).
    """
    logger.info("═" * 60)
    logger.info("⚠️  MODE KRISIS: Simulasi data drift / nasabah berisiko tinggi")
    logger.info("Target: Memicu ALERT approval rate < 10%%")
    logger.info("Jumlah request: %d | Interval: %.2fs", n_requests, interval)
    logger.info("═" * 60)

    approved = rejected = 0

    for i in range(1, n_requests + 1):
        applicant = add_noise(random.choice(CRISIS_APPLICANTS), noise_factor=0.05)
        result    = send_prediction(applicant)

        if result:
            pred = result.get("prediction", -1)
            if pred == 0:
                approved += 1
            else:
                rejected += 1

        if i % 20 == 0 or i == n_requests:
            total = approved + rejected
            rate  = approved / total if total > 0 else 0
            status = "🔴 ALERT ZONE" if rate < 0.10 else "🟡 Monitoring"
            logger.info(
                "%s | Progress %d/%d | Rate: %.1f%% | Approved: %d | Rejected: %d",
                status, i, n_requests, rate * 100, approved, rejected,
            )

        time.sleep(interval)

    logger.info("═" * 60)
    logger.info("SELESAI | Final Approval Rate: %.1f%%", approved / (approved + rejected + 1e-9) * 100)


def run_mixed_simulation(n_requests: int, interval: float) -> None:
    """Simulasi campuran: normal lalu transisi ke krisis untuk demo alerting."""
    normal_count = n_requests // 2
    crisis_count = n_requests - normal_count

    logger.info("MODE MIXED: %d normal + %d crisis requests", normal_count, crisis_count)

    run_normal_simulation(normal_count, interval)
    logger.info("\n🔄 Beralih ke mode krisis untuk memicu alert...\n")
    run_crisis_simulation(crisis_count, interval)


# ─── Entry Point ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference traffic simulator untuk Credit Risk API")
    parser.add_argument(
        "--mode", choices=["normal", "crisis", "mixed"],
        default="normal",
        help="Mode simulasi: normal, crisis, atau mixed (default: normal)"
    )
    parser.add_argument("--requests", type=int, default=200, help="Jumlah total request (default: 200)")
    parser.add_argument("--interval", type=float, default=0.1, help="Interval antar request dalam detik (default: 0.1)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not check_api_health():
        logger.error("API tidak dapat dijangkau. Pastikan 3.prometheus_exporter.py berjalan.")
        sys.exit(1)

    simulators = {
        "normal": run_normal_simulation,
        "crisis": run_crisis_simulation,
        "mixed":  run_mixed_simulation,
    }

    simulators[args.mode](n_requests=args.requests, interval=args.interval)
