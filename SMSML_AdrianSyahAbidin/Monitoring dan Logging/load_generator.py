"""
load_generator.py
=================
Script untuk menghasilkan traffic ke Credit Risk API
sehingga semua 10 metrik Prometheus terisi dengan data.

Cara penggunaan:
    python load_generator.py

Pastikan API sudah berjalan di http://localhost:8000
(jalankan: docker compose up -d)
"""

import time
import random
import requests

API_URL = "http://localhost:8000"

# Payload nasabah normal (income >> loan) -> disetujui (approved)
APPROVED_PAYLOAD = {
    "person_income": 75000.0,
    "person_age": 30,
    "loan_amnt": 5000.0,
    "loan_int_rate": 8.5,
    "person_emp_length": 5.0,
    "loan_percent_income": 0.07,
    "cb_person_cred_hist_length": 4
}

# Payload nasabah berisiko (income << loan) -> ditolak (default)
REJECTED_PAYLOAD = {
    "person_income": 20000.0,
    "person_age": 25,
    "loan_amnt": 50000.0,
    "loan_int_rate": 20.0,
    "person_emp_length": 0.5,
    "loan_percent_income": 0.8,
    "cb_person_cred_hist_length": 1
}

# Payload untuk memicu HIGH LATENCY (person_age = 99)
LATENCY_TRIGGER_PAYLOAD = {
    "person_income": 60000.0,
    "person_age": 99,  # ← trigger: injeksi 3 detik latency
    "loan_amnt": 10000.0,
    "loan_int_rate": 10.0,
    "person_emp_length": 3.0,
    "loan_percent_income": 0.17,
    "cb_person_cred_hist_length": 3
}


def send_request(payload: dict, label: str):
    try:
        start = time.time()
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        elapsed = time.time() - start
        result = r.json()
        print(f"[{label}] status={r.status_code} | "
              f"prediction={result.get('loan_status_label')} | "
              f"latency={elapsed:.3f}s")
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Tidak bisa konek ke API. Pastikan docker compose sudah up.")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
    return True


def main():
    print("=" * 60)
    print("Credit Risk Load Generator")
    print(f"Target: {API_URL}")
    print("=" * 60)

    # Health check dulu
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        print(f"[HEALTH] {r.json()}")
    except Exception:
        print("[ERROR] API belum ready. Jalankan: docker compose up -d")
        print("         Tunggu ~30 detik lalu jalankan script ini lagi.")
        return

    print("\n📊 Phase 1: Mengisi metrik dasar (50 request normal)...")
    for i in range(50):
        payload = APPROVED_PAYLOAD.copy()
        # Variasikan income untuk menghasilkan data drift
        payload["person_income"] = random.uniform(40000, 100000)
        if not send_request(payload, f"NORMAL-{i+1:02d}"):
            return
        time.sleep(0.1)

    print("\n📊 Phase 2: Mengisi metrik default/rejection (30 request berisiko)...")
    for i in range(30):
        if not send_request(REJECTED_PAYLOAD, f"REJECTED-{i+1:02d}"):
            return
        time.sleep(0.1)

    print("\n📊 Phase 3: Trigger HIGH LATENCY (1 request dengan age=99)...")
    send_request(LATENCY_TRIGGER_PAYLOAD, "LATENCY-TRIGGER")

    print("\n📊 Phase 4: Mix traffic (50 request campuran)...")
    for i in range(50):
        payload = random.choice([APPROVED_PAYLOAD, REJECTED_PAYLOAD]).copy()
        payload["person_income"] = random.uniform(15000, 120000)
        payload["loan_amnt"] = random.uniform(1000, 80000)
        if not send_request(payload, f"MIX-{i+1:02d}"):
            return
        time.sleep(0.15)

    print("\n✅ Load generation selesai!")
    print("   Buka Prometheus: http://localhost:9090")
    print("   Buka Grafana:    http://localhost:3000")
    print("   Login Grafana:   admin / admin")
    print("\n   Metrik yang tersedia untuk screenshot Prometheus:")
    metrics = [
        "loan_approval_rate",
        "model_confidence_score",
        "loan_default_prediction_total",
        "api_request_total",
        "api_latency_seconds_bucket",
        "system_cpu_usage_percent",
        "system_memory_usage_percent",
        "http_error_total",
        "api_active_connections",
        "data_drift_score",
    ]
    for i, m in enumerate(metrics, 1):
        print(f"   {i:2d}. {m}")


if __name__ == "__main__":
    main()
