"""
3.prometheus_exporter.py
=========================
FastAPI serving aplikasi untuk model Credit Risk Prediction.
Mengekspos 10 metrik Prometheus (bisnis + sistem) di endpoint /metrics.

Cara menjalankan:
    pip install fastapi uvicorn prometheus-client psutil numpy scikit-learn
    python 3.prometheus_exporter.py

Endpoints:
    POST /predict       — Prediksi loan default dari fitur nasabah
    GET  /metrics       — Prometheus metrics endpoint
    GET  /health        — Health check
"""

import os
import time
import threading
import logging
from typing import Dict, Any

import numpy as np
import psutil
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
    multiprocess,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── Prometheus Registry & Metrik ─────────────────────────────────────────────
registry = CollectorRegistry()

# Metrik 1: loan_approval_rate — Rasio pengajuan kredit yang disetujui (non-default)
LOAN_APPROVAL_RATE = Gauge(
    "loan_approval_rate",
    "Rasio nasabah yang diprediksi tidak default (disetujui)",
    registry=registry,
)

# Metrik 2: model_confidence_score — Rata-rata skor probabilitas prediksi model
MODEL_CONFIDENCE_SCORE = Gauge(
    "model_confidence_score",
    "Rata-rata skor kepercayaan model (probabilitas prediksi)",
    registry=registry,
)

# Metrik 3: loan_default_prediction_count — Jumlah prediksi default yang terdeteksi
LOAN_DEFAULT_COUNT = Counter(
    "loan_default_prediction_total",
    "Total prediksi loan default (status=1) yang terdeteksi",
    registry=registry,
)

# Metrik 4: request_count — Total permintaan API
REQUEST_COUNT = Counter(
    "api_request_total",
    "Total permintaan API yang diterima",
    ["endpoint", "method", "status"],
    registry=registry,
)

# Metrik 5: api_latency — Latensi respons API prediksi (histogram)
API_LATENCY = Histogram(
    "api_latency_seconds",
    "Latensi endpoint prediksi dalam detik",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
    registry=registry,
)

# Metrik 6: cpu_usage — Penggunaan CPU sistem
CPU_USAGE = Gauge(
    "system_cpu_usage_percent",
    "Persentase penggunaan CPU sistem",
    registry=registry,
)

# Metrik 7: memory_usage — Penggunaan memori sistem
MEMORY_USAGE = Gauge(
    "system_memory_usage_percent",
    "Persentase penggunaan memori sistem",
    registry=registry,
)

# Metrik 8: http_errors — Jumlah error HTTP 5xx
HTTP_ERRORS = Counter(
    "http_error_total",
    "Total error HTTP 5xx yang terjadi",
    ["error_type"],
    registry=registry,
)

# Metrik 9: active_connections — Koneksi aktif saat ini
ACTIVE_CONNECTIONS = Gauge(
    "api_active_connections",
    "Jumlah koneksi API yang sedang aktif",
    registry=registry,
)

# Metrik 10: data_drift_score — Skor drift data (simulasi berbasis distribusi)
DATA_DRIFT_SCORE = Gauge(
    "data_drift_score",
    "Skor drift data input (0=tidak drift, 1=drift penuh)",
    registry=registry,
)


# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Risk Prediction API",
    description="MLOps serving API untuk prediksi risiko gagal bayar kredit",
    version="1.0.0",
)

# ─── State Internal ───────────────────────────────────────────────────────────
_state = {
    "total_predictions": 0,
    "total_approvals":   0,
    "total_prob_sum":    0.0,
    "baseline_income":   50000.0,  # Digunakan untuk kalkulasi drift
}


# ─── Model Stub (ganti dengan mlflow.pyfunc.load_model() setelah training) ───

class MockCreditRiskModel:
    """
    Mock model untuk keperluan demonstrasi monitoring.
    Ganti dengan model nyata dari MLflow:
        import mlflow.pyfunc
        self.model = mlflow.pyfunc.load_model("models:/CreditRiskRandomForest/Production")
    """

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Simulasi prediksi berdasarkan fitur input."""
        income_ratio  = features[:, 0] / (features[:, 2] + 1)  # income / loan_amnt
        default_prob  = np.clip(0.15 + 0.3 * (1 - income_ratio.clip(0, 1)), 0.05, 0.95)
        return (default_prob > 0.5).astype(int), default_prob

model = MockCreditRiskModel()


# ─── Schema Request / Response ────────────────────────────────────────────────
class LoanApplication(BaseModel):
    """Schema data pengajuan pinjaman nasabah."""
    person_income:         float = Field(..., gt=0, description="Pendapatan tahunan (Rp)")
    person_age:            int   = Field(..., ge=18, le=100, description="Usia peminjam")
    loan_amnt:             float = Field(..., gt=0, description="Jumlah pinjaman (Rp)")
    loan_int_rate:         float = Field(10.0, ge=0, description="Suku bunga (%)")
    person_emp_length:     float = Field(2.0, ge=0, description="Lama kerja (tahun)")
    loan_percent_income:   float = Field(0.1, ge=0, le=1, description="Rasio pinjaman/pendapatan")
    cb_person_cred_hist_length: int = Field(3, ge=0, description="Riwayat kredit (tahun)")


class PredictionResponse(BaseModel):
    prediction:        int
    default_risk:      str
    confidence_score:  float
    loan_status_label: str
    message:           str


# ─── Background: Update Sistem Metrik ────────────────────────────────────────
def _update_system_metrics():
    """Update metrik CPU dan Memory setiap 5 detik dalam background thread."""
    while True:
        CPU_USAGE.set(psutil.cpu_percent(interval=1))
        MEMORY_USAGE.set(psutil.virtual_memory().percent)
        time.sleep(5)


threading.Thread(target=_update_system_metrics, daemon=True).start()


# ─── Middleware Tracking ───────────────────────────────────────────────────────
@app.middleware("http")
async def track_request_metrics(request: Request, call_next):
    """Middleware untuk melacak request count dan active connections."""
    ACTIVE_CONNECTIONS.inc()
    start_time = time.time()
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        REQUEST_COUNT.labels(
            endpoint=request.url.path,
            method=request.method,
            status=str(response.status_code)
        ).inc()
        if request.url.path == "/predict":
            API_LATENCY.observe(duration)
        if response.status_code >= 500:
            HTTP_ERRORS.labels(error_type="5xx").inc()
        return response
    except Exception as exc:
        HTTP_ERRORS.labels(error_type="unhandled").inc()
        raise exc
    finally:
        ACTIVE_CONNECTIONS.dec()


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "credit-risk-prediction"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_loan_default(application: LoanApplication) -> PredictionResponse:
    """
    Prediksi risiko gagal bayar untuk pengajuan pinjaman.
    Mengupdate metrik Prometheus secara real-time.
    """
    features = np.array([[
        application.person_income,
        application.person_age,
        application.loan_amnt,
        application.loan_int_rate,
        application.person_emp_length,
        application.loan_percent_income,
        application.cb_person_cred_hist_length,
    ]])

    try:
        prediction, probability = model.predict(features)
        pred_label   = int(prediction[0])
        confidence   = float(probability[0])
        is_default   = pred_label == 1

        # Update internal state
        _state["total_predictions"] += 1
        _state["total_prob_sum"]    += confidence
        if not is_default:
            _state["total_approvals"] += 1

        # Update Prometheus Metrics
        approval_rate = _state["total_approvals"] / _state["total_predictions"]
        avg_confidence = _state["total_prob_sum"] / _state["total_predictions"]

        LOAN_APPROVAL_RATE.set(approval_rate)
        MODEL_CONFIDENCE_SCORE.set(avg_confidence)

        if is_default:
            LOAN_DEFAULT_COUNT.inc()

        # Hitung drift score: perbedaan distribusi income dari baseline
        drift = abs(application.person_income - _state["baseline_income"]) / _state["baseline_income"]
        DATA_DRIFT_SCORE.set(min(drift, 1.0))

        return PredictionResponse(
            prediction=pred_label,
            default_risk="HIGH" if is_default else "LOW",
            confidence_score=round(confidence, 4),
            loan_status_label="Default" if is_default else "Non-Default",
            message=(
                "⚠️  Pengajuan berisiko tinggi — kredit ditolak"
                if is_default else
                "✅ Pengajuan disetujui — risiko rendah"
            ),
        )

    except Exception as exc:
        HTTP_ERRORS.labels(error_type="prediction_error").inc()
        logger.error("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediksi gagal: {str(exc)}")


@app.get("/metrics", response_class=PlainTextResponse, tags=["Monitoring"])
def prometheus_metrics():
    """Endpoint Prometheus metrics — /metrics."""
    return PlainTextResponse(
        generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST,
    )


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PORT = int(os.getenv("PORT", 8000))
    logger.info("🚀 Starting Credit Risk Prediction API on port %d", PORT)
    logger.info("   Metrics : http://localhost:%d/metrics", PORT)
    logger.info("   API Docs: http://localhost:%d/docs", PORT)
    uvicorn.run("3.prometheus_exporter:app", host="0.0.0.0", port=PORT, reload=False)
