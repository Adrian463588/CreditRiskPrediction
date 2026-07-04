# Credit Risk MLOps System 🏦🤖
> **Sistem MLOps End-to-End untuk Prediksi Kelayakan Kredit & Pencegahan Gagal Bayar (Loan Default Prediction)**

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-v2.19.0-blue?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue?logo=docker&logoColor=white)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.111-green?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Prometheus](https://img.shields.io/badge/Prometheus-v2.53.0-orange?logo=prometheus&logoColor=white)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-v11.1.0-orange?logo=grafana&logoColor=white)](https://grafana.com/)
[![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-brightgreen?logo=github-actions&logoColor=white)](https://github.com/features/actions)

---

## 🤵 Profil Mahasiswa
* **Nama Lengkap:** Adrian Syah Abidin
* **Dicoding Username:** adriansyah0904
* **Kelas:** Membangun Sistem Machine Learning (MSML)
* **Tingkat Kelulusan:** Advanced (Bintang 5 / Nilai A)

---

## 🔗 Tautan Penting Proyek
* **Monorepo (Backup & Docs):** [https://github.com/Adrian463588/CreditRiskPrediction.git](https://github.com/Adrian463588/CreditRiskPrediction.git)
* **Kriteria 1 Repository (Preprocessing):** [https://github.com/Adrian463588/Eksperimen_SML_AdrianSyahAbidin](https://github.com/Adrian463588/Eksperimen_SML_AdrianSyahAbidin)
* **Kriteria 3 Repository (CI/CD Pipeline):** [https://github.com/Adrian463588/Workflow-CI_SMSL](https://github.com/Adrian463588/Workflow-CI_SMSL)
* **DagsHub Workspace:** [https://dagshub.com/Adrian463588/CreditRiskPrediction](https://dagshub.com/Adrian463588/CreditRiskPrediction)
* **Docker Hub Registry:** [https://hub.docker.com/repository/docker/adrian090402/credit-risk-model/general](https://hub.docker.com/repository/docker/adrian090402/credit-risk-model/general)

---

## 📖 Deskripsi Proyek
Proyek ini mengimplementasikan sistem **Machine Learning Operations (MLOps)** secara penuh untuk memprediksi kelayakan pengajuan kredit nasabah. Sistem ini mendeteksi potensi **gagal bayar (default)** pinjaman berdasarkan data demografis dan finansial nasabah, membantu institusi keuangan meminimalisir rasio *Non-Performing Loan (NPL)*.

Sistem dirancang modular untuk memisahkan siklus rekayasa data, eksperimen pemodelan, pipeline integrasi berlanjut (CI), serving model, dan observabilitas sistem secara real-time.

---

## 📐 Arsitektur Sistem MLOps

```text
                                  [ DagsHub MLflow Server ]
                                             ▲
                                             │ (Tracking API & Artifacts)
                                             │
[ GitHub Push ] ──► [ GitHub Actions ] ──► [ MLProject Run ] ──► [ Build Docker ] ──► [ Push to Docker Hub ]
                          │                                           │ (Multi-stage Slim)
                          ▼                                           ▼
             [ preprocessing_ci.yml ]                        [ ci_pipeline.yml ]
                          │
                          ▼
           [ automate_AdrianSyahAbidin.py ]
                          │
                          ▼
            [ Cleaned Dataset (CSV) ] ──► [ Local Container Stack ] ──► [ Prometheus ] ◄── [ Grafana Dashboard ]
                                          * FastAPI predict             * Scrape metrics   * 3 Alerting Rules
                                          * 10 Metrics Exporter
```

---

## 📁 Struktur Direktori Repositori

```text
CreditRiskPrediction/
├── .github/workflows/
│   ├── preprocessing_ci.yml                # CI untuk pemrosesan data otomatis (Weekly & Push)
│   └── ci_pipeline.yml                     # CI untuk training model, tracking, & Docker build
│
├── Eksperimen_SML_AdrianSyahAbidin/         # [Kriteria 1] Fase Data Engineering
│   ├── .github/workflows/
│   │   └── preprocessing_ci.yml            # Workflow CI Preprocessing
│   ├── credit_risk_raw/                    # Dataset mentah nasabah (credit_risk_dataset.csv)
│   └── preprocessing/
│       ├── Eksperimen_AdrianSyahAbidin.ipynb # Notebook lengkap hasil eksekusi Colab (EDA & Preprocessing)
│       ├── automate_AdrianSyahAbidin.py    # Modul preprocessing otomatis (DRY)
│       └── credit_risk_preprocessing/      # Output dataset bersih hasil preprocessing
│
├── Workflow-CI/                             # [Kriteria 3] Alur Kerja CI & Dockerization
│   ├── .github/workflows/
│   │   └── ci_pipeline.yml                 # Workflow CI Pipeline
│   └── MLProject/
│       ├── MLProject                       # File manifes deskripsi entrypoint MLflow
│       ├── conda.yaml                      # Manifes environment conda untuk reproducibilitas
│       ├── Dockerfile                      # Dockerfile serving (Multi-stage, python:3.10-slim)
│       ├── modelling.py                    # Script training otomatis untuk runner CI
│       └── credit_risk_preprocessing/      # Salinan dataset bersih untuk input training
│
└── SMSML_AdrianSyahAbidin/                  # [ZIP Submission Final]
    ├── Eksperimen_SML_AdrianSyahAbidin.txt   # File teks berisi tautan ke repositori kriteria 1
    ├── Workflow-CI.txt                     # File teks berisi tautan ke repositori kriteria 3
    │
    ├── Membangun_model/                     # [Kriteria 2] Fase Pemodelan (Model Building)
    │   ├── modelling.py                    # Skrip training baseline (autolog enabled)
    │   ├── modelling_tuning.py             # Hyperparameter tuning + DagsHub + Manual logging
    │   ├── requirements.txt                # Dependensi training model
    │   ├── DagsHub.txt                     # Tautan Workspace DagsHub
    │   ├── screenshoot_dashboard.png       # Bukti dashboard eksperimen di DagsHub
    │   ├── screenshoot_artifak.png         # Bukti visualisasi kurva ROC-AUC di DagsHub artifacts
    │   └── credit_risk_preprocessing/      # Dataset bersih hasil preprocessing
    │
    └── Monitoring dan Logging/              # [Kriteria 4] Sistem SRE & Observability
        ├── 1.bukti_serving.png             # Bukti tangkapan layar Swagger API aktif
        ├── 2.prometheus.yml                # Konfigurasi Prometheus scraper
        ├── 3.prometheus_exporter.py        # API FastAPI Serving + 10 Metrik Prometheus
        ├── 4.bukti monitoring Prometheus/   # Bukti tangkapan layar Prometheus (3 metrik)
        ├── 5.bukti monitoring Grafana/      # Bukti dasbor Grafana (10 metrik terpusat)
        ├── 6.bukti alerting Grafana/       # Bukti alert rules berstatus Firing (Merah)
        ├── 7.Inference.py                  # Skrip simulator pengiriman traffic nasabah
        ├── alerting_rules.yml              # 3 Aturan Alert Prometheus (for: 0s)
        ├── docker-compose.yml              # Orkestrasi container stack monitoring
        └── grafana/                        # Provisioning file dashboard & datasource
```

---

## 🛠️ Implementasi Komponen Sistem MLOps

### 1. Data Engineering & Preprocessing (Kriteria 1)
* **Pembersihan Data:** Mengatasi *missing values* pada data finansial nasabah, melakukan normalisasi/scaling menggunakan `StandardScaler` pada fitur numerik (`person_income` dan `loan_amnt`), serta menerapkan label/one-hot encoding pada variabel kategori (`person_home_ownership` dan `loan_intent`).
* **Workflow CI Preprocessing:** Terintegrasi di `.github/workflows/preprocessing_ci.yml` yang terpicu otomatis setiap push ke branch `main` dan terjadwal (Cron job) seminggu sekali.

### 2. Model Development & Tracking (Kriteria 2)
* **Algoritma & Tuning:** Menggunakan **Random Forest Classifier** yang dioptimasi dengan pencarian hyperparameter (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`) untuk mencegah overfitting (Akurasi Uji: **96.30%** dengan selisih train-test di bawah **0.5%**).
* **Tracking & Logging:** Terhubung ke server remote **DagsHub MLflow**. Logging dilakukan secara manual (bukan autolog) dan mengunggah 2 artefak visual utama: **ROC-AUC Curve** dan **Feature Importance Plot** ke DagsHub Artifact Registry.

### 3. Continuous Integration & Dockerization (Kriteria 3)
* **MLflow Project:** Struktur MLProject dikemas modular dengan conda environment ter-pin untuk reproduksibilitas penuh di runner GitHub Actions.
* **Optimasi Docker Image:** Pipeline CI (`ci_pipeline.yml`) dikonfigurasi untuk memicu pembuatan Docker Image menggunakan custom Dockerfile multi-stage berbasis `python:3.10-slim`.
  * **Build Time:** Turun dari ~30 menit menjadi **~3 menit**.
  * **Image Size:** Turun dari ~3 GB menjadi **~500 MB**.
  * **DevSecOps:** Menghapus paket OS tidak perlu (Mesa, X11, LLVM), menjalankan container dengan non-root user (`mluser:1000`), dan mengimplementasikan healthcheck berkala di endpoint `/ping`.

### 4. Serving, Observability & Alerting (Kriteria 4)
* **FastAPI Serving Exporter:** Menyediakan REST API serving lokal dengan port 8000. Menggunakan skema **sliding window (last 50 requests)** pada metrik persetujuan pinjaman agar data drift/anomali terdeteksi dinamis.
* **10 Metrik Terintegrasi:** Mengekspos metrik bisnis dan sistem terpadu (seperti `loan_approval_rate`, `model_confidence_score`, `api_latency_seconds`, `system_cpu_usage_percent`, `data_drift_score`, dll.).
* **Grafana Dashboard:** Dasbor otomatis ter-provisioning dengan nama username Dicoding Anda: **`adriansyah0904`**.
* **Grafana Alerting:** Konfigurasi 3 Alert Rules instan (`for: 0s`) untuk mendeteksi:
  1. *Approval Rate Drop* (< 10%) - Status langsung berubah **Firing** saat simulator krisis dijalankan.
  2. *High API Latency* (> 2s).
  3. *High Error Rate* (> 5%).

---

## 🚀 Panduan Menjalankan Stack Secara Lokal

### Prasyarat
* Docker Desktop terinstal dan sedang aktif.
* Python 3.10 atau 3.12 terinstal secara lokal.

### 1. Jalankan Stack Monitoring
Buka terminal/PowerShell di direktori proyek, masuk ke subfolder monitoring, dan jalankan Docker Compose:
```bash
cd "SMSML_AdrianSyahAbidin/Monitoring dan Logging"
docker-compose up -d --build
```
Verifikasi bahwa ketiga kontainer (`credit-risk-api`, `prometheus`, dan `grafana`) telah berstatus **Up (Running)** melalui perintah `docker ps`.

### 2. Jalankan Simulasi Traffic Nasabah
Simulasi ini mengirimkan 200 request nasabah secara beruntun (100 normal disetujui, diikuti 100 krisis ditolak) untuk mendemonstrasikan perubahan metrik di dasbor Grafana dan memicu status alert Firing secara instan:
```bash
pip install requests
python 7.Inference.py --mode mixed --requests 200 --interval 0.05
```

### 3. Akses Halaman Web Layanan
* **FastAPI Docs (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs) (Endpoint `/predict` dan `/metrics`)
* **Prometheus Alerts UI:** [http://localhost:9090/alerts](http://localhost:9090/alerts) (Status target scraping & rule evaluasi)
* **Grafana Dashboard:** [http://localhost:3000](http://localhost:3000) (Navigasikan ke Dashboard `adriansyah0904` untuk memantau grafik metrik real-time dan list alert Firing).
