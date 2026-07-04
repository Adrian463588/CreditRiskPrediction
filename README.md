# Credit Risk Prediction MLOps System 🏦🤖
### Sistem Prediksi Kelayakan Kredit & Gagal Bayar (End-to-End MLOps)

**Dikerjakan oleh:** Adrian Syah Abidin  
**Dicoding Username:** adriansyah0904  
**Kelas:** Membangun Sistem Machine Learning (MSML)  
**Tingkat Kelulusan:** Advanced (Bintang 5 / Nilai A)

---

## 🔗 Tautan Repositori Utama & Registri

* **GitHub Repository:** [https://github.com/Adrian463588/CreditRiskPrediction.git](https://github.com/Adrian463588/CreditRiskPrediction.git)
* **DagsHub MLflow Tracking:** [https://dagshub.com/Adrian463588/CreditRiskPrediction](https://dagshub.com/Adrian463588/CreditRiskPrediction)
* **Docker Hub Repository:** [https://hub.docker.com/repository/docker/adrian090402/credit-risk-model/general](https://hub.docker.com/repository/docker/adrian090402/credit-risk-model/general)

---

## 📁 Struktur Repositori Proyek

Repositori diatur dengan struktur modular dan rapi sesuai ketentuan PRD dan Brief Dicoding:

```text
CreditRiskPrediction/
├── .github/workflows/
│   ├── preprocessing_ci.yml              # CI untuk Preprocessing Otomatis (Cron & Push)
│   └── ci_pipeline.yml                   # CI untuk Train, Log ke DagsHub, & Build Docker
│
├── Eksperimen_SML_AdrianSyahAbidin/       # [Kriteria 1] Preprocessing & EDA
│   ├── .github/workflows/
│   │   └── preprocessing_ci.yml          # Salinan CI Preprocessing
│   ├── credit_risk_raw/                  # Dataset mentah nasabah (raw)
│   └── preprocessing/
│       ├── Eksperimen_AdrianSyahAbidin.ipynb # Notebook Google Colab lengkap hasil eksekusi
│       ├── automate_AdrianSyahAbidin.py  # Skrip preprocessing otomatis (DRY)
│       └── credit_risk_preprocessing/    # Dataset bersih (siap latih)
│
├── Workflow-CI/                           # [Kriteria 3] MLProject & CI
│   ├── .github/workflows/
│   │   └── ci_pipeline.yml               # Salinan CI Pipeline
│   └── MLProject/
│       ├── MLProject                     # Spesifikasi entrypoint MLProject
│       ├── conda.yaml                    # Environment dependensi conda
│       ├── Dockerfile                    # Dockerfile optimized (python:3.10-slim)
│       ├── modelling.py                  # Skrip latih otomatis MLProject
│       └── credit_risk_preprocessing/    # Dataset bersih input training
│
└── SMSML_AdrianSyahAbidin/                # [ZIP Submission Final]
    ├── Eksperimen_SML_AdrianSyahAbidin.txt # Tautan Repositori Preprocessing (Public)
    ├── Workflow-CI.txt                   # Tautan Repositori CI/CD (Public)
    │
    ├── Membangun_model/                   # [Kriteria 2] Model Building
    │   ├── modelling.py                  # Autolog baseline training
    │   ├── modelling_tuning.py           # Manual log + Hyperparameter Tuning + DagsHub
    │   ├── requirements.txt              # Pinned requirements scikit-learn==1.5.2
    │   ├── DagsHub.txt                   # Tautan DagsHub online
    │   ├── screenshoot_dashboard.png     # Screenshot DagsHub Dashboard
    │   ├── screenshoot_artifak.png       # Screenshot DagsHub Artifacts
    │   └── credit_risk_preprocessing/    # Dataset bersih input training
    │
    └── Monitoring dan Logging/            # [Kriteria 4] Serving & Monitoring
        ├── 1.bukti_serving.png           # Bukti FastAPI Swagger Docs aktif
        ├── 2.prometheus.yml              # Konfigurasi Prometheus scraper
        ├── 3.prometheus_exporter.py      # Serving FastAPI + 10 Metrik Prometheus
        ├── 4.bukti monitoring Prometheus/ # Bukti tangkapan layar Prometheus (3 metrik)
        ├── 5.bukti monitoring Grafana/    # Bukti tangkapan layar Dasbor Grafana (10 metrik)
        ├── 6.bukti alerting Grafana/     # Bukti tangkapan layar Alerts (Firing)
        ├── 7.Inference.py                # Simulator traffic nasabah (mixed/crisis)
        ├── alerting_rules.yml            # 3 Aturan Alert Prometheus (for: 0s)
        ├── docker-compose.yml            # Orkestrasi stack monitoring lokal
        └── grafana/                      # Provisioning dashboard & datasource
```

---

## ⚡ Detail Implementasi MLOps & DevSecOps

### 1. Data Preprocessing (Kriteria 1)
* Pembersihan data historis nasabah (*Credit Risk Dataset*): imputasi *missing values* numerik & kategorikal, standard scaling pada variabel pendapatan (`person_income`) & jumlah pinjaman (`loan_amnt`), serta encoding status kepemilikan rumah (`person_home_ownership`).
* Otomatisasi melalui [automate_AdrianSyahAbidin.py](file:///d:/Dicoding/MembangunSistemMachineLearning/ProyekAkhir/CreditRiskPrediction/Eksperimen_SML_AdrianSyahAbidin/preprocessing/automate_AdrianSyahAbidin.py) dan diintegrasikan dengan GitHub Actions workflow [preprocessing_ci.yml](file:///d:/Dicoding/MembangunSistemMachineLearning/ProyekAkhir/CreditRiskPrediction/.github/workflows/preprocessing_ci.yml).

### 2. Model Training & Tracking (Kriteria 2)
* Pelatihan model prediktif risiko kredit menggunakan algoritma **Random Forest Classifier** dengan hyperparameter tuning.
* Pelacakan model dihubungkan secara online ke **DagsHub** via MLflow.
* logging manual mengunggah dua buah plot visual esensial bisnis: **ROC-AUC Curve** dan **Feature Importance** ke MLflow artifact registry.

### 3. Workflow CI & Docker Build (Kriteria 3)
* Pembuatan struktur MLProject (`conda.yaml`, `MLProject`, `modelling.py`).
* Alur kerja [ci_pipeline.yml](file:///d:/Dicoding/MembangunSistemMachineLearning/ProyekAkhir/CreditRiskPrediction/.github/workflows/ci_pipeline.yml) otomatis dipicu pada setiap push ke branch `main` serta jadwal berkala (Cron job **seminggu sekali setiap hari Minggu jam 23:00 WIB / 16:00 UTC**).
* **Docker Optimization**: Menggunakan custom Dockerfile multi-stage berbasis `python:3.10-slim` untuk mengurangi ukuran image secara drastis dari **~3 GB menjadi ~500 MB** dan mempercepat waktu build dari **30 menit menjadi hanya ~3 menit**.
* Keamanan terjamin dengan tidak ada hardcoded credential, seluruhnya menggunakan enkripsi **GitHub Secrets**.

### 4. Serving & Monitoring (Kriteria 4)
* **API serving** dibuat menggunakan FastAPI yang mengekspos **10 metrik gabungan (bisnis & sistem)** pada endpoint `/metrics`.
* Mengimplementasikan logika **sliding window (50 request terakhir)** pada metrik `loan_approval_rate` agar dasbor merespons perubahan kondisi data drift secara real-time.
* Dasbor Grafana provisioned secara otomatis menggunakan nama username Dicoding Anda: **`adriansyah0904`**.
* Konfigurasi **3 alert rules instan (`for: 0s`)** agar alert langsung berstatus **Firing (Merah)** di Grafana ketika persetujuan pinjaman turun di bawah 10% akibat simulasi krisis.

---

## 🛠️ Cara Menjalankan Stack Serving & Monitoring Lokal

1. **Jalankan Docker Compose**:
   ```bash
   cd "SMSML_AdrianSyahAbidin/Monitoring dan Logging"
   docker-compose up -d --build
   ```
2. **Kirim Traffic Simulasi (Mixed Mode)**:
   ```bash
   pip install requests
   python 7.Inference.py --mode mixed --requests 200 --interval 0.05
   ```
3. **Akses Layanan**:
   * API Serving Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
   * Prometheus UI: [http://localhost:9090/alerts](http://localhost:9090/alerts)
   * Grafana Dashboard & Alerts: [http://localhost:3000/alerting/list](http://localhost:3000/alerting/list)
