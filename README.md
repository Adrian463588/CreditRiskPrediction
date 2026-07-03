# Credit Risk MLOps System
### Prediksi Kelayakan Kredit (Loan Default Prediction)

**Dikerjakan oleh:** Adrian Syah Abidin  
**Dicoding Username:** adriansyah0904  
**Kelas:** Membangun Sistem Machine Learning (MSML)

---

## Deskripsi Proyek

Sistem MLOps end-to-end untuk memprediksi apakah seorang nasabah akan **gagal bayar (default)** pinjaman atau tidak, berdasarkan profil finansial dan demografis mereka. Proyek ini dirancang untuk bank dan institusi keuangan dalam meminimalisir risiko kredit bermasalah (NPL).

### Dataset
- **Sumber:** [Credit Risk Dataset - Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- **Target Variable:** `loan_status` (0 = Non-default, 1 = Default)
- **Fitur Utama:** person_income, loan_amnt, person_home_ownership, loan_intent, dll.

---

## Struktur Repositori

```
CreditRiskPrediction/
├── Eksperimen_SML_AdrianSyahAbidin/      # Kriteria 1: Data Preprocessing
│   ├── .github/workflows/               # GitHub Actions CI Preprocessing
│   ├── namadataset_raw/                 # Raw dataset (credit_risk_dataset.csv)
│   └── preprocessing/
│       ├── Eksperimen_AdrianSyahAbidin.ipynb  # Notebook Colab (EDA + Preprocessing)
│       ├── automate_AdrianSyahAbidin.py       # Automated preprocessing script
│       └── namadataset_preprocessing/         # Output: preprocessed data
│
├── Membangun_model/                      # Kriteria 2: Model Training
│   ├── modelling.py                     # Basic MLflow autolog
│   ├── modelling_tuning.py              # DagsHub + Hyperparameter tuning + manual log
│   ├── namadataset_preprocessing/       # Input: preprocessed data
│   └── requirements.txt
│
├── Workflow-CI/                          # Kriteria 3: CI Pipeline
│   ├── .github/workflows/ci_pipeline.yml # GitHub Actions: train + docker push
│   └── MLProject/
│       ├── MLProject                    # MLflow Project definition
│       ├── conda.yaml                   # Conda environment
│       ├── modelling.py                 # Training entry point
│       └── namadataset_preprocessing/
│
├── Monitoring_dan_Logging/               # Kriteria 4: Serving & Monitoring
│   ├── 2.prometheus.yml
│   ├── 3.prometheus_exporter.py
│   ├── 7.Inference.py
│   ├── docker-compose.yml
│   ├── grafana/
│   ├── 4.bukti_monitoring_Prometheus/
│   ├── 5.bukti_monitoring_Grafana/
│   └── 6.bukti_alerting_Grafana/
│
└── SMSML_AdrianSyahAbidin/             # Paket Submission Final
```

---

## MLOps Architecture

```
[Raw Data] → [GitHub Actions: automate.py] → [Clean Dataset]
                                                    ↓
[Colab: Eksperimen Notebook] → [DagsHub MLflow Tracking] ← [modelling_tuning.py]
                                                    ↓
              [GitHub Actions CI Pipeline] → [mlflow run] → [Docker Build] → [Docker Hub]
                                                    ↓
         [Docker Pull / mlflow serve] → [FastAPI + Prometheus Exporter]
                                                    ↓
                              [Prometheus Server] → [Grafana Dashboard + 3 Alerts]
```

---

## Teknologi yang Digunakan

| Komponen | Teknologi |
|---|---|
| Eksperimen | Jupyter Notebook (Google Colab) |
| Model | RandomForestClassifier / XGBClassifier |
| Tracking | MLflow 2.19.0 + DagsHub |
| CI/CD | GitHub Actions |
| Containerization | Docker + Docker Hub |
| Serving | FastAPI / MLflow Serve |
| Monitoring | Prometheus + Grafana |
| Language | Python 3.12.7 |

---

## Cara Menjalankan

### 1. Preprocessing (Lokal)
```bash
cd Eksperimen_SML_AdrianSyahAbidin/preprocessing
pip install -r requirements.txt
python automate_AdrianSyahAbidin.py
```

### 2. Training (Google Colab)
Buka `Eksperimen_AdrianSyahAbidin.ipynb` di Google Colab, connect ke Google Drive, dan jalankan semua cell.

### 3. CI Pipeline
Push ke branch `main` untuk trigger GitHub Actions CI.

### 4. Monitoring (Lokal)
```bash
cd Monitoring_dan_Logging
docker-compose up -d
python 3.prometheus_exporter.py &
python 7.Inference.py
```
Akses Grafana di `http://localhost:3000`  
Akses Prometheus di `http://localhost:9090`
