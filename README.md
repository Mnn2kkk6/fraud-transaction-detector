# 🛡️ Fraud Transaction Detector

> Hệ thống phát hiện gian lận giao dịch thẻ tín dụng sử dụng Machine Learning + FastAPI Dashboard

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

##  Giới thiệu

**Fraud Transaction Detector** là một ứng dụng web phát hiện gian lận trong giao dịch thẻ tín dụng theo thời gian thực. Hệ thống sử dụng mô hình **RandomForest** và **XGBoost** kết hợp kỹ thuật **SMOTE** để xử lý mất cân bằng dữ liệu, đạt **ROC-AUC 0.983** và **Recall 87.8%** trên tập test.

### Kết quả thực nghiệm

| Metric | RandomForest | XGBoost |
|--------|-------------|---------|
| ROC-AUC | 0.9836 | 0.9834 |
| PR-AUC | 0.8075 | 0.8201 |
| F1-Fraud | 0.5753 | 0.5891 |
| Recall (Fraud) | 87.8% | 88.8% |

---

##  Cấu trúc dự án

```
fraud_detection/
├── data/
│   ├── raw/                    ← creditcard.csv (Kaggle dataset)
│   └── processed/              ← Charts & processed outputs
├── notebooks/
│   ├── 01_eda.ipynb            ← Phân tích dữ liệu (EDA)
│   └── 02_model_experiments.ipynb  ← So sánh RandomForest vs XGBoost
├── src/
│   ├── __init__.py
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── loader.py           ← Đọc & split dataset
│   │   ├── preprocessor.py     ← Scale, SMOTE
│   │   └── feature_engineer.py ← Tạo feature mới
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py       ← Abstract base class
│   │   ├── random_forest.py    ← RF classifier
│   │   ├── xgboost_model.py    ← XGBoost classifier
│   │   └── evaluator.py        ← Metrics: Precision, Recall, AUC
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py             ← FastAPI app entry point
│   │   ├── schemas.py          ← Pydantic request/response
│   │   └── predict.py          ← Endpoint /predict
│   └── utils/
│       ├── __init__.py
│       ├── config.py           ← Đọc .env, hằng số
│       └── logger.py           ← Logging chuẩn
├── templates/
│   └── index.html              ← Dashboard UI
├── tests/
│   ├── test_predict.py         ← Unit tests
│   └── test_api.py             ← API endpoint tests
├── main.py                     ← Entry point (python main.py)
├── train.py                    ← Script train & lưu model
├── requirements.txt
├── .env
└── README.md
```

---

##  Cài đặt

### Yêu cầu
- Python 3.10+
- pip

### Các bước

```bash
# 1. Clone repository
git clone https://github.com/Mnn2kkk6/fraud-transaction-detector.git
cd fraud-transaction-detector

# 2. Tạo virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 3. Cài thư viện
pip install -r requirements.txt
```

---

## Chạy ứng dụng

### Bước 1 — Chuẩn bị dataset

Tải file `creditcard.csv` từ Kaggle:
-----> https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Đặt vào: `data/raw/creditcard.csv`

### Bước 2 — Train model

```bash
# RandomForest (nhanh hơn)
python train.py --model rf

# XGBoost (chính xác hơn, khuyến nghị)
python train.py --model xgb
```

Model được lưu tự động vào `models/saved/`

### Bước 3 — Start server

```bash
python main.py
```

Hoặc dùng uvicorn trực tiếp:

```bash
uvicorn src.api.main:app --reload --port 8000
```

### Bước 4 — Truy cập

| URL | Mô tả |
|-----|-------|
| http://localhost:8000 | Dashboard UI |
| http://localhost:8000/docs | Swagger API docs |
| http://localhost:8000/health | Health check |

---

## 🔌 API

### `POST /predict`

Nhận thông tin giao dịch, trả về kết quả phân tích.

**Request body:**
```json
{
  "Time": 0.0,
  "V1": -1.36, "V2": -0.07, "V3": 2.53,
  "V4": 1.38,  "V5": -0.34, "V6": 0.46,
  "V7": 0.24,  "V8": 0.09,  "V9": 0.36,
  "V10": 0.09, "V11": -0.55,"V12": -0.61,
  "V13": -0.99,"V14": -0.31,"V15": 1.47,
  "V16": -0.47,"V17": 0.21, "V18": 0.02,
  "V19": 0.40, "V20": 0.25, "V21": -0.02,
  "V22": 0.28, "V23": -0.11,"V24": 0.07,
  "V25": 0.13, "V26": -0.19,"V27": 0.13,
  "V28": -0.02,
  "Amount": 149.62
}
```

**Response:**
```json
{
  "is_fraud": false,
  "fraud_probability": 0.0312,
  "risk_level": "LOW",
  "model_used": "XGBClassifier"
}
```

| Field | Type | Mô tả |
|-------|------|-------|
| `is_fraud` | bool | `true` nếu phát hiện gian lận |
| `fraud_probability` | float | Xác suất gian lận (0.0 – 1.0) |
| `risk_level` | string | `LOW` / `MEDIUM` / `HIGH` |
| `model_used` | string | Tên model đang dùng |

---

##  Chạy tests

```bash
# Chạy toàn bộ test
pytest tests/ -v

# Chỉ test schema & feature engineering
pytest tests/test_predict.py -v

# Chỉ test API endpoints
pytest tests/test_api.py -v
```

---

## Kỹ thuật sử dụng

| Kỹ thuật | Mô tả |
|---------|-------|
| **SMOTE** | Oversampling để xử lý mất cân bằng dữ liệu (0.17% fraud) |
| **StandardScaler** | Chuẩn hóa cột `Amount` và `Time` |
| **Feature Engineering** | `log_amount`, `time_period`, interaction features V1×V2, V14×V12 |
| **RandomForest** | `n_estimators=100`, `class_weight=balanced` |
| **XGBoost** | `scale_pos_weight=577`, tối ưu cho imbalanced data |
| **Threshold tuning** | Mặc định 0.5, có thể chỉnh qua `.env` |

---

## Dataset

- **Nguồn:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Kích thước:** 284,807 giao dịch
- **Fraud:** 492 giao dịch (0.17%)
- **Features:** V1–V28 (PCA), Amount, Time
- **Không có missing values**

---

## Tech Stack

- **Backend:** FastAPI, Uvicorn
- **ML:** scikit-learn, XGBoost, imbalanced-learn
- **Data:** pandas, numpy
- **Frontend:** Jinja2, HTML/CSS/JS
- **Testing:** pytest

---

## License

MIT License © 2025
