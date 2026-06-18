# 🛡️ Fraud Detection System

Hệ thống phát hiện gian lận giao dịch thẻ tín dụng sử dụng ML + FastAPI dashboard.

## Stack
- **Model**: RandomForest / XGBoost + SMOTE
- **API**: FastAPI + Uvicorn
- **Dataset**: [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## Cài đặt

```bash
# 1. Tạo virtual environment trong PyCharm hoặc terminal
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 2. Cài thư viện
pip install -r requirements.txt
```

---

## Chạy lần đầu

### Bước 1 — Tải dataset
Tải file `creditcard.csv` từ Kaggle → đặt vào `data/raw/creditcard.csv`

### Bước 2 — Train model
```bash
# RandomForest (nhanh hơn)
python train.py --model rf

# hoặc XGBoost (chính xác hơn)
python train.py --model xgb
```
Model sẽ được lưu vào `models/saved/fraud_model.pkl`

### Bước 3 — Start server
```bash
uvicorn src.api.main:app --reload --port 8000
```

### Bước 4 — Mở dashboard
Truy cập http://localhost:8000

---

## API Endpoints

| Method | URL | Mô tả |
|--------|-----|-------|
| GET | `/` | Dashboard UI |
| POST | `/predict` | Dự đoán giao dịch |
| GET | `/health` | Kiểm tra server |
| GET | `/docs` | Swagger UI tự động |

---

## Cấu trúc thư mục

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

## PyCharm Tips
- Mở thư mục `fraud_detection/` làm project root
- Cấu hình Run Configuration: `uvicorn src.api.main:app --reload`
- Thêm `src/` vào **Sources Root** (chuột phải → Mark Directory As)

