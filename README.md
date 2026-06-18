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
│   ├── raw/            ← creditcard.csv để đây
│   └── processed/
├── src/
│   ├── data_processing/
│   │   ├── loader.py
│   │   ├── preprocessor.py
│   │   └── feature_engineer.py
│   ├── models/
│   │   ├── base_model.py
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   └── evaluator.py
│   ├── api/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── predict.py
│   └── utils/
│       ├── config.py
│       └── logger.py
├── templates/
│   └── index.html      ← Dashboard UI
├── tests/
├── train.py            ← Chạy để train model
├── .env
└── requirements.txt
```

## PyCharm Tips
- Mở thư mục `fraud_detection/` làm project root
- Cấu hình Run Configuration: `uvicorn src.api.main:app --reload`
- Thêm `src/` vào **Sources Root** (chuột phải → Mark Directory As)
